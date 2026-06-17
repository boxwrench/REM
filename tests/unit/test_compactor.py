"""Unit tests for the memory compactor component."""

import threading
import time
import httpx
from filelock import FileLock

from rem.config import Settings
from rem.npu_client import NpuClient
from rem.memory.tiers import MemoryState, Turn, count_tokens
from rem.memory.compactor import should_compact, compact_once, run_background


def test_should_compact():
    """Asserts that should_compact triggers correctly based on token thresholds."""
    settings = Settings(compact_trigger_tokens=100)
    
    # Below threshold (90 tokens)
    state_below = MemoryState(
        turns=[Turn(role="user", content="x" * 360, turn_id=1, tokens=90)]
    )
    assert should_compact(state_below, settings) is False

    # Above threshold (110 tokens)
    state_above = MemoryState(
        turns=[Turn(role="user", content="x" * 440, turn_id=1, tokens=110)]
    )
    assert should_compact(state_above, settings) is True


def test_keep_recent_turns_protected(mock_npu):
    """Asserts that turns within the keep_recent_turns window are never compacted."""
    client = NpuClient()
    settings = Settings(keep_recent_turns=8, compact_span_turns=6)
    
    # State has exactly 8 turns (all protected)
    turns = [
        Turn(role="user", content=f"Turn {i}", turn_id=i, tokens=10)
        for i in range(1, 9)
    ]
    state = MemoryState(turns=turns)
    
    res = compact_once(state, client, settings)
    assert res.compacted is False
    assert len(state.turns) == 8


def test_compaction_swap_and_ordering(mock_npu):
    """Asserts compaction removes the oldest turns and replaces them with a summary."""
    client = NpuClient()
    settings = Settings(keep_recent_turns=4, compact_span_turns=3)
    
    # 7 turns total: 4 protected (5,6,7,8), 3 candidates (1,2,3)
    turns = [
        Turn(role="user", content=f"Turn {i}", turn_id=i, tokens=10)
        for i in range(1, 8)
    ]
    state = MemoryState(turns=turns)

    # Mock NPU responses
    mock_npu.post("/v1/chat/completions").mock(
        side_effect=[
            # Facts extraction response
            httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "content": '[{"kind": "decision", "text": "Planted Decision", "source_turn_id": 1}]'
                            }
                        }
                    ]
                },
            ),
            # Summarization response
            httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "content": "This is a summary of the first three turns."
                            }
                        }
                    ]
                },
            ),
        ]
    )

    res = compact_once(state, client, settings)
    assert res.compacted is True
    assert res.turns_compacted == 3
    assert res.new_summary == "This is a summary of the first three turns."
    assert res.new_facts_count == 1

    # Verbatim turns list should now only contain turns 4 to 7
    assert len(state.turns) == 4
    assert state.turns[0].turn_id == 4
    assert state.turns[-1].turn_id == 7

    # Summary covers turn IDs 1, 2, 3
    assert len(state.summaries) == 1
    assert state.summaries[0].covers_turn_ids == [1, 2, 3]
    
    # Ledger should contain the extracted decision
    assert len(state.ledger.entries) == 1
    assert state.ledger.entries[0].text == "Planted Decision"


def test_token_reduction_after_compaction(mock_npu):
    """Asserts that total token count strictly decreases after compaction."""
    client = NpuClient()
    settings = Settings(keep_recent_turns=1, compact_span_turns=3)
    
    # 4 turns of 20 tokens each = 80 tokens total. 3 oldest turns (60 tokens) will be compacted.
    turns = [
        Turn(role="user", content="x" * 80, turn_id=i, tokens=20)
        for i in range(1, 5)
    ]
    state = MemoryState(turns=turns)

    # Mock NPU: return empty facts and a short 5-token summary
    mock_npu.post("/v1/chat/completions").mock(
        side_effect=[
            httpx.Response(200, json={"choices": [{"message": {"content": "[]"}}]}),
            httpx.Response(200, json={"choices": [{"message": {"content": "Short summary."}}]}),
        ]
    )

    tokens_before = sum(t.tokens for t in state.turns)
    
    res = compact_once(state, client, settings)
    assert res.compacted is True

    # Tokens after = remaining verbatim turns + new summary
    tokens_after = sum(t.tokens for t in state.turns) + sum(s.tokens for s in state.summaries)
    
    assert tokens_after < tokens_before
    # Specifically: 20 (recent turn) + 3 (count_tokens("Short summary.") = 14//4 = 3) = 23 < 80
    assert tokens_after == 20 + count_tokens("Short summary.")


def test_failure_leaves_state_unchanged(mock_npu):
    """Asserts that NPU failures mid-compaction cause no partial state mutations."""
    client = NpuClient()
    settings = Settings(keep_recent_turns=2, compact_span_turns=2)
    
    # 4 turns total
    turns = [
        Turn(role="user", content=f"Turn {i}", turn_id=i, tokens=10)
        for i in range(1, 5)
    ]
    state = MemoryState(turns=turns)

    # Scenario 1: Fact extraction fails (returns invalid JSON)
    mock_npu.post("/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={"choices": [{"message": {"content": "Bad JSON"}}]})
    )

    res1 = compact_once(state, client, settings)
    assert res1.compacted is False
    assert len(state.turns) == 4
    assert len(state.summaries) == 0

    # Scenario 2: Fact extraction succeeds, but summarization fails
    mock_npu.post("/v1/chat/completions").mock(
        side_effect=[
            # Fact extraction works
            httpx.Response(200, json={"choices": [{"message": {"content": "[]"}}]}),
            # Summarization throws an HTTP error
            httpx.Response(500),
        ]
    )

    res2 = compact_once(state, client, settings)
    assert res2.compacted is False
    assert len(state.turns) == 4
    assert len(state.summaries) == 0
    assert len(state.ledger.entries) == 0


def test_run_background_lock_contention(tmp_path, mock_npu):
    """Asserts that run_background respects filelocks and blocks contention."""
    state_file = tmp_path / "state.json"
    lock_file = state_file.with_suffix(".lock")
    client = NpuClient()
    settings = Settings(compact_trigger_tokens=10, keep_recent_turns=0, compact_span_turns=2)

    # Setup state that needs compaction
    turns = [
        Turn(role="user", content="Turn 1 text", turn_id=1, tokens=10),
        Turn(role="user", content="Turn 2 text", turn_id=2, tokens=10),
    ]
    state = MemoryState(turns=turns)
    state.save(state_file)

    # Mock NPU responses
    mock_npu.post("/v1/chat/completions").mock(
        side_effect=[
            httpx.Response(200, json={"choices": [{"message": {"content": "[]"}}]}),
            httpx.Response(200, json={"choices": [{"message": {"content": "Summary"}}]}),
        ]
    )

    # 1. Acquire the lock in the main thread
    lock = FileLock(lock_file)
    lock.acquire()

    # 2. Run run_background in a separate thread - it should block waiting for the lock
    bg_thread_started = False
    bg_thread_completed = False

    def thread_target():
        nonlocal bg_thread_started, bg_thread_completed
        bg_thread_started = True
        run_background(str(state_file), client, settings)
        bg_thread_completed = True

    t = threading.Thread(target=thread_target)
    t.start()

    # Wait briefly to let the thread run and hit the lock
    time.sleep(0.1)
    
    assert bg_thread_started is True
    assert bg_thread_completed is False  # Must be blocked

    # Verify state file remains uncompacted because lock was held
    current_state = MemoryState.load(state_file)
    assert len(current_state.turns) == 2

    # 3. Release the lock - the thread should unblock, run compaction, and finish
    lock.release()
    t.join(timeout=2.0)
    
    assert bg_thread_completed is True

    # Verify state was compacted
    final_state = MemoryState.load(state_file)
    assert len(final_state.turns) == 0
    assert len(final_state.summaries) == 1

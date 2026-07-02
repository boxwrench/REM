"""Unit tests for the memory compactor component."""

import threading
import json
import httpx
from filelock import FileLock

from rem.config import Settings
from rem.npu_client import NpuClient
from rem.memory.tiers import MemoryState, Turn, count_tokens
from rem.memory.compactor import should_compact, compact_once, run_background
from rem.memory.assembler import assemble, assemble_messages


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


def test_compaction_preserves_and_renders_source_provenance(mock_npu):
    client = NpuClient()
    settings = Settings(keep_recent_turns=1, compact_span_turns=3)
    first_date = "2023/05/20 (Sat) 02:21"
    second_date = "2023/06/11 (Sun) 14:05"
    state = MemoryState(turns=[
        Turn(role="user", content="Initial context", turn_id=1, tokens=10,
             session_id="s1", timestamp=first_date),
        Turn(role="user", content="The launch city is Portland", turn_id=2,
             tokens=10, session_id="s1", timestamp=first_date),
        Turn(role="assistant", content="Noted", turn_id=3, tokens=10,
             session_id="s2", timestamp=second_date),
        Turn(role="user", content="Current question", turn_id=4, tokens=10,
             session_id="s3", timestamp="2023/07/01 (Sat) 09:00"),
    ])
    mock_npu.post("/v1/chat/completions").mock(side_effect=[
        httpx.Response(200, json={"choices": [{"message": {"content": json.dumps([{
            "kind": "entity",
            "text": "The launch city is Portland",
            "source_turn_id": 2,
        }])}}]}),
        httpx.Response(200, json={"choices": [{"message": {
            "content": "The launch planning covered Portland."
        }}]}),
    ])

    result = compact_once(state, client, settings)

    assert result.compacted is True
    assert [turn.turn_id for turn in state.turns] == [4]
    fact = state.ledger.entries[0]
    assert fact.session_id == "s1"
    assert fact.timestamp == first_date
    summary = state.summaries[0]
    assert summary.session_ids == ["s1", "s2"]
    assert summary.start_timestamp == first_date
    assert summary.end_timestamp == second_date

    summarizer_request = json.loads(mock_npu.calls.last.request.content)
    summarizer_text = summarizer_request["messages"][1]["content"]
    assert f"Session s1; Timestamp {first_date}" in summarizer_text
    assert f"Session s2; Timestamp {second_date}" in summarizer_text

    prompt = assemble(state, "System", "Task")
    system_message = assemble_messages(state, "System", "Task")[0]["content"]
    for rendered in (prompt, system_message):
        assert f"Timestamp {first_date}" in rendered
        assert f"Timestamps {first_date} to {second_date}" in rendered
        assert "Sessions s1, s2" in rendered


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


def test_run_background_skips_when_compaction_already_running(tmp_path, mock_npu):
    """A second compaction must not overlap a running one.

    The compaction lock is non-blocking: with it already held (simulating an
    in-flight compaction), run_background returns immediately rather than
    queueing, and leaves the state untouched for the running compaction to drain.
    """
    state_file = tmp_path / "state.json"
    lock_file = state_file.with_suffix(".lock")
    client = NpuClient()
    settings = Settings(compact_trigger_tokens=10, keep_recent_turns=0, compact_span_turns=2)

    # Setup state that would otherwise need compaction
    turns = [
        Turn(role="user", content="Turn 1 text", turn_id=1, tokens=10),
        Turn(role="user", content="Turn 2 text", turn_id=2, tokens=10),
    ]
    state = MemoryState(turns=turns)
    state.save(state_file)

    # Mock NPU responses (should not be consumed, since the call must skip)
    mock_npu.post("/v1/chat/completions").mock(
        side_effect=[
            httpx.Response(200, json={"choices": [{"message": {"content": "[]"}}]}),
            httpx.Response(200, json={"choices": [{"message": {"content": "Summary"}}]}),
        ]
    )

    # Simulate a compaction already in progress by holding the compaction lock.
    lock = FileLock(lock_file)
    lock.acquire()
    try:
        bg_thread_completed = False

        def thread_target():
            nonlocal bg_thread_completed
            run_background(str(state_file), client, settings)
            bg_thread_completed = True

        t = threading.Thread(target=thread_target)
        t.start()
        t.join(timeout=2.0)

        # New contract: returns immediately (does not block on the held lock).
        assert bg_thread_completed is True

        # And it did not compact — the in-flight compaction owns that work.
        current_state = MemoryState.load(state_file)
        assert len(current_state.turns) == 2
        assert len(current_state.summaries) == 0
    finally:
        lock.release()

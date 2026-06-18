"""Regression test for the sidecar/compactor lost-update race.

Background compaction loads a snapshot, spends seconds on the NPU, then writes
its result back. If the foreground request path appends a new turn during that
window, a naive ``state.save()`` at the end of compaction clobbers it. The
compactor must instead fold concurrent foreground turns into the latest state
before saving.
"""

from rem.config import Settings
from rem.memory import compactor
from rem.memory.compactor import run_background, CompactionResult
from rem.memory.tiers import MemoryState, SpanSummary, Turn


def test_concurrent_foreground_turn_survives_background_compaction(tmp_path, monkeypatch):
    state_path = tmp_path / "sess_memory_state.json"
    turns = [
        Turn(role="user" if i % 2 else "assistant", content=f"t{i}", turn_id=i, tokens=5)
        for i in range(1, 6)
    ]
    MemoryState(turns=turns).save(state_path)

    calls = {"n": 0}

    def fake_compact_once(state, client, settings=None):
        """Simulate one compaction pass, and on the first call simulate a
        foreground turn landing on disk while we are 'on the NPU' (unlocked)."""
        calls["n"] += 1
        if calls["n"] == 1:
            disk = MemoryState.load(state_path)
            disk.turns.append(Turn(role="user", content="CONCURRENT", turn_id=99, tokens=5))
            disk.save(state_path)

            dropped = state.turns[:2]
            state.summaries.append(
                SpanSummary(covers_turn_ids=[t.turn_id for t in dropped], text="sum", tokens=2)
            )
            state.turns = state.turns[2:]
            return CompactionResult(compacted=True, turns_compacted=2, new_summary="sum")
        return CompactionResult(compacted=False, turns_compacted=0)

    monkeypatch.setattr(compactor, "compact_once", fake_compact_once)

    settings = Settings(compact_trigger_tokens=1, keep_recent_turns=1, compact_span_turns=2)
    run_background(str(state_path), client=object(), settings=settings)

    final = MemoryState.load(state_path)
    contents = [t.content for t in final.turns]

    # The concurrently-appended foreground turn must NOT be clobbered.
    assert "CONCURRENT" in contents
    # The compaction still happened: oldest two turns are gone, one summary added.
    assert "t1" not in contents and "t2" not in contents
    assert len(final.summaries) == 1
    # Surviving verbatim turns keep their order, with the concurrent turn last.
    assert contents == ["t3", "t4", "t5", "CONCURRENT"]

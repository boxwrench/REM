"""--load-state must skip ingest entirely (NPU-free state acquisition)."""

from pathlib import Path

from rem.memory.facts_ledger import FactEntry, FactsLedger
from rem.memory.tiers import MemoryState, Turn
from evals.battery.diagnose_memory import acquire_state


class _ExplodingCM:
    """Stands in for RemContextManager; ingest must never be called."""
    def __init__(self):
        self._state = None
    def ingest(self, *a, **k):
        raise AssertionError("ingest() must not be called when --load-state is set")


def test_acquire_state_loads_without_ingest(tmp_path: Path):
    saved = MemoryState(
        turns=[Turn(role="user", content="hi", turn_id=1, tokens=1)],
        summaries=[],
        ledger=FactsLedger(entries=[FactEntry(kind="entity", text="x", source_turn_id=1)]),
    )
    path = tmp_path / "state.json"
    saved.save(path)

    cm = _ExplodingCM()
    state, ingest_secs = acquire_state(cm, load_state=str(path), item=None, budget_tokens=1000)

    assert ingest_secs == 0.0
    assert len(state.turns) == 1
    assert cm._state is state          # cm now answers from the loaded state

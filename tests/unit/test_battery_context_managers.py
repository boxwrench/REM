import json

from evals.battery.context_managers import RemContextManager, TruncationContextManager
from evals.battery.models import Session
from rem.config import Settings


def _sessions():
    # 3 sessions, oldest first; each turn ~ a few tokens
    return [
        Session("old", [{"role": "user", "content": "alpha " * 50}]),
        Session("mid", [{"role": "user", "content": "bravo " * 50}]),
        Session("new", [{"role": "user", "content": "charlie " * 50}]),
    ]


def test_truncation_keeps_recent_within_budget_drops_oldest():
    cm = TruncationContextManager()
    # budget small enough to hold ~the last session only
    cm.ingest(_sessions(), budget_tokens=150)
    ctx = cm.assemble()
    stats = cm.stats()
    assert "charlie" in ctx          # newest kept
    assert "alpha" not in ctx        # oldest dropped
    assert stats.assembled_tokens <= 150
    assert "new" in stats.retained_session_ids
    assert "old" not in stats.retained_session_ids


def test_truncation_evidence_retention_flag():
    cm = TruncationContextManager()
    cm.ingest(_sessions(), budget_tokens=150)
    assert cm.evidence_retained(["old"]) is False
    assert cm.evidence_retained(["new"]) is True


class FakeNpuClient:
    """Returns a valid facts array, then a summary, for compactor calls."""
    def __init__(self, settings=None):
        self.settings = settings or Settings()

    def chat(self, messages, *, model=None, max_tokens=None, **kw):
        sys = messages[0]["content"] if messages else ""
        if "fact extraction" in sys.lower() or "json array" in sys.lower():
            return json.dumps([{
                "kind": "entity", "source_turn_id": 1,
                "subject": "user", "attribute": "city",
                "value": "Acme", "is_correction": True,
            }])
        return "Summary: the user updated their employer to Acme."


def _many_sessions(n=60):
    return [
        Session(f"s{i}", [{"role": "user", "content": f"fact number {i} " * 20}])
        for i in range(n)
    ]


def test_rem_ingest_triggers_compaction_and_bounds_context():
    budget = 2000
    settings = Settings(
        summarizer_model="gemma4-it:e2b",
        compact_trigger_tokens=budget,
        max_context_tokens=budget * 4,
    )
    cm = RemContextManager(client=FakeNpuClient(), settings=settings)
    cm.ingest(_many_sessions(), budget_tokens=budget)
    st = cm.stats()
    assert st.compactions >= 1            # the trigger fired
    assert st.assembled_tokens > 0
    assert cm.assemble()                  # non-empty assembled context


def test_rem_ingest_is_single_call_and_does_not_mutate_settings():
    import pytest
    settings = Settings(summarizer_model="gemma4-it:e2b", compact_trigger_tokens=99999)
    cm = RemContextManager(client=FakeNpuClient(), settings=settings)
    cm.ingest(_many_sessions(4), budget_tokens=2000)
    assert settings.compact_trigger_tokens == 99999  # caller's object untouched
    with pytest.raises(RuntimeError):
        cm.ingest(_many_sessions(4), budget_tokens=2000)


def _large_compacted_state(n_compactions: int = 82):
    """A synthetic MemoryState the size of a ~500-turn item after compaction:
    ~82 episodic summaries + ~164 ledger facts + a small recent window."""
    from rem.memory.tiers import MemoryState, SpanSummary, Turn, count_tokens
    from rem.memory.facts_ledger import FactsLedger, FactEntry
    state = MemoryState()
    for i in range(n_compactions):
        txt = (f"In session block {i}, the user discussed plans, decisions, and several "
               f"concrete details about their ongoing project and preferences over time.")
        state.summaries.append(
            SpanSummary(covers_turn_ids=list(range(i * 6, i * 6 + 6)), text=txt,
                        tokens=count_tokens(txt)))
    led = FactsLedger()
    for i in range(n_compactions * 2):
        led.add(FactEntry(kind="entity",
                          text=f"Detail number {i} about subsystem {i % 9} is configured to value {i * 7}.",
                          source_turn_id=i + 1))
    state.ledger = led
    for i in range(8):
        t = f"This is a recent verbatim conversational turn number {i} with typical length and content."
        state.turns.append(Turn(role="user", content=t, turn_id=1000 + i, tokens=count_tokens(t)))
    return state


def test_rem_memory_window_fits_realistic_compacted_state():
    """A ~500-turn item's compacted memory overflows the old budget*4 ceiling
    (4000 for budget 1000) but must fit the REM memory window, so the arm can
    assemble and answer instead of raising ContextLimitExceeded."""
    import pytest
    from rem.memory.tiers import count_tokens
    from rem.memory.assembler import assemble, ContextLimitExceeded
    from evals.battery.context_managers import REM_MEMORY_WINDOW_TOKENS

    state = _large_compacted_state()

    with pytest.raises(ContextLimitExceeded):
        assemble(state, "sys", "task", settings=Settings(max_context_tokens=4000))

    prompt = assemble(state, "sys", "task",
                      settings=Settings(max_context_tokens=REM_MEMORY_WINDOW_TOKENS))
    assert count_tokens(prompt) <= REM_MEMORY_WINDOW_TOKENS

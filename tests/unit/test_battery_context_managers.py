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

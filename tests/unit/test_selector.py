"""Unit tests for the bounded read-path selector (NPU-free)."""

from rem.config import Settings
from rem.memory.assembler import assemble
from rem.memory.facts_ledger import FactEntry, FactsLedger
from rem.memory.selector import RecencySelector
from rem.memory.tiers import MemoryState, SpanSummary, Turn, count_tokens


def _big_state(n_summaries: int = 200, n_free_facts: int = 200) -> MemoryState:
    """A state whose summaries + ledger far exceed any small budget."""
    turns = [Turn(role="user", content=f"recent turn {i}", turn_id=900 + i, tokens=4)
             for i in range(4)]
    summaries = [
        SpanSummary(covers_turn_ids=[i], text=f"summary number {i} " * 12, tokens=40)
        for i in range(n_summaries)
    ]
    entries = [
        FactEntry(kind="entity", text=f"free fact {i} about the system " * 3,
                  source_turn_id=i, status="active")
        for i in range(n_free_facts)
    ]
    return MemoryState(turns=turns, summaries=summaries, ledger=FactsLedger(entries=entries))


def test_recency_selector_fits_budget():
    state = _big_state()
    question = "what is the current state?"
    budget = 3000
    fitted = RecencySelector().select(state, question, budget)
    assembled = assemble(fitted, system="", task=question)
    assert count_tokens(assembled) <= budget


def test_recency_selector_keeps_newest_per_slot_and_both_distinct_slots():
    # Two DIFFERENT slot keys must both survive (the 031748ae gold shape):
    # team.size newest = "5 engineers"; team_members.count newest = "4 engineers".
    entries = [
        FactEntry(kind="number", text="team size is 5 engineers", source_turn_id=5,
                  status="active", slot_key="team.size", slot_value="5 engineers"),
        FactEntry(kind="number", text="team size is 3 engineers", source_turn_id=2,
                  status="active", slot_key="team.size", slot_value="3 engineers"),
        FactEntry(kind="number", text="outing had 4 engineers", source_turn_id=12,
                  status="active", slot_key="team_members.count", slot_value="4 engineers"),
    ]
    state = MemoryState(turns=[], summaries=[],
                        ledger=FactsLedger(entries=entries))
    fitted = RecencySelector().select(state, "headcount?", budget_tokens=2000)
    text = assemble(fitted, system="", task="headcount?")
    assert "5 engineers" in text          # newest of team.size kept
    assert "4 engineers" in text          # newest of team_members.count kept
    assert "3 engineers" not in text      # older same-slot value dropped


def test_recency_selector_excludes_stale():
    entries = [
        FactEntry(kind="entity", text="active fact alpha", source_turn_id=10, status="active"),
        FactEntry(kind="entity", text="stale fact beta", source_turn_id=3, status="stale"),
    ]
    state = MemoryState(turns=[], summaries=[], ledger=FactsLedger(entries=entries))
    fitted = RecencySelector().select(state, "q", budget_tokens=2000)
    text = assemble(fitted, system="", task="q")
    assert "active fact alpha" in text
    assert "stale fact beta" not in text


def test_recency_selector_is_deterministic():
    state = _big_state()
    a = RecencySelector().select(state, "q", 3000)
    b = RecencySelector().select(state, "q", 3000)
    assert assemble(a, system="", task="q") == assemble(b, system="", task="q")


def test_recency_selector_prefers_newest_summaries():
    state = _big_state(n_summaries=200, n_free_facts=0)
    fitted = RecencySelector().select(state, "q", budget_tokens=1500)
    kept_ids = {min(s.covers_turn_ids) for s in fitted.summaries}
    # Newest summary (id 199) kept; oldest (id 0) dropped under a tight budget.
    assert 199 in kept_ids
    assert 0 not in kept_ids


def test_recency_selector_caps_oversized_current_slot_tier():
    entries = [
        FactEntry(
            kind="entity", text=(f"slot fact {i} " * 20), source_turn_id=i,
            slot_key=f"subject{i}.value", slot_value=str(i),
        )
        for i in range(200)
    ]
    state = MemoryState(ledger=FactsLedger(entries=entries))
    fitted = RecencySelector().select(state, "current values?", 2000)
    rendered = assemble(fitted, system="", task="current values?")
    assert count_tokens(rendered) <= 2000
    assert len(fitted.ledger.entries) < len(entries)
    assert max(entry.source_turn_id for entry in fitted.ledger.entries) == 199


def test_read_fit_tokens_default():
    assert Settings().read_fit_tokens == 28000
    assert Settings().read_newest_preference is False

from evals.battery.canonicalize_audit import analyze_state
from rem.memory.facts_ledger import FactEntry, FactsLedger
from rem.memory.tiers import MemoryState


def _fact(text, turn, key, value):
    return FactEntry(
        kind="number", text=text, source_turn_id=turn,
        slot_key=key, slot_value=value,
    )


def test_audit_reports_deltas_gold_survival_and_merge_risk():
    state = MemoryState(ledger=FactsLedger(entries=[
        _fact("team was 4 engineers", 1, "team.size", "4 engineers"),
        _fact("team is 5 engineers", 2, "team size.size", "5 engineers"),
        _fact("camera is Sony", 3, "camera.model", "Sony"),
    ]))
    row = analyze_state(state, "q1", "full", ["4 engineers", "5 engineers"])
    assert row["active_reduction"] == 1
    assert row["before"]["ledger_active"] == 3
    assert row["after"]["ledger_active"] == 2
    assert row["all_gold_preserved"] is True
    assert len(row["merge_risk_groups"]) == 1


def test_distinct_slots_with_similar_language_do_not_false_merge():
    state = MemoryState(ledger=FactsLedger(entries=[
        _fact("team size is 5", 1, "team.size", "5"),
        _fact("outing size is 8", 2, "team outing.size", "8"),
    ]))
    row = analyze_state(state, "q2", "full")
    assert row["active_reduction"] == 0
    assert row["merge_risk_groups"] == []

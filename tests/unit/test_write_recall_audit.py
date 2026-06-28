"""write_recall_audit separates write recall and measures slot fragmentation (NPU-free)."""
from rem.memory.facts_ledger import FactEntry, FactsLedger
from rem.memory.tiers import MemoryState, Turn
from evals.battery.write_recall_audit import audit_state, value_fragmentation, needle_in_full


def _state():
    entries = [
        # same value "5 engineers" under TWO distinct slot keys -> fragmentation
        FactEntry(kind="number", text="team size is 5 engineers", source_turn_id=74,
                  status="active", slot_key="team.size", slot_value="5 engineers"),
        FactEntry(kind="number", text="group has 5 engineers", source_turn_id=5,
                  status="active", slot_key="group.headcount", slot_value="5 engineers"),
        # an actually-superseded (stale) entry
        FactEntry(kind="number", text="team size is 3 engineers", source_turn_id=2,
                  status="stale", slot_key="team.size", slot_value="3 engineers"),
    ]
    return MemoryState(turns=[Turn(role="user", content="now?", turn_id=900, tokens=2)],
                       summaries=[], ledger=FactsLedger(entries=entries))


def test_supersession_and_fragmentation_counts():
    a = audit_state(_state(), ["5 engineers"], ["3 engineers"])
    assert a["ledger_total"] == 3
    assert a["ledger_active"] == 2
    assert a["superseded"] == 1
    # "5 engineers" is carried by two distinct active slot keys
    assert a["fragmented_values"] == 1
    assert set(a["fragmentation_examples"]["5 engineers"]) == {"team.size", "group.headcount"}


def test_write_recall_distinguishes_present_from_absent():
    state = _state()
    # gold present in the full state (in a slot); a never-written value is absent
    assert needle_in_full(state, "5 engineers") == "slot"
    assert needle_in_full(state, "9 engineers") == "absent"
    a = audit_state(state, ["5 engineers", "9 engineers"])
    assert a["write_recall_gold"] == {"5 engineers": "slot", "9 engineers": "absent"}


def test_value_fragmentation_ignores_single_key_values():
    # a value under one key only is not fragmentation
    entries = [
        FactEntry(kind="x", text="a", source_turn_id=1, status="active",
                  slot_key="k.one", slot_value="alpha"),
        FactEntry(kind="x", text="b", source_turn_id=2, status="active",
                  slot_key="k.two", slot_value="beta"),
    ]
    state = MemoryState(turns=[], summaries=[], ledger=FactsLedger(entries=entries))
    assert value_fragmentation(state) == {}

"""Post-hoc re-supersession retains ordered history."""

from rem.memory.canonicalize import recanonicalize
from rem.memory.facts_ledger import FactEntry, FactsLedger
from rem.memory.tiers import MemoryState, Turn


def _entry(text, turn, key, value):
    return FactEntry(
        kind="number", text=text, source_turn_id=turn, status="active",
        slot_key=key, slot_value=value,
    )


def test_collapses_fragmented_group_keeps_newest_active():
    state = MemoryState(ledger=FactsLedger(entries=[
        _entry("team size 4", 12, "team.size", "4 engineers"),
        _entry("team size 5", 74, "team size.size", "5 engineers"),
    ]))
    out = recanonicalize(state)
    active = [entry for entry in out.ledger.entries if entry.status == "active"]
    stale = [entry for entry in out.ledger.entries if entry.status == "stale"]
    assert [entry.slot_value for entry in active] == ["5 engineers"]
    assert [entry.slot_value for entry in stale] == ["4 engineers"]
    assert stale[0].superseded_by_turn_id == 74
    assert len(out.ledger.entries) == 2


def test_leaves_singletons_and_unrelated_slots_untouched():
    state = MemoryState(ledger=FactsLedger(entries=[
        _entry("team size 4", 12, "team.members", "4 engineers"),
        FactEntry(kind="entity", text="camera is a Sony", source_turn_id=20,
                  slot_key="camera.model", slot_value="Sony"),
    ]))
    out = recanonicalize(state)
    camera = [entry for entry in out.ledger.entries if entry.slot_key == "camera.model"]
    assert len(camera) == 1
    assert camera[0].status == "active"


def test_does_not_mutate_input_state_and_deep_copies_tiers():
    state = MemoryState(
        turns=[Turn(role="user", content="now?", turn_id=900, tokens=2)],
        ledger=FactsLedger(entries=[
            _entry("team size 4", 12, "team.size", "4 engineers"),
            _entry("team size 5", 74, "team size.size", "5 engineers"),
        ]),
    )
    out = recanonicalize(state)
    out.turns[0].content = "changed"
    assert [(entry.slot_key, entry.status) for entry in state.ledger.entries] == [
        ("team.size", "active"), ("team size.size", "active")
    ]
    assert state.turns[0].content == "now?"

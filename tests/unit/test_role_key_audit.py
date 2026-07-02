"""Path B role-aware re-key stays post-hoc and protects negative roles."""

from evals.memory_methods.run_role_key_audit import role_aware_rekey, same_role
from rem.memory.facts_ledger import FactEntry, FactsLedger
from rem.memory.tiers import MemoryState


def _fact(turn: int, key: str, value: str) -> FactEntry:
    return FactEntry(
        kind="number", text=f"{key}: {value}", source_turn_id=turn,
        slot_key=key, slot_value=value,
    )


def test_known_fragmented_update_keys_collide():
    assert same_role(
        "coffee ratio.tablespoon of coffee per ounces of water",
        "coffee brewing.ratio",
    )
    assert same_role("bird species.count", "species count.total species count")


def test_negative_role_and_instance_sentinels_stay_distinct():
    pairs = (
        ("event dates.start date", "event dates.end date"),
        ("price range.minimum price", "price range.maximum price"),
        ("chicken.refrigerator duration", "chicken.freezer duration"),
        ("plank.sets", "plank.reps"),
        ("onibus coffee.walk distance", "streamer coffee.walk distance"),
    )
    assert all(not same_role(left, right) for left, right in pairs)


def test_rekey_is_non_mutating_and_keeps_ordered_history():
    state = MemoryState(ledger=FactsLedger(entries=[
        _fact(13, "coffee ratio.tablespoon of coffee per ounces of water", "6 ounces"),
        _fact(209, "coffee brewing.ratio", "5 ounces"),
    ]))
    transformed = role_aware_rekey(state)
    assert [entry.status for entry in state.ledger.entries] == ["active", "active"]
    assert [entry.slot_key for entry in state.ledger.entries] == [
        "coffee ratio.tablespoon of coffee per ounces of water", "coffee brewing.ratio",
    ]
    assert [entry.status for entry in transformed.ledger.entries] == ["stale", "active"]
    assert transformed.ledger.entries[0].superseded_by_turn_id == 209
    assert transformed.ledger.entries[0].slot_key == transformed.ledger.entries[1].slot_key


def test_distinct_named_instances_remain_active():
    state = MemoryState(ledger=FactsLedger(entries=[
        _fact(1, "onibus coffee.walk distance", "10 minutes"),
        _fact(2, "streamer coffee.walk distance", "12 minutes"),
    ]))
    transformed = role_aware_rekey(state)
    assert [entry.status for entry in transformed.ledger.entries] == ["active", "active"]


def test_neutral_key_cannot_bridge_start_and_end_roles():
    state = MemoryState(ledger=FactsLedger(entries=[
        _fact(1, "event dates.start date", "May 26"),
        _fact(2, "event dates.date", "May 27"),
        _fact(3, "event dates.end date", "May 28"),
    ]))
    transformed = role_aware_rekey(state)
    active_keys = {
        entry.slot_key for entry in transformed.ledger.entries if entry.status == "active"
    }
    assert len(active_keys) == 2

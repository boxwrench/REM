"""Role-scoped newest-preference: safe reconciliation of Path A + Path B.

The negative/positive sentinels are the SAME human-reviewed families the Path B
audit froze, so these tests assert the read-time mechanism keeps the exact safety
property the audit measured: collapse genuine updates, never cross a role opposite
or a named instance.
"""
from rem.memory.facts_ledger import FactEntry
from rem.memory.role_keys import group_same_role, same_role
from rem.memory.selector import _Candidate, _prefer_newest_role_scoped

# Frozen from evals/memory_methods/run_role_key_audit.py (NEGATIVE/POSITIVE_SENTINELS).
NEGATIVE_SENTINELS = [
    ("event dates.start date", "event dates.end date"),
    ("price range.minimum price", "price range.maximum price"),
    ("chicken.refrigerator duration", "chicken.freezer duration"),
    ("plank.sets", "plank.reps"),
    ("onibus coffee.walk distance", "streamer coffee.walk distance"),
]
POSITIVE_SENTINELS = [
    ("coffee ratio.tablespoon of coffee per ounces of water", "coffee brewing.ratio"),
    ("bird species.count", "species count.total species count"),
]


def test_negative_sentinels_never_group():
    """Role opposites / named instances must stay distinct (no false merge)."""
    for a, b in NEGATIVE_SENTINELS:
        assert not same_role(a, b), (a, b)
        assert group_same_role([a, b]) == [], (a, b)


def test_positive_sentinels_group():
    """Genuine same-slot updates must collapse so newest-preference can fire."""
    for a, b in POSITIVE_SENTINELS:
        assert same_role(a, b), (a, b)
        groups = group_same_role([a, b])
        assert len(groups) == 1 and len(groups[0]) == 2, (a, b, groups)


def test_negatives_do_not_bridge_through_a_middle_spelling():
    """A compatible middle key must not transitively bridge two opposites."""
    # start <-> (compatible generic) <-> end must NOT all land in one group.
    keys = ["event dates.start date", "event dates.date", "event dates.end date"]
    for group in group_same_role(keys):
        members = {keys[i] for i in group}
        assert not ({"event dates.start date", "event dates.end date"} <= members)


def _cand(text, turn, *, key, value):
    entry = FactEntry(kind="number", text=text, source_turn_id=turn,
                      status="active", slot_key=key, slot_value=value)
    return _Candidate(kind="entry", value=entry, text=text, cost=8,
                      turn_id=turn, score=1.0)


def test_prefer_newest_role_scoped_keeps_newest_and_preserves_opposites():
    candidates = [
        # a genuine coffee update (older 6 oz -> newer 5 oz)
        _cand("coffee brewing ratio 1 tbsp per 6 ounces", 13,
              key="coffee ratio.tablespoon of coffee per ounces of water",
              value="6 ounces"),
        _cand("coffee brewing ratio 1 tbsp per 5 ounces", 209,
              key="coffee brewing.ratio", value="5 ounces"),
        # a start/end negative sentinel that must NOT collapse
        _cand("event start date May 26", 5,
              key="event dates.start date", value="May 26"),
        _cand("event end date May 28", 6,
              key="event dates.end date", value="May 28"),
    ]
    out = _prefer_newest_role_scoped(candidates)
    values = sorted(
        c.value.slot_value for c in out
        if isinstance(c.value, FactEntry)
    )
    # coffee collapsed to the newest value only; both start and end survive
    assert "5 ounces" in values
    assert "6 ounces" not in values
    assert "May 26" in values and "May 28" in values
    # the surviving coffee entry is flagged as the current observation
    assert any("LATEST CURRENT OBSERVATION" in c.text for c in out)

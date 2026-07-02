"""Role-aware slot-key compatibility.

Extracted verbatim (behaviour-preserving) from the Path B step-1 role-key audit
(`evals/memory_methods/run_role_key_audit.py`), whose held-out result proved this
rule collapses the two genuine numeric updates (coffee ``6 oz -> 5 oz``, birds
``27 -> 32``) while keeping ALL FIVE negative-sentinel families distinct
(start/end, min/max, fridge/freezer, sets/reps, and per-instance named facts).

Path B was declined for *promotion* only because its aggregate fragmentation
reduction (26.5%) missed the 50% bar — an aggregate write-time metric. That bar is
irrelevant to the read-time use here: the read path's role-scoped newest-preference
only needs, per genuine role-slot, to prefer the newest value, and never to cross a
role opposite or a named instance. This module supplies exactly that safe grouping
so the mechanism lives in ``src`` (the audit kept it in an eval script).
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

from rem.memory.canonicalize import _tokens

# Role opposites: two keys whose token sets land on OPPOSITE sides of any dimension
# are never the same slot (start vs end, min vs max, ...). This veto is what keeps
# the negative sentinels distinct.
ROLE_DIMENSIONS = {
    "boundary": ({"start", "begin", "opening"}, {"end", "finish", "closing"}),
    "range": ({"min", "minimum", "low", "lower"}, {"max", "maximum", "high", "upper"}),
    "storage": ({"fridge", "refrigerator", "refrigerated"},
                {"freezer", "freeze", "frozen"}),
    "exercise": ({"set"}, {"rep", "repetition"}),
}


def _parts(slot_key: str) -> tuple[set[str], set[str]]:
    """(subject tokens, attribute tokens) from a ``subject.attribute`` slot key."""
    subject, separator, attribute = slot_key.rpartition(".")
    if not separator:
        subject, attribute = slot_key, ""
    return _tokens(subject), _tokens(attribute)


def role_conflict(all_a: set[str], all_b: set[str]) -> bool:
    """True if the two token sets sit on opposite sides of any role dimension."""
    for left, right in ROLE_DIMENSIONS.values():
        if (all_a & left and all_b & right) or (all_a & right and all_b & left):
            return True
    return False


def same_role(slot_a: str, slot_b: str) -> bool:
    """Conservative identity: same instance and compatible (non-opposite) role.

    Shared subject evidence is mandatory, so two named instances with the same
    attribute stay separate. A shared non-subject role (or a token moved between
    subject and attribute, or a subset relation) handles extractor rephrasings.
    Explicit role opposites always veto.
    """
    subject_a, attribute_a = _parts(slot_a)
    subject_b, attribute_b = _parts(slot_b)
    all_a, all_b = subject_a | attribute_a, subject_b | attribute_b
    if not all_a or not all_b or role_conflict(all_a, all_b):
        return False
    shared_subject = subject_a & subject_b
    if not shared_subject:
        return False
    if all_a <= all_b or all_b <= all_a:
        return True
    shared_role = (all_a & all_b) - shared_subject
    if not shared_role:
        return False
    # Distinct leftover names on both subjects are evidence for two instances.
    return not (subject_a - all_b and subject_b - all_a)


def group_same_role(slot_keys: Sequence[str]) -> list[list[int]]:
    """Union-find grouping of ``slot_keys`` indices by ``same_role``.

    Returns the multi-member groups only (size >= 2). A merge is admitted only when
    it is cross-compatible with every current member of both roots, so a
    compatible middle spelling cannot bridge two incompatible roles/instances —
    the same guard the Path B audit used.
    """
    n = len(slot_keys)
    parent = list(range(n))
    members: dict[int, set[int]] = {i: {i} for i in range(n)}

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    by_subject_token: dict[str, list[int]] = defaultdict(list)
    compared: set[tuple[int, int]] = set()
    for index in range(n):
        subject, _ = _parts(slot_keys[index])
        for token in sorted(subject):
            for other in by_subject_token[token]:
                pair = (other, index)
                if pair in compared:
                    continue
                compared.add(pair)
                if same_role(slot_keys[other], slot_keys[index]):
                    left, right = find(other), find(index)
                    cross_compatible = all(
                        same_role(slot_keys[a], slot_keys[b])
                        for a in members[left]
                        for b in members[right]
                    )
                    if left != right and cross_compatible:
                        parent[right] = left
                        members[left].update(members.pop(right))
            by_subject_token[token].append(index)

    grouped: dict[int, list[int]] = defaultdict(list)
    for index in range(n):
        grouped[find(index)].append(index)
    return [group for group in grouped.values() if len(group) > 1]

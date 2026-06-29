"""String-first slot identity for measuring supersession fragmentation.

The transform in this module is intentionally post-hoc.  It lets the captured
states establish how much exact-string normalization buys before the write path
or an embedding model is changed.
"""
from __future__ import annotations

import re

from rem.memory.tiers import MemoryState

_STOPWORDS = {
    "a", "an", "the", "of", "to", "for", "in", "on", "per", "and", "or",
    "number", "count", "total", "amount", "value", "range", "type", "new",
    "current",
}
_IRREGULAR = {
    "people": "person", "children": "child", "men": "man", "women": "woman",
}


def _singular(token: str) -> str:
    if token in _IRREGULAR:
        return _IRREGULAR[token]
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith("es") and not token.endswith(
        ("ses", "zes", "ches", "shes")
    ):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _tokens(text: str) -> set[str]:
    """Return normalized identity tokens for a model-emitted key fragment."""
    raw = re.split(r"[^a-z0-9]+", text.lower())
    return {_singular(token) for token in raw if token and token not in _STOPWORDS}


def canonical_slot_key(slot_key: str, granularity: str = "full") -> str:
    """Return a deterministic token signature for a slot key.

    ``full`` combines subject and attribute tokens. ``subject`` deliberately
    ignores the attribute and is therefore an aggressive diagnostic arm.
    """
    if granularity not in {"full", "subject"}:
        raise ValueError("granularity must be 'full' or 'subject'")
    subject, separator, attribute = slot_key.rpartition(".")
    if not separator:
        subject, attribute = slot_key, ""
    tokens = _tokens(subject)
    if granularity == "full":
        tokens |= _tokens(attribute)
    return " ".join(sorted(tokens))


def recanonicalize(state: MemoryState, granularity: str = "full") -> MemoryState:
    """Re-supersede active slotted facts without mutating the captured state.

    Every entry is retained.  In a canonical group, the latest observation is
    active and older observations become ordered history linked to that winner.
    Original slot keys remain on entries so the experiment cannot masquerade as
    write-path integration.
    """
    entries = [entry.model_copy(deep=True) for entry in state.ledger.entries]
    groups: dict[str, list] = {}
    for entry in entries:
        if entry.status == "active" and entry.slot_key:
            key = canonical_slot_key(entry.slot_key, granularity)
            if key:
                groups.setdefault(key, []).append(entry)

    for members in groups.values():
        if len(members) < 2:
            continue
        members.sort(key=lambda entry: entry.source_turn_id)
        newest = members[-1]
        for older in members[:-1]:
            older.status = "stale"
            older.superseded_by_turn_id = newest.source_turn_id

    ledger = state.ledger.model_copy(deep=True)
    ledger.entries = entries
    ledger.rendered_text = None
    return MemoryState(
        schema_version=state.schema_version,
        turns=[turn.model_copy(deep=True) for turn in state.turns],
        summaries=[summary.model_copy(deep=True) for summary in state.summaries],
        ledger=ledger,
    )

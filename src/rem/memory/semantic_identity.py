"""Subject-agnostic full-fact embedding identity for slot supersession (Gate 4).

The writer's exact-string slot keys leave the same attribute fragmented across many
keys ("team.size", "team size.size", "group size.number of engineers"), so genuine
updates never collapse. This module provides a pluggable slot-identity matcher that
decides "same slot" from the cosine similarity of the entries' FULL-FACT text
("natural key: value") — the composition shown to separate same-slot from
different-slot best (see bench/battery/FINDINGS.md, Gate 4 key-composition sweep).

The matcher is injected into a FactsLedger via ``set_slot_matcher`` and only decides
the cross-key case (exact slot_key matches short-circuit in the ledger). It does NOT
import the embedder; callers pass an ``embed(texts) -> list[vec]`` callable, so the
heavy model dependency stays out of the write path.
"""
from __future__ import annotations

import re
from collections.abc import Callable, Sequence

import numpy as np

from rem.memory.facts_ledger import FactEntry, FactsLedger
from rem.memory.tiers import MemoryState

# Spelled-out quantity words: a value carrying one of these (or a digit) is treated
# as quantity-like, so a slot UPDATE (5->5, one->two, 100->150) is allowed while two
# distinct NAMED values (Poffertjes vs apple pie) are read as different instances.
_NUMWORDS = {
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
    "seventeen", "eighteen", "nineteen", "twenty", "thirty", "forty", "fifty",
    "sixty", "seventy", "eighty", "ninety", "hundred", "thousand", "million",
    "half", "quarter", "couple", "few", "several", "dozen", "once", "twice",
}


def quantity_like(value: str | None) -> bool:
    """True if the value reads as a quantity/measurement (digit or number word)."""
    if not value:
        return False
    low = value.lower()
    if re.search(r"\d", low):
        return True
    return any(tok in _NUMWORDS for tok in re.findall(r"[a-z]+", low))


def full_fact_text(entry: FactEntry) -> str:
    """"natural key: value" — dots/underscores in the key become spaces."""
    key = (entry.slot_key or "").replace(".", " ").replace("_", " ").strip()
    value = (entry.slot_value or "").strip()
    if key and value:
        return f"{key}: {value}"
    return key or value


def subject_of(entry: FactEntry) -> str:
    if entry.subject:
        return entry.subject.lower().strip()
    if entry.slot_key:
        return entry.slot_key.split(".")[0].lower().strip()
    return ""


class FullFactEmbeddingMatcher:
    """Same-slot iff cosine(full_fact(a), full_fact(b)) >= threshold.

    ``require_subject_overlap`` (default False) optionally also requires the two
    subjects to share a token — a conservative guard. Threshold-only is the default
    because real fragmentation often splits the subject too ("team" vs "group size").
    Vectors are cached by text; ``preembed`` batches them up front.
    """

    def __init__(
        self,
        embed: Callable[[Sequence[str]], Sequence[Sequence[float]]],
        threshold: float = 0.80,
        require_subject_overlap: bool = False,
        value_aware: bool = False,
    ) -> None:
        self._embed = embed
        self.threshold = threshold
        self.require_subject_overlap = require_subject_overlap
        # Instance-aware gate (option a): when on, a merge between entries with
        # DIFFERENT values is allowed only if both values are quantity-like (a
        # plausible update), blocking distinct-named-instance collisions.
        self.value_aware = value_aware
        self._cache: dict[str, np.ndarray] = {}
        self.merges: list[dict] = []  # audit log of fired merges
        self.blocked: list[dict] = []  # audit log of merges blocked by the value gate

    def preembed(self, texts: Sequence[str]) -> None:
        todo = sorted({t for t in texts if t and t not in self._cache})
        if not todo:
            return
        vecs = self._embed(todo)
        for t, v in zip(todo, vecs):
            arr = np.asarray(v, dtype=np.float32)
            norm = np.linalg.norm(arr)
            self._cache[t] = arr / norm if norm else arr

    def _vec(self, text: str) -> np.ndarray:
        if text not in self._cache:
            self.preembed([text])
        return self._cache[text]

    @staticmethod
    def _subjects_overlap(a: FactEntry, b: FactEntry) -> bool:
        sa = set(subject_of(a).split())
        sb = set(subject_of(b).split())
        generic = {"the", "a", "an", "current", "new", "user", "my", "our"}
        return bool((sa & sb) - generic)

    def same_slot(self, a: FactEntry, b: FactEntry) -> bool:
        if not (a.slot_value and b.slot_value):
            return False  # need a value on both sides for full-fact identity
        if self.require_subject_overlap and not self._subjects_overlap(a, b):
            return False
        ta, tb = full_fact_text(a), full_fact_text(b)
        sim = float(np.dot(self._vec(ta), self._vec(tb)))
        if sim < self.threshold:
            return False
        if self.value_aware:
            va, vb = (a.slot_value or "").strip().lower(), (b.slot_value or "").strip().lower()
            different = va != vb
            if different and not (quantity_like(a.slot_value) and quantity_like(b.slot_value)):
                # Same kind of fact, but two distinct named values -> different
                # instances, not a slot update. Block the merge.
                self.blocked.append({
                    "sim": round(sim, 4),
                    "a_key": a.slot_key, "a_value": a.slot_value,
                    "b_key": b.slot_key, "b_value": b.slot_value,
                })
                return False
        self.merges.append({
            "sim": round(sim, 4),
            "kept_key": b.slot_key, "kept_value": b.slot_value,
            "merged_key": a.slot_key, "merged_value": a.slot_value,
        })
        return True


def resupersede_state(
    state: MemoryState,
    matcher: FullFactEmbeddingMatcher,
) -> tuple[MemoryState, dict]:
    """Replay a captured state's ledger through embedding-matched supersession.

    Resets every entry to active and re-adds them in source-turn order through a
    fresh FactsLedger with ``matcher`` installed, reproducing what ingest would have
    produced with the flag on — NPU-free for the answerer (only the local embedder
    runs). Returns ``(new_state, stats)``. Summaries and verbatim turns are unchanged.
    """
    originals = sorted(state.ledger.entries, key=lambda e: e.source_turn_id)
    matcher.preembed([full_fact_text(e) for e in originals if e.slot_value])

    ledger = FactsLedger(max_stale_entries=10_000)  # keep history for audit/ordering
    ledger.set_slot_matcher(matcher)
    for e in originals:
        clone = e.model_copy(deep=True)
        clone.status = "active"
        clone.superseded_by_turn_id = None
        ledger.add(clone)

    before_active = len(state.ledger.active_entries())
    after_active = len(ledger.active_entries())
    new_state = MemoryState(
        schema_version=state.schema_version,
        turns=list(state.turns),
        summaries=list(state.summaries),
        ledger=ledger,
    )
    stats = {
        "entries_before": len(state.ledger.entries),
        "entries_after": len(ledger.entries),
        "active_before": before_active,
        "active_after": after_active,
        "active_reduction": before_active - after_active,
        "active_reduction_pct": (
            round(100 * (before_active - after_active) / before_active, 2)
            if before_active else 0.0
        ),
        "semantic_merges_fired": len(matcher.merges),
    }
    return new_state, stats

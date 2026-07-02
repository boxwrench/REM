"""Bounded read path: fit a compacted MemoryState to a token budget.

The assembler renders the facts ledger in full and every summary unbounded, so on
long conversations the assembled memory exceeds the answering model's window. A
MemorySelector chooses which summaries / ledger entries to keep so the assembled
memory fits a budget, returning a FILTERED MemoryState that flows through the
existing assemble() (the selector decides what is in; the assembler still decides
how it is rendered, reusing quarantine/stale handling).
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
import re
from typing import Protocol

from rem.memory.facts_ledger import FactEntry, FactsLedger
from rem.memory.query import classify_question
from rem.memory.role_keys import group_same_role
from rem.memory.tiers import MemoryState, SpanSummary, count_tokens

# Reserve inside the budget for section headers + the answer the model must still
# generate. The question's own tokens are reserved separately (it is variable).
SELECTOR_RESERVE_TOKENS = 768


class MemorySelector(Protocol):
    """Chooses which tiers of a compacted state to keep so it fits a budget."""

    def select(self, state: MemoryState, query: str, budget_tokens: int) -> MemoryState:
        ...


def _summary_cost(s: SpanSummary) -> int:
    rendered = s.rendered_text if s.rendered_text is not None else s.text
    provenance = []
    if s.session_ids:
        provenance.append(f"Sessions {', '.join(s.session_ids)}")
    if s.start_timestamp:
        if s.end_timestamp and s.end_timestamp != s.start_timestamp:
            provenance.append(f"Timestamps {s.start_timestamp} to {s.end_timestamp}")
        else:
            provenance.append(f"Timestamp {s.start_timestamp}")
    prefix = f"[{'; '.join(provenance)}] " if provenance else ""
    return count_tokens(f"- {prefix}{rendered}")


def _entry_cost(e: FactEntry) -> int:
    status = "" if e.status == "active" else " stale"
    provenance = [f"Turn {e.source_turn_id}"]
    if e.session_id:
        provenance.append(f"Session {e.session_id}")
    if e.timestamp:
        provenance.append(f"Timestamp {e.timestamp}")
    return count_tokens(
        f"- [{e.kind}{status}] {e.text} ({'; '.join(provenance)})"
    )


class RecencySelector:
    """Fills a hard budget by tier and recency.

    Priority (highest first):
      1. newest verbatim turns and newest active entry per slot_key;
      2. episodic summaries, newest -> oldest;
      3. remaining active ledger entries (no slot_key), newest-first.
    Stale entries are never included. If the current-state tier itself exceeds
    the budget, its newest items win. No query scoring. Deterministic.
    """

    def select(self, state: MemoryState, question: str, budget_tokens: int) -> MemoryState:
        budget = budget_tokens - count_tokens(question) - SELECTOR_RESERVE_TOKENS

        # --- Tier 1: newest verbatim + newest active entry per slot_key ---
        kept_turns_reversed = []
        used = 0
        for turn in reversed(state.turns):
            cost = count_tokens(f"{turn.role.upper()}: {turn.content}")
            if used + cost <= budget:
                kept_turns_reversed.append(turn)
                used += cost
        kept_turns = list(reversed(kept_turns_reversed))

        newest_by_slot: dict[str, FactEntry] = {}
        free_actives: list[FactEntry] = []
        for e in state.ledger.entries:
            if e.status != "active":
                continue
            if e.slot_key:
                cur = newest_by_slot.get(e.slot_key)
                if cur is None or e.source_turn_id > cur.source_turn_id:
                    newest_by_slot[e.slot_key] = e
            else:
                free_actives.append(e)

        kept_entries: list[FactEntry] = []
        for entry in sorted(
            newest_by_slot.values(),
            key=lambda item: (-item.source_turn_id, item.slot_key or "", item.text),
        ):
            cost = _entry_cost(entry)
            if used + cost <= budget:
                kept_entries.append(entry)
                used += cost

        # --- Tier 2: summaries newest-first ---
        kept_summaries: list[SpanSummary] = []
        for s in sorted(
            state.summaries,
            key=lambda s: max(s.covers_turn_ids) if s.covers_turn_ids else 0,
            reverse=True,
        ):
            c = _summary_cost(s)
            if used + c > budget:
                continue
            kept_summaries.append(s)
            used += c

        # --- Tier 3: remaining active ledger entries newest-first ---
        for e in sorted(free_actives, key=lambda e: e.source_turn_id, reverse=True):
            c = _entry_cost(e)
            if used + c > budget:
                continue
            kept_entries.append(e)
            used += c

        return MemoryState(
            schema_version=state.schema_version,
            turns=kept_turns,
            summaries=kept_summaries,
            ledger=FactsLedger(entries=kept_entries),
        )


_QUERY_STOPWORDS = {
    "a", "an", "and", "are", "do", "does", "did", "for", "how", "i", "in",
    "is", "it", "me", "my", "of", "on", "or", "the", "to", "was", "what",
    "when", "where", "which", "who", "with",
}
_TEMPORAL_TERMS = {
    "before", "decrease", "decreased", "earlier", "first", "former", "from",
    "historical", "history", "increase", "increased", "initial", "original",
    "previous", "previously", "prior", "started", "then", "updated", "was",
}
_INVARIANT_NOUNS = {"series", "species"}


def _retrieval_tokens(text: str) -> set[str]:
    """Small dependency-free tokenizer for the lexical evaluation arm."""
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    out = set()
    for token in tokens:
        if token in _QUERY_STOPWORDS:
            continue
        if token not in _INVARIANT_NOUNS and len(token) > 4 and token.endswith("ies"):
            token = token[:-3] + "y"
        elif len(token) > 4 and token.endswith("s") and not token.endswith("ss"):
            token = token[:-1]
        out.add(token)
    return out


def _is_temporal_query(query: str) -> bool:
    return bool(set(re.findall(r"[a-z]+", query.lower())) & _TEMPORAL_TERMS)


@dataclass(frozen=True)
class _Candidate:
    kind: str
    value: FactEntry | SpanSummary
    text: str
    cost: int
    turn_id: int
    score: float = 0.0


def _candidate_text(summary: SpanSummary) -> str:
    return summary.rendered_text if summary.rendered_text is not None else summary.text


def _deduplicated_entries(state: MemoryState, include_history: bool) -> list[FactEntry]:
    source = state.ledger.entries if include_history else state.ledger.active_entries()
    newest_by_identity: dict[tuple[str, str], FactEntry] = {}
    for entry in source:
        normalized_text = " ".join(entry.text.lower().split())
        identity = (entry.slot_key or "", normalized_text)
        current = newest_by_identity.get(identity)
        if current is None or entry.source_turn_id > current.source_turn_id:
            newest_by_identity[identity] = entry
    return list(newest_by_identity.values())


def _scored_candidates(
    state: MemoryState, query: str, *, deduplicate: bool, include_history: bool
) -> list[_Candidate]:
    if deduplicate:
        entries = _deduplicated_entries(state, include_history)
    else:
        entries = (
            list(state.ledger.entries) if include_history
            else list(state.ledger.active_entries())
        )
    raw: list[tuple[str, FactEntry | SpanSummary, str, int, int]] = []
    for entry in entries:
        raw.append(("entry", entry, entry.text, _entry_cost(entry), entry.source_turn_id))
    for summary in state.summaries:
        text = _candidate_text(summary)
        turn_id = max(summary.covers_turn_ids) if summary.covers_turn_ids else 0
        raw.append(("summary", summary, text, _summary_cost(summary), turn_id))

    document_tokens = [_retrieval_tokens(item[2]) for item in raw]
    document_frequency: dict[str, int] = {}
    for tokens in document_tokens:
        for token in tokens:
            document_frequency[token] = document_frequency.get(token, 0) + 1
    query_tokens = _retrieval_tokens(query)
    n_documents = max(1, len(raw))
    max_turn = max((item[4] for item in raw), default=1)
    out = []
    for item, tokens in zip(raw, document_tokens):
        overlap = query_tokens & tokens
        lexical = sum(
            math.log(1.0 + (n_documents + 1) / (document_frequency[token] + 1))
            for token in overlap
        )
        # Recency is only a stable tie-break signal; it cannot outweigh one
        # actual query-term match.
        score = lexical + (item[4] / max_turn) * 0.001
        out.append(_Candidate(*item, score=score))
    return out


def _available_budget(state: MemoryState, query: str, budget_tokens: int) -> tuple[int, list]:
    # Verbatim turns contain the live request and are therefore a protected floor:
    # silently dropping an oversized user message would change the request. Normal
    # compaction keeps this tier small; the assembler raises if the floor alone is
    # larger than the model window.
    turns = list(state.turns)
    turn_cost = count_tokens("\n".join(
        f"{turn.role.upper()}: {turn.content}" for turn in turns
    )) if turns else 0
    return max(
        0,
        budget_tokens
        - count_tokens(query)
        - SELECTOR_RESERVE_TOKENS
        - turn_cost,
    ), turns


_SLOT_TOKEN_ALIASES = {
    "begin": "start",
    "beginning": "start",
    "ending": "end",
    "finish": "end",
    "lower": "minimum",
    "min": "minimum",
    "max": "maximum",
    "upper": "maximum",
    "refrigerated": "refrigerator",
    "fridge": "refrigerator",
    "frozen": "freezer",
    "reps": "repetition",
    "rep": "repetition",
    "repetitions": "repetition",
    "sets": "set",
}
_ROLE_DIMENSIONS = (
    frozenset({"start", "end"}),
    frozenset({"minimum", "maximum"}),
    frozenset({"refrigerator", "freezer"}),
    frozenset({"set", "repetition"}),
    frozenset({"under", "below", "over", "above"}),
    frozenset({"morning", "afternoon", "evening", "night"}),
    frozenset({"left", "right"}),
    frozenset({"front", "rear", "back"}),
    frozenset({"first", "second", "third", "fourth", "fifth"}),
    frozenset({
        "monday", "tuesday", "wednesday", "thursday", "friday",
        "saturday", "sunday",
    }),
)
_SLOT_ROLE_ALIASES = {
    "amount": "count",
    "frequency": "frequency",
    "count": "count",
    "number": "count",
    "total": "count",
    "ratio": "ratio",
    "day": "day",
    "date": "day",
    "time": "time",
    "timing": "time",
    "location": "location",
    "place": "location",
    "status": "status",
    "state": "status",
    "action": "action",
    "result": "result",
    "symptom": "symptom",
    "type": "type",
    "model": "model",
    "capacity": "capacity",
}
_GENERIC_SUBJECT_TOKENS = {
    "change", "detail", "information", "service", "servicing", "update"
}


def _slot_tokens(slot_key: str) -> set[str]:
    tokens = re.findall(r"[a-z0-9]+", slot_key.lower())
    normalized = set()
    for token in tokens:
        token = _SLOT_TOKEN_ALIASES.get(token, token)
        if token not in _INVARIANT_NOUNS and len(token) > 4 and token.endswith("ies"):
            token = token[:-3] + "y"
        elif len(token) > 4 and token.endswith("s") and not token.endswith("ss"):
            token = token[:-1]
        normalized.add(token)
    return normalized


def _role_conflict(left: set[str], right: set[str]) -> bool:
    left_instances = {token for token in left if token.isdigit()}
    right_instances = {token for token in right if token.isdigit()}
    if left_instances and right_instances and left_instances != right_instances:
        return True
    for dimension in _ROLE_DIMENSIONS:
        left_roles = left & dimension
        right_roles = right & dimension
        if left_roles and right_roles and left_roles != right_roles:
            return True
    return False


def _near_duplicate_slot_keys(left: str | None, right: str | None) -> bool:
    """Conservative, role-aware slot-family test used only at read time."""
    if not left or not right:
        return False
    if left == right:
        return True
    left_tokens, right_tokens = _slot_tokens(left), _slot_tokens(right)
    if _role_conflict(left_tokens, right_tokens):
        return False
    shared = left_tokens & right_tokens
    return (
        len(shared) >= 2
        and len(shared) / min(len(left_tokens), len(right_tokens)) >= (2 / 3)
    )


def _value_shape(value: str | None) -> str:
    if not value:
        return "missing"
    if re.search(r"\d", value):
        return "numeric"
    if any(marker in value for marker in ("[", "]", "{", "}")):
        return "collection"
    return "scalar" if len(value.split()) <= 8 else "prose"


def _slot_roles(tokens: set[str]) -> set[str]:
    return {
        _SLOT_ROLE_ALIASES[token]
        for token in tokens
        if token in _SLOT_ROLE_ALIASES
    }


def _subject_anchors(slot_key: str) -> set[str]:
    subject = slot_key.split(".", 1)[0]
    return _slot_tokens(subject) - _GENERIC_SUBJECT_TOKENS


def _near_duplicate_entries(left: FactEntry, right: FactEntry) -> bool:
    if not _near_duplicate_slot_keys(left.slot_key, right.slot_key):
        return False
    if left.slot_key == right.slot_key:
        return True
    left_tokens = _slot_tokens(left.slot_key or "")
    right_tokens = _slot_tokens(right.slot_key or "")
    left_roles, right_roles = _slot_roles(left_tokens), _slot_roles(right_tokens)
    if bool(left_roles) != bool(right_roles):
        return False
    if left_roles and left_roles.isdisjoint(right_roles):
        return False
    if not (_subject_anchors(left.slot_key or "") & _subject_anchors(right.slot_key or "")):
        return False
    left_shape = _value_shape(left.slot_value)
    right_shape = _value_shape(right.slot_value)
    return left_shape == right_shape and left_shape in {"numeric", "scalar"}


def _slot_family_groups(
    candidates: list[_Candidate], query: str, *, active_only: bool
) -> list[list[int]]:
    entry_indexes = [
        index for index, candidate in enumerate(candidates)
        if candidate.kind == "entry"
        and isinstance(candidate.value, FactEntry)
        and (not active_only or candidate.value.status == "active")
    ]
    ordered_indexes = sorted(
        entry_indexes,
        key=lambda index: (-candidates[index].score, -candidates[index].turn_id),
    )
    unassigned = set(entry_indexes)
    groups: list[list[int]] = []
    for seed in ordered_indexes:
        if seed not in unassigned:
            continue
        group = [seed]
        unassigned.remove(seed)
        for other in ordered_indexes:
            if other not in unassigned:
                continue
            other_entry = candidates[other].value
            if all(
                _near_duplicate_entries(other_entry, candidates[index].value)
                for index in group
            ):
                group.append(other)
                unassigned.remove(other)
        groups.append(group)

    query_tokens = _slot_tokens(query)
    anchored = []
    for indexes in groups:
        if len(indexes) < 2:
            continue
        shared_subject = set.intersection(*(
            _subject_anchors(candidates[index].value.slot_key or "")
            for index in indexes
        ))
        if shared_subject & query_tokens:
            anchored.append(indexes)
    return anchored


def _text_contains_value(text: str, value: str | None) -> bool:
    if not value:
        return False
    normalized_text = " ".join(text.lower().split())
    normalized_value = " ".join(value.lower().split())
    if normalized_value.isdigit():
        return bool(re.search(rf"\b{re.escape(normalized_value)}\b", normalized_text))
    return len(normalized_value) >= 3 and normalized_value in normalized_text


def _candidate_with_text(candidate: _Candidate, text: str) -> _Candidate:
    entry = candidate.value.model_copy(update={"text": text})
    return replace(candidate, value=entry, text=text, cost=_entry_cost(entry))


def _prefer_newest_slot_families(
    candidates: list[_Candidate], query: str
) -> list[_Candidate]:
    """Keep the newest active fact and suppress summaries of displaced values.

    The operation filters only the selected read view. The persisted ledger is
    untouched. The newest member inherits the family's best lexical score so a
    key-fragmented update cannot fall below the sparse top-k solely because its
    newer key shares fewer literal words with the question.
    """
    groups = _slot_family_groups(candidates, query, active_only=True)
    discarded: set[int] = set()
    replacements: dict[int, _Candidate] = {}
    obsolete: list[tuple[str, set[str]]] = []
    for indexes in groups:
        newest = max(
            indexes,
            key=lambda index: (
                candidates[index].turn_id,
                candidates[index].text,
            ),
        )
        best_score = max(candidates[index].score for index in indexes)
        newest_entry = candidates[newest].value
        replacements[newest] = replace(
            _candidate_with_text(
                candidates[newest],
                f"LATEST CURRENT OBSERVATION: {newest_entry.text}",
            ),
            score=best_score,
        )
        newest_value = " ".join((newest_entry.slot_value or "").lower().split())
        shared_subject = set.intersection(*(
            _subject_anchors(candidates[index].value.slot_key or "")
            for index in indexes
        ))
        for index in indexes:
            if index == newest:
                continue
            old_value = candidates[index].value.slot_value
            normalized_old = " ".join((old_value or "").lower().split())
            if old_value and normalized_old != newest_value:
                obsolete.append((old_value, shared_subject))
        discarded.update(index for index in indexes if index != newest)

    return [
        replacements.get(index, candidate)
        for index, candidate in enumerate(candidates)
        if index not in discarded
        and not (
            candidate.kind == "summary"
            and any(
                _text_contains_value(candidate.text, value)
                and bool(_slot_tokens(candidate.text) & anchors)
                for value, anchors in obsolete
            )
        )
    ]


def _annotate_temporal_slot_families(
    candidates: list[_Candidate], query: str, question_mode: str
) -> list[_Candidate]:
    """Make fragmented earlier/latest sequences explicit without dropping history."""
    replacements: dict[int, _Candidate] = {}
    for indexes in _slot_family_groups(candidates, query, active_only=False):
        ordered = sorted(indexes, key=lambda index: candidates[index].turn_id)
        earlier = ordered[-2] if question_mode == "previous" else ordered[0]
        latest = ordered[-1]
        earlier_entry = candidates[earlier].value
        latest_entry = candidates[latest].value
        if earlier_entry.slot_value == latest_entry.slot_value:
            continue
        label = "PREVIOUS SEQUENCE" if question_mode == "previous" else "UPDATE SEQUENCE"
        sequence = (
            f"{label}: earlier value was "
            f"{earlier_entry.slot_value} (Turn {earlier_entry.source_turn_id}); "
            f"latest value is {latest_entry.slot_value} "
            f"(Turn {latest_entry.source_turn_id}). {latest_entry.text}"
        )
        replacements[latest] = _candidate_with_text(candidates[latest], sequence)
    return [replacements.get(index, candidate) for index, candidate in enumerate(candidates)]


def _build_selected_state(
    state: MemoryState,
    turns: list,
    selected: list[_Candidate],
    *,
    include_history: bool,
) -> MemoryState:
    entries = [candidate.value for candidate in selected if candidate.kind == "entry"]
    summaries = [candidate.value for candidate in selected if candidate.kind == "summary"]
    return MemoryState(
        schema_version=state.schema_version,
        turns=turns,
        summaries=summaries,
        ledger=FactsLedger(
            entries=entries,
            include_stale_on_render=include_history,
        ),
    )


class LexicalSelector:
    """Rank facts and summaries by query-term overlap, then fill in rank order.

    This arm intentionally avoids dense retrieval, reranking, and knapsack-style
    packing so each later mechanism has an attributable effect.
    """

    def select(self, state: MemoryState, query: str, budget_tokens: int) -> MemoryState:
        include_history = _is_temporal_query(query)
        remaining, turns = _available_budget(state, query, budget_tokens)
        candidates = _scored_candidates(
            state, query, deduplicate=False, include_history=include_history
        )
        candidates.sort(
            key=lambda candidate: (-candidate.score, -candidate.turn_id,
                                   candidate.kind, candidate.text)
        )
        selected = []
        for candidate in candidates:
            if candidate.cost <= remaining:
                selected.append(candidate)
                remaining -= candidate.cost
        return _build_selected_state(
            state, turns, selected, include_history=include_history
        )


class PackedLexicalSelector:
    """Lexical retrieval with deduplication and budget-aware greedy packing."""

    def select(self, state: MemoryState, query: str, budget_tokens: int) -> MemoryState:
        include_history = _is_temporal_query(query)
        remaining, turns = _available_budget(state, query, budget_tokens)
        candidates = _scored_candidates(
            state, query, deduplicate=True, include_history=include_history
        )
        candidates.sort(
            key=lambda candidate: (
                -(candidate.score / max(1, candidate.cost)),
                -candidate.score,
                -candidate.turn_id,
                candidate.kind,
                candidate.text,
            )
        )
        selected = []
        for candidate in candidates:
            if candidate.cost <= remaining:
                selected.append(candidate)
                remaining -= candidate.cost
        return _build_selected_state(
            state, turns, selected, include_history=include_history
        )


def _prefer_newest_role_scoped(
    candidates: list[_Candidate],
) -> list[_Candidate]:
    """Safe newest-preference: prefer the newest value ONLY within a role-slot.

    Unlike ``_prefer_newest_slot_families`` (cross-key grouping, found to risk
    role/instance false merges in adversarial review), this groups active entry
    candidates with ``role_keys.same_role`` — the Path B rule that provably keeps
    all five negative-sentinel families (start/end, min/max, fridge/freezer,
    sets/reps, per-instance) distinct while collapsing genuine updates. Within each
    group the newest source-turn wins and inherits the group's best lexical score;
    displaced entries are dropped from the read view only (the ledger is untouched).
    """
    entry_indexes = [
        index for index, candidate in enumerate(candidates)
        if candidate.kind == "entry"
        and isinstance(candidate.value, FactEntry)
        and candidate.value.status == "active"
        and candidate.value.slot_key
    ]
    if len(entry_indexes) < 2:
        return candidates
    slot_keys = [candidates[index].value.slot_key or "" for index in entry_indexes]
    replacements: dict[int, _Candidate] = {}
    discarded: set[int] = set()
    for group in group_same_role(slot_keys):
        member_indexes = [entry_indexes[g] for g in group]
        newest = max(
            member_indexes,
            key=lambda index: (candidates[index].turn_id, candidates[index].text),
        )
        best_score = max(candidates[index].score for index in member_indexes)
        newest_entry = candidates[newest].value
        replacements[newest] = replace(
            _candidate_with_text(
                candidates[newest],
                f"LATEST CURRENT OBSERVATION: {newest_entry.text}",
            ),
            score=best_score,
        )
        discarded.update(index for index in member_indexes if index != newest)
    return [
        replacements.get(index, candidate)
        for index, candidate in enumerate(candidates)
        if index not in discarded
    ]


# A pure-recency candidate scores at most this (recency term = turn_id/max_turn * 0.001,
# so <= 0.001). One actual query-term match contributes >= log(2) ~= 0.69, far above it.
# A floor here therefore admits only candidates that share at least one query term —
# the "nonzero relevance" the agent's read path requires — and excludes the recency
# fill that lets LexicalSelector pack the whole budget with distractors.
SPARSE_RELEVANCE_FLOOR = 0.01
SPARSE_TOP_K = 24


class SparseChronologicalSelector:
    """Top-k, relevance-floored, chronologically-rendered evidence (gate arm 'sparse').

    Three deliberate differences from LexicalSelector, each targeting a measured
    confound in the current read path:

      * RELEVANCE FLOOR — keep only candidates whose score clears
        ``relevance_floor`` (i.e. that actually matched a query term), instead of
        giving every entry a positive recency score (selector.py score = lexical +
        recency*0.001) and admitting all of them.
      * TOP-K, NOT BUDGET-FILL — cap the evidence at ``top_k`` survivors so a 28k
        budget is never packed full of weak matches; sparse means sparse.
      * CHRONOLOGICAL RENDER — emit the survivors oldest -> newest so the answerer
        sees an ordered then -> now trail (the structure temporal questions need),
        rather than score order.

    Deduplicates near-identical observations. For temporal queries it includes
    ordered history (stale entries) exactly as the lexical arms do, so a then->now
    sequence can surface when one exists.
    """

    def __init__(self, top_k: int = SPARSE_TOP_K,
                 relevance_floor: float = SPARSE_RELEVANCE_FLOOR,
                 prefer_newest: bool = False,
                 newest_scope: str = "cross_key",
                 mode_aware_history: bool = True) -> None:
        self.top_k = top_k
        self.relevance_floor = relevance_floor
        self.prefer_newest = prefer_newest
        # "cross_key" = the existing (unsafe, default-off) family grouping;
        # "role" = role-scoped newest-preference (safe: role_keys.same_role keeps
        # the negative sentinels distinct). Only consulted when prefer_newest is on.
        self.newest_scope = newest_scope
        self.mode_aware_history = mode_aware_history

    def select(self, state: MemoryState, query: str, budget_tokens: int) -> MemoryState:
        question_mode = classify_question(query)
        include_history = _is_temporal_query(query) or (
            self.mode_aware_history
            and question_mode in {"previous", "change", "point-in-time"}
        )
        remaining, turns = _available_budget(state, query, budget_tokens)
        candidates = _scored_candidates(
            state, query, deduplicate=True, include_history=include_history
        )
        if self.prefer_newest and question_mode == "current":
            if self.newest_scope == "role":
                candidates = _prefer_newest_role_scoped(candidates)
            else:
                candidates = _prefer_newest_slot_families(candidates, query)
        elif self.prefer_newest and question_mode in {"previous", "change"}:
            candidates = _annotate_temporal_slot_families(
                candidates, query, question_mode
            )
        relevant = [c for c in candidates if c.score > self.relevance_floor]
        relevant.sort(key=lambda c: (-c.score, -c.turn_id, c.kind, c.text))
        top = relevant[: self.top_k]

        selected = []
        for candidate in top:
            if candidate.cost <= remaining:
                selected.append(candidate)
                remaining -= candidate.cost

        # Render oldest -> newest. _build_selected_state preserves this order into the
        # ledger/summaries, so the assembled evidence reads chronologically.
        selected.sort(key=lambda c: (c.turn_id, c.kind, c.text))
        return _build_selected_state(
            state, turns, selected, include_history=include_history
        )

"""Bounded read path: fit a compacted MemoryState to a token budget.

The assembler renders the facts ledger in full and every summary unbounded, so on
long conversations the assembled memory exceeds the answering model's window. A
MemorySelector chooses which summaries / ledger entries to keep so the assembled
memory fits a budget, returning a FILTERED MemoryState that flows through the
existing assemble() (the selector decides what is in; the assembler still decides
how it is rendered, reusing quarantine/stale handling).
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Protocol

from rem.memory.facts_ledger import FactEntry, FactsLedger
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
    return count_tokens(f"- {rendered}")


def _entry_cost(e: FactEntry) -> int:
    status = "" if e.status == "active" else " stale"
    return count_tokens(f"- [{e.kind}{status}] {e.text} (Turn {e.source_turn_id})")


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


def _retrieval_tokens(text: str) -> set[str]:
    """Small dependency-free tokenizer for the lexical evaluation arm."""
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    out = set()
    for token in tokens:
        if token in _QUERY_STOPWORDS:
            continue
        if len(token) > 4 and token.endswith("ies"):
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
    turns = list(state.turns)
    turn_cost = count_tokens("\n".join(
        f"{turn.role.upper()}: {turn.content}" for turn in turns
    )) if turns else 0
    return max(
        0, budget_tokens - count_tokens(query) - SELECTOR_RESERVE_TOKENS - turn_cost
    ), turns


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
                 relevance_floor: float = SPARSE_RELEVANCE_FLOOR) -> None:
        self.top_k = top_k
        self.relevance_floor = relevance_floor

    def select(self, state: MemoryState, query: str, budget_tokens: int) -> MemoryState:
        include_history = _is_temporal_query(query)
        remaining, turns = _available_budget(state, query, budget_tokens)
        candidates = _scored_candidates(
            state, query, deduplicate=True, include_history=include_history
        )
        relevant = [c for c in candidates if c.score > self.relevance_floor]
        if not relevant:
            # No query-term match at all: fall back to the most recent candidates so
            # the evidence set is not empty, but STILL bounded by top_k (never the
            # whole budget). This keeps the arm honest on questions our tokenizer
            # can't lexically anchor.
            relevant = sorted(candidates, key=lambda c: -c.turn_id)
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

"""Bounded read path: fit a compacted MemoryState to a token budget.

The assembler renders the facts ledger in full and every summary unbounded, so on
long conversations the assembled memory exceeds the answering model's window. A
MemorySelector chooses which summaries / ledger entries to keep so the assembled
memory fits a budget, returning a FILTERED MemoryState that flows through the
existing assemble() (the selector decides what is in; the assembler still decides
how it is rendered, reusing quarantine/stale handling).
"""
from __future__ import annotations

from typing import Protocol

from rem.memory.facts_ledger import FactEntry, FactsLedger
from rem.memory.tiers import MemoryState, SpanSummary, count_tokens

# Reserve inside the budget for section headers + the answer the model must still
# generate. The question's own tokens are reserved separately (it is variable).
SELECTOR_RESERVE_TOKENS = 512


class MemorySelector(Protocol):
    """Chooses which tiers of a compacted state to keep so it fits a budget."""

    def select(self, state: MemoryState, question: str, budget_tokens: int) -> MemoryState:
        ...


def _summary_cost(s: SpanSummary) -> int:
    rendered = s.rendered_text if s.rendered_text is not None else s.text
    return count_tokens(f"- {rendered}")


def _entry_cost(e: FactEntry) -> int:
    status = "" if e.status == "active" else " stale"
    return count_tokens(f"- [{e.kind}{status}] {e.text} (Turn {e.source_turn_id})")


class RecencySelector:
    """Keeps current-state slots + verbatim, then fills the budget newest-first.

    Priority (highest first):
      1. verbatim turns (already bounded) and the newest active entry per slot_key
         (the current-state facts) -- always kept;
      2. episodic summaries, newest -> oldest;
      3. remaining active ledger entries (no slot_key), newest-first.
    Stale entries are never included. No scoring against the question. Deterministic.
    """

    def select(self, state: MemoryState, question: str, budget_tokens: int) -> MemoryState:
        budget = budget_tokens - count_tokens(question) - SELECTOR_RESERVE_TOKENS

        # --- Protected tier: verbatim + newest active entry per slot_key ---
        kept_turns = list(state.turns)
        verbatim_cost = (
            count_tokens("\n".join(f"{t.role.upper()}: {t.content}" for t in kept_turns))
            if kept_turns else 0
        )

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

        kept_entries: list[FactEntry] = list(newest_by_slot.values())
        used = verbatim_cost + sum(_entry_cost(e) for e in kept_entries)

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

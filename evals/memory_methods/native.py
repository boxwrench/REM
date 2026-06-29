"""Evaluation arm for selectors operating over once-captured REM states."""
from __future__ import annotations

from collections.abc import Callable
from time import perf_counter
from typing import Any

from evals.battery.context_managers import REM_SYSTEM
from evals.battery.models import Session
from evals.memory_methods.contracts import RecallResult, SourceReference
from rem.memory.assembler import assemble
from rem.memory.selector import MemorySelector
from rem.memory.tiers import MemoryState, count_tokens


class NativeSelectorArm:
    """Adapt a ``MemorySelector`` to the normalized black-box contract.

    ``capture_loader`` resolves the already-persisted state for a namespace.
    This keeps paired read experiments from repeating expensive NPU ingestion.
    """

    def __init__(
        self,
        name: str,
        selector: MemorySelector,
        capture_loader: Callable[[str, list[Session]], MemoryState],
    ) -> None:
        self.name = name
        self._selector = selector
        self._capture_loader = capture_loader
        self._states: dict[str, MemoryState] = {}
        self._stats: dict[str, dict[str, Any]] = {}

    def ingest(self, namespace: str, sessions: list[Session]) -> None:
        started = perf_counter()
        self._states[namespace] = self._capture_loader(namespace, sessions)
        self._stats[namespace] = {
            "ingest_latency_ms": round((perf_counter() - started) * 1000, 3),
            "read_latencies_ms": [],
            "recalls": 0,
        }

    def await_ready(self, namespace: str, timeout: float) -> None:
        if namespace not in self._states:
            raise KeyError(f"unknown namespace: {namespace}")

    @staticmethod
    def _references(state: MemoryState) -> tuple[SourceReference, ...]:
        refs = []
        for entry in state.ledger.entries:
            refs.append(SourceReference(
                source_id=f"turn:{entry.source_turn_id}",
                kind="fact",
                turn_ids=(entry.source_turn_id,),
                metadata={"slot_key": entry.slot_key or "", "status": entry.status},
            ))
        for summary in state.summaries:
            turn_ids = tuple(summary.covers_turn_ids)
            refs.append(SourceReference(
                source_id="summary:" + ",".join(map(str, turn_ids)),
                kind="summary",
                turn_ids=turn_ids,
            ))
        for turn in state.turns:
            refs.append(SourceReference(
                source_id=f"turn:{turn.turn_id}", kind="verbatim",
                turn_ids=(turn.turn_id,),
            ))
        return tuple(refs)

    def recall(self, namespace: str, query: str, budget_tokens: int) -> RecallResult:
        state = self._states.get(namespace)
        if state is None:
            raise KeyError(f"unknown namespace: {namespace}")
        started = perf_counter()
        selected = self._selector.select(state, query, budget_tokens)
        context = assemble(selected, system=REM_SYSTEM, task=query)
        latency = round((perf_counter() - started) * 1000, 3)
        stats = self._stats[namespace]
        stats["read_latencies_ms"].append(latency)
        stats["recalls"] += 1
        return RecallResult(
            rendered_context=context,
            token_count=count_tokens(context),
            source_references=self._references(selected),
            candidate_count=(
                len(selected.ledger.entries) + len(selected.summaries) + len(selected.turns)
            ),
            latency_ms=latency,
        )

    def stats(self, namespace: str) -> dict[str, Any]:
        if namespace not in self._stats:
            raise KeyError(f"unknown namespace: {namespace}")
        stats = self._stats[namespace]
        latencies = sorted(stats["read_latencies_ms"])
        p95_index = max(0, int(len(latencies) * 0.95 + 0.999) - 1)
        return {
            **stats,
            "read_latency_p95_ms": latencies[p95_index] if latencies else None,
        }

    def reset(self, namespace: str) -> None:
        self._states.pop(namespace, None)
        self._stats.pop(namespace, None)

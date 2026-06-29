"""System-neutral contracts used by native and black-box memory arms."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

from evals.battery.models import Session


@dataclass(frozen=True)
class SourceReference:
    """Stable pointer back to evidence supplied to a memory arm."""

    source_id: str
    kind: str
    turn_ids: tuple[int, ...] = ()
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class RecallResult:
    """Normalized recall output; the answer model consumes rendered_context."""

    rendered_context: str
    token_count: int
    source_references: tuple[SourceReference, ...]
    candidate_count: int
    latency_ms: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MemoryArm(Protocol):
    """Evaluation-only lifecycle for isolated memory systems."""

    name: str

    def ingest(self, namespace: str, sessions: list[Session]) -> None: ...
    def await_ready(self, namespace: str, timeout: float) -> None: ...
    def recall(self, namespace: str, query: str, budget_tokens: int) -> RecallResult: ...
    def stats(self, namespace: str) -> dict[str, Any]: ...
    def reset(self, namespace: str) -> None: ...

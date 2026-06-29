"""Versioned normalized artifact schema for memory-method evaluations."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator


class ModelVersions(BaseModel):
    answer: str = "gemma4-it:e2b"
    judge: str = "claude-haiku-4-5"
    extraction: str | None = None
    embedding: str | None = None
    reranker: str | None = None


class ItemRun(BaseModel):
    question_id: str
    category: str
    arm: str
    budget_tokens: int
    memory_tokens: int
    source_references: list[dict[str, Any]] = Field(default_factory=list)
    candidate_count: int
    ingest_latency_ms: float | None = None
    read_latency_ms: float
    write_recall: bool | None = None
    read_recall: bool | None = None
    judged_correct: bool | None = None
    extraction_failures: int = 0
    context_overflow: bool = False
    provenance_lost: bool = False
    error: str | None = None

    @model_validator(mode="after")
    def validate_accounting(self) -> "ItemRun":
        if self.memory_tokens > self.budget_tokens and not self.context_overflow:
            raise ValueError("over-budget run must set context_overflow=true")
        if self.candidate_count > 0 and not self.source_references and not self.provenance_lost:
            raise ValueError("candidates without sources must set provenance_lost=true")
        return self


class MemoryMethodArtifact(BaseModel):
    schema_version: int = 1
    repository_revision: str
    system_revision: str | None = None
    source_manifest: str
    source_dataset_sha256: str
    models: ModelVersions = Field(default_factory=ModelVersions)
    configuration: dict[str, Any] = Field(default_factory=dict)
    runs: list[ItemRun] = Field(default_factory=list)
    category_accuracy: dict[str, float] = Field(default_factory=dict)
    read_latency_p95_ms: float | None = None
    write_latency_p95_ms: float | None = None
    resource_contention: dict[str, Any] = Field(default_factory=dict)

"""Temporal graph data model: nodes (entities) and edges (facts).

Phase 0: no embeddings, no persistence. See
docs/superpowers/specs/2026-06-20-graph-phase-0-design.md.
"""
from __future__ import annotations

from time import time as _now
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator


def _new_id() -> str:
    return uuid4().hex


class Node(BaseModel):
    """An entity (person, place, object, concept)."""
    id: str = Field(default_factory=_new_id)
    label: str
    aliases: list[str] = Field(default_factory=list)
    embedding: list[float] | None = None
    created_at: float = Field(default_factory=_now)


class Edge(BaseModel):
    """A single fact: subject --relation--> object, over two timelines."""
    id: str = Field(default_factory=_new_id)
    subject_id: str
    relation: str
    object_id: str | None = None
    object_literal: str | None = None
    # event timeline
    valid_from: float
    valid_to: float | None = None
    # transaction timeline
    ingested_at: float = Field(default_factory=_now)
    invalidated_at: float | None = None
    source_turn_id: int | None = None
    kind: Literal["entity", "number", "decision", "quote"] = "entity"
    embedding: list[float] | None = None

    @model_validator(mode="after")
    def _exactly_one_object(self) -> "Edge":
        if (self.object_id is not None) == (self.object_literal is not None):
            raise ValueError("Edge requires exactly one of object_id or object_literal")
        return self

"""Deterministic, model-free serialization of currently-valid edges to text.

Phase 0. See the Phase 0 design spec §6. The serialized form is the token
budget that matters; internal storage is never read by a model.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Callable

from rem.graph.model import Edge
from rem.graph.store import GraphStore


def _iso_month(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m")


def serialize_edges(edges: list[Edge], resolve_label: Callable[[str], str]) -> str:
    """One fact per line: '<subject> <relation> <object> (since <YYYY-MM>)'."""
    lines: list[str] = []
    for e in edges:
        subj = resolve_label(e.subject_id)
        obj = resolve_label(e.object_id) if e.object_id is not None else (e.object_literal or "")
        line = f"{subj} {e.relation} {obj}".rstrip()
        line += f" (since {_iso_month(e.valid_from)})"
        lines.append(line)
    return "\n".join(lines)


def serialize_current(store: GraphStore) -> str:
    """Serialize the store's currently-valid edges, resolving node labels."""
    def resolve(node_id: str) -> str:
        node = store.get_node(node_id)
        return node.label if node else node_id
    return serialize_edges(store.current_edges(), resolve)

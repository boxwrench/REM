"""In-memory temporal graph store with supersession-as-data.

Phase 0: no persistence, no embeddings. See the Phase 0 design spec.
"""
from __future__ import annotations

from rem.graph.model import Node, Edge


def _object_key(edge: Edge) -> tuple[str, str]:
    """A comparable identity for an edge's object (node ref or literal)."""
    if edge.object_id is not None:
        return ("id", edge.object_id)
    return ("lit", edge.object_literal or "")


class GraphStore:
    def __init__(self) -> None:
        self.nodes: dict[str, Node] = {}
        self.edges: list[Edge] = []
        self._label_index: dict[str, str] = {}  # lowercased label -> node id

    # --- nodes ---
    def add_node(self, node: Node) -> Node:
        self.nodes[node.id] = node
        self._label_index.setdefault(node.label.strip().lower(), node.id)
        return node

    def ensure_node(self, label: str, aliases: tuple[str, ...] = ()) -> Node:
        key = label.strip().lower()
        existing_id = self._label_index.get(key)
        if existing_id is not None:
            node = self.nodes[existing_id]
            for a in aliases:
                if a not in node.aliases:
                    node.aliases.append(a)
            return node
        return self.add_node(Node(label=label, aliases=list(aliases)))

    def get_node(self, node_id: str) -> Node | None:
        return self.nodes.get(node_id)

    # --- edges ---
    def _current_for_slot(self, subject_id: str, relation: str) -> list[Edge]:
        return [
            e for e in self.edges
            if e.subject_id == subject_id and e.relation == relation
            and e.valid_to is None and e.invalidated_at is None
        ]

    def add(self, edge: Edge, *, supersede: bool = True) -> Edge:
        if supersede:
            current = self._current_for_slot(edge.subject_id, edge.relation)
            for old in current:
                if _object_key(old) == _object_key(edge):
                    return old  # identical fact already current: no-op
            for old in current:
                old.valid_to = edge.valid_from        # event time
                old.invalidated_at = edge.ingested_at  # transaction time
        self.edges.append(edge)
        return edge

    def current_edges(self) -> list[Edge]:
        return [e for e in self.edges if e.valid_to is None and e.invalidated_at is None]

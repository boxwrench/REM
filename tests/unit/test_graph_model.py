"""Unit tests for the temporal graph data model (Phase 0)."""
import pytest
from rem.graph.model import Node, Edge


def test_node_has_auto_id_and_defaults():
    n = Node(label="Denver")
    assert n.id  # non-empty auto id
    assert n.created_at > 0
    assert n.aliases == []
    assert n.embedding is None


def test_edge_accepts_a_literal_object():
    e = Edge(subject_id="u", relation="lives_in", object_literal="Denver", valid_from=1.0)
    assert e.object_literal == "Denver"
    assert e.object_id is None
    assert e.kind == "entity"
    assert e.valid_to is None and e.invalidated_at is None


def test_edge_accepts_a_node_ref_object():
    e = Edge(subject_id="u", relation="knows", object_id="n2", valid_from=1.0)
    assert e.object_id == "n2"


def test_edge_rejects_both_object_forms():
    with pytest.raises(ValueError):
        Edge(subject_id="u", relation="r", object_id="n", object_literal="x", valid_from=1.0)


def test_edge_rejects_neither_object_form():
    with pytest.raises(ValueError):
        Edge(subject_id="u", relation="r", valid_from=1.0)

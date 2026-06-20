"""Unit tests for the in-memory temporal graph store (Phase 0)."""
from rem.graph.model import Edge
from rem.graph.store import GraphStore


def test_ensure_node_dedups_by_label_case_insensitive():
    s = GraphStore()
    a = s.ensure_node("Denver")
    b = s.ensure_node("denver")
    assert a.id == b.id
    assert len(s.nodes) == 1


def test_ensure_node_accumulates_aliases():
    s = GraphStore()
    s.ensure_node("Denver")
    n = s.ensure_node("Denver", aliases=("Mile High City",))
    assert "Mile High City" in n.aliases


def test_added_edge_appears_in_current():
    s = GraphStore()
    e = Edge(subject_id="u", relation="lives_in", object_literal="Denver", valid_from=1.0)
    s.add(e)
    assert s.current_edges() == [e]


def test_supersession_closes_old_edge_with_different_object():
    s = GraphStore()
    old = Edge(subject_id="u", relation="lives_in", object_literal="Denver",
               valid_from=1.0, ingested_at=1.0)
    s.add(old)
    new = Edge(subject_id="u", relation="lives_in", object_literal="Boulder",
               valid_from=2.0, ingested_at=2.0)
    s.add(new)
    current = s.current_edges()
    assert [e.object_literal for e in current] == ["Boulder"]
    assert old.valid_to == 2.0          # new.valid_from
    assert old.invalidated_at == 2.0    # new.ingested_at


def test_supersession_identical_object_is_a_noop():
    s = GraphStore()
    s.add(Edge(subject_id="u", relation="lives_in", object_literal="Denver", valid_from=1.0))
    s.add(Edge(subject_id="u", relation="lives_in", object_literal="Denver", valid_from=2.0))
    assert len(s.edges) == 1  # duplicate fact not inserted


def test_supersede_false_skips_supersession():
    s = GraphStore()
    s.add(Edge(subject_id="u", relation="lives_in", object_literal="Denver", valid_from=1.0))
    s.add(Edge(subject_id="u", relation="lives_in", object_literal="Boulder", valid_from=2.0),
          supersede=False)
    assert len(s.current_edges()) == 2  # both remain current

"""Unit tests for graph serialization to context text (Phase 0)."""
from datetime import datetime, timezone

from rem.graph.model import Edge
from rem.graph.store import GraphStore
from rem.graph.serialize import serialize_edges, serialize_current
from rem.memory.tiers import count_tokens


def _ts(y: int, m: int, d: int) -> float:
    return datetime(y, m, d, tzinfo=timezone.utc).timestamp()


def test_serialize_edges_resolves_labels_and_renders_since():
    e = Edge(subject_id="n_user", relation="lives_in", object_literal="Denver",
             valid_from=_ts(2024, 3, 1))
    labels = {"n_user": "user"}
    out = serialize_edges([e], lambda nid: labels.get(nid, nid))
    assert out == "user lives_in Denver (since 2024-03)"


def test_serialize_edges_resolves_object_node_label():
    e = Edge(subject_id="n_user", relation="knows", object_id="n_bob",
             valid_from=_ts(2024, 1, 1))
    labels = {"n_user": "user", "n_bob": "Bob"}
    out = serialize_edges([e], lambda nid: labels.get(nid, nid))
    assert out == "user knows Bob (since 2024-01)"


def test_serialize_current_uses_only_valid_edges_and_is_token_counted():
    s = GraphStore()
    u = s.ensure_node("user")
    s.add(Edge(subject_id=u.id, relation="lives_in", object_literal="Denver",
               valid_from=_ts(2024, 3, 1), ingested_at=_ts(2024, 3, 1)))
    s.add(Edge(subject_id=u.id, relation="lives_in", object_literal="Boulder",
               valid_from=_ts(2024, 6, 1), ingested_at=_ts(2024, 6, 1)))
    text = serialize_current(s)
    assert "user lives_in Boulder (since 2024-06)" in text
    assert "Denver" not in text          # superseded, not current
    assert count_tokens(text) > 0

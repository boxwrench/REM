"""Phase 0 gate (architecture spec §10): build a graph by hand, serialize it,
run a query, get the correct currently-valid edges back, deterministically."""
from datetime import datetime, timezone

from rem.graph.model import Edge
from rem.graph.store import GraphStore
from rem.graph.serialize import serialize_current
from rem.memory.tiers import count_tokens


def _ts(y: int, m: int, d: int) -> float:
    return datetime(y, m, d, tzinfo=timezone.utc).timestamp()


def test_phase0_gate_build_supersede_query_serialize():
    s = GraphStore()
    user = s.ensure_node("user")
    rem = s.ensure_node("rem")

    s.add(Edge(subject_id=user.id, relation="lives_in", object_literal="Denver",
               valid_from=_ts(2024, 3, 1), ingested_at=_ts(2024, 3, 1)))
    s.add(Edge(subject_id=rem.id, relation="target_hw", object_literal="strix_halo",
               valid_from=_ts(2024, 1, 1), ingested_at=_ts(2024, 1, 1)))

    # user moves: Denver -> Boulder
    s.add(Edge(subject_id=user.id, relation="lives_in", object_literal="Boulder",
               valid_from=_ts(2024, 6, 1), ingested_at=_ts(2024, 6, 1)))

    # current state
    current = {
        (s.get_node(e.subject_id).label, e.relation): e.object_literal
        for e in s.current_edges()
    }
    assert current[("user", "lives_in")] == "Boulder"
    assert current[("rem", "target_hw")] == "strix_halo"

    # point in time (event time)
    before = {e.object_literal for e in s.state_at_event_time(_ts(2024, 4, 1))
              if e.relation == "lives_in"}
    after = {e.object_literal for e in s.state_at_event_time(_ts(2024, 7, 1))
             if e.relation == "lives_in"}
    assert before == {"Denver"}
    assert after == {"Boulder"}

    # serialized context
    text = serialize_current(s)
    assert "user lives_in Boulder (since 2024-06)" in text
    assert "rem target_hw strix_halo (since 2024-01)" in text
    assert "Denver" not in text
    assert count_tokens(text) > 0

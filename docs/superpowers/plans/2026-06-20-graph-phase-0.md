# Graph Phase 0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an in-memory temporal graph store (facts as edges) with supersession-as-data and deterministic, model-free serialization to context text.

**Architecture:** Three focused pydantic-based modules under `src/rem/graph/`: `model.py` (Node, Edge), `store.py` (GraphStore: add/supersede/temporal queries), `serialize.py` (currently-valid edges → text). In-memory only; no models, no dataset, no persistence. Bitemporal: event time (`valid_from`/`valid_to`) and transaction time (`ingested_at`/`invalidated_at`) are independent.

**Tech Stack:** Python 3.12, pydantic v2 (matching `src/rem/memory/tiers.py`), pytest. Reuses `rem.memory.tiers.count_tokens`.

**Spec:** `docs/superpowers/specs/2026-06-20-graph-phase-0-design.md`

## Global Constraints

- Phase 0 has **no models and no dataset**; tests are deterministic and offline.
- `embedding` fields exist on the models but stay `None` in Phase 0.
- Nothing in `src/rem/graph/` imports from the current compaction path (`rem.memory.facts_ledger`, `compactor`, `assembler`), and vice versa. The only allowed cross-import is `rem.memory.tiers.count_tokens`.
- Node dedup is by exact label, case-insensitive, first-seen canonical.
- Supersession closes a superseded edge with `valid_to = new.valid_from` (event time) and `invalidated_at = new.ingested_at` (transaction time).
- Run tests with `python3 -m pytest` (config sets `-q -m 'not npu'`).
- Match `tiers.py` conventions: pydantic `BaseModel`, `Field(default_factory=...)`.

---

### Task 1: Data model (Node, Edge)

**Files:**
- Create: `src/rem/graph/__init__.py`
- Create: `src/rem/graph/model.py`
- Test: `tests/unit/test_graph_model.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `Node(label: str, id: str=auto, aliases: list[str]=[], embedding: list[float]|None=None, created_at: float=auto)`
  - `Edge(subject_id: str, relation: str, valid_from: float, object_id: str|None=None, object_literal: str|None=None, valid_to: float|None=None, ingested_at: float=auto, invalidated_at: float|None=None, source_turn_id: int|None=None, kind: Literal["entity","number","decision","quote"]="entity", embedding: list[float]|None=None)` — validator requires exactly one of `object_id`/`object_literal`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_graph_model.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/unit/test_graph_model.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'rem.graph'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/rem/graph/__init__.py
"""Temporal graph-resident memory (Phase 0: store + serialization)."""
```

```python
# src/rem/graph/model.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/unit/test_graph_model.py -q`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/rem/graph/__init__.py src/rem/graph/model.py tests/unit/test_graph_model.py
git commit -m "REM(graph): Phase 0 data model — Node and Edge with one-of object"
```

---

### Task 2: Store — nodes, add, supersession, current_edges

**Files:**
- Create: `src/rem/graph/store.py`
- Test: `tests/unit/test_graph_store.py`

**Interfaces:**
- Consumes: `Node`, `Edge` from `rem.graph.model`.
- Produces: `GraphStore` with
  - `add_node(node: Node) -> Node`
  - `ensure_node(label: str, aliases: tuple[str, ...]=()) -> Node`
  - `get_node(node_id: str) -> Node | None`
  - `add(edge: Edge, *, supersede: bool=True) -> Edge`
  - `current_edges() -> list[Edge]`
  - attributes `nodes: dict[str, Node]`, `edges: list[Edge]`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_graph_store.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/unit/test_graph_store.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'rem.graph.store'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/rem/graph/store.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/unit/test_graph_store.py -q`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add src/rem/graph/store.py tests/unit/test_graph_store.py
git commit -m "REM(graph): Phase 0 store — nodes, supersession-as-data, current_edges"
```

---

### Task 3: Temporal point-in-time queries

**Files:**
- Modify: `src/rem/graph/store.py` (add two methods to `GraphStore`)
- Test: `tests/unit/test_graph_store.py` (append)

**Interfaces:**
- Consumes: `GraphStore` from Task 2.
- Produces:
  - `GraphStore.state_at_event_time(t: float) -> list[Edge]`
  - `GraphStore.belief_at_transaction_time(t: float) -> list[Edge]`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/unit/test_graph_store.py
def test_state_at_event_time_returns_old_before_new_after():
    s = GraphStore()
    s.add(Edge(subject_id="u", relation="lives_in", object_literal="Denver",
               valid_from=1.0, ingested_at=1.0))
    s.add(Edge(subject_id="u", relation="lives_in", object_literal="Boulder",
               valid_from=2.0, ingested_at=2.0))
    before = [e.object_literal for e in s.state_at_event_time(1.5)]
    after = [e.object_literal for e in s.state_at_event_time(2.5)]
    assert before == ["Denver"]
    assert after == ["Boulder"]


def test_belief_at_transaction_time_uses_ingestion_not_event_time():
    s = GraphStore()
    s.add(Edge(subject_id="u", relation="lives_in", object_literal="Denver",
               valid_from=1.0, ingested_at=1.0))
    # Became true at event time 2.0, but the system only LEARNED it at txn time 5.0.
    s.add(Edge(subject_id="u", relation="lives_in", object_literal="Boulder",
               valid_from=2.0, ingested_at=5.0))
    # At transaction time 3.0 the system still believed Denver (Boulder learned at 5.0).
    believed = [e.object_literal for e in s.belief_at_transaction_time(3.0)]
    assert believed == ["Denver"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/unit/test_graph_store.py -k "event_time or transaction_time" -q`
Expected: FAIL — `AttributeError: 'GraphStore' object has no attribute 'state_at_event_time'`

- [ ] **Step 3: Write minimal implementation**

Add these two methods to `GraphStore` in `src/rem/graph/store.py` (after `current_edges`):

```python
    def state_at_event_time(self, t: float) -> list[Edge]:
        """Facts true at event time t, per current beliefs."""
        return [
            e for e in self.edges
            if e.invalidated_at is None
            and e.valid_from <= t
            and (e.valid_to is None or t < e.valid_to)
        ]

    def belief_at_transaction_time(self, t: float) -> list[Edge]:
        """What the system believed at transaction time t."""
        return [
            e for e in self.edges
            if e.ingested_at <= t
            and (e.invalidated_at is None or t < e.invalidated_at)
        ]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/unit/test_graph_store.py -q`
Expected: PASS (8 tests)

- [ ] **Step 5: Commit**

```bash
git add src/rem/graph/store.py tests/unit/test_graph_store.py
git commit -m "REM(graph): Phase 0 point-in-time queries (event + transaction time)"
```

---

### Task 4: Serialization to context text

**Files:**
- Create: `src/rem/graph/serialize.py`
- Test: `tests/unit/test_graph_serialize.py`

**Interfaces:**
- Consumes: `Edge` from `rem.graph.model`; `GraphStore` from `rem.graph.store`; `rem.memory.tiers.count_tokens`.
- Produces:
  - `serialize_edges(edges: list[Edge], resolve_label: Callable[[str], str]) -> str`
  - `serialize_current(store: GraphStore) -> str`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_graph_serialize.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/unit/test_graph_serialize.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'rem.graph.serialize'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/rem/graph/serialize.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/unit/test_graph_serialize.py -q`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/rem/graph/serialize.py tests/unit/test_graph_serialize.py
git commit -m "REM(graph): Phase 0 serialization of currently-valid edges to context text"
```

---

### Task 5: Phase 0 gate (end-to-end)

**Files:**
- Test: `tests/unit/test_graph_phase0_gate.py`

**Interfaces:**
- Consumes: `GraphStore`, `Edge`, `serialize_current`, `count_tokens`.
- Produces: nothing (acceptance test only).

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_graph_phase0_gate.py
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
```

- [ ] **Step 2: Run test to verify it fails (then passes)**

Run: `python3 -m pytest tests/unit/test_graph_phase0_gate.py -q`
Expected: PASS immediately if Tasks 1–4 are complete. (If you reached this task first, it FAILs with `ModuleNotFoundError` — implement Tasks 1–4 first.) This task adds no production code; it is the acceptance gate over the prior tasks.

- [ ] **Step 3: Run the full suite (no regressions)**

Run: `python3 -m pytest -q`
Expected: PASS — all prior tests plus the 4 new graph test files.

- [ ] **Step 4: Commit**

```bash
git add tests/unit/test_graph_phase0_gate.py
git commit -m "REM(graph): Phase 0 gate test — build, supersede, query, serialize end-to-end"
```

---

## Self-Review

**Spec coverage:**
- §3 module layout → Tasks 1 (`model`), 2+3 (`store`), 4 (`serialize`). ✓
- §4 data model (Node/Edge, one-of, bitemporal fields, embedding=None) → Task 1. ✓
- §5 store ops (add_node/ensure_node/get_node/add/current_edges) → Task 2; temporal queries → Task 3. ✓
- §5.1 supersession + bitemporal decision → Task 2 (`invalidated_at = new.ingested_at`, tested in `test_supersession_closes_old_edge_with_different_object` and `test_belief_at_transaction_time_uses_ingestion_not_event_time`). ✓
- §6 serialization (label resolution, literal, since-suffix, count_tokens) → Task 4. ✓
- §7 testing (per-module + gate) → Tasks 1–5. ✓
- §8 acceptance (package + tests + isolation) → Tasks 1–5; isolation enforced by Global Constraints + only `count_tokens` imported. ✓

**Placeholder scan:** No TBD/TODO; every code step shows complete code. ✓

**Type consistency:** `Edge`/`Node` signatures in Task 1 match usage in Tasks 2–5. `_object_key`, `_current_for_slot`, `state_at_event_time`, `belief_at_transaction_time`, `serialize_edges`, `serialize_current` names are consistent across tasks. ✓

**Note:** Phase 0 always renders a `(since <YYYY-MM>)` suffix because `valid_from` is required; the spec's mixed example (some lines without `since`) is illustrative. Timeless facts (optional `valid_from`) are deferred — not a Phase 0 requirement.

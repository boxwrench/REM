# Graph Memory — Phase 0 Design (store + serialization)

**Date:** 2026-06-20
**Status:** Design, awaiting approval
**Parent spec:** [`docs/REM-memory-architecture-spec.md`](../../REM-memory-architecture-spec.md) §4, §8, §10
**Phase gate (from parent §10):** "Build a graph by hand, serialize it to context,
run a query against it, get the correct currently-valid edges back, deterministically."

## 1. Goal and scope

Build the data substrate for the graph-resident memory: a temporal graph store
whose edges are facts, with supersession expressed as data (not a prompt
instruction) and a deterministic, model-free serialization into context text.

This is Phase 0 only. It is unit-tested, with **no models and no dataset**. It is
the foundation the later read path (Phase 1) and write path (Phase 2) build on.

Building this now is deliberately ahead of the gate decision: the architecture
spec selects the graph only if the tight-budget battery shows summary/stale-ghost/
read-path failures dominate. Phase 0 is low-risk and reusable regardless, and it
does not touch the current compaction path (the system under test).

## 2. Non-goals (explicitly deferred)

- **Embeddings / entity resolution by meaning** → Phase 1+. Phase 0 dedups nodes
  by exact label only. The `embedding` fields exist on the model but stay `None`.
- **Retrieval (vector / BM25 / traversal / ranking)** → Phase 1.
- **Worker extraction, overflow buffer, ingestion** → Phase 2.
- **Persistence / backend choice** (SQLite, graph db) → later. Phase 0 is
  in-memory behind a clean store interface so a backend can swap in without
  touching callers.
- **Relation normalization / controlled vocabulary** → freeform predicates for now.

## 3. Module layout

New package `src/rem/graph/`, three focused modules (pydantic models, matching
`src/rem/memory/tiers.py` conventions):

| Module | Responsibility | Public surface |
|---|---|---|
| `model.py` | Data model | `Node`, `Edge` |
| `store.py` | Store + supersession + temporal queries | `GraphStore` |
| `serialize.py` | Currently-valid edges → context text | `serialize_edges`, `serialize_current` |

## 4. Data model (`model.py`)

`Node` (entity):
- `id: str` — stable internal id (auto `uuid4().hex` if not supplied)
- `label: str` — canonical display string
- `aliases: list[str] = []` — observed surface forms
- `embedding: list[float] | None = None` — Phase 0: always None
- `created_at: float` — `default_factory=time`

`Edge` (one fact):
- `id: str` — auto `uuid4().hex` if not supplied
- `subject_id: str` — a node id
- `relation: str` — predicate string (freeform)
- object as a validated **one-of**:
  - `object_id: str | None = None` — node id, when the object is an entity
  - `object_literal: str | None = None` — literal value (string/number/date as string)
  - validator: exactly one of the two is non-null
- event timeline:
  - `valid_from: float` — when the fact became true (epoch)
  - `valid_to: float | None = None` — when it stopped being true; None = still true
- transaction timeline:
  - `ingested_at: float` — `default_factory=time`; when the system learned it
  - `invalidated_at: float | None = None` — when the system learned it ended; None = still believed
- `source_turn_id: int | None = None` — provenance
- `kind: Literal["entity", "number", "decision", "quote"] = "entity"`
- `embedding: list[float] | None = None` — Phase 0: always None

Temporal fields are floats for clean, deterministic ordering. A tiny test helper
may convert ISO dates → epoch; production callers (Phase 2) supply timestamps.

## 5. Store + supersession (`store.py`)

`GraphStore` holds `nodes: dict[str, Node]` and `edges: list[Edge]` in memory.

Methods:
- `add_node(node) -> Node` / `ensure_node(label, aliases=()) -> Node` — dedup by
  exact label (case-insensitive); record new surface forms as aliases.
- `get_node(node_id) -> Node | None`
- `add(edge, *, supersede=True) -> Edge` — apply supersession (below), then insert.
- `current_edges() -> list[Edge]` — `valid_to is None and invalidated_at is None`.
- `state_at_event_time(t: float) -> list[Edge]` — facts true at event time `t`
  per current beliefs: `valid_from <= t and (valid_to is None or t < valid_to)`
  and `invalidated_at is None`.
- `belief_at_transaction_time(t: float) -> list[Edge]` — what the system believed
  at transaction time `t`: `ingested_at <= t and (invalidated_at is None or t < invalidated_at)`.

### 5.1 Supersession (parent §4.3, with one decision)

On `add(new_edge)` with `supersede=True`: find currently-valid edges with the same
`(subject_id, relation)`.
- If an existing valid edge has a **different** object → close it, then insert the new edge.
- If the object is **identical** → skip the insert (no-op).
- Otherwise → insert the new edge.

> **DECISION — deviation from parent §4.3 (needs sign-off).**
> The parent spec says close the old edge by setting both `valid_to` *and*
> `invalidated_at` to the new edge's `valid_from`. But those are different
> timelines: `valid_to` is event time (when the fact stopped being true),
> `invalidated_at` is transaction time (when we *learned* it changed). Setting
> `invalidated_at` to `valid_from` makes "what did we believe at transaction
> time T" queries wrong. This design sets:
> - `old.valid_to = new.valid_from` (event time), and
> - `old.invalidated_at = new.ingested_at` (transaction time).
>
> This is the principled bitemporal model and is what makes both point-in-time
> queries correct — the reason we chose to include point-in-time at all. If you
> prefer strict spec fidelity, we set both to `new.valid_from` instead and accept
> that transaction-time queries are not meaningful in Phase 0.

## 6. Serialization (`serialize.py`)

`serialize_edges(edges, resolve_label) -> str` and a convenience
`serialize_current(store) -> str`.

- One fact per line, currently-valid edges only by default.
- Resolve `subject_id`/`object_id` to node labels via the store; render
  `object_literal` verbatim.
- Format: `<subject_label> <relation> <object>`, with an optional `(since <iso-date>)`
  suffix derived from `valid_from` when present.
- Token cost reported via the existing `rem.memory.tiers.count_tokens` (the
  serialized form is the budget that matters, per parent §4.4).

Example output:
```text
user lives_in Denver (since 2024-03)
rem target_hw strix_halo
invoice.q3 owner billing_team
```

## 7. Testing strategy

Unit tests per module, no models, fully deterministic:
- `model.py`: one-of object validator (rejects both-set and neither-set); id/timestamp defaults.
- `store.py`: node dedup + aliases; supersession (different object closes old,
  identical object is a no-op); `current_edges`; `state_at_event_time` returns the
  old value before the update timestamp and the new value after;
  `belief_at_transaction_time` reflects the bitemporal decision.
- `serialize.py`: label resolution, literal rendering, `(since …)` suffix, token count.

**Phase 0 gate test** (end-to-end, the parent §10 gate): hand-build a small graph,
apply a supersession, assert `current_edges()` and `state_at_event_time()` give
old-before / new-after, and that serialization matches expected lines exactly.

## 8. Acceptance criteria

- The package exists with the three modules and their public surface.
- All unit tests pass; the Phase 0 gate test passes deterministically.
- No dependency on any model or dataset; nothing imports from the current
  compaction path (and vice versa).

## 9. Alternatives kept open

These are viable roads not taken in Phase 0. Recorded so we can switch
deliberately rather than rediscover them. None block Phase 0; the clean store
interface and pydantic models are chosen partly to keep these cheap to adopt.

| Decision | Phase 0 choice | Viable alternatives | Revisit when |
|---|---|---|---|
| **Backend** | In-memory (dicts/list) behind a store interface | SQLite edge table; embedded graph store (NetworkX, Kùzu, DuckDB-PGQ) | Phase 0 needs persistence, or edge volume / traversal cost outgrows lists |
| **Supersession timeline** (§5.1) | `invalidated_at = new.ingested_at` (principled bitemporal) | Strict parent-spec (`= new.valid_from`); append-only belief log (never mutate old edges) | A consumer needs strict spec parity, or we want full audit immutability |
| **Object representation** | One-of `object_id` xor `object_literal` (literal as string) | Tagged union `{kind, value}`; typed literals (number/date as real types); reify every literal as a value-node | Literal typing matters for ranking/comparison, or queries need typed math/date ops |
| **Temporal field type** | `float` epoch | ISO-8601 strings; `datetime`; integer logical clock | Human-readable storage or calendar-aware queries become important |
| **Node dedup** | Exact label, case-insensitive, first-seen canonical | Embedding similarity threshold (Phase 1); alias table; normalization (stemming/lowercasing) | Phase 1 entity resolution lands, or fragmentation hurts recall |
| **Relation vocabulary** | Freeform predicates | Controlled vocabulary; relation normalization / aliases (parent §11) | Synonymous relations fragment retrieval |
| **`state_at_event_time` semantics** | Current-beliefs filter (`invalidated_at is None`) | Full bitemporal as-of query (event time T *and* transaction time S) | A test/consumer needs "what we believed at S about T" |
| **Module decomposition** | 3 modules (`model`/`store`/`serialize`) | Single module; merge `model`+`store` | A module stays trivially small or boundaries prove artificial |

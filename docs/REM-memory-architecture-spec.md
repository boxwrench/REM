# REM Memory Architecture

**Version:** 0.1 (draft)
**Status:** Design, pre-implementation
**Target:** AMD Strix Halo (Ryzen AI, XDNA2 NPU + iGPU), single-box local agent

## 1. Goal and scope

Replace the generative-compaction memory path with a graph-resident store. Old
conversation is not summarized into prose by a model. It is decomposed into
facts held as edges in a temporal graph, retrieved per turn by non-model search,
and serialized back into the foreground context on demand.

The previous design's failure mode was write-time and permanent: a 1B model
rewrote history into summaries and a fact ledger, and dropped or corrupted facts
in the process. This design removes model-authored prose summaries from the
memory record. The stored representation is a graph. The worker emits
provenance-backed candidate facts, deterministic code resolves and commits them,
and retrieval reads from the graph with no model in the loop.

Out of scope for v0.1: multi-user graphs, cross-session identity beyond a
single agent, and any community or cluster summarization layer.

## 1.1 Weighted assessment

This design is promising, but it should be treated as a staged candidate
architecture rather than an immediate full rewrite.

| Weight | Insight | Implication |
|---:|---|---|
| 5 | **Stop storing model-written prose as memory.** Prose summaries are compact but hard to audit, hard to supersede, and can permanently rewrite or drop history. | This is the strongest architectural move. Replace summaries with structured, provenance-backed facts if the current compaction path cannot prove recall. |
| 5 | **Separate write recall from read recall.** The graph can be correct but incomplete, or complete but unretrievable. Those are different failures. | Acceptance tests must report write recall and read recall separately, not only final answer accuracy. |
| 5 | **Keep the per-turn read path model-free.** Retrieval by vector, BM25, graph traversal, and deterministic ranking is easier to bound and debug than asking a model to remember. | This should be a hard constraint for the graph design. The foreground model should only consume injected facts. |
| 4 | **Temporal edges are the right primitive for stale facts.** `valid_from` / `valid_to` plus `ingested_at` / `invalidated_at` make current-state and point-in-time behavior testable. | Supersession moves from prompt instruction to data semantics. This directly targets stale-ghost failures. |
| 4 | **The first slice should be read-only and seeded.** A hand-built graph can validate serialization, retrieval, ranking, and stale suppression without model extraction noise. | Build Phase 0 and Phase 1 before adding the worker extractor. Do not debug ingestion and retrieval at the same time. |
| 4 | **Extraction remains the bottleneck.** The worker still has to convert messy conversation into facts. The graph reduces corruption blast radius, but it does not magically recover missed facts. | If the tight-budget battery shows extraction misses dominate, graph work must start with write-recall instrumentation and extractor quality. |
| 3 | **Entity and relation fragmentation can silently kill recall.** `deployment window`, `release window`, and `deploy time` may become different relations unless normalized. | Start with freeform predicates, but measure fragmentation and be ready to introduce relation normalization or aliases. |
| 3 | **Immediate eviction plus async filing is risky.** If evicted turns leave the live window before they are committed, failed extraction can become permanent loss. | The overflow buffer and dropped-fact telemetry are not optional; they are correctness machinery. |
| 2 | **LongMemEval may not cover all narrative memory.** A graph is well matched to fact/update recall, less obviously to long-range narrative or causal context. | Use LongMemEval as the first gate, but do not treat it as the only future memory eval. |

Decision: keep the current tight-budget LongMemEval + JSON robustness gate as the
immediate work. Use this graph design as the next architecture if the current
pipeline loses because summaries, stale ghosts, or read-path ambiguity dominate.
If the current pipeline loses mostly because extraction misses facts, fix write
recall before assuming the graph will help.

## 2. Work placement

Three execution targets, fixed:

- **iGPU:** foreground model, generation only. Per-turn cost is limited to
  prefill of live transcript plus injected facts, and decode of the response.
- **NPU:** the embedding model (query encoding, entity resolution, edge
  embeddings) and the background worker model. Both are encoder or
  small-decoder workloads, the correct fit for this silicon.
- **CPU:** graph store, traversal, ranking, serialization, scheduler.

Hard rule: no generative model runs in the per-turn read path. Retrieval is
search and ranking only.

## 3. Context budget

Foreground window is 32k tokens, split:

- `live_window`: 16k. Most recent turns, verbatim. Preserves immediate
  conversational coherence and exact wording for recent context.
- Remaining 16k: system prompt, retrieved facts for the current turn, and decode
  headroom.

The window is not reduced. Half of it is reserved for retrieved long-term
memory.

## 4. Data model

### 4.1 Node

A node is an entity (person, place, object, concept).

- `id`: stable internal identifier.
- `label`: canonical display string.
- `aliases`: observed surface forms.
- `embedding`: vector for entity resolution and semantic retrieval.
- `created_at`.

### 4.2 Edge (fact)

An edge is a single fact.

- `id`.
- `subject_id`: node.
- `relation`: predicate string.
- `object`: either `object_id` (node) or a literal (string, number, date).
- `valid_from`: event time the fact became true.
- `valid_to`: event time it stopped being true. Null means still true.
- `ingested_at`: transaction time the system learned it.
- `invalidated_at`: transaction time the system learned it was no longer true.
  Null means still believed.
- `source_turn_id`: provenance.
- `kind`: entity | number | decision | quote.
- `embedding`: vector of the edge's serialized text.

Two independent timelines: event time (`valid_from`, `valid_to`) and
transaction time (`ingested_at`, `invalidated_at`). This separation lets the
graph answer both "what is true now" and "what did we believe at time T" without
deleting anything.

### 4.3 Supersession

When a new edge arrives for an existing (`subject`, `relation`) pair with a
different object, the prior edge is not deleted. Set its `valid_to` and
`invalidated_at` to the new edge's `valid_from`, then insert the new edge.
Current state is the set of edges with `valid_to` null and `invalidated_at`
null. History remains queryable.

### 4.4 Serialization (the context format)

Edges are serialized to the foreground model as compact, one-fact-per-line text,
English plus minimal symbols. Currently-valid edges only, by default. Example:

```text
user lives_in Denver (since 2024-03)
rem target_hw strix_halo
invoice.q3 owner billing_team
```

Internal storage may use any denser encoding (integer ids, interned relations).
The model never reads internal storage. It reads the serialized form. The token
budget that matters is the serialized form, not the store.

## 5. Write path (ingestion)

Triggered by the live window crossing its limit.

1. **Evict.** When `live_window` token count exceeds 16k, remove oldest turns
   until the count is at or below `evict_low_water` (12k). Eviction is batched
   so the worker does not run on every turn.
2. **Extract.** The worker model reads the evicted batch and emits candidate
   facts (subject, relation, object, kind), with `valid_from` defaulting to the
   turn's timestamp and `source_turn_id` set.
3. **Resolve entities.** For each subject and object that is an entity, embed
   the surface form and match against existing node embeddings. Above
   `entity_match_threshold`, reuse the node and record the new surface form as
   an alias. Otherwise create a node.
4. **Apply supersession.** For each new edge, find currently-valid edges with
   the same (subject, relation). If the object differs, invalidate the prior
   edge per 4.3 and insert the new edge. If identical, skip.
5. **Commit.** Write nodes and edges. Compute and store edge embeddings.

Ingestion runs in a background worker. The foreground turn does not block on it.
Eviction itself (removing turns from the live window) is immediate. Extraction
and commit happen behind it.

### 5.1 Failure handling

Extraction is the one lossy step in the system. If the worker returns nothing
usable for an evicted batch, those turns have already left the live window, so
silent loss is unacceptable.

- Hold evicted-but-unprocessed turns in a bounded `overflow_buffer`.
- Retry extraction once.
- On repeated failure, drop with a loud warning and increment a dropped-fact
  counter in telemetry. Do not fail silently.

This is a known, accepted lossy boundary. It is the primary thing the acceptance
tests must measure.

## 6. Read path (retrieval)

Runs every turn, before generation. No model.

1. **Encode.** Embed the current query (and optionally the last few verbatim
   turns) with the embedding model.
2. **Generate candidates.** Union of three sources:
   - vector similarity of the query embedding against edge embeddings,
   - keyword (BM25) match against edge text, node labels, and aliases,
   - graph traversal: 1 to 2 hop neighbors of any node named in the query.
3. **Filter.** Keep only currently-valid edges (`valid_to` and
   `invalidated_at` null) unless the query is explicitly point-in-time.
4. **Rank.** Reciprocal rank fusion across the three candidate lists.
5. **Trim.** Reduce to `context_edge_budget` using a diversity pass (MMR) or a
   similarity floor. Retrieve wide for recall, hand over narrow for precision.
6. **Serialize and inject.** Format per 4.4, place in the reserved context
   region.

The foreground model performs final relevance selection by reading the injected
facts. Retrieval's job is recall: get the needed fact into the candidate set
with good ranking. The model's job is precision: use the ones that matter.

## 7. Configuration

Defaults, all tunable:

```text
foreground_context_window   32000
live_window_tokens          16000
evict_low_water_tokens      12000
retrieve_candidate_k        50
context_edge_budget         18      # start; tune 15-20
entity_match_threshold      0.82    # cosine; calibrate on-box
overflow_buffer_turns       20
worker_model                llama3.2:1b   # or smaller; validate on-box
embedding_model             (small encoder; select and validate on-box)
```

Model and library versions are starting points. Validate each on the box before
fixing it in the build. Do not assume a documented version works until it runs
locally.

## 8. Interfaces

Sketch-level. Names indicative.

```python
ingest(turns: list[Turn]) -> IngestResult
    # extract, resolve, supersede, commit. Background.

retrieve(query: str, k: int, budget: int) -> list[Edge]
    # model-free. Returns currently-valid edges, ranked and trimmed.

serialize(edges: list[Edge]) -> str
    # compact context format, currently-valid only.

assemble(system: str, live_turns: list[Turn], facts: str) -> Context
    # system + reserved-region facts + live verbatim transcript.

resolve_entity(surface: str) -> NodeId
    # embed, match above threshold or create.

supersede(new_edge: Edge) -> None
    # invalidate conflicting current edges, insert new.
```

## 9. Acceptance criteria

Measured on the real path, on the target box. A structurally similar test does
not count.

- **Write recall.** Fraction of known injected facts present as correctly-valued,
  currently-valid edges after ingestion. Must exceed the current pipeline's
  rate.
- **Read recall.** Fraction of queries for which the needed valid edge lands in
  the injected context. This is the headline quality number.
- **Supersession correctness.** For scripted updates, a point-in-time query
  returns the old value before the update timestamp and the new value after. No
  stale value surfaces as current.
- **Retrieval latency.** p95 within budget (target 400ms, no model in path).
- **Foreground overhead.** Distribution of injected tokens per turn. Confirm it
  stays well under the reserved 16k.
- **Contention.** Background worker's iGPU decode loss within the existing
  contention budget. Reuse the current harness.

The quality battery must run at a context budget tight enough that retrieval
matters. A budget generous enough for naive truncation to win does not test this
system.

## 10. Build phases

Each gate is a hard stop. Do not start the next phase until the current gate
passes.

- **Phase 0: store and serialization.** Data model, graph store, supersession
  logic, serialization. Unit tests only, no models. Gate: build a graph by hand,
  serialize it to context, run a query against it, get the correct
  currently-valid edges back, deterministically.
- **Phase 1: read path.** Model-free retrieval over a seeded graph. Gate: read
  recall on a fixed seeded graph meets threshold with no ingestion involved. The
  read path is validated in isolation before any extraction exists.
- **Phase 2: write path.** Worker extraction, entity resolution, supersession,
  overflow handling. Gate: write recall and supersession correctness on a
  scripted conversation with known facts and known updates.
- **Phase 3: integration on-box.** End to end on Strix Halo. Gate: latency,
  contention, and the tight-budget battery, beating the current approach on read
  recall.
- **Phase 4 (optional): drop the worker model.** Replace worker extraction with
  structured deltas emitted by the foreground model as a side output, merged
  deterministically. Removes the backend model. Pursue only after Phase 3 holds.

## 11. Open decisions

- **Extraction source.** Default is the background worker model (Phase 2).
  Foreground-emitted deltas (Phase 4) are cleaner and remove a model, at the cost
  of a small tax on the foreground turn. Deferred, not rejected.
- **Model authorship boundary.** The design removes model-authored summaries,
  not model involvement entirely. Decide how strictly worker output must be
  constrained: raw candidate facts, typed schema with validation, or foreground
  tool-call deltas.
- **Relation vocabulary.** Freeform predicates to start. If entity resolution or
  retrieval fragments across synonymous relations, introduce a controlled set.
- **Ambiguous entity matches.** Behavior when a surface form matches two nodes
  near threshold: merge, branch, or defer. Undecided.
- **Prose summaries.** None. The verbatim window plus the graph replace the
  summary tier. Revisit only if read recall on long-range narrative context
  proves insufficient.
- **Eviction semantics.** Decide whether evicted-but-unprocessed turns remain
  serializable as emergency context while the overflow buffer is non-empty, or
  whether retrieval must surface a degraded-memory warning.
- **Graph backend.** Choose the first store only after Phase 0 requirements are
  clear. A simple SQLite-backed edge table may be enough before adopting a graph
  database.
- **Evaluation shape.** LongMemEval `knowledge-update` is the first quality gate,
  but narrative continuity, preference recall, and multi-hop facts may need
  separate seeded tests.

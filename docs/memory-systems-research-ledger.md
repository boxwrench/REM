# REM Memory-Systems Research Ledger

**Status:** Living working document  
**Started:** 2026-06-28  
**Purpose:** Preserve usable findings from incoming research reports without
promoting unverified claims into REM's architecture or roadmap.

This is a synthesis and evidence ledger, not a dump of the source reports. New
reports should update the claim table, comparator list, mechanism backlog, and
open questions below. Conflicting claims stay visible until primary evidence or
an experiment resolves them.

## 1. Evidence labels

| Label | Meaning | Required treatment |
|---|---|---|
| **CODE** | Verified in the current implementation or committed artifact | May describe what exists; does not prove quality. |
| **PRIMARY** | Supported by an official paper, repository, documentation, or benchmark artifact | Record exact version, model, subset, budget, and judge when applicable. |
| **REPRODUCED** | Independently reproduced outside the system's authors or vendor | Strongest external evidence; preserve reproduction details. |
| **REPORTED** | Author, paper, or vendor result without independent reproduction | Useful hypothesis, not a REM decision criterion by itself. |
| **INFERRED** | Engineering expectation derived from architecture or hardware | Must be tested before becoming a claim. |
| **CONTRADICTED** | Conflicts with current code, a stronger source, or REM's measured state | Do not reuse without re-verification. |

Source hierarchy: committed code and raw artifacts; official papers and
benchmark repositories; official documentation; maintainer reports; secondary
analysis; marketing and social posts.

## 2. REM baseline for comparisons

Comparisons must use REM as it exists, not a projected architecture.

- The active runtime stores recent verbatim turns, prose summaries, and a facts
  ledger. Background NPU work performs fact extraction and summarization.
- The current read path is deterministic recency selection followed by
  render-aware trimming. It is **not** vector retrieval.
- The five captured oldest-gold LongMemEval states fit the 28,000-token read
  budget; four produced correct brief answers and one exposed a
  temporal-structure failure.
- Current write recall preserved the audited gold facts. The primary observed
  write defect is fragmented `slot_key` identity, which prevents supersession
  and bloats the ledger.
- The temporal graph is a Phase-0 in-memory prototype, not the active memory
  path.
- The current next increment is post-hoc, NPU-free slot-key canonicalization.

Local evidence:

- [`implementation-roadmap.md`](implementation-roadmap.md)
- [`FINDINGS.md`](../bench/battery/FINDINGS.md)
- [`selector.py`](../src/rem/memory/selector.py)
- [`REM-memory-architecture-spec.md`](REM-memory-architecture-spec.md)

## 3. Accepted findings

These findings are usable for experiment design. They are not approvals to
integrate an external system.

| ID | Finding | Evidence | Implication for REM |
|---|---|---|---|
| F-001 | Hindsight is the substantive memory system behind hal0's memory integration; hal0 supplies deployment, routing, MCP/REST, agents, and UI. | **CODE**: [hal0 memory provider](https://github.com/Hal0ai/hal0/tree/main/src/hal0/memory) | Compare REM with Hindsight directly; do not treat the whole hal0 appliance as the memory comparator. |
| F-002 | Hindsight separates world facts, experiences, opinions, and observations and exposes retain, recall, and reflect. | **PRIMARY**: [Hindsight paper](https://arxiv.org/html/2512.12818v1) | Strong candidate for a full black-box comparator and for studying evidence-versus-inference separation. |
| F-003 | Hindsight retain uses generative extraction; recall combines semantic, BM25, graph, and temporal retrieval, then RRF, cross-encoder reranking, and token-budget filtering. | **PRIMARY**: [Hindsight paper, TEMPR](https://arxiv.org/html/2512.12818v1) | Query-aware hybrid retrieval is the clearest read-path mechanism to test after the current write-side increment. |
| F-004 | Hindsight reports 83.6% LongMemEval accuracy with an open 20B backbone versus 39.0% full context with the same backbone, with larger-backbone results up to 91.4%. | **REPORTED**: [paper](https://arxiv.org/html/2512.12818v1), [benchmark repository](https://github.com/vectorize-io/hindsight-benchmarks) | Sufficient reason to test; not transferable to REM or hal0 without matching model, data, budget, and judge details. |
| F-005 | Graphiti is an open-source temporal context-graph engine with fact validity windows, retained source episodes, provenance, and hybrid retrieval. | **PRIMARY**: [Graphiti repository](https://github.com/getzep/graphiti) | Best mechanism comparator for temporal supersession and provenance. It is not necessary for the first query-aware retrieval experiment. |
| F-006 | LIGHT uses long-term episodic memory, short-term working memory, and a salient-fact scratchpad; BEAM tests histories up to 10M tokens. | **PRIMARY**: [BEAM/LIGHT repository](https://github.com/mohammadtavakoli78/BEAM) | Useful multi-store and extreme-horizon baseline; less direct than Hindsight or Graphiti for REM's immediate slot-identity defect. |
| F-007 | Hindsight's local service is modest in RAM, but retain/reflect require an LLM and local CPU reranking can become a latency bottleneck. | **PRIMARY**: [Hindsight installation guidance](https://hindsight.vectorize.io/developer/installation) | Storage capacity is unlikely to block Strix Halo. Extraction quality, write throughput, recall latency, and contention need on-box measurement. |
| F-008 | hal0's current Hindsight service forces its bundled embeddings and reranker to CPU and records poor latency/structured-output behavior for tested NPU extraction paths. | **CODE**: [hal0 service unit](https://github.com/Hal0ai/hal0/blob/main/installer/systemd/hindsight-api.service) | Do not assume Hindsight-on-NPU works. Test memory quality separately from execution placement. |

### 3.1 hal0 patterns to revisit

This is a capture list, not a weighted recommendation or verification gate. The
point is to retain potentially useful implementation patterns while reports are
still arriving. Revisit each item only when REM reaches the corresponding need.
Links are pinned to the hal0 revision inspected during R-001.

| Pattern to borrow or study | Why it may help REM | hal0 reference |
|---|---|---|
| Engine-neutral memory provider interface | Lets REM compare or swap native memory, Hindsight, Graphiti, or a test double without changing its sidecar/API callers. | [`MemoryProvider` contract](https://github.com/Hal0ai/hal0/blob/6e3ddbbf1b37d6644b88ba6de3c493307cf6f817/src/hal0/memory/provider.py) and [provider factory](https://github.com/Hal0ai/hal0/blob/6e3ddbbf1b37d6644b88ba6de3c493307cf6f817/src/hal0/memory/__init__.py) |
| Shared, private-agent, and project namespaces | Provides a clear future boundary for per-agent memory, shared knowledge, and project-local state without mixing their retrieval or deletion rules. | [namespace policy](https://github.com/Hal0ai/hal0/blob/6e3ddbbf1b37d6644b88ba6de3c493307cf6f817/src/hal0/memory/namespace.py) and [Hindsight bank mapping](https://github.com/Hal0ai/hal0/blob/6e3ddbbf1b37d6644b88ba6de3c493307cf6f817/src/hal0/memory/hindsight_provider.py) |
| One memory contract exposed through REST and MCP | Allows the sidecar, local agents, evaluation harnesses, and external tools to use the same add/search/recall/list/delete behavior. | [REST memory routes](https://github.com/Hal0ai/hal0/blob/6e3ddbbf1b37d6644b88ba6de3c493307cf6f817/src/hal0/api/routes/memory.py) and [MCP memory tools](https://github.com/Hal0ai/hal0/blob/6e3ddbbf1b37d6644b88ba6de3c493307cf6f817/src/hal0/mcp/memory.py) |
| Query-aware, multi-namespace recall with one token budget | Useful reference for moving beyond recency-only selection while keeping the injected memory bounded. | [`HindsightProvider.recall`](https://github.com/Hal0ai/hal0/blob/6e3ddbbf1b37d6644b88ba6de3c493307cf6f817/src/hal0/memory/hindsight_provider.py#L279-L352) |
| Fail-soft optional reranking | A retrieval enhancement should improve order when available without making memory recall fail when the reranker is unavailable. | [`Hal0Reranker`](https://github.com/Hal0ai/hal0/blob/6e3ddbbf1b37d6644b88ba6de3c493307cf6f817/src/hal0/memory/hindsight_provider.py#L39-L75) |
| Agent prefetch plus automatic completed-turn writeback | Shows how a runtime can inject a small recalled slice before a turn and retain the completed user/assistant exchange afterward. | [Hermes memory provider](https://github.com/Hal0ai/hal0/blob/6e3ddbbf1b37d6644b88ba6de3c493307cf6f817/src/hal0/agents/hermes/plugins/memory_hindsight/provider.py#L126-L183) |
| Asynchronous ingestion operation IDs | Gives callers visible progress for slow extraction instead of making newly written memory appear silently absent. | [`HindsightProvider.add`](https://github.com/Hal0ai/hal0/blob/6e3ddbbf1b37d6644b88ba6de3c493307cf6f817/src/hal0/memory/hindsight_provider.py#L139-L188) |
| Configurable extraction model slot | Separates the memory algorithm from whether extraction runs on CPU, iGPU, NPU, or an upstream model. This matches the rule that architecture and placement are independent axes. | [`MemoryGraphConfig.extraction_slot`](https://github.com/Hal0ai/hal0/blob/6e3ddbbf1b37d6644b88ba6de3c493307cf6f817/src/hal0/config/schema.py#L1586-L1633) |
| Memory-engine administration surface | Useful operational reference for inspecting banks, documents, graphs, operations, consolidation, reflection, and engine health without reading store files manually. | [memory admin routes](https://github.com/Hal0ai/hal0/blob/6e3ddbbf1b37d6644b88ba6de3c493307cf6f817/src/hal0/api/routes/memory_admin.py) |
| Explicit agent identity attached to memory calls | Supports provenance, private namespaces, and later audit of which agent wrote or deleted a memory. | [MCP identity handling](https://github.com/Hal0ai/hal0/blob/6e3ddbbf1b37d6644b88ba6de3c493307cf6f817/src/hal0/api/mcp_mount.py) and [memory MCP dispatcher](https://github.com/Hal0ai/hal0/blob/6e3ddbbf1b37d6644b88ba6de3c493307cf6f817/src/hal0/mcp/memory.py) |
| Approval gate for broad destructive memory operations | Single-item cleanup can remain routine while bulk deletion requires explicit authorization. | [memory tool dispatch](https://github.com/Hal0ai/hal0/blob/6e3ddbbf1b37d6644b88ba6de3c493307cf6f817/src/hal0/mcp/memory.py) and [approval queue](https://github.com/Hal0ai/hal0/blob/6e3ddbbf1b37d6644b88ba6de3c493307cf6f817/src/hal0/mcp/approval_queue.py) |
| Separate memory service lifecycle and persistent data root | A useful deployment pattern if REM eventually embeds a database-backed engine while keeping foreground inference independently restartable. | [Hindsight systemd service](https://github.com/Hal0ai/hal0/blob/6e3ddbbf1b37d6644b88ba6de3c493307cf6f817/installer/systemd/hindsight-api.service) |
| Hardware-aware named model slots | May provide a cleaner future placement interface than hard-coding a particular NPU endpoint into each memory mechanism. | [capability orchestrator](https://github.com/Hal0ai/hal0/blob/6e3ddbbf1b37d6644b88ba6de3c493307cf6f817/src/hal0/capabilities/orchestrator.py), [dispatcher](https://github.com/Hal0ai/hal0/blob/6e3ddbbf1b37d6644b88ba6de3c493307cf6f817/src/hal0/dispatcher/router.py), and [FLM provider](https://github.com/Hal0ai/hal0/blob/6e3ddbbf1b37d6644b88ba6de3c493307cf6f817/src/hal0/providers/flm.py) |

### 3.2 R-003 ideas and systems to revisit

These are report-derived leads, not verified findings or roadmap commitments.
They are recorded now so later primary-source review can be targeted rather than
repeating broad discovery work.

| Idea or system | Possible relevance to REM | Link to revisit | Status/caution |
|---|---|---|---|
| Bi-temporal fact identity | Track both event time (when something was true) and transaction time (when REM learned it). This may make corrections, delayed reports, and historical questions less ambiguous than one timestamp. | [Graphiti repository](https://github.com/getzep/graphiti), [Zep paper](https://arxiv.org/html/2501.13956v1) | Revisit after slot identity is stable; do not assume a graph database is required. |
| Typed memory atoms / MemIR | Preserve a structural distinction between raw evidence, retrieval cues, truth-bearing claims, user preferences, and generated inference. This directly addresses provenance-role collapse. | [MemIR paper](https://arxiv.org/html/2605.25869v1) | New lead from R-003; paper and implementation status require review. |
| Hierarchical temporal indexing / MemForest | Time-ordered local summaries may allow incremental writes and interval-directed recall without rewriting a global summary. The canonical leaves-versus-derived-index distinction fits REM's episode/fact provenance needs. | [MemForest paper](https://arxiv.org/html/2605.23986v1), [Hugging Face paper page](https://huggingface.co/papers/2605.23986) | New and currently paper-level; reported 79.8% LongMemEval-S and roughly 6x construction throughput need artifact review. |
| Intent as a separate memory layer / CogniFold | An intent layer could preserve active objectives separately from episodic events and semantic facts, potentially helping long-running tool agents. | [CogniFold paper](https://arxiv.org/html/2605.13438v3) | New and speculative for REM; proactive/test-time-learning claims need careful safety and reproducibility review. |
| Provenance-preserving derived views | Keep source episodes/canonical facts immutable while treating summaries, links, embeddings, heat scores, and graph roots as rebuildable views. | [MemForest paper](https://arxiv.org/html/2605.23986v1), [Graphiti repository](https://github.com/getzep/graphiti) | Architectural pattern worth preserving even if neither external engine is adopted. |
| Write-time entity resolution | Normalize entity and slot identity before or during durable insertion so fragmentation does not propagate into every index. | [Hindsight paper](https://arxiv.org/html/2512.12818v1), [Graphiti repository](https://github.com/getzep/graphiti) | Directly relevant to REM's current defect, but the report's claim that mathematical thresholds alone eliminate fragmentation is unverified. |
| Model-free graph propagation | Personalized PageRank or related spreading activation may retrieve associated evidence without iterative LLM calls. | [HippoRAG 2 paper](https://arxiv.org/html/2502.14802v2), [HippoRAG repository](https://github.com/osu-nlp-group/hipporag) | CPU-side candidate for a later graph experiment; only useful if simpler lexical/vector retrieval misses multi-hop evidence. |
| Governed memory evolution | If REM later rewrites, merges, decays, or deletes memories, add explicit stability rules, audit history, and rollback rather than allowing unconstrained background self-editing. | [SSGM paper](https://arxiv.org/html/2603.11768v2), [A-MEM paper](https://arxiv.org/html/2502.12110v2) | Governance is more reusable than the report's specific A-MEM implementation claims. |
| Heat plus time-decay retrieval | Retrieval frequency and recency could become secondary ranking features for frequently reused operational state. | [A-MEM-derived OpenClaw plugin](https://github.com/heichaowo/openclaw-amem) | Implementation-specific report claim; risks popularity feedback loops and must never replace relevance/provenance. |
| Interval-summary trees | Browse coarse time ranges first, then descend to source episodes, potentially reducing recall work over very long histories. | [MemForest paper](https://arxiv.org/html/2605.23986v1) | Candidate mechanism, not yet a black-box comparator recommendation. |
| Context-engineering controls for tool agents | Memory evaluation should include preserved task state, instructions, tool results, and bounded assembly—not only conversational fact QA. | [context-engineering paper](https://arxiv.org/pdf/2606.10209) | Review alongside agent-task benchmarks; scope is broader than LongMemEval. |

### 3.3 R-004 ideas and systems to revisit

R-004 mostly strengthens or refines earlier leads. These additions remain
unverified until their underlying citations are recovered from the research
session and checked against pinned code or primary artifacts.

| Idea or system | Possible relevance to REM | Link to revisit | Status/caution |
|---|---|---|---|
| Dirty-since-consolidation / observation freshness guard | A compact observation or profile should not be trusted as current when newer source facts have not yet been consolidated. Track the high-water mark used to build every derived summary and supplement or rebuild stale views. | [Hindsight repository](https://github.com/vectorize-io/hindsight), [Hindsight paper](https://arxiv.org/html/2512.12818v1) | Directly fits asynchronous maintenance and is cheaper to test than adopting Hindsight's full observation network. |
| Buffered profile + event-timeline flush | Keep newly inserted blobs in a recent buffer, then asynchronously transform them into a structured user profile and time-aware event history on a threshold, close, idle period, or manual flush. | [Memobase repository](https://github.com/memodb-io/memobase) | Strong architectural fit with REM's background maintenance; exact retention, update, and source-deletion behavior needs code review. |
| Bounded profile/event context API | Compare direct fact retrieval with a compact prompt assembled from stable user background plus recent events under an explicit maximum token size. | [Memobase repository](https://github.com/memodb-io/memobase) | Useful mechanism comparator even if Memobase is not adopted as a full engine. |
| Budget-capped greedy packing | Rank candidates, remove duplicates and superseded values, then greedily pack complete memory units until the token budget is exhausted rather than using a fixed `top-k`. | [Hindsight repository](https://github.com/vectorize-io/hindsight) | Refines backlog items 2-4; must preserve diversity and high-priority instructions rather than only score order. |
| Compact evidence references in the hot path | Keep full raw episodes off-path but inject source IDs, short supporting spans, and epistemic roles with selected facts so answers remain auditable without spending the whole prompt on provenance. | [Graphiti repository](https://github.com/getzep/graphiti), [Cognee repository](https://github.com/topoteretes/cognee) | Candidate compromise between full lineage and token cost. |
| Compact-fact baseline before a graph | Test a fact store with explicit update operations, validity metadata, BM25/dense retrieval, and bounded assembly before accepting graph database complexity. | [Mem0 repository](https://github.com/mem0ai/mem0), [Memobase repository](https://github.com/memodb-io/memobase) | Important control arm; REM's current baseline is still recency selection, not dense/vector retrieval. |
| Mem0 operation classifier | The report describes candidate memories being resolved via `ADD`, `UPDATE`, `DELETE`, or `NOOP`, with the graph variant invalidating conflicting edges. | [Mem0 paper](https://arxiv.org/html/2504.19413), [Mem0 repository](https://github.com/mem0ai/mem0) | Conflicts with R-002/R-003's “ADD-only” description; pin the relevant OSS version before drawing conclusions. |
| LangMem deferred/debounced reflection | Background reflection may benefit from debounce and versioned transformations rather than processing every turn independently. | [LangMem repository](https://github.com/langchain-ai/langmem) | Framework mechanism only; reported third-party latency makes it a poor performance baseline until reproduced. |
| Supermemory relational versioning | Study explicit `updates`, `extends`, and `derives` relationships plus separate document/event dates as an alternative to destructive fact replacement. | [Supermemory repository](https://github.com/supermemoryai/supermemory), [self-hosting documentation](https://supermemory.ai/docs/self-hosting/overview) | Vendor-described semantics and benchmark results require code and artifact verification. |
| Separate retrieval quality from answer quality | Score candidate recall and temporal correctness before the answer model runs, then score the final answer with a fixed model and judge. | [LongMemEval repository](https://github.com/xiaowu0162/LongMemEval) | Prevents stronger answer models from hiding incomplete retrieval. |

### 3.4 Primary-source relevance review (2026-06-28)

This review checks the ledger's decision-sensitive findings against primary
papers, current repositories, and REM's current code. Confidence concerns whether
the mechanism or implementation exists; relevance concerns whether it addresses a
failure REM has actually observed. Author benchmark results remain **REPORTED**
until independently rerun.

| Finding | Primary-source result | Calibrated confidence | Relevance to current REM | Decision consequence |
|---|---|---|---|---|
| Canonical facts and stable identity are prerequisites for temporal memory. | REM's exact-key supersession defect is measured locally. MemForest independently uses canonical facts as stable, temporally anchored write units before indexing. | **HIGH — CODE + PRIMARY**: [REM design](superpowers/specs/2026-06-28-slot-key-canonicalization-design.md), [MemForest paper](https://arxiv.org/html/2605.23986v1) | **Immediate / highest** | Finish the string-first canonicalization experiment. A graph, embedding, or external engine does not remove the need for stable fact identity. |
| Query-aware hybrid retrieval and hard token-budget packing are substantive Hindsight mechanisms. | Hindsight documents semantic, BM25, graph, and temporal channels; RRF; cross-encoder reranking; and greedy packing under a caller-provided budget. | **HIGH — PRIMARY**: [paper](https://arxiv.org/html/2512.12818v1), [repository](https://github.com/vectorize-io/hindsight) | **High, after identity measurement** | The smallest useful borrow is lexical/query-aware candidate retrieval plus budget packing, not the four-network architecture. Add embeddings/graph/reranking only if measured candidate recall requires them. |
| Hindsight maintains asynchronous entity observations. | The paper says observations are regenerated in the background when underlying facts change. It does **not** establish an explicit high-water mark, `proof_count`, evidence quotes, or a read-time stale-observation guard in the reviewed paper text. | **MEDIUM — PRIMARY mechanism; INFERRED freshness policy**: [paper](https://arxiv.org/html/2512.12818v1) | **Medium** | A dirty-since/high-water marker remains a good REM-native design inference, but should not be attributed to Hindsight until code proves it. |
| Graphiti implements temporal validity, invalidation, and episode provenance. | The paper and repository describe four timestamps, contradiction-driven edge invalidation, raw episodes, and bidirectional source links. Its LongMemEval result is 71.2% with GPT-4o versus 60.2% full context at about 1.6k context tokens. | **HIGH — PRIMARY architecture; REPORTED result**: [paper](https://arxiv.org/html/2501.13956v1), [repository](https://github.com/getzep/graphiti) | **Medium now; high if temporal failures recur** | Borrow event/transaction time and provenance semantics before adopting a graph database. Graph construction also adds multiple LLM calls and operational complexity. |
| Memobase uses buffered asynchronous consolidation into a profile and event timeline. | Its repository confirms threshold/idle/manual flushing, asynchronous processing, a profile plus latest-events context, `max_token_size`, fixed three-call workflows in v0.0.40, and deletion of processed source blobs by default. | **HIGH — PRIMARY implementation docs**: [repository](https://github.com/memodb-io/memobase) | **Medium-high mechanism fit** | Useful compact profile/event control arm. REM should retain source episodes by default, unlike Memobase's default, because auditability is a core requirement. |
| Mem0's update semantics are version-specific. | The 2025 paper and current OSS `Memory.add()` describe `ADD/UPDATE/DELETE/NOOP`. The April 2026 README separately advertises a new single-pass ADD-only algorithm with no update/delete. | **HIGH — PRIMARY, but product parity unresolved**: [paper](https://arxiv.org/html/2504.19413), [repository](https://github.com/mem0ai/mem0) | **Medium as a lower-complexity comparator** | Never cite “Mem0 behavior” without naming the revision and algorithm. The 94.8% headline cannot be assumed to describe the default OSS path. |
| MemForest is architecturally close to REM. | It combines parallel extraction, canonical fact consolidation, source references, persistent-versus-derived state, scoped temporal trees, dirty-path refresh, and coarse-to-fine retrieval. Code, configs, and per-question outputs are released. | **HIGH — PRIMARY architecture; REPORTED performance**: [paper](https://arxiv.org/html/2605.23986v1), [artifact](https://github.com/Concyclics/MemForest) | **High as a mechanism source** | Promote above generic graph-RAG systems for design study. Do not treat its roughly 6x throughput or 79.8% result as Strix evidence: runs used Qwen 4B/30B and an embedding model on dedicated H100s, and the artifact has no release. |
| Supermemory has a real local path and relevant temporal mechanisms. | Official docs describe a self-contained local server, local embeddings, OpenAI-compatible local LLMs, and the same API. Its research page describes `updates/extends/derives`, document/event dates, atomic memory plus raw-source chunks, and a 95% LongMemEval-S result at about 720 tokens. | **HIGH — PRIMARY local capability; REPORTED vendor result**: [self-host docs](https://supermemory.ai/docs/self-hosting/overview), [research report](https://supermemory.ai/research/longmembench/), [repository](https://github.com/supermemoryai/supermemory) | **High as an easy black-box comparator** | Worth a local smoke test after REM's native baseline is fixed. Hosted benchmark quality does not establish local-model extraction quality; its evaluation ingests session-by-session and reports Recall@15 with aggregation. |
| A-MEM supports small local backbones and linked evolving notes. | The paper evaluates Qwen 1.5B/3B and Llama 3.2 1B/3B via Ollama and reports 1.2k–2.5k answer-context tokens. It uses top-k retrieval and LLM-driven note/link evolution. No 1.1-second processing result appears in the paper. | **HIGH — PRIMARY mechanism; LOW for latency claim**: [paper](https://arxiv.org/html/2502.12110v2), [repository](https://github.com/agiresearch/A-mem) | **Low now** | Small-model feasibility is worth testing independently. Self-rewriting notes are premature until REM has reliable identity, provenance, rollback, and non-evolving baselines. |
| MemIR structurally separates evidence, cues, and truth-bearing claims. | The preprint defines typed claim atoms with provenance-scoped bundles and reports ablations on LoCoMo and BEAM-100K using mostly GPT-4.1-class construction/answering. No public code was located in the reviewed primary source. | **MEDIUM — PRIMARY preprint, no code found**: [paper](https://arxiv.org/html/2605.25869v1) | **Medium later** | The evidence/claim distinction is relevant to REM's source ledger, but implementing the full compiler/retrieval system now would precede the simpler provenance-ID experiment. |
| CogniFold adds an emergent intent layer and proactive graph folding. | A paper and code exist, but proactive evaluation uses six small synthetic scenarios of roughly 38–41 events generated from predefined gold graphs; the paper itself says larger-scale validity remains open. | **MEDIUM — PRIMARY preprint and code**: [paper](https://arxiv.org/html/2605.13438v3), [repository](https://github.com/OpenNorve/CogniFold) | **Low / outside current failure set** | Keep goals and intent as explicit short-lived task state for now. Do not add autonomous intent emergence to REM's durable memory path without a separate use case and safety model. |
| hal0 offers reusable integration patterns but no independent memory-quality evidence. | Provider abstraction, namespaces, REST/MCP surfaces, agent hooks, and administration were verified in pinned code; Hindsight supplies the actual memory algorithm. | **HIGH — CODE**: [pinned hal0 memory package](https://github.com/Hal0ai/hal0/tree/6e3ddbbf1b37d6644b88ba6de3c493307cf6f817/src/hal0/memory) | **Medium operational relevance** | Borrow interfaces and lifecycle patterns when REM needs them; do not use hal0 as a quality benchmark. |

### 3.5 Decision-ranked relevance

This ranking is an evidence review, not a roadmap change.

1. **Finish fact identity and ordered supersession.** Local evidence and
   MemForest's design both place canonical facts before temporal indexes.
2. **Expand the evaluation beyond five diagnostic states.** One ambiguous
   temporal miss cannot justify a graph or cognitive-memory rewrite.
3. **Test lexical/query-aware selection plus a hard token budget.** This isolates
   the most relevant part of Hindsight without adding embeddings, a graph, or a
   read-time LLM.
4. **Attach source episode IDs and derived-view freshness metadata.** This borrows
   Graphiti/MemForest provenance and the useful part of Hindsight observations
   while preserving REM's current architecture.
5. **Run native black-box comparators.** Hindsight tests a complete hybrid memory;
   Supermemory is the easiest local service; MemForest is the closest research
   architecture but a heavier, immature artifact.
6. **Escalate to validity intervals or temporal trees only on repeated temporal
   failures.** Graphiti and MemForest offer two different designs; neither should
   be selected before a simpler ordered history is measured.
7. **Defer evolving opinions, self-rewriting notes, typed-memory compilers, and
   proactive intent.** These solve later trust or autonomy problems, not REM's
   current slot fragmentation.

### 3.6 Ranked source gaps and overclaim warnings

Highest-impact remaining gaps:

1. **Hindsight benchmark budget — quick source/code audit.** The paper's
   experimental setup literally contains `<add>` placeholders for LongMemEval and
   LoCoMo token budgets. Architecture is well specified; headline efficiency is
   not reproducible from the paper alone.
2. **Mem0 OSS versus April-2026 algorithm — code experiment.** Current OSS code
   still documents update/delete decisions while the README advertises ADD-only.
   Establish which backend the published 94.8% run actually used.
3. **Local Supermemory parity — on-box experiment.** The local server exists, but
   hosted proprietary extraction models differ from the bring-your-own local path.
4. **MemForest portability — on-box experiment.** Verify correctness and write
   throughput with a Strix-feasible model and runtime; H100 results do not predict
   NPU behavior.
5. **Observation freshness — quick Hindsight code audit.** Confirm whether an
   explicit high-water mark/read guard exists or only asynchronous regeneration.
6. **Independent reproduction — deeper reruns, not more web search.** Hindsight,
   Supermemory, MemForest, MemIR, and CogniFold currently rely mainly on author
   artifacts.

Overclaims to avoid:

- “Hindsight is independently reproduced.” Released code and a viewer are not an
  independent reproduction, and some compared baselines use differently sourced
  judge results.
- “Hindsight has proof counts and a stale-observation guard.” The reviewed paper
  supports asynchronous regeneration, not those precise mechanisms.
- “A-MEM processes an item in 1.1 seconds.” The primary paper does not report this;
  the earlier report appears to have misread a table.
- “Mem0 handles updates” or “Mem0 is ADD-only” without a version. Both are true of
  different documented algorithms.
- “Supermemory scores 95% answer accuracy.” Its page frames 95% as Recall@15 with
  aggregation, then uses GPT-4o answer/judge evaluation; ingestion also differs
  from the original round-by-round protocol.
- “Graph retrieval is an NPU workload” or “graph memory is necessarily better.”
  Graph traversal remains CPU-side, and MemForest itself reports mixed LoCoMo
  results despite strong temporal performance.

No additional deep-research prompt is warranted yet. The top gaps now require
code inspection and controlled local experiments rather than more broad search.

## 4. Claims requiring verification

| ID | Claim | Current status | Required evidence |
|---|---|---|---|
| Q-001 | Mem0 v3 achieves the reported LongMemEval, LoCoMo, and BEAM accuracy at about 6.7-7k retrieved tokens and about 1 second p50. | **REPORTED** | Raw artifacts, exact Mem0 revision, ingest/answer/judge models, dataset revisions, per-category results, and latency method. |
| Q-002 | Does the published April-2026 Mem0 ADD-only algorithm exist in, and behave identically to, the default OSS path? | **REPORTED** | The conflict is version/product-specific: the 2025 paper and current OSS method document update/delete, while the current README advertises a new ADD-only benchmark algorithm. Pin and execute the actual benchmark backend. |
| Q-003 | Supermemory is first on LongMemEval, LoCoMo, and ConvoMem and is practical as a local comparator. | **REPORTED** | Reproducible benchmark artifacts; verify the local binary/server path, feature parity, models, and offline behavior. |
| Q-004 | Cognee Memify prunes stale facts and provides reliable temporal correction handling. | **REPORTED** | Current code path, schema, and stale-update tests; architecture marketing is insufficient. |
| Q-005 | LangMem provides a stronger memory policy than REM rather than only an SDK abstraction. | **INFERRED** | A fixed policy/configuration and benchmark result; frameworks are not comparable until instantiated. |
| Q-006 | Full context remains preferable for roughly the first 150 turns. | **REPORTED** | Primary ConvoMem artifact with model, context window, question distribution, and retrieval baseline. |
| Q-007 | Hindsight's headline results have been independently reproduced. | **REPORTED** | A separate team's code, configuration, and output artifacts. Multi-institution authorship is not independent reproduction. |
| Q-008 | MemForest achieves 79.8% pass@1 on LongMemEval-S and about 6x the memory-construction throughput of stateful baselines. | **REPORTED** | Paper artifacts, exact baselines, model calls, concurrency, token accounting, and runnable code. |
| Q-009 | MemIR's typed representation materially prevents provenance-role collapse on BEAM-100K. | **REPORTED** | Primary paper review, code/artifacts, ablations, answer/judge models, and exact task definition. |
| Q-010 | CogniFold's intent layer and continuous cognitive folding improve long-horizon agent behavior. | **REPORTED** | Runnable implementation, benchmark design, safety constraints, compute cost, and comparison against simpler active-task state. |
| Q-011 | Hindsight entity resolution occurs entirely at write time and similarity thresholds eliminate identity fragmentation as the ledger grows. | **REPORTED** | Current retain/entity-resolution code and fragmentation measurements. “Eliminates” is stronger than the available evidence. |
| Q-012 | A-MEM rewrites historical notes, maintains evolution history and soft deletes, uses heat/time-decay ranking, and performs daily consolidation. | **REPORTED** | Separate the A-MEM paper from third-party plugin behavior; inspect each implementation and benchmark independently. |
| Q-013 | Small 1B-3B quantized models are sufficient for MemForest extraction but insufficient for Hindsight, Graphiti, or HippoRAG extraction. | **INFERRED** | Same extraction dataset, schema, quantization, runtime, and scored structured-output quality across model sizes. |
| Q-014 | Graphiti recall is sub-200ms at enterprise scale and its dominant production failures are asyncio conflicts, endpoint-label duplication, and NaN embeddings. | **REPORTED** | Pin specific versions/issues and reproduce under a defined graph size and local backend. |
| Q-015 | Hindsight implements an explicit observation high-water mark, proof counts/evidence quotes, and a read-time freshness guard beyond asynchronous regeneration. | **INFERRED** | The paper confirms background regeneration when facts change but not these precise fields or guard behavior. Locate current schema/read code and test delayed consolidation. |
| Q-016 | Memobase uses a recent-blob buffer, fixed three-call consolidation, profile/event storage, bounded context assembly, and sub-second local retrieval. | **REPORTED** | Pin a release; inspect flush/source-retention behavior; reproduce call count, output quality, and latency locally. |
| Q-017 | A-MEM's small local backbones are sufficiently reliable and efficient for REM-style extraction or evolution on XDNA2. | **INFERRED** | The paper confirms local 1B/3B evaluation and lower answer-context tokens, but provides no 1.1-second processing result or NPU data. Score structured output and latency on REM fixtures. |
| Q-018 | Zep's temporal graph achieved 71.2% LongMemEval with GPT-4o versus 60.2% full context while injecting about 1.6k tokens. | **REPORTED** | Confirm dataset revision, result table, ingestion and answer configuration, judge, average-token calculation, and runnable artifacts. |
| Q-019 | Supermemory's fully local path retains the hosted system's reported 95% LongMemEval-S Recall@15, roughly 720 added tokens, and low retrieval latency. | **REPORTED** | Local operation is documented, but local extraction uses the user's model instead of the hosted proprietary stack. Reproduce with pinned code, local models, dataset revision, hardware, and query accounting. |

## 5. Contradicted or stale claims

| ID | Claim | Why it is rejected |
|---|---|---|
| C-001 | REM currently uses simple vector search. | The active read path is `RecencySelector`; vector retrieval is not implemented. |
| C-002 | BM25, graph traversal, databases, RRF, or PageRank are natural XDNA2 NPU workloads. | These are CPU workloads in the current software stack. Only supported neural inference, such as an embedding encoder or small decoder, is a plausible NPU workload. |
| C-003 | Graphiti's open-source status is unclear. | The temporal engine is openly available in `getzep/graphiti`; Zep is the managed product around it. |
| C-004 | hal0's advertised NPU chat+STT+embedding trio describes current fresh-install provisioning. | Current hal0 provisioning marks the STT/embedding trio passengers dormant and primarily provisions an NPU chat/utility lane. |
| C-005 | hal0 has a durable pgvector fallback if Hindsight is unavailable. | Its current fallback class is an in-memory list, and the provider factory notes that the boot connectivity probe is not wired. |
| C-006 | Qwen 3.6/3.5 27B, Hermes, Luce, DFlash, or PFlash are established parts of this REM repository. | These came from unrelated context and must not appear in REM experiment assumptions. |
| C-007 | REM must abandon its current naive vector store. | REM has no active vector-store read path. The statement diagnoses a system REM does not currently have. |
| C-008 | BM25, vector search, PageRank, SQLite, Kuzu, or working-memory assembly belong on the foreground iGPU. | These are CPU-side operations in the practical REM stack. The iGPU hosts foreground generation; NPU placement is relevant only to supported neural inference. |
| C-009 | LightRAG, HippoRAG, and Graphiti all belong to one “temporal and bi-temporal graph” category. | Graph-enhanced retrieval does not imply validity intervals or bi-temporal semantics. Their temporal behavior must be evaluated separately. |
| C-010 | A 70B BF16 foreground model is an established REM deployment assumption. | It is neither established in this repository nor needed to evaluate memory mechanisms; answer models must be selected and recorded per experiment. |
| C-011 | A-MEM reports approximately 1.1-second local processing with Llama 3.2 1B. | The primary paper contains no such latency result. It evaluates 1B/3B models and reports context-token lengths; the report appears to have misread a table value. |

## 6. Comparator shortlist

### Tier A: run as native black boxes

1. **Hindsight OSS** — strongest complete long-term-memory comparator.
2. **Supermemory self-hosted** — easiest complete local comparator, with local-model quality explicitly separated from hosted claims.
3. **MemForest** — closest research architecture to REM's asynchronous canonical-fact design; immature and heavier to reproduce.
4. **Graphiti** — strongest temporal/provenance engine comparator when temporal failures justify its graph and write cost.
5. **Mem0 OSS** — lower-complexity fact-memory comparator; must not be conflated with the April-2026 managed/benchmark algorithm.

Each black-box comparator should initially own its native ingestion, store, and
retrieval pipeline. Making it read or write REM's ledger would erase the very
architectural differences the comparison is intended to measure.

### Tier B: mechanism sources

- Cognee: graph/vector/relational pipeline and provenance patterns.
- HippoRAG 2: graph propagation and associative retrieval.
- LightRAG: dual-level graph/vector retrieval.
- A-MEM: linked atomic notes and memory evolution.
- MemForest: hierarchical temporal indexing and parallel/localized writes.
- MemIR: typed evidence, cue, claim, and provenance representation.
- CogniFold: intent-layer and continuous-folding research lead.
- LangMem: semantic/episodic/procedural policy abstractions.
- Memobase: buffered profile/event consolidation and bounded profile assembly.
- LIGHT: multi-store and extreme-horizon architecture.
- Letta/MemGPT: agent-managed paging/context baseline.

## 7. Mechanism backlog

Order reflects information value for REM, not general novelty.

1. Finish the NPU-free slot-key canonicalization measurement.
2. Add query-aware lexical retrieval over existing summaries and facts.
3. Measure lexical versus embedding versus fused retrieval under one token
   budget.
4. Add RRF and optional fail-soft reranking only if candidate recall warrants it.
5. Preserve source episodes behind every derived fact.
6. Test explicit validity intervals on clean temporal-update fixtures.
7. Test episodic/semantic/working-store separation.
8. Test observations or reflection only after fact extraction and identity are
   trustworthy.
9. Test typed epistemic/provenance roles before allowing derived summaries to
   influence durable truth-bearing facts.
10. Test hierarchical temporal indexes only after simpler query-aware retrieval
    is measured at a scale where linear/flat indexing is a demonstrated cost.
11. Add a freshness high-water mark to every derived summary/profile experiment
    so asynchronous lag is visible and testable.
12. Compare fixed `top-k` with budget-capped greedy packing only after candidate
    ranking and supersession filtering are held constant.

No item in this list is assigned to the NPU by default. Placement is a separate
experiment after the mechanism works and its workload is known.

## 8. Comparison rules

Every comparative run must record:

- repository revision and configuration;
- dataset name, revision, subset, and item IDs;
- ingestion/extraction model;
- embedding and reranking models;
- answer model and judge model;
- total available context and injected-memory token budget;
- write recall, read recall, final answer accuracy, and abstention;
- stale-value, temporal-order, contradiction, and provenance failures;
- ingest latency/throughput and recall p50/p95;
- CPU, iGPU, NPU, RAM, power, and foreground contention where applicable;
- hard failures, retries, malformed structured output, and dropped records.

Architecture and placement are independent axes:

1. First establish memory quality with fixed compute placement.
2. Then compare CPU, iGPU, and NPU placement for supported components.
3. Do not attribute an accuracy change to placement or a throughput change to
   architecture without a controlled arm.

## 9. Initial experiment sequence

1. Complete the current canonicalization increment against the five captured
   states.
2. Build an NPU-free lexical/query-aware read-path spike over those same states.
3. Run Hindsight as a native black-box comparator on a fixed, unambiguous
   LongMemEval subset.
4. Run Graphiti on the same subset if temporal-update misses remain material.
5. Add Mem0 only after pinning a reproducible OSS revision and configuration.
6. Add a compact profile/event arm inspired by Memobase before committing to a
   full graph database.
7. Expand beyond the five oldest-gold states before making an architecture
   decision.

The five captured states are diagnostic fixtures, not a representative benchmark.
The ambiguous `031748ae` item must not decide the architecture by itself.

## 10. Report intake log

| Report | Scope | Usable contribution | Main cautions |
|---|---|---|---|
| R-001 | hal0 repository comparison | Located Hindsight as the actual memory engine; exposed useful provider, namespace, and token-budget patterns. | hal0 README and NPU-trio claims are partly stale; hal0 does not validate Hindsight quality on Strix Halo. |
| R-002 | Broad memory-system survey | Produced the initial taxonomy, comparator candidates, and mechanism list. | Mixes primary and secondary sources, vendor metrics, unsupported NPU placement, and unrelated REM assumptions. |
| R-003 | [Architectures for Long-Term Memory in LLM Agents](/home/keith/Downloads/LLM%20Agent%20Memory%20Systems%20Review.md), received 2026-06-28 | Added MemForest, MemIR, CogniFold, governed memory evolution, bi-temporal identity, and provenance-preserving derived views as leads. | Repeats the false vector-store baseline; mixes papers, vendor material, secondary articles, GitHub issues, and third-party implementations; overstates reproduction and NPU/iGPU suitability. |
| R-004 | [Long-Term Memory Systems for LLM Agents](/home/keith/Downloads/deep-research-report.md), received 2026-06-28 | Added observation-freshness guards, Memobase's buffered profile/event design, compact-fact-before-graph control, explicit greedy budget packing, and a conflicting account of Mem0 updates. | Better evidence discipline than prior reports, but its `turn...` citation markers are not portable; source URLs and pinned revisions must be recovered before detailed claims can be audited. |

## 11. Intake template for new reports

For each incoming report:

1. Record its scope and date in the intake log.
2. Extract atomic claims rather than preserving its narrative wholesale.
3. Assign each claim an evidence label.
4. Replace secondary links with primary sources where possible.
5. Check claims against current source code and pinned revisions.
6. Add genuinely new systems to the shortlist at the appropriate tier.
7. Add mechanisms only when they address an observed REM failure mode.
8. Add experiments only when they isolate one architecture or placement variable.
9. Preserve contradictions in section 5 rather than silently deleting them.
10. Do not change REM's roadmap until evidence or a controlled experiment supports it.

## 12. Open research questions

- Can a small NPU-resident model match the structured extraction reliability
  required by Hindsight or Graphiti?
- Does lexical/query-aware retrieval fix REM's remaining read failures before
  embeddings or a graph are needed?
- Is string canonicalization sufficient for fact identity, or is semantic
  identity required?
- Do explicit validity intervals improve clean knowledge-update questions enough
  to justify graph complexity?
- What is the crossover point where bounded memory beats recent-context
  truncation for real agent workloads?
- Which components can actually run through the available XDNA2 runtime, and
  which must remain CPU/iGPU workloads?
- How much extraction backlog can accumulate before asynchronous maintenance
  stops being operationally safe?
- Is event time plus transaction time materially better than REM's simpler
  supersession metadata on delayed corrections and historical questions?
- Can typed evidence/claim/inference roles reduce unsupported answers without
  consuming too much of the bounded injection budget?
- At what history size does a hierarchical temporal index outperform a flat
  facts/episodes index enough to justify its write and maintenance complexity?
- Does intent require durable long-term memory, or should it remain explicit,
  short-lived task state with separate expiry and authorization rules?
- How should a derived summary advertise that it is stale while background
  consolidation is behind the source-event high-water mark?
- Does a profile plus recent-event timeline recover most temporal-update value
  without a graph database?
- Which source references are sufficient in the hot prompt for auditability,
  while the complete raw episode remains available off-path?

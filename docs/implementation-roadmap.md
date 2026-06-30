# REM Implementation Roadmap

Status: active, evidence-gated roadmap (updated 2026-06-28).

The default rule is to test the smallest mechanism matching an observed failure.
External engines remain evaluation-only until they pass both development and
confirmation gates. The fixed answer model is `gemma4-it:e2b`; the fixed judge is
`claude-haiku-4-5`.

## Gate 0 — Native fact identity

The post-hoc string experiment is complete over the five captured states. Its raw
artifact is [`canonicalize_audit.json`](../bench/battery/canonicalize_audit.json).

| arm | active entries reduced | fragmented values reduced | merge-risk groups | gold retained |
|---|---:|---:|---:|---:|
| full key | 0.96% | 3.30% | 36 | yes |
| subject only | 25.06% | 38.68% | 592 | yes |

Neither arm reaches the required 50% fragmentation reduction. Subject-only
identity also collapses many distinct current values. Therefore string
canonicalization is **not promoted into the write path**. The residual is
predominantly semantic, so the embedding-identity experiment is activated: a
Qwen embedding baseline must be tested before DREAM. `031748ae` remains a
diagnostic item because its expected temporal inference is ambiguous; no paid
answer rerun was needed after the quantitative promotion criterion failed.

Implemented support:

- `canonical_slot_key` and non-mutating `recanonicalize`;
- same-slot and distinct-slot negative fixtures;
- a reproducible five-state audit with gold-survival and merge-risk reporting.

## Gate 1 — Normalized evaluation foundation

Infrastructure is implemented under `evals/memory_methods/` and artifacts belong
under `bench/memory_methods/`.

- `freeze_manifest.py` deterministically selects 10 knowledge-update, 10
  temporal-reasoning, and 10 multi-session items, preferring distributed then old
  evidence and excluding adjudicated ambiguous IDs.
- `capture_states.py` persists each selected REM state once and skips existing
  captures.
- Every native development run uses both 8,000 and 28,000 injected-memory-token
  budgets.
- `MemoryMethodArtifact` records repository/system revisions, fixed models,
  source references, ingest/read latency, memory tokens, write/read recall,
  category accuracy, extraction failures, overflow/provenance failures, and
  contention data.

The LongMemEval-S source file is not in this repository (gitignored), so the 30 IDs
and states are materialized from the locally supplied dataset rather than invented.
The manifest is now **frozen** at `bench/memory_methods/development_manifest.json`
(10/10/10, `031748ae` excluded, `source_sha256` pinned to the dataset). State capture
is resumable (`capture_states.py --limit` caps NEW captures per run; existing-state
skips are free) and runs the ~75-min/item NPU compaction. Until the states are
captured **and** the mechanisms are validated on this held-out suite, no development
accuracy claim is valid — every Gate-4 number remains diagnostic/overfit.

## Gate 2 — Native read-path ladder

Paired native arms now implement the common
`select(state, query, budget_tokens) -> MemoryState` contract:

1. `RecencySelector` (current baseline).
2. `LexicalSelector` (query-aware lexical rank, rank-order fill).
3. `PackedLexicalSelector` (deduplication, supersession filtering, and
   budget-aware greedy packing).

Temporal queries may retrieve explicitly stale facts and render their stale
status and source turn. Current-state queries continue to filter stale facts.
Dense retrieval is activated only when gold exists in the full state and lexical
selection misses paraphrased evidence. RRF or fail-soft reranking is activated
only when fusion raises candidate recall but final ranking/packing still drops
gold.

A mechanism advances on the frozen suite only when it produces at least three
more wins than regressions, regresses no category by more than one item, fits both
budgets, keeps recall p95 at or below one second, and loses neither provenance nor
context safety. `run_development.py` writes normalized paired artifacts;
`promotion.py` evaluates this bar.

## Gate 3 — Initial black-box comparators

Evaluation-only adapters implement the same `MemoryArm` lifecycle for:

1. Hindsight OSS, using a separate bank per question and native retain/recall.
2. Self-hosted Supermemory, using a separate container tag per question and
   native document readiness/search.

Both adapters preserve returned source IDs, enforce the caller's output budget,
support cleanup, and expose ingest/read stats. Hindsight uses synchronous native
retain; Supermemory polls document readiness with timeout and retry behavior.

Before a full run, each pinned local installation must pass a three-item smoke.
Unsupported native setup is recorded as unsupported; it is never silently
replaced. A controlled extraction pass may use REM's Gemma endpoint only where
the pinned system documents that configuration. Hosted benchmark numbers are not
local results.

## Gate 4 — Conditional specialized experiments

| Measured condition | Activated experiment |
|---|---|
| Semantic slot fragmentation after Gate 0 | **Active:** Qwen embedding identity baseline, then DREAM challenger |
| Gold exists in full state but lexical misses paraphrase | Dense retrieval and lexical/dense fusion |
| Candidate contains gold but ranking/packing drops it | RRF, optional reranker, diversity-aware packing |
| Two unambiguous stale/current or predecessor errors | Ordered validity intervals; Graphiti and MemForest comparison |
| Two source-attribution or unsupported-answer failures | Episode IDs and evidence spans; later MemIR-style typed claims |
| Derived summaries become stale during async lag | Source high-water marks and dirty-derived-view handling |
| Profile redundancy dominates token use | Memobase-style profile plus recent-event control |
| Write maintenance misses one session per 30 seconds | MemForest-style parallel extraction and localized refresh |
| Foreground decode degrades by more than 5% | Component placement and scheduling review |
| Gold reaches context but the answer model still fails | Fix or replace the answerer; memory remains unchanged |

A-MEM evolution, Hindsight opinions/reflection, MemIR's full compiler, CogniFold
intent emergence, HippoRAG, LightRAG, and Cognee remain research references until
a measured trigger applies.

## Gate 5 — Confirm and promote

Before REM's default changes, run current REM, candidate REM, and truncation on
every available LongMemEval-S knowledge-update, temporal-reasoning, and
multi-session item with fixed answer/judge models. Require:

- at least five points overall improvement, or ten points in the targeted
  category without overall regression;
- no write-recall reduction or hard-extraction-failure increase;
- recall p95 at or below one second and compliance with the 8k budget;
- foreground decode degradation at or below 5%;
- restart, persistence, duplicate-ingest, delayed-indexing, and partial-failure
  tests.

Native mechanisms ship behind configuration flags first. A production provider
abstraction is justified only if an entire external engine passes this gate.

## Historical evidence — completed tight-budget gate

The original tight-budget milestone is complete and remains useful evidence:

- robust extraction diagnostics and malformed-JSON recovery are covered;
- a bounded `RecencySelector` path turned 36,977–58,150-token states into
  model-consumable slices;
- five oldest-gold captured states fit the 28k read budget and produced four
  correct answers, with `031748ae` retained as an ambiguous temporal diagnostic;
- the write audit found 212 fragmented active values across the five states and
  only 53 previously superseded entries.

See [`FINDINGS.md`](../bench/battery/FINDINGS.md) for the measurement history.
The graph architecture specification is now conditionally triggered, not the
default successor to the bounded read path.

## Commands

Local dataset path on the dev box: `/home/keith/datasets/longmemeval/longmemeval_s`
(raw HF download, no `.json` extension; the loader reads by path). Substitute your own.

```bash
DATA=/home/keith/datasets/longmemeval/longmemeval_s

# Freeze the 30-item development set + pin the dataset SHA (NPU-free, instant).
PYTHONPATH=.:src python3 evals/memory_methods/freeze_manifest.py --data "$DATA"

# Capture states (resumable; --limit caps NEW captures/run, existing skips are free;
# this invokes the ~75-min/item NPU compaction). KU items come first in the manifest.
PYTHONPATH=.:src python3 evals/memory_methods/capture_states.py --data "$DATA" --limit 6

# Replay supersession NPU-free straight from the frozen manifest (go/no-go).
PYTHONPATH=.:src python3 evals/memory_methods/run_supersession_instanceaware.py \
  --manifest bench/memory_methods/development_manifest.json
PYTHONPATH=.:src python3 evals/memory_methods/run_supersession_endtoend.py \
  --manifest bench/memory_methods/development_manifest.json   # add --answer for the NPU mix

# Typed-judge write-cost (NPU-free): how many judge calls/ingest the band would need.
PYTHONPATH=.:src python3 evals/memory_methods/run_typed_band_cost.py \
  --manifest bench/memory_methods/development_manifest.json

# Paired recall-only run, then the fixed answer/judge run.
PYTHONPATH=.:src python3 evals/memory_methods/run_development.py
PYTHONPATH=.:src python3 evals/memory_methods/run_development.py --score
```

TencentDB and DREAM remain in the deprioritized research queue. DREAM activates
only after the Qwen embedding identity baseline is materialized and scored.

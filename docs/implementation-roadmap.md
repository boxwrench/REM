# REM Implementation Roadmap

> NOTE (2026-07-02): the memory-quality work has reached a deliberate stopping point.
> See **[docs/rem-closeout-and-revisit.md](rem-closeout-and-revisit.md)** for what
> shipped, why each remaining avenue stopped (the *walls*), and the concrete condition
> that would make each worth revisiting.

Status: active, evidence-gated roadmap (updated 2026-07-02).

This document is the authoritative forward plan. Historical measurements remain
below as evidence, but they do not override the current A/B/C/D disposition or
the frozen confirmation criteria. Live capture counts are recorded in
[`capture_integrity.json`](../bench/memory_methods/capture_integrity.json); the
session handoff and detailed chronology remain in
[`progress.md`](../.superpowers/sdd/progress.md).

The default rule is to test the smallest mechanism matching an observed failure.
External engines remain evaluation-only until they pass both development and
confirmation gates. The fixed answer model is `gemma4-it:e2b`; the fixed judge is
`claude-haiku-4-5`.

## Current execution state — A/B/C/D

| path | current disposition | next admissible action |
|---|---|---|
| A — native read path | Safe sparse retrieval plus the shared question taxonomy are implemented. Cross-key newest preference is experimental and **default off** after adversarial false-merge review. The earlier 5/6 result is diagnostic, not promotion evidence. | After Path C is complete, run the frozen 30-item safe-vs-candidate confirmation. Promote the candidate only if every frozen accuracy, safety, provenance, context, and latency check passes. |
| B — write-time keying | **NO-GO / parked.** Role-aware post-hoc re-keying reduced fragmentation from 302 to 222 (26.49%), below the pre-registered 50% bar, despite preserving all gold and negative sentinels. Similarity and typed-judge write-path variants remain off. | No production extraction or supersession change. Reopen only with new held-out evidence for an instance- and attribute-aware mechanism with bounded write cost. |
| C — provenance and capture | Date/session provenance now survives loading, compaction, persistence, selection, and rendering. Capture is resumable and records extraction telemetry. The last integrity snapshot reports 10 valid, 1 in progress, 19 missing, and 0 invalid. | Finish the 20 temporal/multi-session captures sequentially, then require a clean 30/30 integrity report before Path A confirmation. |
| D — one-call episode card | Implemented behind `episode_card_consolidation=False`. Offline replay preserves ledger, summary, provenance, and atomic failure behavior while reducing logical NPU calls from two to one. | After Path C releases the NPU, run the paired live wall-clock benchmark. Keep the flag off unless all benchmark checks pass. |

The immediate critical path is therefore **C → A confirmation → D live gate →
Gate 5 promotion checks**. Path B and destructive supersession remain outside that
path.

## Gate 0 — Native fact identity (historical diagnosis; complete)

The post-hoc string experiment is complete over the five captured states. Its raw
artifact is [`canonicalize_audit.json`](../bench/battery/canonicalize_audit.json).

| arm | active entries reduced | fragmented values reduced | merge-risk groups | gold retained |
|---|---:|---:|---:|---:|
| full key | 0.96% | 3.30% | 36 | yes |
| subject only | 25.06% | 38.68% | 592 | yes |

Neither arm reached the required 50% fragmentation reduction. Subject-only
identity also collapsed many distinct current values. Therefore string
canonicalization was **not promoted into the write path**.

The subsequent Qwen embedding baseline, richer-key experiments, full-state
replays, numeric value gate, typed-identity probe, and typed-band cost audit are
also complete. They established that similarity measures resemblance rather
than instance/attribute identity: global full-fact matching over-merged real
states, the value gate left unsafe numeric collisions, and the useful typed
judge band was too expensive for write-time use. Path B's final role-aware audit
then missed its 50% fragmentation bar (26.49%). These results park write-time
identity work; they do not activate DREAM or authorize a destructive merge.
`031748ae` remains a historical diagnostic and is excluded from the frozen
30-item suite.

Implemented support:

- `canonical_slot_key` and non-mutating `recanonicalize`;
- same-slot and distinct-slot negative fixtures;
- a reproducible five-state audit with gold-survival and merge-risk reporting.
- committed Qwen and role-key audit artifacts, including the final
  [`path_b_role_key_audit.json`](../bench/memory_methods/path_b_role_key_audit.json)
  **NO-GO**.

## Gate 1 — Normalized evaluation foundation / Path C

Infrastructure is implemented under `evals/memory_methods/` and artifacts belong
under `bench/memory_methods/`.

- `freeze_manifest.py` deterministically selects 10 knowledge-update, 10
  temporal-reasoning, and 10 multi-session items, preferring distributed then old
  evidence and excluding adjudicated ambiguous IDs.
- `capture_states.py` persists each selected REM state once, records extraction
  telemetry in the manifest, and skips existing captures.
- Every native development run uses both 8,000 and 28,000 injected-memory-token
  budgets.
- `MemoryMethodArtifact` records repository/system revisions, fixed models,
  source references, ingest/read latency, memory tokens, write/read recall,
  category accuracy, extraction failures, overflow/provenance failures, and
  contention data.

The LongMemEval-S source file is not in this repository (gitignored), so the 30 IDs
and states are materialized from the locally supplied dataset rather than invented.
The manifest is now **frozen** at
[`development_manifest.json`](../bench/memory_methods/development_manifest.json)
(10/10/10, `031748ae` excluded, `source_sha256` pinned to the dataset). State capture
is resumable (`capture_states.py --limit` caps new captures per run; existing-state
skips are free) and runs the ~75-min/item NPU compaction. The integrity validator
requires loadable state structure and, for Path C temporal/multi-session items,
complete date/session provenance and capture metadata. Its latest recorded snapshot
is 10 valid, 1 in progress, 19 missing, and 0 invalid.

Until all 30 states pass integrity and the mechanisms are validated on this suite,
the earlier six-item Path A and five-state Gate 4 numbers remain development or
diagnostic evidence, not promotion evidence.

## Gate 2 — Native read-path ladder / Path A

Paired native arms now implement the common
`select(state, query, budget_tokens) -> MemoryState` contract:

1. `RecencySelector` (historical bounded baseline).
2. `LexicalSelector` (historical query-aware rank-order fill).
3. `PackedLexicalSelector` (deduplication, supersession filtering, and
   budget-aware greedy packing).
4. `SparseChronologicalSelector(prefer_newest=False)` (`safe-sparse`, the
   production Path A arm).
5. `SparseChronologicalSelector(prefer_newest=True)` (`path-a-candidate`,
   experimental and default-off).

Temporal queries may retrieve explicitly stale facts and render their stale
status and source turn. Current-state queries continue to filter stale facts.
Dense retrieval is activated only when gold exists in the full state and lexical
selection misses paraphrased evidence. RRF or fail-soft reranking is activated
only when fusion raises candidate recall but final ranking/packing still drops
gold.

The shipping sidecar now uses the safe sparse selector and shared question taxonomy.
The experimental `path-a-candidate` differs only by enabling cross-key newest
preference; it remains default-off because adversarial review found role/instance
false-merge risks. The historical six-item candidate result (5/6 versus sparse 3/6
in that rerun) cannot promote it.

The authoritative confirmation protocol is frozen in
[`path_a_confirmation_criteria.json`](../bench/memory_methods/path_a_confirmation_criteria.json)
and explained in [`path-a-confirmation-freeze.md`](path-a-confirmation-freeze.md).
It requires exact 30-item coverage for both arms at 8k and 28k, three scored
repetitions at 8k, at least three candidate wins, zero item or category regressions,
and every safety/provenance/context/latency check. Incomplete capture exits before
model clients are constructed. `run_development.py` and `promotion.py` remain the
general development ladder; the frozen Path A wrapper is the promotion authority
for newest preference.

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
| Semantic slot fragmentation after Gate 0 | **Completed / parked:** Qwen similarity and Path B role-aware re-key were insufficient or unsafe. DREAM/typed claims require a new, bounded, held-out instance-and-attribute identity proposal. |
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

Path D is a separate throughput experiment, not a correctness substitute. Its
one-call episode card is implemented default-off and may advance only after a live,
paired NPU benchmark confirms equivalent fact signatures, nonempty summaries,
successful compaction, exact 2-to-1 call counts, and a lower episode-card median wall
time. Offline replay validates harness plumbing but cannot promote the flag.

## Gate 5 — Confirm and promote

The general gate for a new external engine or full architecture remains a run of
current REM, candidate REM, and truncation on every available LongMemEval-S
knowledge-update, temporal-reasoning, and multi-session item with fixed
answer/judge models. That gate requires:

- at least five points overall improvement, or ten points in the targeted
  category without overall regression;
- no write-recall reduction or hard-extraction-failure increase;
- recall p95 at or below one second and compliance with the 8k budget;
- foreground decode degradation at or below 5%;
- restart, persistence, duplicate-ingest, delayed-indexing, and partial-failure
  tests.

Native mechanisms ship behind configuration flags first. A production provider
abstraction is justified only if an entire external engine passes this general
gate.

For the present A/B/C/D work, Gate 5 starts only after Path C integrity is 30/30.
Path A's later, mechanism-specific frozen protocol supersedes the generic improvement
threshold and truncation arm for newest-preference promotion: it supplies the paired
30-item accuracy, context, provenance, and recall-latency decision. Path D has its
own live wall-clock decision and does not share promotion with Path A. The NPU-free
operational harness covers restart/persistence, duplicate-ingest idempotency, and
partial-failure atomicity; delayed indexing is explicitly not applicable because REM
persists and reads `MemoryState` synchronously. Extraction-failure and foreground-
contention evidence are still separate required inputs before any default flips.

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

# Historical/recovery only: recreate the already-frozen manifest and dataset pin.
# Do not re-freeze during the current confirmation run.
PYTHONPATH=.:src python3 evals/memory_methods/freeze_manifest.py --data "$DATA"

# Path C: resume the remaining captures sequentially. Existing files are skipped;
# this invokes the ~75-min/item NPU compaction and updates manifest telemetry.
PYTHONPATH=.:src python3 evals/memory_methods/capture_states.py --data "$DATA"

# Path C: read-only integrity report. During capture, identify the one active item;
# after capture, omit --in-progress-id and add --require-complete.
PYTHONPATH=.:src python3 evals/memory_methods/validate_capture_integrity.py \
  --in-progress-id gpt4_45189cb4 \
  --out bench/memory_methods/capture_integrity.json
PYTHONPATH=.:src python3 evals/memory_methods/validate_capture_integrity.py \
  --require-complete --out bench/memory_methods/capture_integrity.json

# Path A: NPU-free preflight. It exits 2 without constructing model clients until
# all 30 state files satisfy the frozen protocol.
PYTHONPATH=.:src python3 evals/memory_methods/run_path_a_confirmation.py

# Once preflight is ready: recall/context at 8k + 28k, then the frozen three-rep
# answer/judge comparison at 8k. The latter needs the NPU and ANTHROPIC_API_KEY.
PYTHONPATH=.:src python3 evals/memory_methods/run_path_a_confirmation.py --run
PYTHONPATH=.:src python3 evals/memory_methods/run_path_a_confirmation.py --score

# Path D: offline plumbing is safe at any time. Run --run only after Path C has
# released the NPU; the live result alone can promote the default-off flag.
PYTHONPATH=.:src python3 evals/memory_methods/run_episode_card_benchmark.py \
  --out bench/memory_methods/episode_card_wallclock_offline.json
PYTHONPATH=.:src python3 evals/memory_methods/run_episode_card_benchmark.py --run \
  --out bench/memory_methods/episode_card_wallclock.json

# Gate 5 operational checks (NPU-free). Add --workspace DIR to retain fixtures.
PYTHONPATH=.:src python3 evals/memory_methods/run_gate5_operational.py \
  --out bench/memory_methods/gate5_operational.json

# Historical Path B reproduction only; this is expected to remain NO-GO.
PYTHONPATH=.:src python3 evals/memory_methods/run_role_key_audit.py
```

TencentDB and DREAM remain in the deprioritized research queue. The Qwen baseline
has already been materialized and shown insufficient; DREAM no longer activates
merely because that baseline exists. Reopening it requires new held-out evidence
and a pre-registered instance/attribute identity and write-cost proposal.

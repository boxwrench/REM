# REM Implementation Roadmap

Status: active handoff note.

This file tracks the next implementation gate for REM. Broader hardware placement
work can continue later, but the immediate priority is proving that the memory
system wins a user-visible recall task under a tight context budget.

Architecture direction note: [`REM-memory-architecture-spec.md`](REM-memory-architecture-spec.md)
captures a larger graph-resident memory design that removes prose summaries from
the stored representation and makes the per-turn read path model-free. Treat it
as the candidate architecture after the current recall gate, unless the
tight-budget battery shows the existing compaction path cannot be made credible.
The spec now includes a weighted assessment and open questions; the short
version is that graph memory is strongest against summary corruption, stale
ghosts, and read-path ambiguity, but it does not remove the need to measure and
improve write recall.

## Current Gate: Tight-Budget Recall Win

**Milestone:** REM beats naive truncation on LongMemEval `knowledge-update` when
the context budget is tight enough that truncation drops the gold evidence
session, and JSON extraction failures are measured and reduced enough that they
are not the dominant failure mode.

This is the right next gate because the NPU placement result is already
measured. The unresolved question is whether REM's compaction path preserves
useful facts better than simply truncating old context. Until that is shown, the
elegant placement is solving a problem whose end-user recall advantage remains
unproven.

## Progress (updated 2026-06-20)

The measurement apparatus for this gate is now in place; the remaining work is
running the battery on real data and reading the result.

- **Done:** validity rule, extraction observability, JSON robustness coverage,
  and the failure classifier (priority items 1, 2, 3, and the item-6 tooling).
  See the `recall-gate-extraction-observability` branch. Full unit suite green.
- **In progress:** tight-budget smokes on `longmemeval_s` (priority item 4).
- **Pending real data:** the valid subset run and the architecture decision
  (items 5 and 7), plus applying the classifier to actual misses (item 6).

## Priority Order

1. **Lock the validity rule.** _[done]_
   A battery run only counts if truncation drops the gold evidence session.
   Invalid runs should still be saved as artifacts, but they must not be used as
   evidence that REM beats or loses to truncation.
   Implemented as the retention ceiling guard in `evals/battery/aggregate.py`.

2. **Make extraction failures observable.** _[done]_
   The battery artifact should expose fact-extraction diagnostics: strict parse,
   repair, retry, salvage, truncation, loop detection, and final extraction
   failure. A REM miss caused by dropped facts should be visible as an extraction
   failure, not hidden inside answer accuracy.
   The diagnostic taxonomy is folded into `_extraction_stats`
   (`reset_extraction_stats` / `get_extraction_stats`) and surfaced per question
   on `ArmRun.extraction`, summed into `arm_extraction` in the artifact.

3. **Harden JSON robustness.** _[done — pipeline already handled all six; locked
   with mutation-verified regression tests in `tests/unit/test_robust_extract.py`]_
   Add regression coverage for the malformed outputs the small NPU model
   produces in practice:
   - markdown-fenced JSON
   - truncated arrays or objects
   - sibling JSON objects without an enclosing list
   - repeated/looping objects
   - partly salvageable responses with at least one valid fact
   - unrecoverable responses that should fail cleanly without compaction

4. **Run tight-budget smokes.** _[in progress]_
   Use small limits first (`--limit 3` or `--limit 5`) and sweep budgets until
   the run is valid. Candidate budgets: `4000`, `3000`, `2000`, and lower if
   needed.

5. **Run the real valid subset.** _[pending real data]_
   Use the first stable budget where truncation drops the evidence and judge
   outputs are parseable. Commit the result artifact under `bench/battery/`.

6. **Classify failures before broadening scope.** _[tooling done; application
   pending a valid run]_
   If REM loses or only ties truncation, keep the artifact and classify misses:
   extraction drop, summary loss, stale ghost, answerer failure, judge ambiguity,
   context overflow, or budget invalidity.
   `evals/battery/classify.py` buckets misses and maps the dominant bucket to the
   item-7 branch; the runner embeds this classification in every artifact.

7. **Choose the next architecture from the failure mix.** _[pending real data]_
   If failures are mostly malformed JSON or extraction drops, keep the near-term
   work on write robustness. If failures are mostly prose-summary corruption,
   stale ghosts, or retrieval/read-path ambiguity, start the graph-resident
   architecture with Phase 0 and Phase 1 from
   [`REM-memory-architecture-spec.md`](REM-memory-architecture-spec.md).

## Acceptance Criteria

The milestone is complete when the repo contains:

- [x] unit tests that pin the JSON robustness behavior (mutation-verified, since
  the pipeline already handled the malformed-output cases);
- [x] a valid tight-budget LongMemEval artifact (`bench/battery/valid_b1000_oldgold.json`,
  reached via gold-recency selection — see `bench/battery/FINDINGS.md`);
- [x] aggregate REM vs truncation accuracy and evidence-retention numbers
  (`evals/battery/aggregate.py`, in every artifact);
- [x] extraction diagnostics summarized in the result (`arm_extraction` plus the
  embedded `classification` block);
- [x] a short doc stating what the valid battery did or did not prove
  (`bench/battery/FINDINGS.md`).

Outstanding before a memory-quality verdict: the first valid run is blocked by
REM `context_overflow`. Its compacted memory on the five oldest-gold items is
36,977–58,150 tokens (the assembler renders the full ledger + all summaries with
no size bound), so it scores 0/5 without ever answering. Raising the assemble
ceiling does NOT fix this: the memory exceeds 16k on all five, and the larger
items exceed the answering model's own ~32–40k window (HTTP 400 "Max length
reached!"). The real fix is a bounded read path (retrieval/eviction that fits
memory to the model window), which is the item-7 architecture gate — not a
ceiling tweak. See `bench/battery/FINDINGS.md`.

## Suggested Commands

Install eval dependencies:

```bash
pip install -e ".[dev,eval]"
```

Run default tests:

```bash
python3 -m pytest
```

Run tight-budget smokes:

```bash
PYTHONPATH=.:src python3 evals/battery/run_battery_spike.py \
  --data /path/to/longmemeval_s.json \
  --limit 5 \
  --budget 3000 \
  --out bench/battery/tight_smoke_b3000.json
```

Run the selected valid subset:

```bash
PYTHONPATH=.:src python3 evals/battery/run_battery_spike.py \
  --data /path/to/longmemeval_s.json \
  --budget 3000 \
  --out bench/battery/tight_valid_b3000.json
```

The budget above is a placeholder. The artifact only counts if `valid: true`.

## Deferred Until After This Gate

- foreground inter-token p50/p99 under contention;
- balanced operating point sweeps across span size, output cap, power, and
  thermal behavior;
- scheduler polish;
- embedding-based semantic identity beyond the JSON robustness and current
  slot-supersession path.
- full graph-resident memory replacement. If selected, begin with only the
  seeded graph store, serialization, supersession, and model-free read path from
  [`REM-memory-architecture-spec.md`](REM-memory-architecture-spec.md), before
  adding worker extraction.

Those are important, but they should not displace the recall proof.

## Post-Gate Method Evaluation Queue (added 2026-06-27)

These evaluations begin only after the bounded read path produces a gradeable
native REM result. They are comparisons, not permission to replace the current
architecture. The full intake and acceptance contracts live in the Tesla repo
under `projects/rem-methods-evaluation/`.

1. **Run TencentDB Agent Memory as a black-box comparator.** Feed it the same
   fixed valid LongMemEval `knowledge-update` subset, budget, prompts, and
   scoring used for native REM. Record recall, retained evidence, stale-ghost
   failures, injected tokens, latency, and provenance. End with a
   keep/borrow/reject note. Do not integrate its plugin or storage code during
   the comparison.
2. **Test DREAM-0.5B as an embedding challenger.** After the current
   Qwen3-Embedding-0.6B baseline is wired into the relevant identity/retrieval
   path, compare both models on a frozen paraphrase, unseen-vocabulary,
   same-slot/different-slot, and stale-update fixture. Run DREAM through
   Transformers/PEFT first. Convert to GGUF only if it wins the preregistered
   quality threshold without exceeding the 400 ms p95 retrieval target.
3. **Borrow mechanisms only on measured evidence.** Candidate mechanisms are
   provenance drill-down, progressive disclosure, and hybrid lexical/dense
   retrieval. Any architectural change driven by these comparisons requires a
   new versioned proposal; do not silently revise the current spec.

DSpark, JetSpec, FastContext, TS-ICL, MiniMax MSA, DAPO, and the social-archive
design are not REM roadmap items.

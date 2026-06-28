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

## Progress (updated 2026-06-27)

The gate is resolved. The bounded read path was built and validated, the first
valid run answers instead of overflowing, and the five-item failure mix has been
read. The architecture decision (item 7) is settled for now: keep the tuned read
path; the mix does not justify the graph rebuild.

- **Bounded read path (Step 0): PASS.** `RecencySelector` + `REM_READ_FIT_TOKENS`
  fit compacted memory to budget; `fit_with_selector` measures the assembled cost
  and trims the lowest-priority tiers so it honestly fits, leaving the protected
  slot+verbatim floor (the gold) intact. On 031748ae: 40,626 → 27,996 tokens, gold
  preserved, model answers (no HTTP 400). See `bench/battery/FINDINGS.md` "Step 0".
- **Failure mix (five oldest-gold items): read path holds 4/5.** All five fit the
  budget (27,891–27,999) — size is not a failure mode. Mix: 4 pass, 1
  temporal-structure (031748ae, the dataset-ambiguous item), 0 retrieval-recall.
  Per-item states captured NPU-free-reusable under `bench/battery/states/`; labels
  in `bench/battery/mix_report.json`. See `bench/battery/FINDINGS.md` "Failure mix".
- **Next lever (not the graph): slot-key canonicalization.** The write-recall audit
  (`evals/battery/write_recall_audit.py`) shows write recall is good but supersession
  fires on only ~1.1–1.3% of entries — the extractor assigns a fresh slot key per
  mention, so one value lands under up to 7 keys (32–55 fragmented values/item).
  That fragmentation is both the 950-entry ledger bloat and the lone
  temporal-structure miss (031748ae's 4 and 5 active under different keys, unordered).
  Normalize keys / add semantic supersession so updates collapse — the most likely
  single fix, without a graph. Secondary: a larger token-matched *unambiguous* item
  set, and slot-value-aware needle matching (methodology). See `bench/battery/FINDINGS.md`.

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

7. **Choose the next architecture from the failure mix.** _[resolved 2026-06-27 —
   keep the read path]_
   The five-item mix is in: the bounded read path fits all five and answers 4/5
   correctly; the lone miss is a then→now link on the dataset-ambiguous item, not
   summary corruption or read-path size. That is thin evidence for the graph
   rebuild, and the architecture spec's own guidance (rows 36, 42–45) is to
   validate read recall and fix write recall before assuming the graph helps. The
   capture surfaced write-side noise (malformed entries, slot-key fragmentation),
   so the near-term work stays on write robustness and a cleaner item set, not the
   graph. Revisit the graph if temporal-structure failures recur on unambiguous
   items. See `bench/battery/FINDINGS.md` "Failure mix".

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

Resolved (2026-06-27): the `context_overflow` block is cleared. Compacted memory on
the five oldest-gold items was 36,977–58,150 tokens (the assembler renders the full
ledger + all summaries with no size bound), so the first valid run scored 0/5
without ever answering. The fix was a bounded read path (Step 0): the selector fits
memory to the model window and the consumer trims the assembled slice to budget. All
five now fit (27,891–27,999) and answer; the read path holds 4/5 on the failure mix.
See `bench/battery/FINDINGS.md` ("Step 0", "Failure mix").

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

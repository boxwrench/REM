# Failure Mix — Post-Step-0 Design (capture states, then iterate NPU-free)

**Date:** 2026-06-27
**Status:** Design, awaiting approval
**Parent:** `docs/superpowers/specs/2026-06-26-bounded-read-path-design.md` §8 (open question)
**Predecessor result:** `bench/battery/FINDINGS.md` "Step 0: bounded read path" — Step 0 PASS on item 031748ae, with the caveat that a needle present in the fitted slice does not guarantee the model can use it.

## 1. Goal and scope

Step 0 proved the bounded read path fits one item and preserves its gold. It did
not choose the architecture. Spec §8 says the choice between **continuing to tune
the read path** (retrieval/eviction variants) and **committing to the
graph-resident read path** waits on the *failure mix* across the five oldest-gold
knowledge-update items: for each item, is the miss driven by **size** (cannot fit),
**retrieval-recall** (recency selects the wrong facts), or **temporal/structure**
(the right facts are present but not linked then→now)?

This increment produces that mix. It has two halves:

1. **Capture (one NPU cost).** Run the real ~75-min compaction once per item and
   persist each compacted `MemoryState` to disk. Five items ≈ ~6h of NPU; item
   `031748ae` is already captured, so only four remain (~5h). This is the only NPU
   spend in this increment beyond brief per-item answer inferences.
2. **Iterate (NPU-free).** Load the saved states and run the read path / selector
   analysis over all five with zero ingest, classifying each item's miss into the
   §8 taxonomy.

## 2. Non-goals (deferred)

- **New selector variants (lexical, structure-aware).** The mix tells us which
  lever to pull; building a `LexicalSelector` is the *next* increment if the mix
  says retrieval-recall dominates (spec D2). This increment characterizes misses
  with the existing `RecencySelector`.
- **The graph-resident read path.** That is the alternative the mix decides for or
  against; it is not built here.
- **Judged-correctness as the bar.** Per the Step 0 spec §2, the items' gold can
  rest on dataset inferences. The mix uses gold-survival and tier provenance as the
  primary signals; judged correctness is recorded where a brief answer is taken but
  does not gate the taxonomy.
- **The 950-entry ledger bloat.** Unchanged; acknowledged separately.

## 3. The five items (deterministic selection)

`load_knowledge_update(data, max_gold_recency=0.33)` returns these five, sorted
oldest-gold first (verified NPU-free):

| question_id | gold_recency | sessions | turns | gold answer (abbrev.) |
|---|---|---|---|---|
| 031748ae | 0.163 | 50 | 494 | started 4 engineers → now 5 |
| 3ba21379 | 0.178 | 46 | 431 | Ford F-150 pickup truck |
| cc5ded98 | 0.264 | 54 | 493 | about two hours/day coding |
| c6853660 | 0.277 | 48 | 477 | increased coffee limit 1→2 cups |
| 9bbe84a2 | 0.283 | 54 | 459 | previous Apex goal: level 100 |

## 4. Capture format

States live under `bench/battery/states/`:

- `bench/battery/states/<question_id>_state.json` — the compacted `MemoryState`
  (via `MemoryState.save`), one per item, ~0.8–1.2 MB each.
- `bench/battery/states/manifest.json` — a list of per-item records, written
  **incrementally** (after each item completes) so a crash mid-ingest keeps the
  states already captured. Each record:

  ```json
  {
    "question_id": "...", "question": "...", "answer": "...",
    "answer_session_ids": ["..."], "gold_recency": 0.178,
    "n_sessions": 46, "n_turns": 431,
    "state_file": "bench/battery/states/3ba21379_state.json",
    "assembled_total_tokens": 50529, "ingest_secs": 4480.0,
    "captured_at": 1719500000.0
  }
  ```

Capture settings match the existing 031748ae diagnostic so all five are
comparable: `budget_tokens=1000`, `max_context_tokens=DIAG_WINDOW_TOKENS=64000`,
`summarizer_model=gemma4-it:e2b`. Item `031748ae`'s existing state
(`bench/battery/diag_031748ae_w64k_state.json`) is pre-seeded into the states dir;
capture is **idempotent** (skips any item whose state file already exists), so it
ingests only the four missing items and is resumable.

## 5. Mix analysis (NPU-free)

For each captured state, load it and run the bounded read path
(`fit_with_selector` + `RecencySelector`, the Step-0 read path), recording:

- `fitted_tokens` and `fits_budget` (≤ `REM_READ_FIT_TOKENS`).
- Per gold needle: present in the fitted slice, and **which tier carries it**
  (protected slot / summary / free entry / absent). Needles are curated per item
  from the gold answer (e.g. `3ba21379` → "F-150"; `cc5ded98` → "two hours";
  `c6853660` → "two cups", "increased"; `9bbe84a2` → "level 100"; `031748ae` →
  "4 engineers", "5 engineers") and stored alongside the manifest.
- A **failure-mode label** per item, by this rule:
  - `size` — the protected floor alone exceeds the budget (cannot fit even minimal
    current-state), i.e. fitting is impossible without dropping a current-state slot.
  - `retrieval-recall` — fits the budget, but a gold needle is **absent** from the
    fitted slice (recency dropped the carrying summary/entry).
  - `temporal-structure` — fits and all needles are **present**, yet a brief answer
    fails to connect them then→now (the 031748ae shape: needles in the slice, model
    still cannot answer or answers wrong on the then/now relation).
  - `pass` — fits, needles present, brief answer correct.
- Optionally one brief answer inference per item (the only NPU in this half), to
  separate `temporal-structure` from `pass`.

Output: `bench/battery/mix_report.json` (per-item records + a counts summary) and a
printed table.

## 6. Success criterion

This increment succeeds when all five states are captured and `mix_report.json`
assigns each item a failure-mode label with its supporting evidence (fitted tokens,
per-needle tier provenance, brief answer). The **mix** — how the five distribute
across `size` / `retrieval-recall` / `temporal-structure` — is the deliverable.
There is no PASS/FAIL bar; the distribution is the evidence §8 waits on.

## 7. Decision log (for backtracking)

| # | Decision | Chosen | Rejected | Reverse if… |
|---|---|---|---|---|
| E1 | Capture vehicle | New `capture_states.py` (ingest + save per item, idempotent) | Extend `diagnose_memory.py` (single-item, `limit=1`) | A single multi-item entry point is wanted; then fold capture into diagnose. |
| E2 | Capture settings | Match 031748ae (budget 1000, window 64000) | Re-tune per item | A different budget is the actual deployment target; recapture all five to match. |
| E3 | Idempotent skip | Skip items whose state file exists (resumable) | Always re-ingest | A state file can be stale (compaction changed); then add a `--force` re-ingest. |
| E4 | Mix taxonomy | size / retrieval-recall / temporal-structure (rule in §5) | Judged-correct vs not | The three-way split proves too coarse; refine once the first five are labelled. |
| E5 | Needle source | Curated per item from the gold answer | Auto-extract salient tokens | Curation does not scale past five; then automate needle extraction. |
| E6 | Variants now? | No — characterize with RecencySelector only | Build LexicalSelector in the same increment | The mix is already obviously retrieval-recall on capture; then promote the variant build. |

## 8. What this unblocks

The labelled mix decides the architecture question §8 carries: a mix dominated by
`retrieval-recall` argues for a `LexicalSelector` (keep tuning the read path); one
dominated by `temporal-structure` argues for the graph-resident read path
(then→now links the flat ledger cannot represent); a mix dominated by `size` argues
for bounding the ledger writer first. The five-item distribution is the evidence;
this increment is the instrument that produces it.

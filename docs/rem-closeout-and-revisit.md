# REM — close-out & revisit map (2026-07-02)

Deliberate stopping point for the memory-quality work. **The bottleneck has moved off
memory and onto the 2B answerer.** Knowledge-update recall is memory/read-bound and is
fixed + shipped; everything else parked at a *real* wall (a model or methodology limit,
not a "needs more effort" limit). This doc records the walls AND the concrete condition
that would make each worth revisiting, so a future advance can resume with full context.

Branch `recall-gate-extraction-observability` @ `2f08732`. Suite 267 green. Supersession
flag OFF; nothing destructive shipped. Everything below is pushed.

---

## 1. What shipped (the result)
- **Sparse-chronological read path + question taxonomy**, wired into the serving path
  (`sidecar.py` uses `SparseChronologicalSelector`): relevance floor + top-k + oldest→
  newest render, replacing the 28k budget-fill that dumped distractors. Non-destructive.
- **Provenance schema** (nullable, additive): `Turn.session_id/timestamp`,
  `FactEntry.modality`; threaded through capture → compaction.
- **Knowledge-update result** (dev/small-n but consistent): gold present AND active in
  the ledger 6/6 (extraction is not the bottleneck); decision gate current 2 < sparse
  3–4 < oracle 5–6 (/6). Read-side is where the KU gains live, safely.
- Supporting fixes: capture diagnostic-assemble ceiling 64k→1M (temporal states save);
  `source_turn_id` "Turn 3"→3 coercion; episode-card prompt realigned to the extractor.

## 2. The walls (why each avenue stopped) + revisit trigger

### A. Temporal reasoning — ANSWERER-BOUND (the big one)
- Evidence: held-out temporal items (ordering / abstention / date-arithmetic) scored
  **0/3 for every arm INCLUDING oracle** (`heldout_temporal_eval.md`). Perfect evidence
  still fails — the 2B model can't reconstruct an order, won't abstain, won't compute a
  date delta. Retrieval cannot lift a reasoning ceiling.
- Cheap partial win found: an **abstention scaffold instruction rescues abstention**
  (0→1); ordering/arithmetic stay bound (`temporal_scaffold_probe.json`).
- **REVISIT WHEN:** a stronger answerer (bigger/better-reasoning model) or explicit
  tool-use (a date-delta calculator, an ordering step) is available. The memory/retrieval
  side is already able to deliver oracle-quality evidence; only the reasoner is missing.
  Also: fold the abstention instruction into the shipped answer prompt if abstention
  matters (cheap, prompt-only).

### B. Write-side supersession / canonical keying (Path B) — near SAFE CEILING
- Similarity fails in BOTH directions on the numeric class: ~11% precision AND ~0 recall
  on genuine updates (`FINDINGS.md` held-out check; `path_b_role_key_audit.*`). Held-out
  numeric audit: 291 candidates → ~258 false / ~30 same-value dedup / **~0–3 genuine**.
- The safe deterministic role-keyer cut fragmentation only 26.5% (< the 50% bar) — but
  the residual is ~entirely distinct-instance coincidences that MUST NOT merge, so the
  keyer is AT the safe ceiling and the 50% bar was miscalibrated
  (`path_b_residual_fragmentation.md`).
- **REVISIT WHEN:** a cheap, reliable **typed identity judge** exists (a model that can
  cheaply decide SAME/DIFFERENT slot at write time — the scaffold `TypedIdentityMatcher`
  + `make_gemma_slot_judge` are built and parked), OR role-aware canonical keying is moved
  into the extraction prompt. Do NOT chase aggregate fragmentation reduction — score on
  the sentinel families (start/end, min/max, fridge/freezer, sets/reps, per-instance),
  which are frozen in the audits as a labeled regression set.

### C. Read-time newest-preference (safe stale→current at read) — UNFINISHED
- Cross-key version is unsafe (role/instance false merges) → **default OFF**. The
  role-scoped prototype (`src/rem/memory/role_keys.py`) is provably safe on all 7
  sentinels but over-merges on rich multi-attribute subjects (grouped `count` with
  `woodpecker types`), so it isn't yet a net win
  (`role_scoped_newest_pref_prototype.md`).
- **REVISIT / FINISH:** add an attribute-head constraint to `group_same_role` (only
  prefer-newest among entries sharing the same attribute head), re-check the 7 sentinels,
  and validate on **held-out KU** (NOT the overfit dev items). This is the closest thing
  to more read-side headroom for KU.

### D. Throughput / one-call episode card (Path D) — MARGINAL, shelved
- Call count halves (2→1) but wall-clock barely moves: steady-state median 4.14s vs
  3.93s (~5%); the one big call ≈ two small calls (same total work). The 8s→3.7s outlier
  was a cold-start artifact. And the one-call card decomposes subject/attribute worse →
  messier slot_keys, which hurts the read side. `episode_card_wallclock.json`.
- **REVISIT IF:** per-call overhead turns out to dominate on much longer conversations
  (the benchmark fixture is only 3 turns, so scaling is unmeasured), or a cheaper single-
  call path emerges. Mechanism is built behind a flag (default OFF).

## 3. Durable, model-independent insights (survive any revisit)
- **similarity ≠ identity** (shown 4 independent ways). Do not build write-time identity
  on embedding similarity alone.
- Put uncertain decisions on the **read side** (reversible, query-aware); keep truth
  **immutable + additive**. "Compact access paths, not truth."
- On this model: **KU recall = memory/read-bound (fixable); temporal reasoning =
  answerer-bound (not memory).** Extraction faithfully captured the gold in every KU case.
- **Extraction is nondeterministic**, so exact-equivalence promotion gates are the wrong
  bar (killed both Path B's 50% and Path D's fact-signature gate). Score on labeled
  sentinels / semantic equivalence, not byte-identical output.

## 4. If someone resumes: how to validate the KU read-side win properly
The frozen 30-item held-out suite is 10 KU + 10 temporal + 10 multi-session. Temporal is
answerer-bound and will NOT show the read-side signal, so do not use it to validate the
read path. Capture held-out **KU** items (LongMemEval-S pool beyond the manifest) and/or
**multi-session LOOKUP** items (e.g. `a96c20ee` "which university…") — those are the
retrieval-bound cases where sparse-vs-current actually separates. Harness ready:
`evals/memory_methods/heldout_eval.py` (arms + `--read-fit` to cap to the flm window).

## 5. State on disk / to know
- 13 captured states: 10 KU (6 fresh fixed-extractor + 4 reused pre-fix) + 3 temporal
  held-out. 17 temporal/multi uncaptured (intentionally — not worth the NPU).
- Answerer = `gemma4-it:e2b` on the NPU via `flm` (port 13306). No `ANTHROPIC_API_KEY`
  in-env, so the Claude judge path needs a key; a strong in-conversation judge is the
  audited fallback.
- Scheduled `rem-supersession-heldout-audit` task may still be enabled — disable it.

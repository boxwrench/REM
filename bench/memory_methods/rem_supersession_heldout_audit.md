# REM supersession held-out audit — fresh recapture (NPU-free)

**Run:** 2026-06-30 · 6/6 fresh held-out states present · recapture COMPLETE
**Models:** Qwen3-Embedding-0.6B (CPU), threshold 0.8

## Verdict (lead)

The ce6d2d27 numeric over-merge problem **reproduces on every one of the five newer fresh
states.** In all six states the value-gate's headline holds — `textual_distinct` collapses
plain→instance_aware to **0** — but the `numeric_update` merges remain **dominated by FALSE
merges of distinct facts**, not genuine then→now updates. The count alone is still misleading.
The false-merge classes are identical to ce6d2d27: start-date↔end-date, min↔max of one range,
fridge↔freezer duration, sets↔reps, different instances (shops, social accounts, companies,
salary sectors, devices, trip options), and unrelated percentages. Genuine updates are rare to
nonexistent; the only benign merges are a minority of same-fact unit/format dedups.

## Per-state

| state | textual_distinct (plain→IA) | numeric_update | ~false / distinct-fact merges | genuine then→now |
|-------|------------------------------|----------------|-------------------------------|------------------|
| ce6d2d27 | 304 → 0 ✓ | 46 | ~40 (≈87%) | ~0 |
| 945e3d21 | 273 → 0 ✓ | 45 | ~33 (≈75%) | ~1 |
| 6071bd76 | 318 → 0 ✓ | 47 | ~42 (≈89%) | ~1 (coffee ratio) |
| 22d2cb42 | 329 → 0 ✓ | 78 | ~68 (≈87%) | ~2 |
| dfde3500 | 230 → 0 ✓ | 45 | ~33 (≈73%) | ~1 |
| affe2881 | 276 → 0 ✓ | 30 | ~24 (≈80%) | ~0 |

`textual_distinct` → 0 confirmed on all six (instance_aware reassigns those to `blocked`).
Counts are my by-eye read of the dumped pairs, not a token heuristic.

## Representative false merges (read from the pairs)

- **start/end & min/max:** `dates.start=May 26` ↔ `dates.end=May 28`; `price.min=40` ↔ `price.max=50`;
  `score.min=1` ↔ `score.max=5`; DV-2024 registration start↔end; exhibition start↔end dates;
  calvin&hobbes 1985↔1995; birth-date↔death-date (multiple people).
- **fridge vs freezer:** `refrigerator=5 days` ↔ `freezer=3 months`; `40°F` ↔ `0°F`;
  refrigerated broth 5–7 days ↔ frozen 3–4 months.
- **different instances:** daily-grind 10-min ↔ coffee-club 15-min walk; google 24wk ↔ facebook 26wk
  parental leave; finance ↔ government ↔ healthcare data-scientist salary bands; running-hits 2M ↔
  fitness-beats 1.5M ↔ running-jams 200K followers; fitbit-charge-3 ↔ inspire-hr battery; ski resorts.
- **sets vs reps / attribute mixups:** `russian twists.sets=3` ↔ `.reps=12-15`; `glute bridges.reps`
  ↔ `.rest`; whatsapp `members=256` ↔ `name=25 chars` ↔ `description=512 chars`; n_estimators ↔ max_depth.
- **unrelated percentages:** women-in-energy 27% ↔ construction 12%; white 34.3% ↔ black 57.4%;
  probability bands (very-unlikely/unlikely/even/probable) all cross-merged.

## Benign (true) merges — the minority

Same fact in different units/wording: 68–72°F ↔ 20–22°C; `f(x)=e^{-x*x/2.}` ↔ `f(x)=e^(-x^2/2)`;
whatsapp `256` ↔ `256`; file-sharing `100 MB` ↔ `100 MB`; linux-lab "30 commands" descriptions;
kdp royalty `70%/35%` ↔ `35% and 70%`. These are harmless dedups, not then→now updates.

## Bottom line

Across all six fresh held-out states the value-gate cannot tell a real numeric supersession from
two distinct facts that merely share a numeric attribute. The `numeric_update` figure (~30–78/state)
overstates real updates by roughly 3–9×; the true then→now update rate is near zero. The instance-aware
gate fixes only the textual-distinct class. Numeric merging still needs reasoning, not embedding
similarity + a token rule.

Outputs:
- `bench/memory_methods/supersession_instanceaware_freshcheck.json`
- `bench/memory_methods/numeric_merge_audit_fresh.json`

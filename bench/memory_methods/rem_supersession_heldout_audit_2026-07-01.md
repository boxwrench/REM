# REM held-out supersession audit — all six fresh states (2026-07-01)

NPU-free. Model `Qwen/Qwen3-Embedding-0.6B`, threshold 0.80. No answerer, no NPU, nothing committed.
Recapture **complete**: all six fresh held-out ids present (ce6d2d27, 945e3d21, 6071bd76, 22d2cb42, dfde3500, affe2881).

## Verdict (lead)

**The ce6d2d27 numeric over-merge problem reproduces on every one of the five newer fresh states.**
The value-gate headline still looks clean everywhere (textual_distinct → 0 under instance_aware), but the `numeric_update` bucket is dominated by FALSE merges — distinct facts that merely share a numeric-looking attribute. Genuine then→now updates (the same fact's value corrected across the conversation) are essentially absent: at most 0–1 plausible ones per state, and even those are ambiguous. The count alone remains misleading; the gate cannot block these and the token heuristic mislabels them.

## Per-state read

| state | textual_distinct plain → instance_aware | numeric_update count | est. FALSE merges | genuine then→now updates |
|---|---|---|---|---|
| ce6d2d27 | 304 → 0 | 46 | ~43 (≈93%) | ~0 |
| 945e3d21 | 273 → 0 | 45 | ~36 (≈80%) | ~0 |
| 6071bd76 | 318 → 0 | 47 | ~43 (≈91%) | ~0 |
| 22d2cb42 | 329 → 0 | 78 | ~70 (≈90%) | 0–2 (ambiguous exhibition date edits) |
| dfde3500 | 230 → 0 | 45 | ~38 (≈84%) | ~0 |
| affe2881 | 276 → 0 | 30 | ~24 (≈80%) | 0–1 (gyro taco price, ambiguous) |
| **total** | **1730 → 0** | **291** | **~254 (≈87%)** | **~0–3** |

The non-false remainder in each state is not genuine supersession either — it is same-fact dedup / unit- or phrasing-restatement (e.g. Fahrenheit↔Celsius of one range, "100 MB"↔"100 MB in size", KU 90-day↔90-day), which the numeric bucket also mislabels.

## The recurring false-merge families (identical across all six states)

- **start ↔ end of the same span** — start date/end date, start time/end time, start age/end age, publication start/end year, registration open/close. (e.g. `dates.start date=May 26 ← end date=May 28`; `techxpo start Feb 10 ← end Feb 12`; `calvin and hobbes start 1985 ← end 1995`; `spartan start age 7 ← end age 30`)
- **min ↔ max / low ↔ high of one range** — price min/max, capacity min/max, score min/max, total-cost min/max, NYC spring-high/summer-high/winter-low.
- **different instances of the same category** — different meals' protein, different coffee shops' walk distance, different IG accounts' follower counts, different salary sectors, different ski resorts' acreage, different fish sizes' feeding amounts, different fitbit models' battery life, different road-trip options' total cost, different probability-band definitions.
- **two different attributes of one entity** — sets vs reps, water-resistance vs battery-life, member-limit vs name-length, screen-size vs price, followers vs engagement-rate, birth-date vs death-date, credit% vs end-date.
- **unrelated numbers colliding at threshold** — 27% women-in-energy ← 12% women-in-construction; salt 1/4 tsp ← milk 1/4 cup; watermelon 1–2 oz ← mango 1–2 cups.

22d2cb42 (78 merges) is the worst offender purely because it packs many exercise routines (sets-vs-reps), salary sectors, and ski-resort terrains — all classic false-merge shapes — not because the behavior differs.

## Bottom line

Headline metric (textual collapse → 0) is confirmed on all six fresh, non-overfit states. But numeric supersession is not working: ~87% of numeric merges are false, ~0% are real updates. The problem is systematic and reproduces cleanly beyond ce6d2d27 — it needs a typed judge (distinguish then→now update from distinct same-attribute fact), not a similarity threshold or token rule.

Sources: `bench/memory_methods/supersession_instanceaware_freshcheck.json`, `bench/memory_methods/numeric_merge_audit_fresh.json` (both regenerated 2026-07-01 15:10 covering all six ids).

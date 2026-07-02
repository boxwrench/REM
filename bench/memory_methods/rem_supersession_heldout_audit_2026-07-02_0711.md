# REM supersession held-out AUDIT — 2026-07-02 07:11 (NPU-free)

## Verdict (numeric false-merge — read first)

**The ce6d2d27 numeric over-merge problem REPRODUCES on all five newer fresh states.**
On every one of the six freshly captured held-out states, the `numeric_update` merges are dominated by FALSE merges — distinct facts that merely share a numeric attribute type, not genuine then→now updates. Genuine temporal supersessions are nearly absent in the high-similarity band. The value-gate cannot touch these (its `numeric_update` count is identical, 291, with or without the instance-aware gate) because they are not textual near-duplicates — they are distinct-attribute numeric collisions the token heuristic mislabels as updates.

Recurring false-merge families observed across states: start-date ↔ end-date, min ↔ max, refrigerator ↔ freezer duration/temperature, two different shops / social accounts / companies / devices, different demographic groups, different probability bands, sets ↔ reps, water-resistance ↔ battery-life, undergrad ↔ grad.

## Headline vs. reality

All six states pass the textual headline and all fail the numeric substance:

| state | textual_distinct plain→IA | numeric_update count | false-merge read (top-12) |
|---|---|---|---|
| ce6d2d27 | 304 → 0 ✓ | 46 | ~11/12 false, ~0 genuine |
| 945e3d21 | 273 → 0 ✓ | 45 | ~9/12 false, ~3 dedup, ~0–1 genuine |
| 6071bd76 | 318 → 0 ✓ | 47 | ~11/12 false, ~1 dedup, ~0 genuine |
| 22d2cb42 | 329 → 0 ✓ | 78 | ~10/12 false, ~1 dedup, ~1–2 borderline |
| dfde3500 | 230 → 0 ✓ | 45 | ~7/12 false, ~4 dedup (same value), ~0 genuine |
| affe2881 | 276 → 0 ✓ | 30 | ~9/12 false, ~2 dedup, ~0 genuine |

Fresh runs reproduced the prior numbers exactly (textual → 0 on all six; numeric 46/45/47/78/45/30). "dedup" above = same value restated under a reworded attribute (harmless); "genuine" = a real then→now update (rare).

## Per-state evidence (representative merged pairs)

**ce6d2d27 (46)** — start/end dates (May 26 ↔ May 28); price min ↔ max (40 ↔ 50); capacity min ↔ max (200 ↔ 1,500); fridge ↔ freezer duration (5 days ↔ 3 months); two dishes' protein (30g ↔ 35g); two databases' counts (GeoNames 11M ↔ Getty 1M); screen-size ↔ price on the same laptop. Essentially all false.

**945e3d21 (45)** — TechXpo start ↔ end (Feb 10 ↔ Feb 12); market hours start ↔ end (9AM ↔ 1PM); Spartan training start-age ↔ end-age (7 ↔ 30); followers ↔ likes (200 ↔ 50); Saginaw under-18 ↔ over-65 counts (16,667 ↔ 4,017); squirrel leap-distance ↔ fall-height. A few same-fact restatements (marriage year, stay duration).

**6071bd76 (47)** — worst for cross-instance follower merges: running hits 2M ↔ running followers 1M ↔ fitness beats 1.5M ↔ workout faves 500K ↔ running jams 200K all merged as "updates." Also undergrad 17% ↔ grad 36% intl students; probability bands (unlikely ↔ very unlikely, probable ↔ very probable); fridge ↔ freezer shelf life; succulent summer ↔ winter watering. Almost entirely false.

**22d2cb42 (78, highest)** — DV-2024 registration start ↔ end; exhibition start/end dates; score min ↔ max (1 ↔ 5); Iguazú minimum ↔ recommended stay; finance ↔ government salary ranges; Google 24-wk ↔ Facebook 26-wk parental leave; total works ↔ works-on-display (200 ↔ 60); sets ↔ reps. Mostly false; a couple of conflicting-exhibition-date pairs are borderline.

**dfde3500 (45)** — Calvin & Hobbes start ↔ end year (1985 ↔ 1995); small ↔ medium ↔ large fish feeding amounts; meditation daily-duration ↔ weekly-frequency; Verizon monthly ↔ pay-as-you-go pricing; WhatsApp member-limit (256) ↔ group-name char-limit (25). Higher dedup share (4 same-value restatements) but the distinct-fact merges are still false.

**affe2881 (30, lowest)** — under-35 ↔ 38-40 IVF success rates (40–50% ↔ 20–30%); refrigerated ↔ frozen broth; HD 1080p ↔ 4K 2160p; cost min ↔ max ($10k ↔ $20k); two Fitbit models' battery life (7d ↔ 5d); water-resistance ↔ battery on the same device. Almost all false.

## Status & next step

- **Recapture COMPLETE** — all six fresh held-out ids present (ce6d2d27, 945e3d21, 6071bd76, 22d2cb42, dfde3500, affe2881) plus the four reused dev states. Capture log: "limit reached (6 new capture(s) this run)".
- You can now **DISABLE this recurring task** from the Scheduled panel.
- Constraints honored: NPU-free (no `--answer`, no answerer); nothing committed.

Outputs written (uncommitted):
`bench/memory_methods/supersession_instanceaware_freshcheck.json`,
`bench/memory_methods/numeric_merge_audit_fresh.json`.

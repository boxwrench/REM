# Held-out supersession audit — 6 fresh states (NPU-free)

> CORRECTION (see bench/battery/FINDINGS.md, "Held-out check ALL SIX fresh states"): the "~genuine"
> column below (~33 total) is the NON-FALSE count with same-value dedups and genuine then->now
> CONFLATED. Split by the careful 07-01/07-02 reads it is ~30 same-value dedup PLUS ~0-3 genuine
> then->now. Do NOT cite "~33 genuine" — genuine updates are ~0-3. FINDINGS is authoritative.

Run: 2026-07-02. Embedding model Qwen/Qwen3-Embedding-0.6B (local CPU), threshold 0.80.
Manifest: bench/memory_methods/development_manifest.json
Outputs: supersession_instanceaware_freshcheck.json, numeric_merge_audit_fresh.json

## Verdict (lead)

The ce6d2d27 numeric over-merge problem **reproduces on all six fresh held-out states — it is systemic, not a one-state artifact.** The value-gate's headline is real (textual_distinct 1730 -> 0 across the six), but the numeric_update merges it leaves untouched are **dominated by false merges everywhere** — roughly **85–90% false** by reading the pairs. 22d2cb42 is the worst (78 numeric merges). The value-gate cannot block these; the numeric_update count is a misleading "win."

## Per-state results

| state | textual_distinct plain->IA | numeric_update | ~false merges | ~genuine | reproduces? |
|-------|---------------------------|----------------|---------------|----------|-------------|
| ce6d2d27 | 304 -> 0 | 46 | ~43 (93%) | ~3 | yes |
| 945e3d21 | 273 -> 0 | 45 | ~39 (87%) | ~6 | yes |
| 6071bd76 | 318 -> 0 | 47 | ~42 (90%) | ~5 | yes |
| 22d2cb42 | 329 -> 0 | 78 | ~70 (90%) | ~8 | yes (worst) |
| dfde3500 | 230 -> 0 | 45 | ~36 (80%) | ~9 | yes |
| affe2881 | 276 -> 0 | 30 | ~28 (93%) | ~2 | yes |
| **TOTAL** | **1730 -> 0** | **291** | **~258** | **~33** | — |

textual_distinct goes to 0 on every state (headline holds). numeric_update is identical plain vs instance_aware (291) — the gate does not touch it.

## The false-merge taxonomy from ce6d2d27 recurs on every state

- **start-date <-> end-date**: techxpo Feb 10/12, market 9:00 AM/1:00 PM (945e3d21); DV-2024 registration & exhibition start/end, vacation Jun 15/20 (22d2cb42); Calvin&Hobbes 1985/1995, rush-hour AM/PM (dfde3500); NYC spring/summer/winter temps (affe2881).
- **min <-> max / range endpoints**: score min/max, iguazú min/recommended stay (22d2cb42); total-cost min/max (affe2881); price min/max, capacity min/max (ce6d2d27).
- **fridge <-> freezer duration**: chicken parmesan refrigerated 3-4d vs frozen 3mo (6071bd76); broth 5-7d vs 3-4mo (affe2881); ce6d2d27 fridge 5d vs freezer 3mo.
- **different instances sharing an attribute**: social-media follower accounts cross-merged en masse (6071bd76: running hits/fitness beats/beast mode/…), data-scientist salary by sector (22d2cb42), ski-resort terrain acreage (22d2cb42), fish feeding by size (dfde3500), Fitbit/wearable devices (affe2881), two coffee shops' walk distances (ce6d2d27).
- **unrelated percentages**: probability-definition buckets (6071bd76), engagement rate 2-3% vs 30% & racial % (945e3d21), women-in-tech % (22d2cb42), intl-student % undergrad vs grad (6071bd76).

New/expanded classes on the fresh states:
- **sets <-> reps within an exercise** (both numeric attributes of same entity): russian twists sets=3 vs reps=12-15, calf raises, glute bridges, donkey kicks, hip thrusts, box jumps… (22d2cb42, 6071bd76).
- **attribute cross-merge within one entity**: wearable battery-life vs water-resistance (affe2881), laptop screen-size vs price (ce6d2d27), person marriages vs children (affe2881), birth-date vs death-date (945e3d21).

## Genuine merges are the minority (true dedups / then->now)

Examples that are legitimate: file-sharing 100 MB duplicate, evaluation-metrics duplicate, WhatsApp 256-member duplicate, white-rice 3-cups duplicate (dfde3500); credit-card delivery 7-10 days duplicate, CNN accuracy 75%->93% before/after tuning (6071bd76); linux-lab-exercise "30 commands" duplicates, stay-duration variants (945e3d21); calf-raises 12-15 duplicate, registration/entry-period time duplicates (22d2cb42). These total roughly 30-35 across all six — the rest are false.

## Bottom line

A token heuristic that just counts numeric_update as "good merges" will overstate the value-gate by ~10x. Blocking the false numeric merges needs typed reasoning (start/end, min/max, per-instance keys, sets/reps, distinct attributes of one entity) — not a similarity threshold, and not the current gate.

Recapture complete: all six fresh ids captured (capture_limit6 run hit its 6-item limit; 20 items still uncaptured but the held-out six are done).

# REM held-out supersession audit — fresh recapture (2026-06-30)

**NPU-free.** Six fresh held-out knowledge-update states recaptured on the fixed extractor and audited.
Recapture is **complete** (all six present; `capture_limit6.log`: "limit reached (6 new capture(s) this run)").

## Verdict (lead): the numeric over-merge reproduces — and gets worse

The value-gate's headline is real but misleading. On every fresh state the instance-aware
gate drives `textual_distinct` merges to **0**, exactly as designed. But the `numeric_update`
class it *preserves* is **dominated by false merges** — distinct facts that merely share a
numeric attribute, collapsed as if one supersedes the other. Reading the top-similarity pairs
(the ones the gate is most confident about), ~**55 of 72** shown merges are false. The ce6d2d27
problem is not a one-off; it is the general behavior, and **6071bd76** and **22d2cb42** are
worse than ce6d2d27.

Almost none of the "genuine" residue is an actual then→now temporal update. The non-false
pairs are overwhelmingly harmless re-dedups (unit conversions, reformatting, restatements of the
*same* value) — not the knowledge-updates the numeric class is supposed to represent.

## Per-state results

| id | textual_distinct plain→IA | numeric_update count | false among top-12 shown | representative false merges |
|----|---------------------------|----------------------|--------------------------|------------------------------|
| ce6d2d27 | 304 → 0 | 46 | ~9/12 | start↔end date; price min↔max; capacity min↔max; fridge↔freezer duration; geonames↔getty counts; two dishes' protein |
| 945e3d21 | 273 → 0 | 45 | ~7/12 | market start↔end time; training start↔end age; followers↔likes; under-18↔over-65 residents; squirrel leap↔fall |
| 6071bd76 | 318 → 0 | 47 | ~11/12 | undergrad↔grad intl %; probability buckets; fridge↔freezer; multiple distinct accounts' follower counts (2M/1.5M/1M/500K/200K) |
| 22d2cb42 | 329 → 0 | 78 | ~10/12 | reg start↔end date (×2); min↔recommended stay; finance↔govt salary; score min↔max; google↔facebook leave; sets↔reps |
| dfde3500 | 230 → 0 | 45 | ~8/12 | pub start↔end year; small/medium/large fish feeding; daily↔weekly meditation; monthly↔PAYG plan; members↔name-length |
| affe2881 | 276 → 0 | 30 | ~10/12 | age-bracket success rates; fridge↔freezer (×2); HD↔4K; cost min↔max; two Fitbit models; water-resist↔battery |

Totals across the six: `textual_distinct` 1730 → **0** (gate works); `numeric_update` **291**,
of which the readable sample says the large majority are false.

## What the gate cannot fix
The value-gate blocks *textual* distinctness but has no way to tell a then→now numeric update
from two distinct facts that happen to share a numeric attribute. A token heuristic mislabels
these (start vs end, min vs max, instance A vs instance B all look like "5→7"). Separating them
needs typed reasoning (attribute-role awareness: start/end, min/max, per-instance keys), not a
count and not a token rule.

## Artifacts
- `bench/memory_methods/supersession_instanceaware_freshcheck.json`
- `bench/memory_methods/numeric_merge_audit_fresh.json`

Recapture complete — the recurring **rem-supersession-heldout-audit** task can be disabled from the Scheduled panel.

# Extraction-recall on the 6 fresh KU states (NPU-free, 2026-07-02)

De-confounds the decision gate: is a gate miss a WRITE-side loss (gold never
extracted) or a READ-side miss (gold present but not retrieved/reasoned)?

## Result: extraction-recall = 6/6 — write side is NOT the bottleneck
For every one of the six fresh knowledge-update states, the gold value is extracted
into the ledger **and active**:

| item | gold | ledger entry (turn, status) |
|---|---|---|
| ce6d2d27 | Friday | `class preparation time frequency.day = Fridays` (t163, active) |
| 945e3d21 | three times a week | `yoga classes frequency.frequency per week = three times a week` (t189, active) |
| 6071bd76 | 5 ounces / less water | `coffee brewing.ratio = 1 tablespoon of coffee for every 5 ounces of water` (t209, active) |
| 22d2cb42 | Main St | `guitar servicing.location = Main St` (t102, active) |
| dfde3500 | Wednesday | `language exchange class.day = Wednesday evening` (t154, active) |
| affe2881 | 32 | `species count.total species count = 32` (t274, active) |

Malformed-fact drops during capture: 17 across the six states (~2.8/state, from the
capture log) — none of them dropped the gold.

## Interpretation
Every KU gate miss is therefore downstream of extraction. The current arm's 2/6 and
the residual to oracle are read/serialization/reasoning failures, not write loss.
This confirms the standing read-side direction and, importantly, shows the gold is
already present AND active — so no destructive write-time supersession/keying is
needed to make it available; the coffee `5 oz` (t209) simply coexists with the stale
`6 oz` (t13), which is exactly the read-time stale-vs-current problem.

## Caveats / notes
- The 6 fresh captures predate the telemetry-persistence commit (f933e81), so no
  structured extractor counters (attempts/repaired/failed) exist for them; the
  running Path C temporal captures DO persist telemetry, so richer per-state
  extraction stats will be available for the 20 held-out items.
- This is a stage-1 (retrieval-independent) measure: "is the value in the ledger,
  active." It does not measure whether every *supporting* fact for multi-hop items
  survived — but for these single-value KU golds it is the right stage-1 signal.

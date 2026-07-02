# Path B residual-fragmentation analysis (NPU-free, 2026-07-02)

Question: after the role-aware re-key cut fragmented values 302 -> 222 (26.5%, below
the 50% bar -> no-go), WHERE is the residual, and could a stronger keyer safely clear
the bar?

## Finding: the residual is ~entirely distinct facts that MUST NOT merge
The 222 residual "fragmented values" resolve into **30 groups** (a value shared by
>=2 slot keys) spanning **123 keyed facts**. Categorising by whether the keys share a
subject/instance:

- **24/30 (80%) different-subject** — distinct named instances that merely share a
  value: `35g` across six different meals' protein, `shinjuku-ku` across four coffee
  shops, `7` across agreement-rating / user-interest / grooming-kit-pieces. Merging
  these would be a FALSE merge (the exact error the value-gate made).
- **6/30 (20%) nominal "same-subject"** — but on inspection these are ALSO distinct
  instances sharing only a generic token: `5-7 pages` across financial-projections /
  market-analysis / marketing sections; `15-30 seconds` across calf / chest / glute
  stretch; `12-15 reps` across calf-raises / deadlifts / leg-raises. Not genuine
  single-slot fragmentation.

So effectively **~0 of the residual is safely collapsible.** The role-keyer already
collapsed the genuine updates (both positive sentinels collide) and correctly left
the distinct-instance coincidences apart (all five negative sentinels + gold
preserved).

## Implication: the 50% target was miscalibrated, not the keyer too weak
`fragmented_values` counts any value appearing under >=2 keys — but most such values
are distinct facts coinciding on a number/string, which no safe mechanism may merge.
A keyer that reached 50% would necessarily be **over-merging distinct instances**,
reintroducing the ~88%-false numeric-merge problem. The role-keyer's 26.5% with all
sentinels preserved is therefore near the SAFE CEILING, not a weak result.

Recommendation: do NOT chase a stronger keyer against the 50% aggregate bar. Re-state
the Path B success criterion in terms of the sentinels (collapse genuine updates,
preserve negatives + instances), which the current keyer already meets — and let the
answer-level payoff decide, noting extraction-recall is already 6/6 (gold present AND
active), so the KU leverage is read-side, not write-side. The typed judge stays
parked; there is little safe write-time headroom to justify it for this class.

## Method
`after.fragmentation_examples` (value -> [slot_keys]) from `path_b_role_key_audit.json`,
subject anchors via `role_keys._parts`, generic tokens filtered. Reproducible; no NPU.

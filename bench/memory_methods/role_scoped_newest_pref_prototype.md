# Role-scoped newest-preference — prototype + finding (NPU-free, 2026-07-02)

Reconciles Path A (cross-key newest-preference, **unsafe** per adversarial review,
default-off) with Path B (role-aware keyer that **preserved all five negative
sentinels** but missed the 50% aggregate fragmentation bar). Idea: at read time,
prefer the newest value ONLY within a genuine role-slot, so it can fix updates
without the cross-role/instance false merges that made Path A's version unsafe.

## Shipped in this prototype (all default-off; shipped behaviour unchanged)
- `src/rem/memory/role_keys.py` — Path B's `same_role` / `group_same_role` extracted
  into `src` (was trapped in the eval script), reusing `canonicalize._tokens`.
- `SparseChronologicalSelector(prefer_newest=True, newest_scope="role")` — new
  `newest_scope` switch; `"cross_key"` (default) keeps the existing behaviour,
  `"role"` uses `_prefer_newest_role_scoped`.
- `tests/unit/test_role_keys.py` — 4 tests. Suite 267 green; new files ruff-clean.

## Safety: PASS (the point of the exercise)
All five negative-sentinel families stay distinct and both positive sentinels
collapse, asserted directly on the audit's frozen slot keys:
- start/end, min/max, fridge/freezer, sets/reps, per-instance → `group_same_role` = []
- coffee `6oz→5oz`, birds `27→32` → one group each.
Also: a compatible middle spelling does not transitively bridge start↔end.

## Effectiveness on a REAL state: NOT YET A WIN — needs attribute-head scoping
On `affe2881` (birds, gold 32), `newest_scope="role"` REGRESSES vs `"cross_key"`:
cross-key surfaces `LATEST CURRENT OBSERVATION: ...total species found: 32`; role
surfaces a woodpecker-types line and drops the 32. Diagnosis — the grouping is
subject-anchored but not attribute-head-scoped, so it over-merges under a shared
subject:

    GROUP: bird species.count = 27 (t178)  +  bird species.woodpecker types = [...] (t275)

`count` and `woodpecker types` are different attributes of the same subject; the
newest (woodpecker, t275) is promoted and the count is discarded. (The genuine
count update actually lives in a *separate* clean group — `species count.number of
species=27` → `...total species count=32` — which resolves correctly; the harm is
the unrelated over-merge next to it.)

## Next step (defer; do NOT tune on affe2881)
Add an attribute-head / value-type constraint to `group_same_role` so newest-
preference only fires among entries sharing the same attribute head (or value type),
not merely the same subject — then re-check the sentinels still pass. Validate the
refinement on **Path C held-out** states, not on affe2881 (which is now a dev item);
designing the fix against it would overfit. The answerer-level confirm remains the
frozen 30-item Path-A protocol after capture completes.

## Bottom line
A provably-safe role-scoped newest-preference exists and is wired behind a flag, but
pure subject-anchored role grouping is too coarse to beat the (unsafe) cross-key
version on rich real states. The safe path forward is attribute-head scoping,
confirmed on held-out data. Nothing here changes shipped defaults.

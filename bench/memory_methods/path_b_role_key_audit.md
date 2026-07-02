# Path B step-1 role-aware re-key audit

## Decision: NO-GO

The NPU-free post-hoc role-aware keyer reduced raw fragmented values from 302 to
222 across the six fresh captured ledgers: **26.5%**, below the pre-registered
**50%** requirement. It preserved all six active gold needles and all five
negative-sentinel families (start/end, min/max, fridge/freezer, sets/reps, and
named per-instance facts). Both positive update sentinels—the coffee ratio and
bird species count—did collide.

Because the fragmentation bar failed, do not move this keyer into extraction or
recapture states for it. The persisted captures were not modified.

## Per-state raw fragmentation

| state | before | after | reduction |
|---|---:|---:|---:|
| ce6d2d27 | 66 | 47 | 28.8% |
| 945e3d21 | 60 | 42 | 30.0% |
| 6071bd76 | 48 | 39 | 18.8% |
| 22d2cb42 | 52 | 38 | 26.9% |
| dfde3500 | 31 | 26 | 16.1% |
| affe2881 | 45 | 30 | 33.3% |
| **total** | **302** | **222** | **26.5%** |

## Reproduce

```bash
PYTHONPATH=.:src python3 evals/memory_methods/run_role_key_audit.py
```

Machine-readable result: `bench/memory_methods/path_b_role_key_audit.json`.
That artifact records SHA-256 hashes for the three held-out audit reports used
as regression evidence and gives every gold and sentinel result explicitly.

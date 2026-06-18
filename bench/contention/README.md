# Contention evidence — raw artifacts for the canonical placement result

These are the **raw per-run measurements** behind the headline contention number
(NPU **3.81%** vs CPU **4.77%** iGPU decode loss, ~3× background throughput, ~3×
perf/watt). They make that claim self-auditable from this repository.

## Provenance

The measurements were produced with the [**xdna-top**](https://github.com/boxwrench/xdna-top)
contention harness (`bench/contention_benchmark.py`) on an AMD Ryzen AI MAX+ 395
(Strix Halo, gfx1151), kernel 6.17.0-35, xrt 2.21.75 — the same monitor REM uses
for its scheduler gauge. `METHODOLOGY.md` here is a verbatim copy of that
experiment's write-up. REM's own `evals/contention/run_contention_benchmark.py`
is a thin wrapper around the same measurement approach.

## Files

```
measurements/
  run{1,2,3}.m1_contention.json   # NPU arm: baseline vs concurrent-NPU, N=20 decode samples
  run{1,2,3}.m2_cpu_arm.json      # CPU control arm: concurrent-CPU, N=20 decode samples
evidence/
  baseline.snapshot.json          # xdna-top telemetry snapshots for each condition
  npu.snapshot.json
  cpu.snapshot.json
METHODOLOGY.md                    # full experiment write-up (frontmatter lists every artifact)
verify_contention.py              # re-derives 3.81% / 4.77% from the raw samples
```

## Audit it yourself

```bash
python bench/contention/verify_contention.py
```

This recomputes pooled `(baseline − concurrent) / baseline` decode loss across the
three runs (60 samples per arm) and asserts it matches the published 3.81% / 4.77%
within 0.05 pp. Per-run figures: NPU 4.19 / 3.79 / 3.45, CPU 5.09 / 5.04 / 4.19.

## Note on N

The canonical result pools **3 independent runs of N=20 = 60 decode samples** per
arm (DECISIONS D18). An earlier single N=5 run (MP1) did **not** reproduce and is
superseded — see the project history. Prefill throughput is reported separately
from decode and is intentionally **not** averaged into the headline (prefill is
high-variance; decode is what an interactive user feels).

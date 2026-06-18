#!/usr/bin/env python3
"""Re-derive the canonical contention headline from the committed raw artifacts.

Reads the six per-run measurement files in ``measurements/`` (3 NPU arm +
3 CPU arm, N=20 decode samples each) and recomputes the pooled iGPU decode
loss for each placement. This is the audit path for the headline number:
run it and confirm the repo's published 3.81% / 4.77% fall out of the raw data.

    python bench/contention/verify_contention.py

Exits non-zero if the recomputed means drift from the published values.
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MEAS = HERE / "measurements"

# Published canonical values (docs/npu-placement-benchmark.md, DECISIONS D18).
PUBLISHED = {"npu_loss_pct": 3.81, "cpu_loss_pct": 4.77}
TOL = 0.05  # percentage points


def _decode_mean(node: dict) -> float:
    return node["decode"]["igpu_throughput_tok_s"]["mean"]


def main() -> int:
    npu_loss, cpu_loss = [], []
    for run in (1, 2, 3):
        m1 = json.loads((MEAS / f"run{run}.m1_contention.json").read_text())
        m2 = json.loads((MEAS / f"run{run}.m2_cpu_arm.json").read_text())
        base = _decode_mean(m1["baseline"])
        npu_loss.append((base - _decode_mean(m1["concurrent_npu"])) / base * 100)
        cpu_loss.append((base - _decode_mean(m2["concurrent_cpu"])) / base * 100)

    npu_mean = statistics.mean(npu_loss)
    cpu_mean = statistics.mean(cpu_loss)
    print(f"NPU iGPU decode loss: {npu_mean:.2f}%  (runs: {', '.join(f'{x:.2f}' for x in npu_loss)})")
    print(f"CPU iGPU decode loss: {cpu_mean:.2f}%  (runs: {', '.join(f'{x:.2f}' for x in cpu_loss)})")

    ok = (
        abs(npu_mean - PUBLISHED["npu_loss_pct"]) <= TOL
        and abs(cpu_mean - PUBLISHED["cpu_loss_pct"]) <= TOL
    )
    if ok:
        print(f"\nOK — matches published {PUBLISHED['npu_loss_pct']}% / {PUBLISHED['cpu_loss_pct']}% within {TOL}pp.")
        return 0
    print(
        f"\nMISMATCH — published {PUBLISHED['npu_loss_pct']}% / {PUBLISHED['cpu_loss_pct']}% "
        f"not reproduced within {TOL}pp.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

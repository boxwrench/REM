# Strix Halo Placement & Contention Benchmark

> **Canonical result (3 independent runs of N=20, 60 decode samples).** For a
> latency-tolerant background generation job running next to an interactive iGPU
> model: **NPU 3.81% (3.4–4.2%) vs CPU 4.77% (4.2–5.1%) iGPU decode loss** — the NPU
> contends *slightly less* than spare CPU cores — at **~3× background throughput**
> (12.15 ± 1.20 vs 4.11 ± 0.80 tok/s) and **~3× total-board perf/watt** (0.143 vs
> 0.049). An earlier, smaller N=5 run had suggested ~2.8% NPU / near-zero CPU
> contention; that **did not reproduce** at N=20 and is not used. Measured with the
> [xdna-top](https://github.com/boxwrench/xdna-top) monitor; full per-run data lives
> with that project's contention experiment.

**What this is.** Honest, reproducible numbers for deciding *what to run where* on
an AMD Strix Halo APU — XDNA2 NPU + gfx1151 iGPU + CPU sharing one unified
~212 GB/s memory bus. The engines are **compute-isolated but bandwidth-shared**,
so the whole story is memory-bandwidth contention. Built and reproduced with our
own monitor, **xdna-top** (which exists because amd-smi is broken on gfx1151).

**Status:** living document. Methodology is fixed below; result sections are
filled by the benchmark tasks as they land. Numbers are reported as
**mean ± stddev** over N≥5 trials; anything below measurement resolution is
labelled as such rather than published with false precision.

**Audience:** developers building NPU/iGPU/CPU use-cases on this hardware.

**Scope note:** this is a general LLM **generation-speed and concurrency**
benchmark. The workload is plain text generation on each engine; it is not tied
to any particular application.

---

## Test platform
- **SoC / APU**: AMD RYZEN AI MAX+ 395 (Strix Halo)
- **NPU**: RyzenAI-npu5 (gfx1151)
- **iGPU**: Radeon 8060S (gfx1151)
- **OS/Kernel**: Ubuntu 24.04, Linux Kernel 6.17.0-35-generic
- **XRT version**: 2.21.75
- **NPU firmware**: 1.1.2.65
- **FLM/Lemonade version**: FLM v0.9.39
- **Unified memory**: 128 GB LPDDR5x (unified bus bandwidth ~212 GB/s)

## Engines under test
- **NPU** — FLM/Lemonade OpenAI-compatible endpoint (port 13306).
- **iGPU** — llama.cpp (treated as the interactive "main lane").
- **CPU** — llama.cpp on spare cores.

## Model ladder
1B / 3B / 4B. Same quantization across engines where possible; deviations noted
per row.

---

## The metric matrix (what we measure and why it helps)

### A. Single-engine baselines (speed tables)
Per engine, alone: **prefill tok/s** and **decode tok/s** across the model ladder.
The denominator for everything else.

### B. Pairwise contention matrix (the core result)
Who degrades whom, measured in **both directions** and **split by phase** (prefill
vs decode — they contend differently and must never be averaged).

### Contention Results (Decode Slice, pooled over 3×N=20 = 60 samples)
Generation speed and contention for the baseline and concurrent conditions:

| Condition | iGPU Decode Loss % | Background Throughput (tok/s) | Gen-tok/s per Total Board Watt |
| :--- | :---: | :---: | :---: |
| **Baseline** (iGPU only) | 0.00% | — | — |
| **NPU (concurrent)** | 3.81% (3.4–4.2%) | 12.15 ± 1.20 | 0.143 |
| **CPU (concurrent)** | 4.77% (4.2–5.1%) | 4.11 ± 0.80 | 0.049 |

Per-run ranges are shown so run-to-run variance is explicit. Prefill contention is
below measurement resolution (very high stddev) and is not reported. Full per-run
data and the power columns live with the xdna-top contention experiment.

#### Caveats & Methodology Notes
1. **Prefill Resolution**: Prefill throughput measurements carry extremely high variance (~8-12k tok/s standard deviation) due to the fast prefill pass and HTTP network latency. Because the prefill contention signal is smaller than the standard deviation, prefill contention is considered **below the measurement resolution** of this harness. No prefill contention loss % is reported.
2. **Performance/Watt Denominator**: The efficiency metric is calculated as `Background Throughput (tok/s) / Total Board Decode Power (W)`. Board power is read from the package power sysfs node, representing total system cost, not isolated engine efficiency.

### C. Beyond throughput
- **Main-lane latency p50/p99** (inter-token) under contention — a 3% throughput
  loss can still spike tail latency, which is what an interactive user feels.
- **Power & perf/watt** per condition + **marginal watts** of NPU vs CPU offload.
  Note: perf/watt here is generation-tok/s per *total board watt*, not isolated
  engine efficiency.
- **Sustained vs burst** — multi-minute runs to catch thermal throttling (a
  background job may run for minutes).

### D. The unifying lens — memory bandwidth (probe)
Attribute contention to achieved GB/s so the numbers stop being a "3% mystery."
**Feasibility risk:** amd-smi is broken on gfx1151; a clean BW counter may not
exist. Treated as a probe — recorded either way.

---

## Instrumentation: xdna-top
Every run captures xdna-top telemetry as the attribution layer — PID → AIE
partition activity, NPU/iGPU utilization, board power series — proving the work
ran where claimed. xdna-top is also the tool readers install to reproduce. A
machine-readable export backs the published numbers (so they trace to the capture,
not a separate path).

---

## How to interpret this for your use-case
The data confirms a substantial advantage when offloading background generation jobs to the NPU rather than spare CPU cores (pooled over 3×N=20):
- **Throughput Advantage**: The background job on the NPU achieves **12.15 ± 1.20 tok/s**, compared to **4.11 ± 0.80 tok/s** on 4 CPU cores (a **≈3× throughput speedup**).
- **Efficiency Advantage**: The NPU achieves **0.143 tok/s/W** on a total-board power basis, compared to **0.049 tok/s/W** for the CPU control arm (a **≈3× efficiency improvement**).
- **Slightly Lower Contention than CPU**: A concurrent NPU generation job costs **~3.8%** (3.4–4.2%) iGPU decode loss — *slightly less* than the same job on spare CPU cores (**~4.8%**, 4.2–5.1%). Both offloads cost the main lane a real single-digit %; the NPU is the cheaper of the two (NOT free).

## Not yet measured (boundaries)
- Deep tile-level utilization beyond existence-of-activity.
- NPU native-shape role: embedding/retrieval vs autoregressive generation.

---

## Planned extensions
- Single-engine prefill/decode speed tables across a small model ladder (the denominator for everything else).
- Full pairwise contention grid + main-lane inter-token latency p50/p99 under load.
- Power / perf-watt curve, sustained/thermal behavior, and a memory-bandwidth probe.

Reproduce command:
```bash
PYTHONPATH=src python3 evals/contention/run_contention_benchmark.py --trials 5
```

## Reproducibility & honesty rules (per CONTRIBUTING.md)
- Every published number is backed by a committed `bench/*.json` artifact and its
  per-trial samples. No fabricated precision; no "calibrated" unless measured.
- Real hardware only; no mock-as-evidence. Commit-then-run (no dirty-tree evals).
- Numbers carry dispersion (mean ± stddev); sub-resolution signals are labelled.

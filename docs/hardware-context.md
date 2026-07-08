# Hardware context — why the NPU is a real home for memory work

REM's thesis is a *placement* claim: while the big model is awake on the iGPU, the
NPU can quietly own the agent's memory-maintenance work. That claim only makes
sense if you know what the NPU is, how it relates to the CPU and iGPU it shares a
chip with, and which of its behaviors you can actually *see*. This page is that
context — enough to evaluate REM, and enough to start building your own NPU
use-case on the same foundation.

It is deliberately concise. For the full hardware spec/capability reference (every
engine, every readable and unreadable sensor, with sources), see the companion
doc in the telemetry tool REM is measured with:
**[xdna-top → docs/HARDWARE.md](https://github.com/boxwrench/xdna-top/blob/main/docs/HARDWARE.md)**.

---

## The platform in one picture

REM runs on an **AMD Ryzen AI MAX+ 395 ("Strix Halo")** — a *monolithic APU* where
the CPU, iGPU, NPU, and memory controller all sit on one die and share **one pool
of LPDDR5x memory**. There is no separate VRAM. That single fact is why REM's
honest framing matters: the three engines are **compute-isolated but
bandwidth-shared**, so placing work on the NPU is never "free" — it's a question
of *which* engine carries a job most cheaply.

| Block | This chip | What REM uses it for |
| --- | --- | --- |
| **CPU** | 16× Zen 5 | the baseline alternative — "just run compaction on spare cores" |
| **iGPU** | Radeon 8060S, 40 RDNA 3.5 CUs (`gfx1151`) | the **foreground model** — the lane we protect |
| **NPU** | XDNA 2 (`RyzenAI-npu5`), 50 TOPS | the **background memory compactor** — REM's whole point |
| **Memory** | up to 128 GB LPDDR5x-8000, 256-bit bus | shared by all three; the source of all contention |

- **Unified bus:** ~**256 GB/s** theoretical (256-bit × LPDDR5x-8000), **~212–215
  GB/s** measured in practice. (Earlier REM/xdna-top docs cited these as if they
  competed — they don't; one is the ceiling, the other the achieved rate.)
- **This box, exactly:** Ubuntu 24.04, kernel 6.17.0-35-generic, XRT 2.21.75, NPU
  firmware 1.1.2.65, FLM v0.9.39.

---

## What the NPU actually is

The NPU is **XDNA 2** — a *spatial dataflow* engine, not a smaller GPU. Instead of
cores stepping through instructions, a 2D array of **AI Engine tiles (32 tiles,
8 columns × 4 rows on this chip)** is configured into a pipeline that data streams
through. AMD rates it at **50 peak TOPS**, and unusually at that figure in **both
INT8 and Block FP16**.

What this means for a builder: the NPU is **not** where you'd run your largest,
latency-critical model — the iGPU has far more raw throughput. The NPU's edge is
**perf/watt** and **occupying a different execution resource than the iGPU**. That
makes it an excellent home for work that is (a) latency-tolerant, (b) runs in the
background, and (c) you'd rather not steal iGPU cycles for. Memory compaction is
exactly that shape — which is the bet REM is testing.

---

## How the engines compare (REM's lens)

| | CPU (Zen 5) | iGPU (RDNA 3.5) | NPU (XDNA 2) |
| --- | --- | --- | --- |
| Role in REM | control arm | foreground model | memory compactor |
| Background-gen throughput\* | 4.11 ± 0.80 tok/s | (the main lane) | 12.15 ± 1.20 tok/s |
| Total-board perf/watt\* | 0.049 tok/s/W | — | 0.143 tok/s/W |
| Cost to the main lane\* | 4.77% decode loss | — | 3.81% decode loss |

\* Pooled over 3 runs × N=20 = 60 decode samples on this box. The NPU does **~3×
the background throughput at ~3× the total-board perf/watt** of spare CPU cores,
while contending **slightly less** with the iGPU. Full methodology, caveats, and
committed raw artifacts: [npu-placement-benchmark.md](npu-placement-benchmark.md)
and [`bench/contention/`](../bench/contention/). The headline reproduces from
committed data with `python bench/contention/verify_contention.py` — no hardware
needed.

The takeaway is not "the NPU is fastest." It's "for a latency-tolerant background
job sitting next to an interactive iGPU model, the NPU is the cheapest *and* most
efficient place to put it." REM is one instance of that pattern; yours could be
another.

---

## What you can — and can't — see the NPU doing

Building on the NPU means being honest about its observability, because it's the
most opaque engine on the chip. On this stack:

- ✅ **Per-context activity** — which PID owns an NPU hardware context, and its
  submission/completion counters over time. This is how REM *proves* compaction
  ran on the NPU rather than asserting it.
- ❌ **Isolated NPU power** — not exposed; only whole-SoC package power (PPT) is
  readable, which is why REM reports *total-board* perf/watt and never claims a
  marginal-watt number.
- ❌ **NPU temperature / memory bandwidth** — no sensor on this stack.

REM captures the per-context evidence with **[xdna-top](https://github.com/boxwrench/xdna-top)**
(`snapshot` / `record` / `assert`), the same monitor that produced the telemetry
behind every number above. If you're prototyping your own NPU workload, that tool
is the fastest way to answer the first question you'll have — *"is the NPU actually
executing my job?"* — and to turn the answer into reproducible evidence.

---

## Build on this

- **Watch the NPU live / capture evidence:** [xdna-top](https://github.com/boxwrench/xdna-top).
- **Full hardware reference:** [xdna-top → docs/HARDWARE.md](https://github.com/boxwrench/xdna-top/blob/main/docs/HARDWARE.md).
- **REM's placement methodology and open problems:** [npu-placement-benchmark.md](npu-placement-benchmark.md)
  and the [README](../README.md)'s "what's demonstrated vs. open" section.

If REM's pattern fits a workload you have — anything latency-tolerant you'd rather
keep off the iGPU — this hardware will likely carry it well. The telemetry to
prove it already exists.

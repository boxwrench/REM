# Paper Evaluation: Striking the Balance

Date: 2026-06-17

Paper: "Striking the Balance: GEMM Performance Optimization Across Generations
of Ryzen AI NPUs"

Source: local PDF `2512.13282v1.pdf`; arXiv metadata:
https://arxiv.org/abs/2512.13282

## Summary

This paper is directly useful for REM, but not because it gives a headline TOPS
number to reuse. Its real value is the systems framing: useful NPU performance
comes from finding a balanced operating point between compute, explicit data
movement, memory layout, DMA scheduling, and shared DRAM bandwidth.

That is also REM's central question at the application layer. REM is asking
whether a background memory-maintenance workload can run on the XDNA2 NPU while
the iGPU serves the foreground model, with the shared-memory penalty kept inside
a measured contention budget.

The paper strengthens REM's current framing as a placement and scheduling
feasibility study. It also raises the bar for reproducibility: the repo should
make raw contention, power, and eventually bandwidth evidence easy to audit.

## What The Paper Shows

The authors optimize GEMM on AMD Ryzen AI NPUs across XDNA and XDNA2. Their main
technical result is not simply that XDNA2 is faster. It is that the fastest
kernel is found at a balance point:

- If tile choices favor compute too aggressively, the NPU can become memory
  bound.
- If tile choices favor memory too aggressively, compute efficiency falls.
- The best point is where compute time and off-chip memory time are balanced.

The paper backs this with analytical modeling plus hardware profiling. It also
shows that XDNA2's higher peak compute makes effective DRAM bandwidth and layout
more important, not less important.

Important implementation details:

- XDNA/XDNA2 use explicit data movement through DMA engines.
- Local L1/L2 memories and multi-dimensional DMA addressing are central.
- Matrix layout matters. Column-major access for one GEMM operand improves
  contiguous access and can materially change performance.
- Overlapping DMA transfers with buffer-descriptor reconfiguration matters.
- Reconfiguring whole NPU designs across GEMM shapes can cost milliseconds, so
  stable reusable parameters matter.

Reported top GEMM sweep values include up to 6.76 TOPS int8 on XDNA, 38.05 TOPS
int8 on XDNA2, 3.14 TOPS bf16 on XDNA, and 14.71 TOPS bf16 on XDNA2. These are
close-to-metal GEMM results and should not be treated as end-to-end REM
compaction throughput.

## Lens For REM

REM is the application-level version of the paper's "balanced point" problem.

For REM, the balance point is not `m_ct`, `k_ct`, and `n_ct`. It is:

- NPU compaction drain rate exceeds context arrival rate.
- iGPU foreground decode loss remains inside a measured budget.
- Shared DRAM pressure does not create unacceptable latency tails.
- Board power and thermals remain stable during sustained background work.
- Memory quality remains acceptable after compression and supersession.

This makes REM's strongest claim:

> The NPU is not free compute. It is a better placement for latency-tolerant
> memory maintenance when its service rate, power cost, and shared-memory
> contention sit at a stable operating point.

That wording is more defensible than "spare NPU capacity" because it matches the
paper's actual lesson.

## What This Strengthens In The Repo

The README's "not contention-free" framing is correct and should stay. The paper
supports the point that XDNA2 is compute-isolated but not memory-isolated.

The current headline contention result also has the right shape:

- NPU background job: about 3.81 percent iGPU decode loss.
- CPU background job: about 4.77 percent iGPU decode loss.
- NPU background throughput: about 3x CPU control throughput.
- NPU total-board perf/watt: about 3x CPU control.

Through this paper's lens, that is not a claim that the NPU avoids contention.
It is a claim that the NPU reaches a better application-level balance point than
the CPU control for this latency-tolerant background workload.

The compaction throughput probe also maps well to the paper. REM's span-size
sweep is analogous to a tiling sweep: span size changes how much useful context
is absorbed per compaction call, how much output is produced, and how much fixed
generation overhead is amortized.

## Claims To Avoid

Avoid using XDNA2 peak TOPS as direct evidence for REM. The paper itself shows
that realized performance depends on data layout, arithmetic intensity, DMA
scheduling, and effective DRAM bandwidth.

Avoid implying REM is doing custom kernel optimization. Today REM uses the
FLM/Lemonade NPU model-serving stack. That makes REM an end-to-end placement
study, not a close-to-metal GEMM paper.

Avoid saying the NPU is bandwidth-isolated from the iGPU. The paper reinforces
the opposite: XDNA2 performance is sensitive to the SoC fabric and DRAM path.

Avoid treating per-call compaction latency as a foreground latency win. REM wins
only when async background service keeps up and does not overflow the context
budget.

## Experiments Suggested By The Paper

1. Add a REM balance sweep.

   Sweep compaction span size, output token cap, and budget. Report drain rate,
   output size, compression ratio, iGPU decode loss, power, and quality. The goal
   is to show where REM's operating point saturates or regresses.

2. Add bandwidth or best-available memory-pressure telemetry.

   The paper treats effective DRAM bandwidth as decisive. If direct counters are
   unavailable on gfx1151, document that and capture the closest proxies:
   iGPU decode loss, board power, NPU activity, and process attribution.

3. Split prefill and decode impacts where possible.

   The repo already notes prefill variance in the contention harness. Keep
   decode separate from prefill, and do not average them into one number.

4. Measure output-cap sensitivity.

   In the paper, output movement is amortized by the reduction dimension. In REM,
   summary/fact output length is the analogous pressure point. If the output cap
   is too high, the compactor may waste NPU service time; if too low, quality may
   fail.

5. Treat structured extraction reliability as part of the balance point.

   A faster compactor that drops facts due to malformed JSON is not a stable
   operating point. The sampling sweep and repair/retry path belong in the same
   evidence chain as throughput.

6. Add tail latency for the foreground lane.

   A small average decode loss can still hurt inter-token p99. The paper's
   system-level approach argues for measuring the user-visible tail, not only
   mean throughput.

## Suggested Repo Framing

Short version:

> REM searches for a stable background-memory operating point on Strix Halo:
> the NPU compactor must drain context faster than the agent produces it, while
> shared-memory pressure keeps foreground iGPU decode loss inside a measured
> budget.

Longer version:

> This is not a claim that the NPU is contention-free. XDNA2 shares the SoC
> memory system with the iGPU and CPU. REM evaluates whether memory-maintenance
> work has a better system-level balance on the NPU than on spare CPU cores:
> higher background service rate, lower total-board energy per generated token,
> bounded iGPU decode loss, and stable sustained thermals.

## Repo Implications

The paper does not invalidate any core REM claim found in the review. It does
make the raw-evidence gaps more important:

- Commit or immutably link the canonical contention artifacts.
- Make the reproduction command match the published 3xN=20 result.
- Commit the retention/quality artifact or soften the retention claim.
- Keep power and thermal artifacts tied to each headline number.
- Add bandwidth or memory-pressure telemetry if the platform exposes it.

The cleanest next move is to add a "balanced operating point" section to the
benchmark doc after the missing raw numbers are committed. That section can
connect the paper's hardware theory to REM's measured application behavior
without overstating what the current implementation controls.

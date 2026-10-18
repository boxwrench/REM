# The Strix Halo XDNA2 NPU: a measured map

What the NPU on a Ryzen AI MAX+ 395 actually does for LLM work on Linux, measured
on one box. This page collects the device envelope (rate by shape, compiler gap,
per-call cost, numerical error), contention in both directions, the settings that
keep a GPU+NPU run from hanging, and a plain "good for / not good for" table.

Every number is measured unless a row says otherwise. The conditions matter: an
NPU rate is only meaningful with its shape, data format, toolchain, and safety
state. Where a conclusion is reasoning rather than a measurement, it is marked
**[reasoning]**.

Companion pages: [keeping the NPU without slowing the iGPU](strix-halo-iommu-npu.md),
[which models can use the NPU for prefill](npu-prefill-offload-fit.md), and the
[placement benchmark](npu-placement-benchmark.md). The same envelope from the
monitoring side, including what telemetry can and cannot see, is in
[xdna-top's Strix Halo NPU envelope](https://github.com/boxwrench/xdna-top/blob/main/docs/experiments/strix-halo-npu-envelope.md).

---

## The box

| | |
|---|---|
| SoC | AMD Ryzen AI MAX+ 395 (Strix Halo), 128 GB LPDDR5x |
| iGPU | Radeon 8060S, `gfx1151`, 40 CUs |
| NPU | XDNA 2 / NPU5 `RyzenAI-npu5`, aie2p 6x8, firmware 1.1.2.65 |
| Software | XRT 2.25, `amdxdna` 7.0.0-rc1, ROCm 7.2.2, kernel `7.3.0-rc6-perfopt+`, `iommu=pt` |
| Data format | bfp8 (block floating point, 8-bit), 128x64x128 tiles |
| Safety state | FCLK held at 2000 MHz, NPU pmode `performance`, NPU `power/control=on` |

The NPU has no native BF16 or INT8 kernel in the fixtures we tested; bfp8 is the
working format, and it is approximate. All GPU rates below are BF16 unless stated.

## Rate vs shape

The array wants rows. At K=2560, N=1024, bfp8, the NPU is useful from about
128 rows up, and its rate climbs steeply with M:

| M | NPU ms | NPU TFLOPS | GPU BF16 ms | GPU TFLOPS | Note |
|---:|---:|---:|---:|---:|---|
| 128 | 0.211 | 3.2 | 0.063 | 10.7 | smallest buildable tile |
| 256 | 0.216 | 6.2 | 0.062 | 21.5 | same physical tile as 128 |
| 512 | 0.225 | 11.9 | 0.121 | 22.2 | |
| 4096 | 0.811 | 26.5 | 0.524 | 41.0 | the shape the array is built for |

Rows 128–512 are the 8-core small-M kernel; the M=4096 cell is the 32-core shape.
On that 8-core kernel, a request for fewer than 128 rows still executes as 128; a block of 8
therefore gets about 0.2 useful TFLOPS on the NPU while the GPU gets 0.64. Across
matched dense GEMMs the NPU is **1.55–5.63× slower than this iGPU**, and
**3.2–3.5× slower at a drafter's block shape**.

At the dense 27B FFN shapes, the open-toolchain kernel measures 17.3–19.3 TFLOPS
(M=512/2048/4096, K=5120 or 17408, N=1024–4096). The shipped kernel reaches
26.5–26.8 TFLOPS but only at M=4096, K=2560 (it can K-chunk to cover larger K).
This is not a hardware ceiling: it is the state of the installed kernels.

### Shipped vs open toolchain

| Kernel | Rate | Condition |
|---|---:|---|
| Mike Alani's shipped bfp8 GEMM | 26.5–27.7 TFLOPS | M=4096, K=2560, verify PASS |
| Rebuilt with mlir-aie 1.4.4 + Peano 22 | 17.3–19.6 TFLOPS | same topology, K=2560–8192 |
| Older Peano pin / default lowering | 14.7–16.9 TFLOPS | same session, worse |

The ~35% deficit lives in the core program, not the host wrapper: the shipped
`xclbin` stores its AIE program as a CDO-encoded PDI with no compiler banner, and
the open Peano toolchain we tested tops out near 18 TFLOPS. The shipped binary is
consistent with a proprietary compiler. **We did not close the gap with the open
toolchain**, so treat 18–19 TFLOPS as the practical open-build rate and 27 as what
the vendor-side build achieves.

### Per-call cost

- Host A-encode plus C-decode through XRT: **~0.86 ms per call** (GLM 32K q/k/v shapes). This
  is what kills one-call-per-GEMM designs on that path; calls must be batched per layer chunk.
- Through HRX (AMD FastFlow-SDK) the dispatch cost drops by an order of magnitude: **66 µs** for a
  single dispatch and **~18 µs** per dispatch when chained (2026-10-18).
- Switching between two xclbins costs **~2–3 ms**; two instruction streams on the same xclbin
  alternate for free, so a whole model can run from one resident program.
- A compact return (147 KB) from NPU-owned memory: **8.5 µs**.
- Warm launch: 2–60 µs; first call after load: 0.87–3.7 ms.
- Effective B (weight) streaming in the small-M fixture: 6–22 GB/s. There is no
  resident-weight path installed; B is re-read each call.

### Numerical error

bfp8 with BF16 accumulation, against an FP32 reference, max relative error:

| K | Max relative error |
|---:|---:|
| 2560 | 1.0% |
| 4096 | 1.2% |
| 8192 | 2.6% |

The error grows with K. Any engine integration needs a quality gate, not just a
speed check. On a full dense 27B model the measured perplexity was unchanged
within noise (6.191 with the NPU slice vs 6.218 without, 16×2048 wikitext-2).

## Contention, both directions

The NPU and iGPU share one LPDDR5x bus. Contention was measured with both engines
busy at 0/50/100% NPU duty:

| Workload | iGPU effect | NPU effect |
|---|---:|---:|
| 321B MoE prefill, 32K, 50% / 100% NPU duty | −3.7% / −10.1% | 20.5 / 25.8 TFLOPS (28.8 alone) |
| Dense 27B prefill, 4K and 32K, 50–100% duty | −1.0 to −3.6% | −13 to −28% |
| Decode under 100% small-M NPU load | −4.8% | — |
| Decode under 100% large-M NPU load | −16.4% | — |

The dense-27B GPU loss is real but small (per-unit-duty p=0.024–0.037); the NPU
loses more when the GPU is busy. Package power stayed flat in these runs, so the
loss is bandwidth and scheduling, not a power cap. Peak NPU rate alone is not the
rate you get under load: plan with the loaded column.

## Safety envelope

### IOMMU tax, and PerfOpt recovery

The NPU needs the IOMMU (SVA/PASID); with `amd_iommu=off` the device fails to open
with `ENODEV`. On a stock 6.17 kernel, `iommu=pt` costs iGPU prefill:

| GPU prefill | `amd_iommu=off` | `iommu=pt` | `iommu=pt` + PerfOpt |
|---|---:|---:|---:|
| ~32K prompt | 387.0 tok/s | 356.4 tok/s (−7.9%) | **390.3 tok/s** |
| ~64K prompt | 377.3 tok/s | 350.0 tok/s (−7.3%) | **378.3 tok/s** |
| decode | 54.11 ms/tok | 54.43 ms/tok | 54.19 ms/tok |

PerfOpt (Mario Limonciello, AMD; queued for Linux 7.4, on by default) switches the
iGPU's ATS/PRI/PASID off after init while its IOMMU domain is identity-mapped. With
it, the NPU is available at effectively no iGPU cost. The test kernel also changed
the kernel version, so the comparison is PerfOpt plus a newer kernel; both are
reported rather than separated.

### The hold

Before any concurrent GPU+NPU run:

```sh
echo manual | sudo tee /sys/class/drm/card1/device/power_dpm_force_performance_level
echo 7      | sudo tee /sys/class/drm/card1/device/pp_dpm_fclk     # 2000 MHz here
sudo xrt-smi configure --pmode performance                          # never turbo
echo on     | sudo tee /sys/bus/pci/devices/0000:c6:00.1/power/control
```

All four reset at reboot. [`scripts/npu-hold/`](../scripts/npu-hold/) applies them
at boot and after resume. A fixed top FCLK also gives the GPU **+1.6–2.2%** prefill
on its own at no measurable power cost.

Without the hold, concurrent GPU+NPU load is reported to hard-hang Strix Halo
([amd/xdna-driver#1605](https://github.com/amd/xdna-driver/issues/1605)). With the
hold, our runs had **0 machine checks, 0 `amdxdna` timeouts, 0 GPU resets** across
a 71-minute soak (NPU at 100% duty for 64.5 of them), about an hour of dense
offload overlap, and the contention runs. That is a data point, not a proof of safety: a hang is still possible, and
FCLK *switching* may be the actual trigger (untested).

Two habits from the field: leave the NPU AXI QoS at the vendor default 0, and if a
copy fault appears, retry with `HSA_ENABLE_SDMA=0` before debugging
([gufo#569](https://github.com/gufo-org/gufo/pull/569)).

## What it is good for

| Job | Verdict | What we measured |
|---|---|---|
| Decode / draft heads | **No** | A full draft head on the NPU is ~21.8 ms/step (M=128 floor, K=4096, 8-core bfp8) against ~1.0 ms for the same Q4_K head on the iGPU. Memory-bound work, and the NPU has no bandwidth advantage. |
| Block drafters | **No** | 1.55–5.63× slower than the iGPU on matched GEMMs, 3.2–3.5× at a block shape. Weight-retention tricks reached 1.17–1.48×, below the 2× gate. A cost model leaves only a 50M INT8 / k=2 student at +2.4%, under the ≥5% gate **[reasoning, not run end-to-end]**. |
| Large-MoE prefill | **No** | Offloading the last two 1024-wide N blocks (25% of N) of each GLM q/k/v projection moved end-to-end prefill by +0.1% at 32K and +1.7% at 64K. The offloadable share is too small and the experts are small and variable. |
| Sparse-attention indexer | **No** | 2.3% / 4.3% / 7.9% of prefill at 32K / 64K / 120K, and 2.8–7.4% of long-context decode. Under the 10% gate at every phase. |
| Dense prefill, synchronous split | **No** | A dense 27B hybrid model in a llama.cpp-based fork, FFN split by intermediate channels: output correct (PPL 6.191 vs 6.218), speed **−6.2% / −7.6% / −6.2%** at 4K / 8K / 32K. Only ~43% of NPU time was hidden; the GPU needed the offloaded half immediately. |
| Dense prefill, deferred consumers | **Open** | Not yet measured here. The idea is to offload rows the GPU consumes later. gufo measured **+7.8–9.6%** prefill on Qwen3.8 Flash-Next in their engine with this design; our port was started and not finished. This is the only prefill lead with measured upside elsewhere. |
| A second, latency-tolerant job beside the iGPU | **Yes** | On a 6.17/XRT 2.21.75 stack, a background generation job did ~3× the throughput of 4 CPU cores at ~3× total-board perf/watt, with 3.8% iGPU decode loss vs 4.8% for the CPU arm. See the [placement benchmark](npu-placement-benchmark.md). |
| Whole-model background decode (MoE writer) | **Yes, runs** | Hy-Memory-30B-A3B with every linear layer on the NPU as a fused 4-bit GEMV in one resident program: **9.4 tok/s** decode, greedy output matching llama.cpp Q4_K_M on the check prompt ([NPU Wrench](https://github.com/boxwrench/npu-wrench), 2026-10-18). Weight streaming is data-movement bound at ~22 GB/s. iGPU contention not yet measured. |
| Encoders / side models (STT, embeddings) | **Untested** | Desk analysis only; never run beside the main model on this box. No measurement either way. |

## What this does not establish

- These are installed-toolchain rates, not XDNA2's hardware ceiling. A better
  compiler or a native INT8/BF16 kernel could move the envelope.
- One box, one software stack. Contention and safety results may differ on another
  board, kernel, or firmware.
- The NPU's own power is not observable (`xrt-smi` reports N/A; no sensor). All
  power statements are package-level.
- No side-model or encoder job was run. The "good for" row is untested.
- The deferred-consumer result is gufo's, on their engine and model, not ours.

## Credits

- **Mike Alani** ([mikealanni/gufo](https://github.com/mikealanni/gufo), upstreaming
  as [gufo-org/gufo#568](https://github.com/gufo-org/gufo/pull/568)) wrote the bfp8
  GEMM kernel and the prefill-offload design this work follows, and documented the
  fabric-clock hold.
- **Mario Limonciello** (AMD) wrote the IOMMU PerfOpt series.
- **hipfire** ([warpfront/hipfire](https://github.com/warpfront/hipfire)) published
  the safety rules (AXI QoS, per-run clock check, bounded memlock) that the holds
  here follow.
- The reporters in
  [amd/xdna-driver#1605](https://github.com/amd/xdna-driver/issues/1605) documented
  the GPU+NPU hang.

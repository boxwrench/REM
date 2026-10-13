# Strix Halo: which models can use the NPU for prefill

We tried to speed up prompt processing (prefill) for a large model by giving part
of its matrix math to the NPU while the iGPU did the rest. The mechanism worked,
but the model was the wrong kind. This page records what we measured and, from
that, **which models are likely to benefit**. The second half is reasoning, not
measurement, and is marked as such.

Same box and settings as [strix-halo-iommu-npu.md](strix-halo-iommu-npu.md):
Ryzen AI MAX+ 395, 128 GB, gfx1151 iGPU + XDNA 2 NPU, Linux 7.3-rc6 with the AMD
IOMMU PerfOpt series, `iommu=pt`, FCLK held at the top level.

---

## 1. What we measured

The test model was GLM-5.3-Flash (321B mixture-of-experts) in
[Maya](https://github.com/mw00/project-maya). The NPU took the last 1–2 of the
eight 1024-row blocks of each attention q/k/v projection (8192×4096). It used an
open-toolchain bfp8 GEMM kernel (8-bit block floating point), zero-copy pinned
buffers, and up to 4 submits in flight. The design follows Mike Alani's gufo
offload.

| | Result |
|---|---|
| NPU GEMM rate, alone (async ×4) | ~28.8 TFLOPS |
| iGPU on the same full shape (tuned hipBLASLt, T=8192) | 21.4 ms ≈ 25.7 TFLOPS |
| iGPU slowdown while the NPU runs (shared memory) | −3.7% at ~50% NPU duty, −10.1% at 100% |
| Encode/decode cost per NPU call | ~0.9 ms |
| GPU time spent waiting for the NPU | ~0 (fully hidden) |
| Accuracy | bfp8 level (rel. L2 ~1.2e-2); generated text coherent |
| **Net prefill change, best split** | **+0.1% at 32K, +1.7% at 64K tokens** |

Two practical notes from the run:

- **Re-tune the GPU for the split shapes.** Once part of a GEMM moves to the NPU,
  the iGPU's share has a new size. If that size isn't in the hipBLASLt tuning
  table, the GPU falls back to an untuned path that is ~5× slower. Our first run
  lost 25–28% of prefill speed to this alone.
- **Contention, not the NPU, sets the limit.** Offloading 25% of q/k/v cut the
  GPU's GEMM time by about 2.1 s of an 86 s prefill (~2.5%). Memory contention and
  the per-call copies took most of that back.

**Why this model doesn't benefit:** the dense projections the NPU can take are
only ~18% of prefill time. The rest is spread over many small, variable-size
expert GEMMs and attention.

## 2. Which models are likely to benefit (speculation)

The deciding factor is **how much of prefill is spent in large, fixed-shape dense
GEMMs**. On this box the NPU and the iGPU are roughly equal on that work, so
the NPU can take a large share of it. Each kernel is compiled for one shape,
and bfp8 is approximate, so every candidate still needs a quality check.

| Likely to fit | Why | Rough guess |
|---|---|---|
| Dense transformers, ~4B–32B | Prefill is mostly large fixed-shape GEMMs; the NPU could take ~40% of them | +10–25% prefill at long prompts after contention |
| Small hidden size (e.g. 2560) | The iGPU is least efficient on small GEMMs, so the NPU is relatively stronger | Probably the best ratio |
| Encoders: embeddings, rerankers, Whisper, vision towers | Pure GEMM work, fixed shapes, no token generation | Good fit, especially as a separate job |
| A separate model next to the main one | No splitting, no waiting: the only cost is contention (REM measured −3.8% iGPU decode, [npu-placement-benchmark.md](npu-placement-benchmark.md)) | The most practical use today |

| Unlikely to fit | Why |
|---|---|
| Large mixture-of-experts models | Small dense-GEMM share; experts are small and variable (section 1) |
| Token generation (decode), including draft models | Memory-bound; the iGPU wins easily. A whole draft head on the NPU measured ~21.8 ms per step, against ~1–2.3 ms on the iGPU |
| Short prompts | The fixed per-call cost dominates |

The +10–25% figure assumes ~70% of prefill is offloadable GEMM, the NPU holds
~0.8× the iGPU's rate under contention, and every projection is offloaded. None
of that is measured yet.

**Rule of thumb before building anything:** profile the model's prefill on the
iGPU alone. If large dense GEMMs are more than ~60% of it, a one-GEMM prototype is
worth building. Below ~30%, skip it.

## What's next

- Profile one dense model's prefill on the iGPU to test the rule above.
- Measure a separate encoder job (speech-to-text or embeddings) on the NPU next to
  a model on the iGPU.
- Mike Alani's Qwen kernel (hidden size 2560) already fits the "small hidden size"
  row; porting that path is the main prefill-offload target.

## Credits

- **Mike Alani** ([mikealanni/gufo](https://github.com/mikealanni/gufo); upstreaming
  as [gufo-org/gufo#568](https://github.com/gufo-org/gufo/pull/568)) designed the
  NPU prefill offload this test follows: pinned shared buffers, async submits, and
  the fabric-clock hold.
- **Mario Limonciello** (AMD) wrote PerfOpt, which lets the NPU run without slowing
  the iGPU.

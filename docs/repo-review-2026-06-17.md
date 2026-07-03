# Repo Review - 2026-06-17

Scope: review of repo claims, readability, reproducibility, supporting artifacts,
and implementation risks. This review did not modify code paths.

## Summary

The repo is readable and the main framing is unusually honest for a feasibility
study: it clearly says the NPU placement is not contention-free, the quality work
is unfinished, and the project is pre-1.0. The strongest locally supported claims
are compaction throughput and thermal behavior. The weakest part is evidence
traceability for the headline contention table: the numbers are repeated across
README, docs, glossary, and the report page, but the raw artifacts for that
canonical result are not committed here.

I would not call the main placement result false based on this pass, but I would
not ship it as fully reproducible from this repository yet.

## Findings

### High: headline contention numbers are not self-auditable here

`docs/npu-placement-benchmark.md` publishes the canonical 3 x N=20 result:

- NPU: 3.81% iGPU decode loss, 12.15 +/- 1.20 tok/s, 0.143 tok/s/W
- CPU: 4.77% iGPU decode loss, 4.11 +/- 0.80 tok/s, 0.049 tok/s/W

But the same doc says the full per-run data lives with the `xdna-top`
contention experiment, not in this repo. Meanwhile its reproducibility rules say
every published number should be backed by a committed `bench/*.json` artifact.

Observed locally: no `bench/m1_contention.json`, `bench/m2_cpu_arm.json`, or
`bench/RESULTS.md` is committed.

Recommended fix: either commit the raw contention artifacts here, or link to a
specific immutable commit/file in `xdna-top` and state that the contention
evidence is external.

### High: the advertised reproduction command does not reproduce the canonical result

`README.md` shows:

```bash
PYTHONPATH=src python3 evals/contention/run_contention_benchmark.py --trials 5
```

The published result is 3 independent runs of N=20. The command also omits
required setup for the iGPU llama.cpp server and the CPU control arm. The script
requires `--cpu-model-path` and `--llama-server-path` when `--skip-cpu` is not
used.

Recommended fix: add a "Reproduce canonical contention result" section with:

- exact iGPU server launch command
- exact NPU FLM/Lemonade setup
- exact CPU GGUF and `llama-server` command/path
- exact `--trials 20` command
- how to repeat it 3 times and aggregate
- expected output filenames

### High: retention 1.0 is claimed but not backed by a committed result artifact

`README.md` and `index.html` say buried facts survive compaction with retention
1.0 at roughly 10x compression. The committed `throughput_probe.json` supports
throughput and compression, not judged evidence retention.

Recommended fix: commit a battery result JSON from `run_battery_spike.py`, or
soften the claim to say the throughput probe reached 9.9x compression and the
quality battery is still pending / budget-conditional.

### Medium: dependency and install instructions are incomplete

`src/rem/scheduler/gauge.py` imports `xdna_top.gauge`, but `xdna-top` is not in
`pyproject.toml`. This machine has `xdna-top` installed editable from a sibling
repo, so local tests can mask the missing dependency.

The battery judge requires `anthropic`, and `evals/battery/README.md` tells the
reader to install it manually. The top-level README only says:

```bash
pip install -e ".[dev]"
```

That is not enough for the full advertised battery.

Recommended fix: add extras such as `.[dev,eval]`, declare `anthropic` in the
eval extra, and document `xdna-top` installation or make the scheduler import
degrade gracefully when it is absent.

### Medium: LongMemEval link is stale

The docs point to `xiaowu0162/longmemeval`. Hugging Face now marks that dataset
as deprecated and replaced by `xiaowu0162/longmemeval-cleaned`.

Recommended fix: either pin the exact original revision used for the committed
results, or update the instructions to use `longmemeval-cleaned` and note that
old benchmark numbers used the deprecated source.

### Medium: thermal wording is inconsistent

Local artifact check:

- `bench/battery/thermal_trace.csv`: 2,682 samples, 192.2 minutes, mean 40.89 C,
  max 61.0 C
- `bench/battery/thermal_trace_b2000.csv`: 1,392 samples, 116.2 minutes, mean
  41.34 C, max 62.0 C

`README.md` says "~2.5 h" and max 62 C. `index.html` is more precise: it says
the 192-minute run peaked at 61 C and a separate 116-minute run peaked at 62 C.

Recommended fix: make README match the HTML wording.

### Medium: sidecar state writes can race background compaction

The background compactor uses a `FileLock`, but the sidecar request/response path
saves state without that same lock. A foreground request or response can save
while a background compaction is holding an older in-memory state, and the later
background save can overwrite newer turns.

Relevant files:

- `src/rem/memory/sidecar.py`
- `src/rem/memory/compactor.py`

Recommended fix: centralize state load/mutate/save behind the same lock, or make
the compactor reload/merge just before saving.

### Low: README references an untracked asset

The modified README references `docs/assets/headline.svg`, but `docs/assets/`
is currently untracked. If README is committed without that asset, the top image
will break.

Recommended fix: `git add docs/assets/headline.svg` with the README change, or
remove the reference before committing.

### Low: teaching/glossary docs have stale or missing references

Several docs point to paths not present in this repo:

- `research/toolchain-npu-stack.md`
- `research/tooling-gap-scan.md`
- `research/sources.md`
- `research/papers/`
- `paths/A-compaction-channel/`
- `paths/B-wiki-memory/`
- `paths/C-scheduler-substrate/`
- `docs/superpowers/specs/2026-06-16-rem-standardized-battery-spike-design.md`

`TEACHING.md` also still cites older narrative numbers, including 132.4 tok/s
and a 4.41x keep-up rate, while the current committed probe reports 74.86 tok/s.

Recommended fix: mark `TEACHING.md` as historical, update it, or move missing
internal-only references out of public docs.

## What is well supported

- Default tests pass outside the sandbox: 93 passed, 1 deselected.
- `bench/battery/throughput_probe.json` supports the throughput story:
  - drain rate: 74.86 tok/s
  - median compaction call: 18.605 s
  - compression ratio: 9.9x
  - no fallback compactions in that run
- `bench/battery/sweep/*.json` mostly supports the "~73 tok/s plateau" wording.
  Span 4 is lower at 61.86 tok/s, so the plateau claim should stay scoped to
  the observed span range.
- `bench/battery/thermal_trace.csv` supports the 192-minute thermal story:
  mean 40.89 C, max 61.0 C, no obvious upward drift.
- The README's "not contention-free" and "not a latency-reduction trick"
  framing is good and should stay.

## Verification Performed

Commands run:

```bash
python3 -m pytest
PYTHONPATH=src python3 evals/contention/run_contention_benchmark.py --dry-run --output-dir /tmp/rem-review-contention
PYTHONPATH=.:src python3 evals/battery/throughput_probe.py --help
PYTHONPATH=.:src python3 evals/battery/run_battery_spike.py --help
```

Results:

- `python3 -m pytest`: 93 passed, 1 deselected, 1 warning.
- Dry-run contention script completed and wrote simulated output.
- Battery/probe CLI help works.
- `ruff` could not be run here because it was not installed in the current
  environment, despite being listed in the dev extra.

External checks:

- Parallel Context Compaction paper exists: https://arxiv.org/abs/2605.23296
- Original LongMemEval dataset is deprecated:
  https://huggingface.co/datasets/xiaowu0162/longmemeval
- Replacement dataset:
  https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned

## Suggested Priority Order

See also [`docs/implementation-roadmap.md`](implementation-roadmap.md), which
promotes the tight-budget LongMemEval + JSON robustness work to the current
implementation gate.

1. Commit or link immutable raw contention evidence.
2. Fix the canonical reproduction instructions.
3. Commit/qualify the retention-1.0 evidence.
4. Add an eval extra and document `xdna-top`.
5. Align README/HTML/TEACHING numbers.
6. Fix sidecar state locking before treating the sidecar as production-ready.

## Advice Taken Together

The repo review and the Ryzen AI NPU GEMM paper point in the same direction:
REM's strongest story is not "free NPU compute." It is a measured placement
story: a latency-tolerant memory-maintenance workload can run on the NPU when
its drain rate, shared-memory contention, power draw, thermal behavior, and
quality all land inside a stable operating range.

### Now: make the current claim auditable

Do not broaden the project yet. The immediate win is to make the existing
feasibility claim self-auditing.

Recommended now:

1. Commit or immutably link the raw 3xN=20 contention artifacts.
2. Update the reproduction command so it actually reproduces the canonical
   result, including iGPU server, NPU endpoint, CPU control, trial count, and
   expected output files.
3. Either commit the retention/quality artifact or soften the retention-1.0
   language until that artifact is present.
4. Add `docs/paper-evaluation-2512.13282v1.md` to the evidence trail and link it
   from the benchmark doc once the raw numbers are in.
5. Keep the README's "not contention-free" language. It is correct and becomes
   stronger under the paper's memory-bandwidth lens.

The near-term public posture should be: "this is a real feasibility result with
clear caveats," not "this is a finished memory system."

### Short Term: turn the benchmark into a balance sweep

After the missing artifacts are in, the next step is to show REM's operating
range rather than only one headline point.

Recommended short-term work:

1. Add a balanced operating point table to `docs/npu-placement-benchmark.md`:
   context arrival rate, NPU drain rate, iGPU decode loss, board power, thermal
   max/mean, compression ratio, and quality/retention.
2. Sweep compaction span size, budget, and output token cap. Treat this like the
   paper's tile-size sweep: the goal is to find where service rate improves,
   saturates, or regresses.
3. Add foreground inter-token p50/p99 under NPU and CPU background load. Mean
   decode loss is useful, but tail latency is what an interactive user feels.
4. Make structured extraction reliability part of the same evidence chain. A
   fast compactor that drops facts because JSON failed is not a stable operating
   point.
5. Fix the sidecar state-locking race before presenting the sidecar as anything
   more than experimental.

This phase should produce a defensible claim like: "on this Strix Halo system,
REM is stable for agent context growth up to X tok/s under Y output cap, costing
Z percent iGPU decode loss."

### Long Term: make REM more native to the hardware

Longer term, the opportunity is to stop treating the NPU only as a small LLM
serving endpoint and move more memory work into shapes that suit XDNA2.

Recommended long-term directions:

1. Split REM into two lanes: generative compaction for prose summaries, and
   structured memory operations for facts, embeddings, identity, and retrieval.
2. Move fact identity and supersession toward embeddings or other semantic
   matching so stale facts can be retired by meaning, not just string labels.
3. Explore NPU-native or NPU-friendly kernels for embeddings, reranking,
   clustering, or structured extraction. This is where the GEMM paper's lessons
   about layout, contiguous access, and memory movement become directly
   relevant.
4. Build a contention-aware scheduler that admits background memory work only
   when the foreground lane has budget. The policy should reason over measured
   drain rate, backlog, iGPU loss, thermals, and power.
5. Treat memory bandwidth as a first-class resource. If direct counters are not
   available on gfx1151, document that and keep the best available proxies:
   iGPU decode loss, board power, NPU activity, and workload attribution.

The long-term research claim should be that REM is a memory-system architecture
for local agents on heterogeneous APUs, not just a prompt-compression trick. The
paper gives that claim a stronger hardware-theory backbone, as long as the repo
keeps separating measured evidence from plausible future work.

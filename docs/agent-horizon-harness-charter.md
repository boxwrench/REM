# Agent-horizon harness — charter & dispatch brief (2026-07-03)

Companion to [direction-review-2026-07-03.md](direction-review-2026-07-03.md) (rev. 2,
narrowed mission). This is the brief for the design/prep agent: enough context,
constraints, and acceptance criteria to produce the full designs without re-deriving
the project. Read the direction review and
[rem-closeout-and-revisit.md](rem-closeout-and-revisit.md) first.

## 1. Mission

**Claim to prove:** a unified-memory box — Strix Halo first; Apple-silicon Mac and
NVIDIA DGX Spark as peers — runs a local agent *longer* than the same models run
naively, because REM keeps the working context bounded and does the memory maintenance
on silicon that is otherwise idle (the NPU).

**Headline metric — horizon multiplier:** max agent steps achievable inside a fixed
window / RAM / pp-latency budget with REM in the loop, divided by the same limit for
the naive arm. Flat injected-context is the ideal; **any multiplier > 1 is shipped
capability.** Correctness is a floor (REM arm must not score below baseline on the
task grader), not the objective.

**Second claim (graceful stop):** when the budget gauge nears the wall, the agent
checkpoints and touches base with the user instead of crashing. Even before the
multiplier is optimized, "extends your horizon and *tells you* before it runs out" is
a product claim no raw context window makes.

## 2. Why this doesn't exist yet

Everything validated so far is single-question recall over a long chat history
(LongMemEval-shaped). No agent loop has ever run through REM. The compounding problem
is real and already observed: unbounded assembly hit 37k–58k tokens against a
~32–40k window → HTTP 400 → `context_overflow` 0/5 (`bench/battery/FINDINGS.md`).
Tokens-per-step over an N-step task is unmeasured because the harness doesn't exist.
This charter creates it.

## 3. What exists to build on (do not rebuild these)

| Piece | Where | Status |
|---|---|---|
| Sidecar intercept (message rewrite in serving path) | `src/rem/memory/sidecar.py` | shipped, uses `SparseChronologicalSelector` |
| Bounded read path (fit-to-budget, protected floor) | `src/rem/memory/selector.py` + Step 0 (`docs/superpowers/specs/2026-06-26-bounded-read-path-design.md`) | passing: 40.6k → ≤28k, gold kept |
| Selector protocol (pluggable strategies) | `MemorySelector` in `selector.py` (`RecencySelector`, `LexicalSelector`, sparse) | shipped |
| NPU compactor (~10× compression, ~73 tok/s drain, bg thread) | `src/rem/memory/compactor.py`, `bench/battery/throughput_probe.json` | measured |
| Budget gauge / scheduler scaffolding | `src/rem/scheduler/` (`gauge.py`, `admission.py`, `queue.py`) | exists, not wired to a halt |
| Frozen eval harness + arms pattern (current/sparse/oracle) | `evals/memory_methods/heldout_eval.py` | shipped; reuse the arm/report pattern |
| Contention + thermal evidence | `bench/contention/`, thermal traces | measured, publishable |

Known leaks the harness must tolerate (not fix): malformed-JSON fact drops, stale-value
supersession, unbounded ledger+summaries store (93% of memory — store growth is fine,
*read* must stay bounded).

## 4. Harness design requirements

### 4.1 Two modes

**Mode A — trace replay (build first).** Replay a recorded or synthesized N-step
transcript through each arm's context-assembly path; at scripted checkpoints, ask probe
questions graded deterministically. No live environment, no agent-competence confound,
NPU-free iteration against persisted states (the Step 0 pattern). Mode A alone produces
the context-growth curves, the crash points, and a defensible first multiplier.

**Mode B — live loop.** A real tool-using loop against a deterministic, scripted
environment. Measures what Mode A can't: coherence under the model's own compounding
output, backlog behavior under real timing, and the halt/resume experience.

### 4.2 Task suite (Mode B) — requirements, not prescriptions

Three tasks, each: deterministic environment (same action → same observation),
programmatic grader, natural length ≥100 steps, and step observations big enough to
stress context (hundreds of tokens). Candidate shapes (design agent to finalize):
a seeded file-tree reorganization with a manifest to satisfy; a research-over-local-
corpus task where required facts are spread across many reads; a stateful inventory/
booking simulation with late steps depending on early facts. Every task embeds
**recall probes** (late steps that fail unless an early fact survived compaction) and
**contradiction probes** (grader detects incoherent/repeated actions — the soft-failure
signal).

### 4.3 Arms

1. **naive** — full history appended every step (the compounding baseline).
2. **truncate** — sliding window, newest-N tokens. *The honest baseline: it beat REM
   0.6 vs 0.4 at a generous budget. REM's win must show where truncation drops facts.*
3. **truncate+summary** — rolling summary + window (common practice baseline).
4. **REM** — NPU compaction + bounded sparse read via the sidecar.
5. **REM-cpu** (Strix only) — same compaction on CPU, isolating the NPU placement value.

### 4.4 Metrics — the budget line (per step, JSON schema to be designed)

tokens-injected, pp-latency, decode tok/s, peak unified-RAM (honest KV accounting is a
named open question), compaction backlog depth, NPU/iGPU utilization. Per run:
max horizon before failure, failure class (**hard**: overflow/OOM; **soft**: grader
incoherence; **clean**: checkpoint fired), task score, horizon multiplier vs naive,
halt quality (gauge fired before crash? resume-from-state-card succeeded?).

### 4.5 Checkpoint-and-touch-base protocol

Gauge threshold (default 80% of budget) → freeze a **state card** (goal, progress,
open items, key facts — the compactor already produces the ingredients) → surface to
user: continue trimmed / re-scope / stop. Acceptance: fires before any hard failure on
all tasks; a fresh session resuming from the state card alone passes the remaining
grader steps at ≥ the uninterrupted arm's rate.

### 4.6 Portability

Foreground and compactor models addressed as OpenAI-compatible endpoints; the
background-compactor runner is pluggable (XDNA NPU / Mac via MLX or llama.cpp /
Spark via CUDA). One results schema so a Mac or Spark owner produces a comparable
table from the same frozen tasks. Strix-specific code stays behind the runner
interface.

**Interface standards (evaluate, don't invent):**
- **Mode B environments should target the [OpenEnv](https://github.com/huggingface/OpenEnv)
  interface** (Meta-PyTorch + Hugging Face: Gymnasium-style `step()/reset()/state()`,
  containerized env servers, typed HTTP APIs, shareable via the HF hub). Frozen tasks
  published as OpenEnv environments make the M4 community ask trivial — pull and run.
  Design agent to confirm the spec fits fat-observation, ≥100-step tasks before
  committing.
- **Mode A traces as an event-sourced log**: append-only per-step events (action,
  observation, tokens), replayable, branchable from any step k with a different arm.
  Prior-art pattern (not a dependency): the replay module in
  [cognicore-my-openenv](https://github.com/cognicore-dev/cognicore-my-openenv) —
  note that project's *memory* approach (cross-episode answer memorization, unbounded
  context growth, no token/latency accounting) is not relevant to REM.

## 5. Milestones (part-time, each ends usable or decisively informative)

- **M1 (~1–2 wk):** Mode A + instrumentation + naive/truncate arms. Output: context
  curves, crash points, first budget lines. *Decisive even if REM never runs.*
- **M2 (~1 wk):** REM arm via sidecar, Mode A. Output: **first horizon multiplier.**
- **M3 (~1–2 wk):** checkpoint halt wired to the gauge + resume test; Mode B task #1.
  Output: clean-stop demo — the dogfoodable moment.
- **M4 (~1 wk):** publish harness + frozen tasks + schema; invite Mac/Spark runs.
  REM-cpu arm for the NPU-placement headline.

Direction 1 (temporal tool scaffold) proceeds in parallel; Mode B ordering probes will
lean on it.

## 6. Deliverables expected from the design agent

1. Harness architecture spec (Mode A/B, arm interface, runner interface) — follow the
   Step 0 spec's format and its selector-protocol style.
2. Task-suite spec: 3 tasks with environments, probe placement, graders.
3. Budget-line JSON schema + results-table format (extend `heldout_eval.py` reporting).
4. Checkpoint/state-card format + halt protocol spec.
5. Implementation plan sequenced M1→M4 with test gates (suite 267 stays green;
   sidecar changes flag-gated, non-destructive).
6. Risk register: honest KV/RAM accounting on unified memory; step-token distribution
   assumptions; backlog overflow under fast step rates; grader gameability;
   **non-determinism in the NPU compaction step breaking clean Mode-A replay** — the
   compactor calls a model mid-assembly, so its outputs must be recorded-and-replayed
   or its sampling pinned, or the "only the memory arm changed" guarantee silently fails.

## 7. Non-goals

No QA-benchmark chasing, no answerer upsizing (budget-gated per the direction review),
no appliance work, no fixing write-side supersession here (tolerate, measure, move on).

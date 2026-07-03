# REM — direction review & decision (2026-07-03)

Companion to [rem-closeout-and-revisit.md](rem-closeout-and-revisit.md). That doc records
where the memory-quality work stopped and why. This one records the decision about what
comes next, the options that were on the table, and where outside help would genuinely
move things.

REM is an open project. This review is published as-is — including the open questions —
because the open questions are the contribution surface. Prepared with an AI review of
the close-out state; the decision and constraints are mine.

> **Revised same day.** The first draft chose "swap the 2B answerer for a 26–32B" as
> direction 1 — optimizing QA accuracy on LongMemEval rather than what REM is for
> (context economy over long horizons). This revision makes tool-scaffolding the
> front-runner, gates any answerer change behind a budget rule, and promotes the
> context-economy measurement to direction 2. The facts didn't change; the ranking
> axis did.

---

## Where things stand

Three facts, all traceable to the close-out doc:

1. The knowledge-update read-side win is shipped (sparse-chronological read path +
   question taxonomy, non-destructive, flag-gated). Dev-scale numbers: current 2 <
   sparse 3–4 < oracle 5–6 out of 6. Together with the
   [bounded-read-path spec](superpowers/specs/2026-06-26-bounded-read-path-design.md)
   ("inject fewer tokens, bound what's read"), this is the thread that directly serves
   the long-horizon thesis — it is context-economy work, not just a QA score.
2. Held-out testing moved the bottleneck off memory. Temporal items (ordering,
   abstention, date arithmetic) scored 0/3 for every arm **including oracle** — perfect
   evidence in, wrong answer out. The ~2B on-NPU answerer is the ceiling
   (`heldout_temporal_eval.md`).
3. The close-out doc's own revisit trigger for that wall is "a stronger answerer or
   explicit tool-use." As of this review, stronger local models are already on disk:
   qwen2.5-32B-instruct-q8 (ollama), gemma-4-26B/31B and qwen3.6-27B/35B (lemonade),
   with ~76 GB RAM free. The trigger is satisfied. Nobody has run the experiment yet.
   Note the trigger names *two* branches — a stronger answerer **or** explicit tool-use —
   and only the tool-use branch preserves REM's context/pp/RAM budget. Upsizing the
   answerer spends exactly the budget REM exists to protect (see next section).

Constraints shaping the decision: solo developer directing AI coding agents, part-time
bandwidth, one Ryzen AI box (XDNA NPU + Radeon iGPU, ~128 GB RAM), no cloud/API budget.
The target for the next step: a 2–4 week part-time milestone that ends in something
usable or decisively informative.

## What REM is optimizing (the constraint that ranks these)

REM exists for long-horizon task capability on one box: let a small, fast, on-NPU model
stay coherent across a long task without the working context growing until it blows the
window, exhausts RAM, or drags prompt-processing latency. The scarce resource is *context
budget over time*. The headline number is the **horizon multiplier** — how many more
agent steps fit inside the same window / RAM / pp-latency budget with REM in the loop
vs the same models run naively (≈ raw tokens produced per step ÷ tokens injected per
step). Flat injection is the ideal, but it is not the bar: **any multiplier > 1 is
shipped capability** — 10× compression is roughly a 10× longer task on the same box.
Long-term recall alone is table stakes; vector stores and RAG already do it. REM's
differentiator is a cheap, bounded working state that a weak local model can ride for a
long horizon — plus a **graceful stop**: when the budget gauge nears the wall, the agent
checkpoints and touches base with the user instead of crashing mid-task
(`ContextLimitExceeded` becomes a user touchpoint, not a failure).

That makes correctness a **constraint, not the objective**. "Did a bigger model answer
more temporal questions" is a QA score; "how many steps did REM buy the 2B before the
context wall, and did it stop cleanly" is the actual game. The directions below are
ranked by that constraint: an option that spends context/pp/RAM budget to buy a
benchmark point ranks *below* an option that holds the budget and still clears the task.
Any change to the answerer must therefore report a budget line (tokens-into-context/step,
pp-latency, peak RAM), not just an accuracy number.

**Narrowed mission (2026-07-03):** the target machine is the unified-memory box —
Strix Halo first (where the NPU placement is the differentiator), Apple-silicon Macs
and NVIDIA DGX Spark as portable peers. The claim to prove: *these boxes run a local
agent longer than the same models run naively.* The concrete vehicle is the
agent-horizon harness — see
[agent-horizon-harness-charter.md](agent-horizon-harness-charter.md), which supersedes
direction 2's harness sketch below with a dispatch-ready design brief.

## The four directions considered

### 1. Clear the temporal wall while holding the context budget (chosen)

**The wall:** temporal items (ordering, abstention, date arithmetic) score 0/3 for every
arm including oracle — perfect evidence in, wrong answer out. The ~2B on-NPU answerer
can't reason over even correct evidence. This ceiling must be cleared for long-horizon
work: a task that can't order its own history won't stay coherent over many steps.

**Primary probe — tool-scaffolding (keeps the thesis intact):** give the 2B answerer a
deterministic date-delta / ordering / abstention step it calls, instead of asking the
model to do the arithmetic in-weights. This lifts correctness *without* growing the
model, the injected context, or pp latency — the only branch that clears the wall while
respecting the budget REM exists to protect. There is direct evidence it bites: the
close-out §A scaffold already rescued abstention 0→1. Reliable tool calls at ~2B are the
known risk; the day-1–2 work below is to make them robust and tested.

**Secondary, gated — a larger answerer as a measured-cost experiment (not a default):**
the stronger local models on disk (qwen2.5-32B, gemma-4-26B/31B, etc.) are worth exactly
*one* sweep — to bound the oracle-reasoning ceiling, i.e. learn whether the temporal task
is solvable at all on this hardware. But upsizing the shipped answerer runs directly
against REM's point: a 26–32B model on iGPU/CPU is slower pp and heavier RAM, the exact
failure modes REM removes. So this arm is an experiment to bound the ceiling, not a
promotion path. It only becomes a promotion candidate if it clears the budget gate below
— which, on this hardware, it is not expected to.

**Honest cost:** tool-scaffold is days of design + tests, budget-neutral by construction.
The model sweep is a single measured run; its pp-latency and RAM cost get recorded, not
waved past.

**Decision rule, committed before results:** promote a change to the shipped answerer
only if it clears **all** of:
- temporal-with-oracle ≥2/3, AND held-out KU sparse ≥ current (correctness constraint),
- tokens-into-context per step ≤ the current 2B path, AND answerer pp-latency within a
  pre-committed multiple of the 2B-on-NPU baseline, AND peak RAM within the box's
  headroom (budget gate).

Tool-scaffolding on the 2B is expected to pass the budget gate by construction; the
larger-model arm is expected to fail it and therefore stays an informative experiment,
not a promotion. If neither clears temporal ≥2/3, the wall is a reasoning limit for this
hardware and the next probe is a richer deterministic scaffold — still on the small model.

### 2. Measure the long-horizon claim, then publish it (context economy)

**What:** Make the north-star metric a measured result instead of a stated principle.
Build the agent-horizon harness (design brief:
[agent-horizon-harness-charter.md](agent-horizon-harness-charter.md)): run replayable
N-step agent tasks under naive / truncation / REM arms and log tokens-into-context/step,
working-set size, pp-latency, and peak RAM per step — the output is the **horizon
multiplier** at a fixed budget. Two slices ride along: the **checkpoint-and-touch-base
stop** (budget gauge ≥ threshold → checkpoint state card, hand control to the user —
days of work, converts the hard wall into a feature) and the held-out KU capture
(close-out §4) as the **correctness floor**: sparse ≥ current at larger n, checked
alongside the budget curve, not instead of it.

**Why it matters:** this is REM's differentiator made falsifiable. The publishable claim
becomes "working context held flat (or sub-linear) over N steps at a fixed pp/RAM
budget, with accuracy no worse than baseline" — a claim no QA benchmark number can
express, and one the shipped sparse path plus the bounded-read spec are already most of
the way to supporting. Held-out replication of the KU numbers rides along and either
hardens the shipped result or catches a non-replication before it goes public; either
outcome is worth publishing in an open project.

**Why second:** direction 1 is smaller, removes the temporal confound from any public
demo, and its budget instrumentation (day 1 below) is deliberately the first brick of
this direction's harness. They interleave rather than compete.

### 3. Dogfood REM as a daily memory layer (queued behind 1)

**What:** Wire REM's capture+recall sidecar into a daily-driver chat surface
(open-webui is already in the AI-Box bundle) and use it for real, every day, for two
weeks.

**Why it matters:** REM's biggest strategic unknown is use-case, and the cheapest honest
answer is lived usage. The success signal is qualitative and strict: REM recalls
something I'd forgotten and needed, roughly weekly, without steering.

**Why it waits:** dogfooding with known-broken temporal handling confounds "REM isn't
useful" with "temporal questions fail." Direction 1 removes the confound — by
scaffolding, not by upsizing. Assumption to verify on day one of this direction:
open-webui exposes a usable pipeline/filter hook for the sidecar.

### 4. Harden the AI-Box into an appliance (parked)

**What:** one-command bring-up of the full service bundle (flm, ollama, comfyui, kokoro,
open-webui, xdna-top), pinned versions, REM as the memory spine.

**Why parked:** largest surface, highest scope-creep exposure for a solo part-timer, and
no 2–4-week slice of it is decisively informative about REM itself. It becomes attractive
after directions 1–2 establish what the box is actually for.

### Explicitly out of scope

Finishing Wall C (attribute-head constraint for read-time newest-preference) stays
parked. It is the closest remaining read-side headroom, and it is also exactly the
memory micro-optimization chapter that was deliberately closed. The close-out doc records
how to resume it if the KU held-out floor check (direction 2) shows the headroom matters.

## First week (direction 1)

| Day | Item | Owner | Output |
|-----|------|-------|--------|
| 1 | Disable the stale `rem-supersession-heldout-audit` scheduled task (close-out §5) | me | confirmed off |
| 1 | Add budget instrumentation to `heldout_eval.py`: log tokens-into-context/step, answerer pp-latency, peak RAM per arm | agent | every arm reports a budget line, not just accuracy |
| 1–2 | Build the deterministic temporal tool (date-delta + ordering + abstention) callable by the 2B; unit-test standalone | agent | tool + tests green; suite 267 green |
| 2 | Run 3 frozen temporal items on the 2B **with** the tool, oracle arm first | agent | temporal-with-tool result + budget line |
| 2–3 | One gated model sweep: add `--answerer-endpoint` (OpenAI-compatible), run the same 3 temporal items on qwen2.5-32B / gemma-4-26B to bound the oracle-reasoning ceiling; record pp-latency + RAM | agent | `heldout_answerer_sweep.json` incl. budget columns |
| 3 | Write the decision rule (incl. the budget gate) into the results doc *before* reading results | me | rule on record |
| 4 | Fold the abstention scaffold instruction into the shipped answer prompt (close-out §A; rescued abstention 0→1) | agent | prompt updated, tests green |
| 5 | Go/no-go memo: chosen temporal fix, accuracy **and** budget (tokens/pp/RAM), week-2 plan | me | memo appended to the close-out doc as "Revisit A: executed" |

## Where community help would actually move this

Concrete, self-contained asks — each one lands even without context on the rest:

- **Tool-use scaffolding for small models (the front-runner).** The primary temporal fix
  is a deterministic date-delta/ordering/abstention tool step callable by a ~2B model,
  because it clears the wall without spending context/pp/RAM budget. Prior art or a
  working pattern for reliable tool calls at that size would save weeks.
- **Run the sweep on your hardware — and report the budget, not just accuracy.** The
  harness takes an OpenAI-compatible endpoint (after the day-1–2 change above). If you
  have a different GPU/NPU and a 7–70B local model, your temporal-with-oracle numbers
  *plus* tokens-into-context/step, pp-latency and peak RAM extend the answer beyond one
  box. Accuracy without the budget line doesn't tell us whether the model is viable for
  long-horizon work; the frozen items and arms make results comparable.
- **A cheap typed identity judge** (Wall B's revisit trigger). The scaffold
  (`TypedIdentityMatcher`, `make_gemma_slot_judge`) is built and parked; the labeled
  sentinel families are frozen as a regression set. This is a well-bounded ML problem
  with its eval already written.
- **Long-horizon traces for the N-step harness** (direction 2). Multi-hundred-step
  agent task traces suitable for replay — or prior art on tokens-per-step accounting —
  are the raw material for measuring "working context held flat to N steps." Even
  pointers help.
- **open-webui integration experience.** If you've wired a memory sidecar into
  open-webui's pipeline/filter hooks, direction 3's day-one question is one you can
  answer in a comment.
- **Held-out item capture** (direction 2's correctness floor): more KU / multi-session
  LOOKUP items from the LongMemEval-S pool, captured per close-out §4, directly raise
  the n on the shipped result.

Issues and discussion welcome. The walls are documented precisely so that someone else's
advance can plug in without re-deriving the context.

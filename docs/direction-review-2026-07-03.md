# REM — direction review & decision (2026-07-03)

Companion to [rem-closeout-and-revisit.md](rem-closeout-and-revisit.md). That doc records
where the memory-quality work stopped and why. This one records the decision about what
comes next, the options that were on the table, and where outside help would genuinely
move things.

REM is an open project. This review is published as-is — including the open questions —
because the open questions are the contribution surface. Prepared with an AI review of
the close-out state; the decision and constraints are mine.

---

## Where things stand

Three facts, all traceable to the close-out doc:

1. The knowledge-update read-side win is shipped (sparse-chronological read path +
   question taxonomy, non-destructive, flag-gated). Dev-scale numbers: current 2 <
   sparse 3–4 < oracle 5–6 out of 6.
2. Held-out testing moved the bottleneck off memory. Temporal items (ordering,
   abstention, date arithmetic) scored 0/3 for every arm **including oracle** — perfect
   evidence in, wrong answer out. The ~2B on-NPU answerer is the ceiling
   (`heldout_temporal_eval.md`).
3. The close-out doc's own revisit trigger for that wall is "a stronger answerer or
   explicit tool-use." As of this review, stronger local models are already on disk:
   qwen2.5-32B-instruct-q8 (ollama), gemma-4-26B/31B and qwen3.6-27B/35B (lemonade),
   with ~76 GB RAM free. The trigger is satisfied. Nobody has run the experiment yet.

Constraints shaping the decision: solo developer directing AI coding agents, part-time
bandwidth, one Ryzen AI box (XDNA NPU + Radeon iGPU, ~128 GB RAM), no cloud/API budget.
The target for the next step: a 2–4 week part-time milestone that ends in something
usable or decisively informative.

## The four directions considered

### 1. Lift the answerer ceiling (chosen)

**What:** Point the frozen 30-item held-out harness (`evals/memory_methods/heldout_eval.py`)
at the larger local models, oracle arm first. Capture and extraction stay on the NPU 2B;
only the answerer role is swapped.

**Why it wins:** It executes the close-out doc's revisit trigger directly, with assets
already paid for — frozen items, an oracle arm that isolates reasoning from retrieval,
downloaded models. Every outcome is decisive. If a 26–32B model clears temporal
reasoning, REM's retrieval finally has a reasoner that can use it, and dogfooding
becomes a fair test. If every local model fails, that closes the question for this
hardware and tool-use scaffolding (a date-delta calculator, an explicit ordering step)
becomes the clear next probe.

**Honest cost:** wall-clock. A 32B answerer on iGPU/CPU will be slower than the 2B on
NPU. How much slower is unmeasured; it gets measured alongside accuracy in the same
sweep. Effort: small — days, not weeks.

**Decision rule, committed before results:** ≥2/3 on temporal-with-oracle AND held-out
KU sparse ≥ current → promote that model to the answerer role. Otherwise → tool-scaffold
probe.

### 2. Dogfood REM as a daily memory layer (queued behind 1)

**What:** Wire REM's capture+recall sidecar into a daily-driver chat surface
(open-webui is already in the AI-Box bundle) and use it for real, every day, for two
weeks.

**Why it matters:** REM's biggest strategic unknown is use-case, and the cheapest honest
answer is lived usage. The success signal is qualitative and strict: REM recalls
something I'd forgotten and needed, roughly weekly, without steering.

**Why it waits:** dogfooding with a 2B answerer confounds "REM isn't useful" with "the
answerer can't reason." Direction 1 removes the confound first. Assumption to verify on
day one of this direction: open-webui exposes a usable pipeline/filter hook for the
sidecar.

### 3. Validate, then publish the read-side result

**What:** Capture held-out KU and multi-session LOOKUP items (close-out §4 names the
pools), run sparse-vs-current arms at larger n, and publish the result with the launch
materials that already exist.

**Why it matters:** the shipped KU numbers are dev-scale and small-n. Held-out
replication either hardens them or catches a non-replication before it goes public.
Either result is worth publishing in an open project.

**Why it's third:** it inherits value from direction 1 — a public demo speaks with the
answerer's voice, and "REM plus a local reasoner that clears temporal tasks" is a much
stronger first impression than the same retrieval behind a model that can't order two
events.

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
how to resume it if the KU held-out validation (direction 3) shows the headroom matters.

## First week (direction 1)

| Day | Item | Owner | Output |
|-----|------|-------|--------|
| 1 | Disable the stale `rem-supersession-heldout-audit` scheduled task (close-out §5) | me | confirmed off |
| 1 | Verify candidate answerers serve; note rough tok/s: qwen2.5-32B (ollama), gemma-4-26B/31B (lemonade) | me + agent | model/endpoint/tok-s table |
| 1–2 | Add `--answerer-endpoint` (OpenAI-compatible URL + model) to `heldout_eval.py`; smoke-test on ollama | agent | eval runs on a non-flm endpoint; suite 267 green |
| 2–3 | Run 3 frozen temporal items, oracle arm first, per candidate; then 10 KU items (sparse vs current) on the best temporal performer | agent | `heldout_answerer_sweep.json` + results table |
| 3 | Write the decision rule into the results doc *before* reading results | me | rule on record |
| 4 | Fold the abstention scaffold instruction into the shipped answer prompt (close-out §A calls this out as cheap; it rescued abstention 0→1) | agent | prompt updated, tests green |
| 5 | Go/no-go memo: chosen answerer or no-go, accuracy + latency, week-2 plan | me | memo appended to the close-out doc as "Revisit A: executed" |

## Where community help would actually move this

Concrete, self-contained asks — each one lands even without context on the rest:

- **Run the answerer sweep on your hardware.** The harness takes an OpenAI-compatible
  endpoint (after the day-1–2 change above). If you have a different GPU/NPU and a 7–70B
  local model, your temporal-with-oracle numbers extend the answer beyond one box.
  The frozen items and arms make results comparable.
- **Tool-use scaffolding for small models.** If the sweep no-goes, the next probe is a
  deterministic date-delta/ordering tool step callable by a ~2B model. Prior art or a
  working pattern for reliable tool calls at that size would save weeks.
- **A cheap typed identity judge** (Wall B's revisit trigger). The scaffold
  (`TypedIdentityMatcher`, `make_gemma_slot_judge`) is built and parked; the labeled
  sentinel families are frozen as a regression set. This is a well-bounded ML problem
  with its eval already written.
- **open-webui integration experience.** If you've wired a memory sidecar into
  open-webui's pipeline/filter hooks, direction 2's day-one question is one you can
  answer in a comment.
- **Held-out item capture** (direction 3): more KU / multi-session LOOKUP items from the
  LongMemEval-S pool, captured per close-out §4, directly raise the n on the shipped
  result.

Issues and discussion welcome. The walls are documented precisely so that someone else's
advance can plug in without re-deriving the context.

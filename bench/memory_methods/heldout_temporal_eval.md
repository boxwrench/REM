# Held-out temporal-reasoning eval (n=3, 2026-07-02) — answerer-bound; capture STOPPED

Early read on the first 3 held-out states the Path C capture produced (all
temporal-reasoning). Ran the answerer gate (current / sparse / oracle / candidate)
with the fixed gemma4-it:e2b answerer. Harness: `evals/memory_methods/heldout_eval.py`;
answers: `bench/memory_methods/heldout_eval_answers.json`.

## Result: 0/3 for EVERY arm, including oracle

| item | type | gold | current | sparse | oracle | candidate |
|---|---|---|---|---|---|---|
| gpt4_45189cb4 | ordering | NBA→CFB champ→NFL playoffs | "no info" | "no info" | names 2/3 events, no order | "no info" |
| gpt4_fe651585_abs | abstention | "not enough info (Tom never mentioned)" | hallucinates Alex | hallucinates Alex | states an Alex fact, no abstain | hallucinates Alex |
| gpt4_7a0daae1 | arithmetic | "1 week" | "no dates" | surfaced 03/10 + 03/15, no compute | "no dates" | surfaced dates, no compute |

## Conclusion: on temporal reasoning the 2B ANSWERER is the bottleneck, not retrieval
This is the OPPOSITE of the knowledge-update pattern (there oracle scored 5–6/6, so
good evidence sufficed and the read path was the lever). Here even *perfect* oracle
evidence yields ~0/3: the model can't reconstruct an order, won't abstain, and won't
do date arithmetic. Retrieval improvements cannot lift a reasoning ceiling.

Two sharp sub-findings:
- On the arithmetic item, **sparse out-retrieved oracle** — the clean extracted date
  slots (03/10 bought, 03/15 received) beat the raw prose sessions. Retrieval did its
  job; the model simply didn't subtract. Read side works; reasoning doesn't.
- **Abstention fails universally** — no arm says "not enough info about Tom." This is
  exactly the regression category the KU set structurally couldn't show, and it's a
  model limitation, not a retrieval one.

## Decision (Keith, 2026-07-02): STOP the temporal/multi capture
The 20 held-out states are all temporal/multi. Since temporal reasoning is answerer-
bound, a frozen confirm on them would mainly measure the 2B model's reasoning ceiling
(≈flat across arms), NOT the knowledge-update read-side win. So the ~25h capture was
validating the wrong thing for the decision at hand. Stopped at 3 held-out states.

## What still stands (do not let "it's not working" flatten this)
- The KNOWLEDGE-UPDATE read-side result is unchanged and real: gold present+active
  6/6, current 2 < sparse 3–4 < oracle 5–6, safe sparse+taxonomy shipped, non-
  destructive. That win is category-specific to KU.
- Temporal reasoning (ordering / abstention / arithmetic) is out of reach for the 2B
  answerer regardless of retrieval. Moving it needs a stronger answerer or explicit
  reasoning scaffolding (e.g. compute-the-delta / abstention prompts), not memory work.
- To validate the KU read-side win on held-out data, capture held-out *KU* items
  (LongMemEval-S pool beyond the manifest), not temporal/multi.

## Follow-up: cheap reasoning-scaffold answer prompt (probe, same evidence)
Re-ran the 3 items with a scaffold system prompt (list events by timestamp for order;
find both dates and compute for arithmetic; if a named entity isn't in memory, say not
enough info), on sparse + oracle. Harness: `evals/memory_methods/temporal_scaffold_probe.py`;
answers: `temporal_scaffold_probe.json`. Result vs the 0/3 base:
- **Abstention: RESCUED.** Sparse now cleanly abstains ("not enough information to
  determine who became a parent first"); oracle correctly reasons Tom is never
  mentioned. This category was purely prompt-addressable.
- **Ordering: still fails.** The model engages but sparse doesn't surface the events
  and oracle names only one — retrieval + reasoning bound.
- **Arithmetic: still fails.** It now hunts for both dates and reasons, but tangles on
  an ambiguous "received" date rather than computing 1 week.
=> ~1/3 with the scaffold vs 0/3 base. Takeaway: an abstention/"say-not-enough-info"
instruction is a cheap real win worth folding into the shipped answer prompt; ordering
and date-arithmetic stay model/retrieval bound and are not worth chasing with memory work.

## Caveats
n=3, all temporal-reasoning (no multi-session item was reached — those may be more
lookup-like and read-relevant). Strict grading. The flm serving window had shrunk to
<27k, so `current` was capped at 13k — but it failed by saying "no info", not by
truncation, so the cap did not cause the failures.

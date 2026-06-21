# Battery findings: tight-budget recall gate

Status as of 2026-06-21. Artifacts in this directory; numbers below come straight
from them.

## The gate

Does REM's compaction preserve a user-relevant fact better than naive truncation,
at a context budget tight enough that truncation drops the gold evidence? A run
only counts if truncation actually drops the gold (validity guard in
`evals/battery/aggregate.py`).

## Runs

| Artifact | items | budget | valid | truncation kept gold | REM acc | dominant miss | extraction |
|---|---|---|---|---|---|---|---|
| `tight_smoke_b3000_limit1.json` | 1 | 3000 | no | 100% | 0/1 | — | 181 att / 0 fail |
| `tight_sweep_b2000_limit3.json` | 3 | 2000 | no | 67% | 0/3 | budget_invalid | 957 att / 2 fail |
| `tight_sweep_b1000_limit3.json` | 3 | 1000 | no | 67% | 0/3 | budget_invalid | 1311 att / 4 fail |
| `valid_b1000_oldgold.json` | 5 | 1000 | **yes** | 0% | 0/5 | context_overflow | 2220 att / 1 fail |

## What the budget sweep showed

The first three runs are invalid: truncation kept the gold, so the comparison is
trivial. Tightening the budget from 2000 to 1000 did not move truncation
retention (67% both times). Cause: for `knowledge-update`, the gold is the latest
update, which sits in recent sessions — exactly what truncation preserves. Across
all 78 knowledge-update items the latest gold session has median normalized
timeline position 0.86, and 53 of 78 have it in the newest third. The budget knob
cannot manufacture a valid comparison on recent-gold items.

## The selection fix

`--max-gold-recency 0.33` selects the 5 items whose latest gold is in the oldest
third (`gold_recency` 0.16–0.28). Truncation drops the gold on all 5 (0%
retention), so `valid_b1000_oldgold.json` is the first valid run.

## What the valid run did and did not prove

- It is valid: truncation retains 0% and scores 0/5, as intended.
- It does **not** yet give a memory-quality verdict. REM also scored 0/5, and the
  classifier attributes all 5 misses to `context_overflow`: REM raised
  `ContextLimitExceeded` during assembly and never produced an answer. These items
  are large (431–494 turns); REM's compacted memory (summaries + facts ledger +
  recent window) exceeds the assembly ceiling `max_context_tokens = budget×4 =
  4000`.
- Extraction held up: 2220 attempts, 1 hard failure. The JSON-robustness and
  malformed-entry recovery work is doing its job. One new mangling shape still
  slips through (`"source_turn_id":443,subject"...`).

The classifier's recommendation stands: fix the harness before diagnosing memory.

## The overflow: root cause and repair

Reproduced NPU-free: a ~500-turn item compacts to ~82 episodic summaries (~2949
tokens) + ~164 ledger facts (~3475 tokens) = ~6642 tokens, against the REM arm's
`max_context_tokens = budget×4 = 4000`. The assembler renders the ledger in full
and all summaries with no size bound, so REM's compacted memory grows with
conversation length. This is a scaling gap, not merely a small config ceiling:
the verbatim tier is bounded but summaries + ledger are not.

Repair (diagnostic-first): assemble the REM arm within a fixed memory window
`REM_MEMORY_WINDOW_TOKENS = 16000` (the reserved memory region from the
architecture spec §3) instead of `budget×4`. This unblocks assembly so the arm
can answer, and isolates write/read recall from token-efficiency.

Caveat: the REM arm now gets up to 16k of context while truncation gets the
budget, so the comparison is **not token-matched**. It answers "does the gold
survive compaction at all?" Budget-bounded memory (eviction so summaries + ledger
fit ~budget) is the follow-up if write recall holds.

## Next

Re-run `--budget 1000 --max-gold-recency 0.33`. Expect the REM misses to move out
of `context_overflow` into a diagnostic bucket: `summary_loss`/`stale_ghost`
(gold lost in compaction → graph-architecture signal) or `answerer_failure`
(gold present, model missed it → memory is not the bottleneck).

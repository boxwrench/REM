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

## Next

1. Determine whether the overflow is a config ceiling (`max_context_tokens` too
   small for ~500-turn items) or unbounded growth of REM's compacted memory.
2. Fix it, then re-run `--budget 1000 --max-gold-recency 0.33` for an actual
   REM-vs-truncation number.

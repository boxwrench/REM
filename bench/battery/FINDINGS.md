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

## The overflow: root cause, and why the 16k "repair" does not fix it

Root cause: the assembler renders the facts ledger in full and every episodic
summary with no size bound, so REM's compacted memory grows with conversation
length. The verbatim tier is bounded but summaries + ledger are not. This is a
scaling gap, not a small config ceiling.

The actual magnitude (from the artifact's own overflow messages, corroborated by
the canary and `diag_031748ae_w64k.json`): on the five oldest-gold items REM's
assembled memory is **36,977 – 58,150 tokens**:

| item | assembled tokens |
|---|---|
| 9bbe84a2 | 36,977 |
| 031748ae | 40,565 |
| 3ba21379 | 50,529 |
| cc5ded98 | 54,790 |
| c6853660 | 58,150 |

An earlier note here estimated ~6.6k from a synthetic reproduction; that
under-counted by 6×. Raising the assemble ceiling from `budget×4 = 4000` to
`REM_MEMORY_WINDOW_TOKENS = 16000` (spec §3 reserved region) **does not unblock
assembly** — all five still exceed 16k, so REM still raises
`ContextLimitExceeded` and answers nothing. The 16k change only moves the
ceiling; it does not bound the memory.

Worse, the larger items (50k–58k) exceed the **answering model's own context
window** (~32–40k): `gemma4-it:e2b` returns HTTP 400 `"Max length reached!"`
above that. So even with an unbounded assemble ceiling, the model cannot read
REM's compacted memory on long items. Compaction here re-encodes the history at
similar scale rather than compressing it to a consumable size.

## What this means for the gate

REM cannot answer these items until it has a **bounded read path** — retrieval
or eviction that fits the assembled memory to the model window (~28k after
question/answer headroom), selecting which summaries/facts to include. That is a
design change, not a ceiling tweak, and is the next architecture gate (roadmap
item 7). A token-matched REM-vs-truncation comparison is only meaningful once
that exists.

## Next

1. Diagnostic (`evals/battery/diagnose_memory.py`, artifact
   `diag_031748ae_w64k.json`): with a large assemble window, measure the per-tier
   breakdown (which store dominates the 40k), whether the gold survived
   compaction, and whether a model-fitting head-slice can still recall it. This
   decides whether the problem is read-path-only (gold survived) or also a
   write-recall/summary-fidelity problem (gold lost).
2. Then choose the P1 fix scope: a bounded read path so the five-item battery can
   produce a token-matched comparison, vs. recording the read-path overflow as
   the architecture verdict.

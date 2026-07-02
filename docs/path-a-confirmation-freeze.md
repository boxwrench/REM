# Path A 30-item confirmation freeze

The machine-readable source of truth is
`bench/memory_methods/path_a_confirmation_criteria.json` (`path-a-confirmation-30-v1`).
It was frozen before any 30-item answer generation. The six earlier Path-A answers
did not set these thresholds.

The paired arms are `safe-sparse` and `path-a-candidate`. Both use the same shared
question taxonomy; the only selector difference is `prefer_newest=false` versus
`true`. Every one of the frozen 30 items must appear once per arm at 8k and 28k:
10 knowledge-update, 10 temporal-reasoning, and 10 multi-session. Three fixed
Gemma answers and Claude judgments are taken at 8k, with majority correctness.
The 28k pass is recall/context robustness only.

Experimental newest preference may be promoted only with at least three 8k wins,
zero item regressions, and no category accuracy reduction. It must also introduce
no abstention, stale-failure, gold-source-loss, or empty-retrieval regression;
lose no source provenance; retain date/session provenance for temporal and
multi-session evidence; fit both context budgets without overflow; produce no run
errors; and keep candidate recall p95 at or below 1,000 ms at each budget.

Preflight is deliberately NPU-free:

```bash
PYTHONPATH=.:src python3 evals/memory_methods/run_path_a_confirmation.py
```

If any capture is missing or invalid, it writes a report listing every affected
state and exits 2 before constructing answerer or judge clients. It never writes
a partial run artifact. Once all captures exist, `--run` executes recall only;
`--score` executes the frozen three-repetition 8k answer/judge comparison plus
recall at both budgets. The candidate remains default-off unless every frozen
check passes.

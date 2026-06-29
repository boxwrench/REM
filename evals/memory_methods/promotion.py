"""Pre-registered promotion checks for the 30-item development suite."""
from __future__ import annotations

from collections import defaultdict

from evals.memory_methods.artifacts import ItemRun


def promotion_decision(
    baseline: list[ItemRun], candidate: list[ItemRun], *, latency_limit_ms: float = 1000
) -> dict:
    baseline_by_key = {
        (run.question_id, run.budget_tokens): run for run in baseline
    }
    candidate_by_key = {
        (run.question_id, run.budget_tokens): run for run in candidate
    }
    if baseline_by_key.keys() != candidate_by_key.keys():
        raise ValueError("baseline and candidate must cover identical item/budget pairs")
    wins = regressions = 0
    category_delta: dict[str, int] = defaultdict(int)
    for key, base in baseline_by_key.items():
        other = candidate_by_key[key]
        if base.judged_correct is None or other.judged_correct is None:
            continue
        delta = int(other.judged_correct) - int(base.judged_correct)
        wins += delta > 0
        regressions += delta < 0
        category_delta[base.category] += delta
    latencies = sorted(run.read_latency_ms for run in candidate)
    p95_index = max(0, int(len(latencies) * 0.95 + 0.999) - 1)
    p95 = latencies[p95_index] if latencies else None
    checks = {
        "three_more_wins_than_regressions": wins - regressions >= 3,
        "category_regression_at_most_one": all(
            delta >= -1 for delta in category_delta.values()
        ),
        "within_token_budgets": all(
            run.memory_tokens <= run.budget_tokens for run in candidate
        ),
        "recall_p95_at_most_one_second": p95 is not None and p95 <= latency_limit_ms,
        "no_context_overflow": not any(run.context_overflow for run in candidate),
        "no_provenance_loss": not any(run.provenance_lost for run in candidate),
    }
    return {
        "promote": all(checks.values()),
        "wins": wins,
        "regressions": regressions,
        "category_delta": dict(category_delta),
        "read_latency_p95_ms": p95,
        "checks": checks,
    }

"""Aggregate ArmRuns into per-arm accuracy + evidence-retention, with a validity guard."""
from __future__ import annotations

from collections import defaultdict

from evals.battery.models import ArmRun, BatteryResult

TRUNCATION_RETENTION_CEILING = 0.5  # if truncation kept the gold on >= half the
# questions, the budget is too generous and the comparison is trivial. REM-independent
# on purpose: REM's evidence_retained is a heuristic that reads False when REM correctly
# compacts the gold into a summary, so it must NOT gate validity.


def aggregate(runs: list[ArmRun], n_questions: int) -> BatteryResult:
    by_arm: dict[str, list[ArmRun]] = defaultdict(list)
    for r in runs:
        by_arm[r.arm].append(r)

    acc, ret, extraction = {}, {}, {}
    for arm, rs in by_arm.items():
        graded = [r for r in rs if r.judged_correct is not None]
        acc[arm] = (sum(1 for r in graded if r.judged_correct) / len(graded)) if graded else 0.0
        ret[arm] = (sum(1 for r in rs if r.evidence_retained) / len(rs)) if rs else 0.0

        # Sum per-question extraction telemetry; omit arms with no extraction stage.
        with_extraction = [r.extraction for r in rs if r.extraction]
        if with_extraction:
            summed: dict[str, int] = {}
            for diag in with_extraction:
                for key, value in diag.items():
                    summed[key] = summed.get(key, 0) + value
            extraction[arm] = summed

    result = BatteryResult(
        arm_accuracy=acc, arm_evidence_retention=ret,
        n_questions=n_questions, runs=runs, arm_extraction=extraction,
    )

    if "truncation" in ret and ret["truncation"] >= TRUNCATION_RETENTION_CEILING:
        result.valid = False
        result.invalid_reason = (
            f"Budget too generous: truncation retained the gold evidence on "
            f"{ret['truncation']:.0%} of questions (>= {TRUNCATION_RETENTION_CEILING:.0%}). "
            f"Lower the budget so truncation drops the gold session, else the comparison "
            f"is trivial."
        )
    return result

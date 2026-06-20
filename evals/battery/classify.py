"""Classify REM misses in a battery artifact into actionable failure buckets.

Roadmap item 6: a single accuracy number cannot tell you *why* REM lost. This
module reads a battery artifact (the JSON written by run_battery_spike) and
buckets each REM miss so the failure mix can drive item 7's architecture choice:

  - extraction_drop  -> keep near-term work on write robustness
  - summary_loss / stale_ghost / read-path ambiguity -> start the graph architecture

The classifier is triage, not ground truth. In particular, without separate
write-recall instrumentation, `summary_loss` cannot be distinguished from a
*silent* extraction miss (extraction succeeded but never captured the gold fact).
The report says so in its caveats.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

REM_ARM = "rem"

# Ordered by classification precedence (see classify_miss).
CATEGORIES = (
    "budget_invalid",     # run is invalid; the comparison does not count
    "context_overflow",   # the arm could not even assemble within the budget
    "extraction_drop",    # measured: extraction hard-failed on a span (facts lost)
    "judge_ambiguity",    # the judge itself was unsure; the "miss" is untrustworthy
    "stale_ghost",        # an old/superseded value surfaced as current
    "answerer_failure",   # gold text was in context, the foreground model still missed
    "summary_loss",       # gold absent from context, no extraction hard-failure recorded
)

_OVERFLOW_MARKER = "context overflow"
_AMBIGUOUS_MARKERS = (
    "ambiguous", "unclear", "cannot determine", "can't determine",
    "unable to determine", "partially",
)
_STALE_MARKERS = (
    "stale", "outdated", "old value", "previous value", "superseded",
    "no longer", "earlier value",
)


def classify_miss(run: dict, *, run_valid: bool = True) -> str:
    """Bucket one REM miss. Precedence is high-confidence/structural first."""
    reason = (run.get("judge_reason") or "").lower()

    if not run_valid:
        return "budget_invalid"
    if _OVERFLOW_MARKER in reason:
        return "context_overflow"

    extraction = run.get("extraction") or {}
    if extraction.get("failures", 0) > 0:
        return "extraction_drop"

    if any(m in reason for m in _AMBIGUOUS_MARKERS):
        return "judge_ambiguity"
    if any(m in reason for m in _STALE_MARKERS):
        return "stale_ghost"

    if run.get("evidence_retained"):
        return "answerer_failure"
    return "summary_loss"


@dataclass
class ClassificationReport:
    valid: bool
    n_rem_runs: int
    n_rem_correct: int
    n_rem_misses: int
    counts: dict[str, int]
    dominant: str | None
    misses: list[dict] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)

    def recommendation(self) -> str:
        """Map the dominant bucket to roadmap item 7's branch."""
        if not self.valid:
            return ("Run is budget-invalid; rerun at a tighter budget before drawing "
                    "any architecture conclusion.")
        if self.dominant is None:
            return "No REM misses to classify."
        if self.dominant in ("extraction_drop",):
            return ("Misses dominated by extraction drops: keep near-term work on "
                    "write/JSON robustness before changing architecture.")
        if self.dominant in ("summary_loss", "stale_ghost"):
            return ("Misses dominated by summary loss / stale ghosts: this is the "
                    "empirical trigger to start the graph-resident architecture "
                    "(Phase 0/1 in the architecture spec).")
        if self.dominant == "answerer_failure":
            return ("Misses dominated by answerer failures: the gold fact reached "
                    "context but the foreground model missed it; memory architecture "
                    "is not the bottleneck.")
        if self.dominant in ("context_overflow", "judge_ambiguity"):
            return (f"Misses dominated by {self.dominant}: fix the harness "
                    "(budget/assembly or judge prompt) before diagnosing memory.")
        return "Mixed failure modes; inspect per-miss detail."


def classify_battery(artifact: dict) -> ClassificationReport:
    valid = bool(artifact.get("valid", True))
    runs = artifact.get("runs", []) or []
    rem = [r for r in runs if r.get("arm") == REM_ARM]
    correct = [r for r in rem if r.get("judged_correct") is True]
    miss_runs = [r for r in rem if r.get("judged_correct") is False]

    counts: Counter[str] = Counter()
    misses: list[dict] = []
    for r in miss_runs:
        cat = classify_miss(r, run_valid=valid)
        counts[cat] += 1
        misses.append({
            "question_id": r.get("question_id"),
            "category": cat,
            "judge_reason": r.get("judge_reason", ""),
        })

    dominant = counts.most_common(1)[0][0] if counts else None

    caveats: list[str] = []
    if not valid:
        caveats.append(
            "Run is invalid ({}); miss classification is inconclusive.".format(
                artifact.get("invalid_reason", "budget too generous")
            )
        )
    if counts.get("summary_loss"):
        caveats.append(
            "summary_loss also captures silent extraction misses; separating them "
            "needs write-recall instrumentation."
        )

    return ClassificationReport(
        valid=valid,
        n_rem_runs=len(rem),
        n_rem_correct=len(correct),
        n_rem_misses=len(miss_runs),
        counts={c: counts.get(c, 0) for c in CATEGORIES},
        dominant=dominant,
        misses=misses,
        caveats=caveats,
    )


def _format_report(report: ClassificationReport) -> str:
    lines = [
        f"valid: {report.valid}",
        f"REM runs: {report.n_rem_runs}  correct: {report.n_rem_correct}  misses: {report.n_rem_misses}",
        "miss breakdown:",
    ]
    for cat in CATEGORIES:
        n = report.counts.get(cat, 0)
        if n:
            lines.append(f"  {cat:18s} {n}")
    lines.append(f"dominant: {report.dominant}")
    for c in report.caveats:
        lines.append(f"caveat: {c}")
    lines.append(f"recommendation: {report.recommendation()}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="Classify REM misses in a battery artifact")
    ap.add_argument("artifact", help="Path to a battery result JSON")
    ap.add_argument("--json", action="store_true", help="Emit the report as JSON")
    args = ap.parse_args()

    artifact = json.loads(Path(args.artifact).read_text(encoding="utf-8"))
    report = classify_battery(artifact)

    if args.json:
        payload = {
            "valid": report.valid,
            "n_rem_runs": report.n_rem_runs,
            "n_rem_correct": report.n_rem_correct,
            "n_rem_misses": report.n_rem_misses,
            "counts": report.counts,
            "dominant": report.dominant,
            "misses": report.misses,
            "caveats": report.caveats,
            "recommendation": report.recommendation(),
        }
        print(json.dumps(payload, indent=2))
    else:
        print(_format_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

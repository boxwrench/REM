"""Failure classifier for battery artifacts (roadmap item 6).

Buckets each REM miss so item 7's architecture decision is driven by the failure
mix, not a single accuracy number.
"""

from evals.battery.classify import (
    CATEGORIES,
    classify_battery,
    classify_miss,
)


def _miss(**over):
    base = {
        "question_id": "q",
        "arm": "rem",
        "judged_correct": False,
        "judge_reason": "incorrect",
        "evidence_retained": False,
        "model_answer": "wrong",
        "extraction": {"attempts": 1, "failures": 0, "truncations": 0},
    }
    base.update(over)
    return base


def test_extraction_drop_when_extraction_recorded_failures():
    run = _miss(extraction={"attempts": 2, "failures": 1, "truncations": 0})
    assert classify_miss(run, run_valid=True) == "extraction_drop"


def test_context_overflow_from_judge_reason():
    run = _miss(judge_reason="context overflow: assembled 41000 > 12000", model_answer="")
    assert classify_miss(run, run_valid=True) == "context_overflow"


def test_stale_ghost_when_judge_flags_outdated_value():
    run = _miss(judge_reason="Incorrect: the model gave the stale value 1,200 instead of 950.")
    assert classify_miss(run, run_valid=True) == "stale_ghost"


def test_answerer_failure_when_evidence_present_but_wrong():
    # Gold text survived verbatim into context, no extraction failure, not stale.
    run = _miss(evidence_retained=True)
    assert classify_miss(run, run_valid=True) == "answerer_failure"


def test_summary_loss_when_evidence_absent_and_no_extraction_failure():
    run = _miss(evidence_retained=False)
    assert classify_miss(run, run_valid=True) == "summary_loss"


def test_genuine_judge_uncertainty_is_judge_ambiguity():
    # The judge itself is unsure -> judge_ambiguity is correct.
    for reason in (
        "The grading is ambiguous; the answer could match either value.",
        "It is unclear whether the answer refers to the updated figure.",
        "I cannot determine whether this answer is correct from the gold.",
    ):
        assert classify_miss(_miss(judge_reason=reason), run_valid=True) == "judge_ambiguity"


def test_model_cannot_determine_is_not_judge_ambiguity():
    # The judge is CERTAIN the model failed; "cannot determine" describes the
    # MODEL's answer, not the judge's uncertainty. Must not be judge_ambiguity.
    reason = "Incorrect: the model cannot determine the number of engineers from the memory."
    assert classify_miss(_miss(judge_reason=reason, evidence_retained=False),
                         run_valid=True) == "summary_loss"
    assert classify_miss(_miss(judge_reason=reason, evidence_retained=True),
                         run_valid=True) == "answerer_failure"


def test_budget_invalid_takes_precedence():
    # An invalid run cannot be diagnosed, even if it has an extraction failure.
    run = _miss(extraction={"attempts": 2, "failures": 1, "truncations": 0})
    assert classify_miss(run, run_valid=False) == "budget_invalid"


def test_all_categories_are_known():
    run = _miss()
    assert classify_miss(run, run_valid=True) in CATEGORIES


def test_classify_battery_tallies_and_picks_dominant():
    artifact = {
        "valid": True,
        "runs": [
            # Truncation arm must be ignored entirely.
            {"arm": "truncation", "question_id": "q1", "judged_correct": False,
             "judge_reason": "wrong", "evidence_retained": False},
            # A correct REM run is not a miss.
            {"arm": "rem", "question_id": "q1", "judged_correct": True,
             "judge_reason": "correct", "evidence_retained": True,
             "extraction": {"failures": 0}},
            # Two summary-loss misses, one extraction-drop miss -> summary_loss dominates.
            _miss(question_id="q2", evidence_retained=False,
                  extraction={"attempts": 1, "failures": 0}),
            _miss(question_id="q3", evidence_retained=False,
                  extraction={"attempts": 1, "failures": 0}),
            _miss(question_id="q4", extraction={"attempts": 2, "failures": 1}),
        ],
    }
    report = classify_battery(artifact)

    assert report.n_rem_misses == 3
    assert report.counts["summary_loss"] == 2
    assert report.counts["extraction_drop"] == 1
    assert report.dominant == "summary_loss"


def test_classify_battery_flags_invalid_run():
    artifact = {
        "valid": False,
        "invalid_reason": "Budget too generous",
        "runs": [_miss(question_id="q1")],
    }
    report = classify_battery(artifact)
    assert report.counts["budget_invalid"] == 1
    assert any("invalid" in c.lower() for c in report.caveats)

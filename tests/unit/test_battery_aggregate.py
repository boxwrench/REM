from evals.battery.aggregate import aggregate
from evals.battery.models import ArmRun


def _runs():
    return [
        ArmRun("q1", "truncation", 100, evidence_retained=False, model_answer="Globex", judged_correct=False),
        ArmRun("q1", "rem", 90, evidence_retained=True, model_answer="Acme", judged_correct=True),
        ArmRun("q2", "truncation", 100, evidence_retained=False, model_answer="old", judged_correct=False),
        ArmRun("q2", "rem", 90, evidence_retained=True, model_answer="new", judged_correct=True),
    ]


def test_aggregate_computes_accuracy_and_retention():
    res = aggregate(_runs(), n_questions=2)
    assert res.arm_accuracy["rem"] == 1.0
    assert res.arm_accuracy["truncation"] == 0.0
    assert res.arm_evidence_retention["truncation"] == 0.0
    assert res.arm_evidence_retention["rem"] == 1.0
    assert res.valid is True


def test_aggregate_flags_trivial_comparison():
    # truncation retains as much evidence as REM -> budget too generous
    runs = [
        ArmRun("q1", "truncation", 100, evidence_retained=True, model_answer="x", judged_correct=True),
        ArmRun("q1", "rem", 90, evidence_retained=True, model_answer="x", judged_correct=True),
    ]
    res = aggregate(runs, n_questions=1)
    assert res.valid is False
    assert "budget" in res.invalid_reason.lower()


def test_aggregate_valid_when_truncation_drops_most_evidence():
    runs = [
        ArmRun("q1", "truncation", 100, evidence_retained=False, model_answer="x", judged_correct=False),
        ArmRun("q2", "truncation", 100, evidence_retained=True, model_answer="y", judged_correct=True),
        ArmRun("q3", "truncation", 100, evidence_retained=False, model_answer="z", judged_correct=False),
        ArmRun("q1", "rem", 90, evidence_retained=False, model_answer="x", judged_correct=True),
    ]
    res = aggregate(runs, n_questions=3)
    # truncation retention = 1/3 ≈ 0.33 < 0.5 -> valid even though REM retention is 0.0
    assert res.valid is True

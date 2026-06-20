"""The battery artifact must carry its own failure classification (item 6 wired
into the runner), so a sweep produces the diagnosis alongside the raw results."""

from evals.battery.aggregate import aggregate
from evals.battery.models import ArmRun
from evals.battery.run_battery_spike import build_result_payload


def test_payload_embeds_classification_block():
    runs = [
        ArmRun("q1", "truncation", 100, evidence_retained=False, model_answer="x", judged_correct=False),
        ArmRun("q1", "rem", 90, evidence_retained=False, model_answer="x", judged_correct=False,
               extraction={"attempts": 1, "failures": 0}),
        ArmRun("q2", "rem", 90, evidence_retained=False, model_answer="y", judged_correct=False,
               extraction={"attempts": 1, "failures": 0}),
    ]
    result = aggregate(runs, n_questions=2)

    payload = build_result_payload(result, budget=3000, answerer="gemma", judge="haiku")

    # Classification travels with the artifact.
    assert "classification" in payload
    c = payload["classification"]
    assert c["dominant"] == "summary_loss"
    assert c["n_rem_misses"] == 2
    assert c["recommendation"]
    assert isinstance(c["counts"], dict)

    # Existing fields are preserved.
    assert payload["budget_tokens"] == 3000
    assert payload["valid"] is True
    assert payload["arm_extraction"]["rem"]["attempts"] == 2
    assert len(payload["runs"]) == 3

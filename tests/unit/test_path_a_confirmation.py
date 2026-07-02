import json
import sys

from evals.memory_methods.artifacts import ItemRun, MemoryMethodArtifact
from evals.memory_methods.confirmation import (
    capture_preflight,
    confirmation_decision,
    load_criteria,
)
from evals.memory_methods import run_path_a_confirmation
from rem.memory.tiers import MemoryState


def _manifest():
    criteria = load_criteria()
    categories = (
        ["knowledge-update"] * 10
        + ["temporal-reasoning"] * 10
        + ["multi-session"] * 10
    )
    return {
        "source_sha256": load_criteria()["source_dataset_sha256"],
        "items": [
            {
                "question_id": criteria["suite"]["question_ids"][index],
                "category": category,
                "state_file": f"unused-{index}.json",
            }
            for index, category in enumerate(categories)
        ],
    }


def _reference(category):
    metadata = {}
    if category in {"temporal-reasoning", "multi-session"}:
        metadata = {"session_id": "s1", "timestamp": "2026-01-01"}
    return {
        "source_id": "turn:1",
        "kind": "fact",
        "turn_ids": [1],
        "metadata": metadata,
    }


def _successful_artifact(manifest):
    runs = []
    for index, item in enumerate(manifest["items"]):
        for arm in ("safe-sparse", "path-a-candidate"):
            for budget in (8000, 28000):
                scored = budget == 8000
                correct = None
                if scored:
                    correct = index >= 3 if arm == "safe-sparse" else True
                runs.append(ItemRun(
                    question_id=item["question_id"],
                    category=item["category"],
                    arm=arm,
                    budget_tokens=budget,
                    memory_tokens=100,
                    source_references=[_reference(item["category"])],
                    candidate_count=1,
                    read_latency_ms=10,
                    write_recall=True,
                    read_recall=True,
                    judged_correct=correct,
                    model_answers=["answer"] * 3 if scored else [],
                    judge_reasons=["matches gold"] * 3 if scored else [],
                ))
    return MemoryMethodArtifact(
        repository_revision="abc",
        source_manifest="manifest.json",
        source_dataset_sha256=manifest["source_sha256"],
        configuration={
            "arms": ["safe-sparse", "path-a-candidate"],
            "budgets": [8000, 28000],
            "scored_budgets": [8000],
            "answer_repetitions": 3,
            "answer_taxonomy": True,
            "confirmation_criteria_id": "path-a-confirmation-30-v1",
            "implementation_files_sha256": load_criteria()[
                "implementation_files_sha256"
            ],
        },
        runs=runs,
    )


def test_capture_preflight_reports_all_missing_then_accepts_valid_state(tmp_path):
    state_path = tmp_path / "q-state.json"
    criteria = {
        "criteria_id": "test",
        "source_dataset_sha256": "sha",
        "suite": {
            "item_count": 1,
            "categories": {"knowledge-update": 1},
            "question_ids": ["q"],
        },
        "implementation_files_sha256": {},
        "incomplete_capture_behavior": {"exit_code": 2},
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({
        "source_sha256": "sha",
        "items": [{
            "question_id": "q", "category": "knowledge-update",
            "state_file": str(state_path),
        }],
    }), encoding="utf-8")

    report = capture_preflight(str(manifest_path), criteria)
    assert report["ready"] is False
    assert report["missing_state_files"] == [str(state_path)]

    MemoryState().save(state_path)
    report = capture_preflight(str(manifest_path), criteria)
    assert report["ready"] is True
    assert report["captured_state_count"] == 1


def test_frozen_decision_requires_exact_matrix_and_three_zero_regression_wins():
    criteria = load_criteria()
    manifest = _manifest()
    artifact = _successful_artifact(manifest)
    decision = confirmation_decision(artifact, manifest, criteria)
    assert decision["promote_experimental_candidate"] is True
    assert decision["metrics"]["wins"] == 3
    assert decision["metrics"]["regressions"] == 0
    assert all(decision["checks"].values())

    artifact.runs.pop()
    decision = confirmation_decision(artifact, manifest, criteria)
    assert decision["status"] == "not-evaluable"
    assert decision["checks"]["exact_full_matrix"] is False


def test_frozen_decision_blocks_category_stale_and_abstention_regressions():
    criteria = load_criteria()
    manifest = _manifest()
    artifact = _successful_artifact(manifest)
    candidate = next(
        run for run in artifact.runs
        if run.question_id == manifest["items"][10]["question_id"]
        and run.arm == "path-a-candidate"
        and run.budget_tokens == 8000
    )
    candidate.judged_correct = False
    candidate.model_answers = ["I don't know"] * 3
    candidate.judge_reasons = ["uses the previous stale value"] * 3
    decision = confirmation_decision(artifact, manifest, criteria)
    assert decision["promote_experimental_candidate"] is False
    assert decision["checks"]["no_item_regressions"] is False
    assert decision["checks"]["no_category_regression"] is False
    assert decision["checks"]["no_new_abstentions"] is False
    assert decision["checks"]["no_new_stale_failures"] is False


def test_scored_wrapper_stops_before_run_when_capture_is_incomplete(
    tmp_path, monkeypatch,
):
    state_path = tmp_path / "missing.json"
    criteria = {
        "criteria_id": "test",
        "source_dataset_sha256": "sha",
        "suite": {
            "item_count": 1,
            "categories": {"knowledge-update": 1},
            "question_ids": ["q"],
        },
        "implementation_files_sha256": {},
        "arms": {"safe": "safe-sparse", "candidate": "path-a-candidate"},
        "budgets_tokens": [8000, 28000],
        "scored_budget_tokens": 8000,
        "answer_repetitions": 3,
        "incomplete_capture_behavior": {"exit_code": 2},
    }
    criteria_path = tmp_path / "criteria.json"
    criteria_path.write_text(json.dumps(criteria), encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({
        "source_sha256": "sha",
        "items": [{
            "question_id": "q", "category": "knowledge-update",
            "state_file": str(state_path),
        }],
    }), encoding="utf-8")
    called = False

    def fail_if_called(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("run must not start")

    monkeypatch.setattr(run_path_a_confirmation, "run_development", fail_if_called)
    monkeypatch.setattr(sys, "argv", [
        "run_path_a_confirmation.py",
        "--criteria", str(criteria_path),
        "--manifest", str(manifest_path),
        "--preflight-out", str(tmp_path / "preflight.json"),
        "--score",
    ])
    assert run_path_a_confirmation.main() == 2
    assert called is False

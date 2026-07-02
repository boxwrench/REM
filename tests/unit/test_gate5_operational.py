"""Tests for the NPU-free Gate 5 operational harness."""
import json

from evals.memory_methods.run_gate5_operational import (
    NOT_APPLICABLE,
    PASS,
    run,
)


def test_gate5_operational_checks_pass_and_write_artifact(tmp_path):
    output = tmp_path / "gate5.json"

    payload = run(output, workspace=tmp_path / "fixtures")

    assert payload["gate_status"] == PASS
    assert payload["applicable_checks_pass"] is True
    assert payload["checks"]["restart_persistence"]["status"] == PASS
    assert payload["checks"]["duplicate_ingest_idempotency"]["status"] == PASS
    assert payload["checks"]["partial_failure_atomicity"]["status"] == PASS
    assert payload["checks"]["compaction_failure_rollback"]["status"] == PASS
    assert payload["checks"]["delayed_indexing"]["status"] == NOT_APPLICABLE
    assert payload["checks"]["delayed_indexing"]["async_indexing_detected"] is False
    assert json.loads(output.read_text(encoding="utf-8")) == payload


def test_gate5_duplicate_ingest_evidence_is_exact(tmp_path):
    payload = run(tmp_path / "gate5.json", workspace=tmp_path / "fixtures")

    evidence = payload["checks"]["duplicate_ingest_idempotency"]["evidence"]
    assert evidence == {
        "turn_count_after_first_ingest": 1,
        "turn_count_after_duplicate_ingest": 1,
        "state_unchanged": True,
    }


def test_gate5_partial_failure_preserves_original_state(tmp_path):
    payload = run(tmp_path / "gate5.json", workspace=tmp_path / "fixtures")

    evidence = payload["checks"]["partial_failure_atomicity"]["evidence"]
    assert evidence == {
        "failure_propagated": True,
        "original_state_preserved": True,
        "temporary_file_removed": True,
    }


def test_gate5_compaction_failure_rolls_back_memory_and_disk(tmp_path):
    payload = run(tmp_path / "gate5.json", workspace=tmp_path / "fixtures")

    evidence = payload["checks"]["compaction_failure_rollback"]["evidence"]
    assert evidence == {
        "failed_call_count": 1,
        "reported_compacted": False,
        "in_memory_model_preserved": True,
        "in_memory_bytes_preserved": True,
        "persisted_model_preserved": True,
        "persisted_bytes_preserved": True,
    }

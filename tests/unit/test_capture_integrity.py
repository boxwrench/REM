import json

from evals.memory_methods.validate_capture_integrity import validate_manifest
from rem.memory.facts_ledger import FactEntry, FactsLedger
from rem.memory.tiers import MemoryState, SpanSummary, Turn


def _record(tmp_path, question_id, category, *, capture=True):
    record = {
        "question_id": question_id,
        "category": category,
        "state_file": str(tmp_path / f"{question_id}_state.json"),
    }
    if capture:
        record["capture"] = {
            "ingest_secs": 1.0,
            "assembled_total_tokens": 100,
            "captured_at": 123.0,
            "extraction": {"failures": 0},
        }
    return record


def _write_manifest(tmp_path, records):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"items": records}), encoding="utf-8")
    return path


def _save_path_c_state(path):
    timestamp = "2023/05/20 (Sat) 02:21"
    MemoryState(
        turns=[Turn(
            role="user",
            content="recent",
            turn_id=3,
            tokens=2,
            session_id="s2",
            timestamp=timestamp,
        )],
        summaries=[SpanSummary(
            covers_turn_ids=[1],
            text="summary",
            tokens=2,
            session_ids=["s1"],
            start_timestamp=timestamp,
            end_timestamp=timestamp,
        )],
        ledger=FactsLedger(entries=[FactEntry(
            kind="decision",
            text="fact",
            source_turn_id=1,
            session_id="s1",
            timestamp=timestamp,
        )]),
    ).save(path)


def test_valid_path_c_state_reports_full_provenance(tmp_path):
    record = _record(tmp_path, "temporal-1", "temporal-reasoning")
    _save_path_c_state(tmp_path / "temporal-1_state.json")

    report = validate_manifest(_write_manifest(tmp_path, [record]))

    assert report["summary"] == {
        "expected": 1,
        "available": 1,
        "valid": 1,
        "invalid": 0,
        "in_progress": 0,
        "missing": 0,
    }
    item = report["items"][0]
    assert item["checks"]["question_id"]["matches"] is True
    assert item["checks"]["json_loadable"] is True
    assert item["checks"]["memory_state_loadable"] is True
    assert item["checks"]["provenance"]["facts"]["coverage"] == 1.0
    assert item["checks"]["provenance"]["summaries"]["coverage"] == 1.0
    assert item["checks"]["provenance"]["recent_turns"]["coverage"] == 1.0
    assert item["checks"]["capture_metadata"]["present"] is True
    assert item["checks"]["capture_metadata"]["extraction_failures"] == 0


def test_invalid_json_and_question_filename_are_reported(tmp_path):
    record = _record(tmp_path, "expected", "temporal-reasoning")
    record["state_file"] = str(tmp_path / "wrong_state.json")
    (tmp_path / "wrong_state.json").write_text("{bad", encoding="utf-8")

    report = validate_manifest(_write_manifest(tmp_path, [record]))

    item = report["items"][0]
    assert item["status"] == "invalid"
    assert item["checks"]["question_id"]["matches"] is False
    assert item["checks"]["json_loadable"] is False
    assert any("question_id" in error for error in item["errors"])
    assert any("JSON" in error for error in item["errors"])


def test_in_progress_and_missing_are_separate(tmp_path):
    records = [
        _record(tmp_path, "active", "multi-session", capture=False),
        _record(tmp_path, "later", "multi-session", capture=False),
    ]

    report = validate_manifest(
        _write_manifest(tmp_path, records),
        in_progress_ids={"active"},
    )

    assert report["summary"]["in_progress"] == 1
    assert report["summary"]["missing"] == 1
    assert report["question_ids"]["in_progress"] == ["active"]
    assert report["question_ids"]["missing"] == ["later"]


def test_path_c_requires_provenance_and_capture_metadata(tmp_path):
    record = _record(tmp_path, "temporal-1", "temporal-reasoning", capture=False)
    MemoryState(
        turns=[Turn(role="user", content="undated", turn_id=1, tokens=2)],
    ).save(tmp_path / "temporal-1_state.json")

    report = validate_manifest(_write_manifest(tmp_path, [record]))

    item = report["items"][0]
    assert item["status"] == "invalid"
    assert any("provenance incomplete" in error for error in item["errors"])
    assert any("capture metadata missing" in error for error in item["errors"])


def test_legacy_state_provenance_gaps_are_warnings(tmp_path):
    record = _record(tmp_path, "legacy", "knowledge-update", capture=False)
    MemoryState(
        turns=[Turn(role="user", content="old", turn_id=1, tokens=1)],
        summaries=[SpanSummary(covers_turn_ids=[1], text="old", tokens=1)],
        ledger=FactsLedger(entries=[
            FactEntry(kind="decision", text="old", source_turn_id=1)
        ]),
    ).save(tmp_path / "legacy_state.json")

    report = validate_manifest(_write_manifest(tmp_path, [record]))

    item = report["items"][0]
    assert item["status"] == "valid"
    assert len(item["warnings"]) == 2


def test_path_c_extraction_failures_are_visible_warnings(tmp_path):
    record = _record(tmp_path, "temporal-1", "temporal-reasoning")
    record["capture"]["extraction"]["failures"] = 2
    _save_path_c_state(tmp_path / "temporal-1_state.json")

    report = validate_manifest(_write_manifest(tmp_path, [record]))

    item = report["items"][0]
    assert item["status"] == "valid"
    assert item["checks"]["capture_metadata"]["extraction_failures"] == 2
    assert any("2 extraction failure" in warning for warning in item["warnings"])

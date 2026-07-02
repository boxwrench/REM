"""Read-only integrity checks for resumable memory-method state captures."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rem.memory.tiers import MemoryState

PATH_C_CATEGORIES = frozenset({"temporal-reasoning", "multi-session"})
CAPTURE_METADATA_FIELDS = (
    "ingest_secs",
    "assembled_total_tokens",
    "captured_at",
    "extraction",
)


def _resolve_state_path(manifest_path: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or path.exists():
        return path
    return manifest_path.parent / path


def _coverage(objects: list[Any], fields: tuple[str, ...]) -> dict[str, Any]:
    complete = sum(
        all(getattr(obj, field, None) not in (None, [], "") for field in fields)
        for obj in objects
    )
    total = len(objects)
    return {
        "total": total,
        "complete": complete,
        "missing": total - complete,
        "coverage": complete / total if total else 1.0,
        "required_fields": list(fields),
    }


def _capture_metadata(record: dict) -> dict[str, Any]:
    capture = record.get("capture")
    missing = [
        field
        for field in CAPTURE_METADATA_FIELDS
        if not isinstance(capture, dict) or capture.get(field) is None
    ]
    return {
        "present": not missing,
        "missing_fields": missing,
        "values": capture if isinstance(capture, dict) else None,
        "extraction_failures": (
            capture.get("extraction", {}).get("failures")
            if isinstance(capture, dict)
            and isinstance(capture.get("extraction"), dict)
            else None
        ),
    }


def _validate_available(record: dict, manifest_path: Path) -> dict[str, Any]:
    question_id = str(record.get("question_id", ""))
    category = record.get("category")
    state_file = record.get("state_file")
    result: dict[str, Any] = {
        "question_id": question_id,
        "category": category,
        "state_file": state_file,
        "status": "invalid",
        "path_c_required": category in PATH_C_CATEGORIES,
        "errors": [],
        "warnings": [],
        "checks": {},
    }
    errors: list[str] = result["errors"]
    warnings: list[str] = result["warnings"]

    if not state_file:
        errors.append("manifest record has no state_file")
        return result
    path = _resolve_state_path(manifest_path, state_file)
    expected_name = f"{question_id}_state.json"
    path_matches = path.name == expected_name
    result["checks"]["question_id"] = {
        "expected": question_id,
        "expected_filename": expected_name,
        "actual_filename": path.name,
        "matches": path_matches,
    }
    if not path_matches:
        errors.append("state filename does not match expected question_id")

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        result["checks"]["json_loadable"] = True
    except (OSError, json.JSONDecodeError) as exc:
        result["checks"]["json_loadable"] = False
        errors.append(f"state JSON is not loadable: {exc}")
        return result

    embedded_id = raw.get("question_id") if isinstance(raw, dict) else None
    result["checks"]["question_id"]["embedded"] = embedded_id
    if embedded_id is not None and str(embedded_id) != question_id:
        errors.append("embedded question_id does not match manifest")

    structure = {
        "schema_version_present": isinstance(raw, dict) and "schema_version" in raw,
        "turns_present": isinstance(raw, dict) and isinstance(raw.get("turns"), list),
        "summaries_present": (
            isinstance(raw, dict) and isinstance(raw.get("summaries"), list)
        ),
        "ledger_present": isinstance(raw, dict) and isinstance(raw.get("ledger"), dict),
        "ledger_entries_present": (
            isinstance(raw, dict)
            and isinstance(raw.get("ledger"), dict)
            and isinstance(raw["ledger"].get("entries"), list)
        ),
    }
    result["checks"]["extraction_state_structure"] = structure
    missing_structure = [name for name, present in structure.items() if not present]
    if missing_structure:
        errors.append(
            "state is missing extraction/state fields: " + ", ".join(missing_structure)
        )

    try:
        state = MemoryState.load(path)
        result["checks"]["memory_state_loadable"] = True
    except (OSError, ValueError, TypeError) as exc:
        result["checks"]["memory_state_loadable"] = False
        errors.append(f"MemoryState is not loadable: {exc}")
        return result

    provenance = {
        "facts": _coverage(
            state.ledger.entries,
            ("session_id", "timestamp"),
        ),
        "summaries": _coverage(
            state.summaries,
            ("session_ids", "start_timestamp", "end_timestamp"),
        ),
        "recent_turns": _coverage(
            state.turns,
            ("session_id", "timestamp"),
        ),
    }
    result["checks"]["provenance"] = provenance
    result["checks"]["extraction_counts"] = {
        "facts": len(state.ledger.entries),
        "summaries": len(state.summaries),
        "recent_turns": len(state.turns),
    }

    metadata = _capture_metadata(record)
    result["checks"]["capture_metadata"] = metadata

    incomplete_tiers = [
        tier for tier, coverage in provenance.items() if coverage["missing"]
    ]
    if result["path_c_required"]:
        if incomplete_tiers:
            errors.append(
                "Path C provenance incomplete for: " + ", ".join(incomplete_tiers)
            )
        if not metadata["present"]:
            errors.append(
                "Path C capture metadata missing: "
                + ", ".join(metadata["missing_fields"])
            )
        elif metadata["extraction_failures"]:
            warnings.append(
                f"capture recorded {metadata['extraction_failures']} "
                "extraction failure(s)"
            )
    else:
        if incomplete_tiers:
            warnings.append(
                "legacy capture has incomplete provenance for: "
                + ", ".join(incomplete_tiers)
            )
        if not metadata["present"]:
            warnings.append(
                "legacy capture metadata missing: "
                + ", ".join(metadata["missing_fields"])
            )

    result["status"] = "invalid" if errors else "valid"
    return result


def validate_manifest(
    manifest: str | Path,
    *,
    in_progress_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Inspect manifest states without modifying state or manifest files."""
    manifest_path = Path(manifest)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = payload.get("items") if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        raise ValueError("manifest must be a list or an object with an items list")
    in_progress_ids = in_progress_ids or set()

    items = []
    for record in records:
        question_id = str(record.get("question_id", ""))
        state_file = record.get("state_file")
        state_path = (
            _resolve_state_path(manifest_path, state_file) if state_file else None
        )
        if question_id in in_progress_ids:
            items.append({
                "question_id": question_id,
                "category": record.get("category"),
                "state_file": state_file,
                "status": "in_progress",
                "path_c_required": record.get("category") in PATH_C_CATEGORIES,
                "errors": [],
                "warnings": [],
                "checks": {"state_exists": bool(state_path and state_path.exists())},
            })
        elif state_path is None or not state_path.exists():
            items.append({
                "question_id": question_id,
                "category": record.get("category"),
                "state_file": state_file,
                "status": "missing",
                "path_c_required": record.get("category") in PATH_C_CATEGORIES,
                "errors": [],
                "warnings": [],
                "checks": {"state_exists": False},
            })
        else:
            items.append(_validate_available(record, manifest_path))

    by_status = {
        status: [item["question_id"] for item in items if item["status"] == status]
        for status in ("valid", "invalid", "in_progress", "missing")
    }
    summary = {
        "expected": len(items),
        "available": len(by_status["valid"]) + len(by_status["invalid"]),
        **{status: len(ids) for status, ids in by_status.items()},
    }
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "manifest": str(manifest_path),
        "summary": summary,
        "question_ids": by_status,
        "items": items,
    }


def _print_report(report: dict) -> None:
    summary = report["summary"]
    print(
        "capture-integrity: "
        f"expected={summary['expected']} available={summary['available']} "
        f"valid={summary['valid']} invalid={summary['invalid']} "
        f"in_progress={summary['in_progress']} missing={summary['missing']}"
    )
    for status in ("invalid", "in_progress", "missing"):
        ids = report["question_ids"][status]
        if ids:
            print(f"{status}: {ids}")
    for item in report["items"]:
        if item["status"] == "invalid":
            print(f"[{item['question_id']}] " + "; ".join(item["errors"]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default="bench/memory_methods/development_manifest.json",
    )
    parser.add_argument(
        "--in-progress-id",
        action="append",
        default=[],
        help="question ID currently being captured; repeat when needed",
    )
    parser.add_argument("--out", help="optional JSON report path")
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="also fail when any state is missing or in progress",
    )
    args = parser.parse_args()

    report = validate_manifest(
        args.manifest,
        in_progress_ids=set(args.in_progress_id),
    )
    _print_report(report)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if report["summary"]["invalid"]:
        return 1
    if args.require_complete and (
        report["summary"]["missing"] or report["summary"]["in_progress"]
    ):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

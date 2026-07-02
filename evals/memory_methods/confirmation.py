"""Frozen Path-A confirmation preflight and paired decision logic."""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from evals.memory_methods.artifacts import MemoryMethodArtifact
from rem.memory.tiers import MemoryState


DEFAULT_CRITERIA = "bench/memory_methods/path_a_confirmation_criteria.json"

_ABSTENTION_PATTERNS = (
    re.compile(r"\b(?:cannot|can't|unable to) (?:answer|determine|find|tell)\b", re.I),
    re.compile(r"\b(?:not enough|insufficient) (?:context|information|memory)\b", re.I),
    re.compile(r"\b(?:memory|context) (?:does not|doesn't) (?:contain|mention|provide)\b", re.I),
    re.compile(r"\b(?:i do not|i don't) know\b", re.I),
)
_STALE_REASON = re.compile(r"\b(?:stale|outdated|older|previous)\b", re.I)


def load_criteria(path: str = DEFAULT_CRITERIA) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def capture_preflight(
    manifest_path: str, criteria: dict[str, Any]
) -> dict[str, Any]:
    """Validate frozen manifest shape and every captured state without model calls."""
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    items = manifest.get("items", [])
    expected = criteria["suite"]
    expected_categories = expected["categories"]
    qids = [item.get("question_id") for item in items]
    category_counts = Counter(item.get("category") for item in items)

    protocol_errors = []
    if manifest.get("source_sha256") != criteria["source_dataset_sha256"]:
        protocol_errors.append("source dataset SHA-256 differs from frozen criteria")
    if len(items) != expected["item_count"]:
        protocol_errors.append(
            f"manifest has {len(items)} items; expected {expected['item_count']}"
        )
    if len(set(qids)) != len(qids):
        protocol_errors.append("manifest contains duplicate question IDs")
    if qids != expected["question_ids"]:
        protocol_errors.append("manifest question IDs/order differ from frozen criteria")
    if dict(category_counts) != expected_categories:
        protocol_errors.append(
            f"category coverage is {dict(category_counts)}; expected {expected_categories}"
        )

    implementation_hashes = {
        path: file_sha256(path)
        for path in criteria["implementation_files_sha256"]
        if Path(path).is_file()
    }
    for path, expected_hash in criteria["implementation_files_sha256"].items():
        actual_hash = implementation_hashes.get(path)
        if actual_hash is None:
            protocol_errors.append(f"frozen implementation file is missing: {path}")
        elif actual_hash != expected_hash:
            protocol_errors.append(f"frozen implementation file changed: {path}")

    missing = []
    invalid = []
    for item in items:
        state_path = Path(item["state_file"])
        if not state_path.is_file():
            missing.append(str(state_path))
            continue
        try:
            MemoryState.load(state_path)
        except Exception as exc:  # report every bad capture in one preflight
            invalid.append({
                "state_file": str(state_path),
                "error": f"{type(exc).__name__}: {exc}",
            })

    ready = not protocol_errors and not missing and not invalid
    return {
        "criteria_id": criteria["criteria_id"],
        "ready": ready,
        "status": "ready" if ready else "capture-incomplete",
        "manifest": manifest_path,
        "manifest_item_count": len(items),
        "category_counts": dict(category_counts),
        "captured_state_count": len(items) - len(missing) - len(invalid),
        "missing_state_files": missing,
        "invalid_state_files": invalid,
        "protocol_errors": protocol_errors,
        "implementation_files_sha256": implementation_hashes,
        "behavior": criteria["incomplete_capture_behavior"],
    }


def _is_abstention(answers: list[str]) -> bool:
    if not answers:
        return False
    matches = sum(
        any(pattern.search(answer or "") for pattern in _ABSTENTION_PATTERNS)
        for answer in answers
    )
    return matches > len(answers) / 2


def _is_stale_failure(correct: bool | None, reasons: list[str]) -> bool:
    if correct is not False or not reasons:
        return False
    return sum(bool(_STALE_REASON.search(reason)) for reason in reasons) > len(reasons) / 2


def _has_temporal_provenance(reference: dict[str, Any]) -> bool:
    metadata = reference.get("metadata") or {}
    has_session = bool(metadata.get("session_id") or metadata.get("session_ids"))
    has_time = bool(metadata.get("timestamp") or metadata.get("start_timestamp"))
    return has_session and has_time


def confirmation_decision(
    artifact: MemoryMethodArtifact,
    manifest: dict[str, Any],
    criteria: dict[str, Any],
) -> dict[str, Any]:
    """Apply the frozen criteria; malformed or partial matrices cannot promote."""
    arms = criteria["arms"]
    safe_arm = arms["safe"]
    candidate_arm = arms["candidate"]
    budgets = criteria["budgets_tokens"]
    primary_budget = criteria["scored_budget_tokens"]
    repetitions = criteria["answer_repetitions"]
    manifest_items = {item["question_id"]: item for item in manifest["items"]}
    manifest_matches = (
        manifest.get("source_sha256") == criteria["source_dataset_sha256"]
        and [item["question_id"] for item in manifest["items"]]
        == criteria["suite"]["question_ids"]
        and dict(Counter(item["category"] for item in manifest["items"]))
        == criteria["suite"]["categories"]
    )
    expected_keys = {
        (qid, arm, budget)
        for qid in manifest_items
        for arm in (safe_arm, candidate_arm)
        for budget in budgets
    }

    rows: dict[tuple[str, str, int], Any] = {}
    duplicates = []
    unexpected = []
    category_mismatches = []
    for run in artifact.runs:
        key = (run.question_id, run.arm, run.budget_tokens)
        if key not in expected_keys:
            unexpected.append(key)
            continue
        if key in rows:
            duplicates.append(key)
            continue
        rows[key] = run
        expected_category = manifest_items[run.question_id]["category"]
        if run.category != expected_category:
            category_mismatches.append({
                "key": key,
                "actual": run.category,
                "expected": expected_category,
            })

    missing = sorted(expected_keys - rows.keys())
    exact_coverage = not missing and not duplicates and not unexpected
    metadata_matches = (
        artifact.source_dataset_sha256 == criteria["source_dataset_sha256"]
        and artifact.models.answer == criteria["models"]["answer"]
        and artifact.models.judge == criteria["models"]["judge"]
        and artifact.configuration.get("arms") == [safe_arm, candidate_arm]
        and artifact.configuration.get("budgets") == budgets
        and artifact.configuration.get("scored_budgets") == [primary_budget]
        and artifact.configuration.get("answer_repetitions") == repetitions
        and artifact.configuration.get("answer_taxonomy") is True
        and artifact.configuration.get("confirmation_criteria_id")
        == criteria["criteria_id"]
        and artifact.configuration.get("implementation_files_sha256")
        == criteria["implementation_files_sha256"]
    )
    scored_rows_complete = exact_coverage and all(
        rows[(qid, arm, primary_budget)].judged_correct is not None
        and len(rows[(qid, arm, primary_budget)].model_answers) == repetitions
        and len(rows[(qid, arm, primary_budget)].judge_reasons) == repetitions
        for qid in manifest_items
        for arm in (safe_arm, candidate_arm)
    )

    wins = []
    regressions = []
    new_abstentions = []
    new_stale_failures = []
    gold_source_losses = []
    write_recall_losses = []
    empty_retrieval_regressions = []
    category_delta: dict[str, int] = defaultdict(int)
    if exact_coverage:
        for qid, item in manifest_items.items():
            safe = rows[(qid, safe_arm, primary_budget)]
            candidate = rows[(qid, candidate_arm, primary_budget)]
            if safe.judged_correct is False and candidate.judged_correct is True:
                wins.append(qid)
            if safe.judged_correct is True and candidate.judged_correct is False:
                regressions.append(qid)
            if safe.judged_correct is not None and candidate.judged_correct is not None:
                category_delta[item["category"]] += (
                    int(candidate.judged_correct) - int(safe.judged_correct)
                )
            if _is_abstention(candidate.model_answers) and not _is_abstention(
                safe.model_answers
            ):
                new_abstentions.append(qid)
            if _is_stale_failure(
                candidate.judged_correct, candidate.judge_reasons
            ) and not _is_stale_failure(safe.judged_correct, safe.judge_reasons):
                new_stale_failures.append(qid)

            for budget in budgets:
                safe_budget = rows[(qid, safe_arm, budget)]
                candidate_budget = rows[(qid, candidate_arm, budget)]
                if safe_budget.read_recall is True and candidate_budget.read_recall is not True:
                    gold_source_losses.append({"question_id": qid, "budget": budget})
                if safe_budget.write_recall is True and candidate_budget.write_recall is not True:
                    write_recall_losses.append({"question_id": qid, "budget": budget})
                if safe_budget.candidate_count > 0 and candidate_budget.candidate_count == 0:
                    empty_retrieval_regressions.append({
                        "question_id": qid, "budget": budget,
                    })

    all_expected_rows = list(rows.values()) if exact_coverage else []
    candidate_rows = [run for run in all_expected_rows if run.arm == candidate_arm]
    p95_by_budget = {}
    for budget in budgets:
        values = sorted(
            run.read_latency_ms for run in candidate_rows
            if run.budget_tokens == budget
        )
        index = max(0, int(len(values) * 0.95 + 0.999) - 1)
        p95_by_budget[str(budget)] = values[index] if values else None

    temporal_categories = set(criteria["safety"]["provenance_categories"])
    temporal_refs = [
        reference
        for run in all_expected_rows
        if run.category in temporal_categories
        for reference in run.source_references
    ]
    checks = {
        "manifest_matches_frozen_suite": manifest_matches,
        "run_metadata_matches_frozen_protocol": metadata_matches,
        "exact_full_matrix": exact_coverage,
        "categories_match_manifest": not category_mismatches,
        "primary_scores_complete": scored_rows_complete,
        "minimum_primary_wins": len(wins) >= criteria["comparison"]["minimum_wins"],
        "no_item_regressions": len(regressions) <= criteria[
            "comparison"
        ]["maximum_item_regressions"],
        "no_category_regression": all(
            category_delta.get(category, 0)
            >= -criteria["comparison"]["maximum_category_regressions"]
            for category in criteria["suite"]["categories"]
        ) if scored_rows_complete else False,
        "no_new_abstentions": len(new_abstentions)
        <= criteria["safety"]["maximum_new_abstentions"],
        "no_new_stale_failures": len(new_stale_failures)
        <= criteria["safety"]["maximum_new_stale_failures"],
        "no_gold_source_loss": len(gold_source_losses)
        <= criteria["safety"]["maximum_gold_source_losses"],
        "no_write_recall_loss": not write_recall_losses,
        "write_recall_measured": exact_coverage and all(
            run.write_recall is not None for run in all_expected_rows
        ),
        "no_empty_retrieval_regression": len(empty_retrieval_regressions)
        <= criteria["safety"]["maximum_empty_retrieval_regressions"],
        "no_run_errors": exact_coverage and not any(run.error for run in all_expected_rows),
        "within_context_budgets": exact_coverage and all(
            run.memory_tokens <= run.budget_tokens and not run.context_overflow
            for run in all_expected_rows
        ),
        "no_provenance_loss": exact_coverage and all(
            not run.provenance_lost
            and run.candidate_count == len(run.source_references)
            for run in all_expected_rows
        ),
        "temporal_provenance_complete": bool(temporal_refs) and all(
            _has_temporal_provenance(reference) for reference in temporal_refs
        ),
        "candidate_latency_within_limit": exact_coverage and all(
            value is not None and value <= criteria["safety"]["read_latency_p95_ms"]
            for value in p95_by_budget.values()
        ),
    }
    promotable = all(checks.values())
    status = "promote-experimental-candidate" if promotable else (
        "not-evaluable" if not exact_coverage or not scored_rows_complete
        else "keep-safe-sparse"
    )
    return {
        "criteria_id": criteria["criteria_id"],
        "status": status,
        "promote_experimental_candidate": promotable,
        "checks": checks,
        "metrics": {
            "wins": len(wins),
            "win_question_ids": wins,
            "regressions": len(regressions),
            "regression_question_ids": regressions,
            "category_delta_items": dict(category_delta),
            "new_abstention_question_ids": new_abstentions,
            "new_stale_failure_question_ids": new_stale_failures,
            "gold_source_losses": gold_source_losses,
            "write_recall_losses": write_recall_losses,
            "empty_retrieval_regressions": empty_retrieval_regressions,
            "candidate_read_latency_p95_ms": p95_by_budget,
        },
        "coverage": {
            "expected_runs": len(expected_keys),
            "observed_runs": len(rows),
            "missing": missing,
            "duplicates": duplicates,
            "unexpected": unexpected,
            "category_mismatches": category_mismatches,
        },
    }

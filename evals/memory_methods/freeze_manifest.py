"""Freeze the balanced 30-item LongMemEval-S development manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from evals.battery.longmemeval_loader import MEMORY_METHOD_CATEGORIES

DEFAULT_EXCLUDED = {"031748ae"}


def _recency(entry: dict) -> float:
    ids = entry["haystack_session_ids"]
    positions = [ids.index(value) for value in entry.get("answer_session_ids", []) if value in ids]
    if not positions or len(ids) <= 1:
        return 1.0
    return max(positions) / (len(ids) - 1)


def _gold_turn_groups(entry: dict) -> list[list[int]]:
    """Map each gold session to the one-based turn IDs used by REM capture."""
    answer_ids = set(entry.get("answer_session_ids", []))
    groups = []
    next_turn_id = 1
    for session_id, turns in zip(
        entry["haystack_session_ids"], entry["haystack_sessions"]
    ):
        turn_ids = list(range(next_turn_id, next_turn_id + len(turns)))
        if session_id in answer_ids:
            groups.append(turn_ids)
        next_turn_id += len(turns)
    return groups


def build_manifest(
    raw: list[dict],
    *,
    per_category: int = 10,
    excluded: set[str] | None = None,
) -> list[dict]:
    excluded = DEFAULT_EXCLUDED if excluded is None else excluded
    selected = []
    for category in MEMORY_METHOD_CATEGORIES:
        candidates = [
            entry for entry in raw
            if entry.get("question_type") == category
            and entry.get("question_id") not in excluded
        ]
        # Prefer evidence distributed across sessions, then older evidence;
        # question_id makes the freeze deterministic when scores tie.
        candidates.sort(key=lambda entry: (
            len(entry.get("answer_session_ids", [])) < 2,
            _recency(entry),
            entry["question_id"],
        ))
        if len(candidates) < per_category:
            raise ValueError(
                f"need {per_category} unambiguous {category} items, found {len(candidates)}"
            )
        for entry in candidates[:per_category]:
            question_id = entry["question_id"]
            selected.append({
                "question_id": question_id,
                "category": category,
                "question": entry["question"],
                "answer": entry["answer"],
                "answer_session_ids": entry.get("answer_session_ids", []),
                "gold_source_turn_groups": _gold_turn_groups(entry),
                "gold_recency": _recency(entry),
                "distributed_evidence": len(entry.get("answer_session_ids", [])) >= 2,
                "n_sessions": len(entry["haystack_session_ids"]),
                "state_file": f"bench/memory_methods/states/{question_id}_state.json",
            })
    return selected


def freeze(data_path: str, output_path: str, excluded: set[str] | None = None) -> None:
    source = Path(data_path)
    raw_bytes = source.read_bytes()
    items = build_manifest(json.loads(raw_bytes), excluded=excluded)
    payload = {
        "schema_version": 1,
        "suite": "rem-memory-methods-development",
        "development_only": True,
        "source_file": source.name,
        "source_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "selection": {
            "per_category": 10,
            "categories": list(MEMORY_METHOD_CATEGORIES),
            "excluded_ambiguous_question_ids": sorted(
                DEFAULT_EXCLUDED if excluded is None else excluded
            ),
            "preference": "distributed evidence, then oldest gold, then question_id",
        },
        "items": items,
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument(
        "--out", default="bench/memory_methods/development_manifest.json"
    )
    parser.add_argument("--exclude", action="append", default=[])
    args = parser.parse_args()
    excluded = DEFAULT_EXCLUDED | set(args.exclude)
    freeze(args.data, args.out, excluded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

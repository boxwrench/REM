"""Preflight and run the frozen 30-item Path-A confirmation protocol."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from evals.memory_methods.artifacts import MemoryMethodArtifact
from evals.memory_methods.confirmation import (
    DEFAULT_CRITERIA,
    capture_preflight,
    confirmation_decision,
    file_sha256,
    load_criteria,
)
from evals.memory_methods.run_development import run as run_development


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--criteria", default=DEFAULT_CRITERIA)
    parser.add_argument(
        "--manifest", default="bench/memory_methods/development_manifest.json"
    )
    parser.add_argument(
        "--preflight-out",
        default="bench/memory_methods/path_a_confirmation_preflight.json",
    )
    parser.add_argument(
        "--out", default="bench/memory_methods/path_a_confirmation_run.json"
    )
    parser.add_argument(
        "--decision-out",
        default="bench/memory_methods/path_a_confirmation_decision.json",
    )
    parser.add_argument(
        "--run", action="store_true",
        help="Run the recall-only matrix after preflight. Default is preflight only.",
    )
    parser.add_argument(
        "--score", action="store_true",
        help="Run fixed answerer/judge repetitions at the frozen scored budget.",
    )
    args = parser.parse_args()

    criteria = load_criteria(args.criteria)
    preflight = capture_preflight(args.manifest, criteria)
    preflight_path = Path(args.preflight_out)
    preflight_path.parent.mkdir(parents=True, exist_ok=True)
    preflight_path.write_text(json.dumps(preflight, indent=2), encoding="utf-8")
    if not preflight["ready"]:
        print(
            "[path-a-confirmation] capture incomplete: "
            f"{len(preflight['missing_state_files'])} missing, "
            f"{len(preflight['invalid_state_files'])} invalid; no run started",
            flush=True,
        )
        return 2
    if not args.run and not args.score:
        print("[path-a-confirmation] preflight ready; no run requested", flush=True)
        return 0

    criteria_hash = file_sha256(args.criteria)
    run_development(
        args.manifest,
        args.out,
        [criteria["arms"]["safe"], criteria["arms"]["candidate"]],
        criteria["budgets_tokens"],
        score=args.score,
        score_budgets=[criteria["scored_budget_tokens"]],
        answer_repetitions=criteria["answer_repetitions"],
        configuration_extra={
            "confirmation_criteria_id": criteria["criteria_id"],
            "confirmation_criteria_sha256": criteria_hash,
            "answer_taxonomy": True,
            "implementation_files_sha256": preflight[
                "implementation_files_sha256"
            ],
        },
    )
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    artifact = MemoryMethodArtifact.model_validate_json(
        Path(args.out).read_text(encoding="utf-8")
    )
    decision = confirmation_decision(artifact, manifest, criteria)
    decision["criteria_sha256"] = criteria_hash
    Path(args.decision_out).write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )
    print(
        f"[path-a-confirmation] {decision['status']} -> {args.decision_out}",
        flush=True,
    )
    return 0 if decision["status"] != "not-evaluable" else 3


if __name__ == "__main__":
    raise SystemExit(main())

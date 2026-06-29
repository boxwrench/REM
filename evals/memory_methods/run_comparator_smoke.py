"""Run a three-item local viability smoke for one pinned external comparator."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from dataclasses import asdict
from pathlib import Path

from evals.battery.longmemeval_loader import MEMORY_METHOD_CATEGORIES, load_categories
from evals.memory_methods.artifacts import ItemRun, MemoryMethodArtifact, ModelVersions
from evals.memory_methods.external import HindsightArm, SupermemoryArm


def _make_arm(provider: str, base_url: str):
    if provider == "hindsight":
        from hindsight_client import Hindsight
        return HindsightArm(Hindsight(base_url=base_url))
    from supermemory import Supermemory
    kwargs = {
        "api_key": os.environ.get("SUPERMEMORY_API_KEY", "local"),
        "base_url": base_url,
    }
    return SupermemoryArm(Supermemory(**kwargs))


def run(provider: str, revision: str, base_url: str, data: str, manifest: str,
        output: str, timeout: float = 600) -> int:
    if not revision.strip():
        raise ValueError("a pinned comparator revision is required")
    manifest_payload = json.loads(Path(manifest).read_text(encoding="utf-8"))
    dataset_sha = hashlib.sha256(Path(data).read_bytes()).hexdigest()
    if dataset_sha != manifest_payload["source_sha256"]:
        raise ValueError("dataset SHA-256 does not match the frozen manifest")
    selected = manifest_payload["items"][:3]
    source_items = {
        item.question_id: item
        for item in load_categories(data, MEMORY_METHOD_CATEGORIES)
    }
    repository_revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()
    try:
        arm = _make_arm(provider, base_url)
    except Exception as exc:
        unavailable = MemoryMethodArtifact(
            repository_revision=repository_revision,
            system_revision=revision,
            source_manifest=manifest,
            source_dataset_sha256=dataset_sha,
            models=ModelVersions(),
            configuration={
                "provider": provider, "base_url": base_url,
                "smoke_items": 3, "native_configuration": True,
                "status": "unsupported",
                "setup_error": f"{type(exc).__name__}: {exc}",
            },
        )
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(unavailable.model_dump_json(indent=2), encoding="utf-8")
        return 2
    runs = []
    cleanup_errors = []
    for record in selected:
        item = source_items[record["question_id"]]
        namespace = f"rem-eval-{item.question_id}"
        completed_budgets = set()
        try:
            arm.ingest(namespace, item.sessions)
            arm.await_ready(namespace, timeout)
            for budget in (8000, 28000):
                result = arm.recall(namespace, item.question, budget)
                runs.append(ItemRun(
                    question_id=item.question_id,
                    category=item.question_type,
                    arm=provider,
                    budget_tokens=budget,
                    memory_tokens=result.token_count,
                    source_references=[asdict(ref) for ref in result.source_references],
                    candidate_count=result.candidate_count,
                    ingest_latency_ms=arm.stats(namespace).get("ingest_latency_ms"),
                    read_latency_ms=result.latency_ms,
                    context_overflow=result.token_count > budget,
                    provenance_lost=(
                        result.candidate_count > 0
                        and (
                            not result.source_references
                            or any(
                                ref.metadata.get("provenance") == "memory_only"
                                for ref in result.source_references
                            )
                        )
                    ),
                ))
                completed_budgets.add(budget)
        except Exception as exc:
            for budget in sorted({8000, 28000} - completed_budgets):
                runs.append(ItemRun(
                    question_id=item.question_id,
                    category=item.question_type,
                    arm=provider,
                    budget_tokens=budget,
                    memory_tokens=0,
                    candidate_count=0,
                    read_latency_ms=0,
                    error=f"{type(exc).__name__}: {exc}",
                ))
        finally:
            try:
                arm.reset(namespace)
            except Exception as exc:
                cleanup_errors.append(f"{namespace}: {type(exc).__name__}: {exc}")
    artifact = MemoryMethodArtifact(
        repository_revision=repository_revision,
        system_revision=revision,
        source_manifest=manifest,
        source_dataset_sha256=dataset_sha,
        models=ModelVersions(),
        configuration={
            "provider": provider,
            "base_url": base_url,
            "smoke_items": 3,
            "native_configuration": True,
            "status": "completed" if not any(run.error for run in runs) else "failed",
            "cleanup_errors": cleanup_errors,
        },
        runs=runs,
    )
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(artifact.model_dump_json(indent=2), encoding="utf-8")
    return 1 if cleanup_errors else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", required=True, choices=["hindsight", "supermemory"])
    parser.add_argument("--revision", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument(
        "--manifest", default="bench/memory_methods/development_manifest.json"
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--timeout", type=float, default=600)
    args = parser.parse_args()
    return run(
        args.provider, args.revision, args.base_url, args.data, args.manifest,
        args.out, args.timeout,
    )


if __name__ == "__main__":
    raise SystemExit(main())

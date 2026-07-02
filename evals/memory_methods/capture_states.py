"""Capture every development item once for paired read-path experiments."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from evals.battery.capture_states import capture_item
from evals.battery.longmemeval_loader import MEMORY_METHOD_CATEGORIES, load_categories
from evals.battery.context_managers import RemContextManager
from rem.config import Settings
from rem.npu_client import NpuClient

GEMMA = "gemma4-it:e2b"


def run(data: str, manifest: str, budget_tokens: int = 1000, make_cm=None,
        limit: int | None = None) -> int:
    manifest_path = Path(manifest)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    actual_sha = hashlib.sha256(Path(data).read_bytes()).hexdigest()
    if actual_sha != payload["source_sha256"]:
        raise ValueError("dataset SHA-256 does not match the frozen manifest")
    wanted = {item["question_id"]: item for item in payload["items"]}
    source_items = {
        item.question_id: item
        for item in load_categories(data, MEMORY_METHOD_CATEGORIES)
        if item.question_id in wanted
    }
    missing = sorted(set(wanted) - set(source_items))
    if missing:
        raise ValueError(f"manifest IDs missing from dataset: {missing}")
    if make_cm is None:
        client = NpuClient(Settings(summarizer_model=GEMMA))

        def make_cm():
            # The capture-time ceiling only gates ingest's final DIAGNOSTIC assemble
            # (self._assembled, used for stats/evidence). Temporal/multi items render a
            # full ledger > 64k and crashed capture before the (valid) state was saved.
            # Raise it so any item ingests and the full immutable state persists; the
            # read path fits it down at query time. Does not change captured content.
            return RemContextManager(client, Settings(
                summarizer_model=GEMMA, max_context_tokens=1_000_000
            ))
    captured = 0
    for question_id, record in wanted.items():
        state_path = Path(record["state_file"])
        if state_path.exists():
            print(f"[{question_id}] state exists, skipping")
            continue
        if limit is not None and captured >= limit:
            remaining = sum(
                1 for it in wanted.values()
                if not Path(it["state_file"]).exists()
            )
            print(f"limit reached ({limit} new capture(s) this run); "
                  f"{remaining} item(s) still uncaptured")
            break
        result = capture_item(
            source_items[question_id], state_path.parent, make_cm, budget_tokens
        )
        produced = Path(result["state_file"])
        if produced != state_path:
            state_path.parent.mkdir(parents=True, exist_ok=True)
            produced.replace(state_path)
        record["capture"] = {
            "ingest_secs": result["ingest_secs"],
            "assembled_total_tokens": result["assembled_total_tokens"],
            "captured_at": result["captured_at"],
            "extraction": result["extraction"],
        }
        manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        captured += 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument(
        "--manifest", default="bench/memory_methods/development_manifest.json"
    )
    parser.add_argument("--budget", type=int, default=1000)
    parser.add_argument(
        "--limit", type=int, default=None,
        help="max NEW captures this run (skips of existing states don't count); "
             "resumable, so repeated runs advance through the manifest",
    )
    args = parser.parse_args()
    return run(args.data, args.manifest, args.budget, limit=args.limit)


if __name__ == "__main__":
    raise SystemExit(main())

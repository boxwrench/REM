"""Triggered embedding-identity experiment for semantic slot fragmentation."""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from collections.abc import Callable
from pathlib import Path

from rem.config import Settings
from rem.npu_client import NpuClient


def cosine(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    denominator = math.sqrt(sum(a * a for a in left)) * math.sqrt(
        sum(b * b for b in right)
    )
    return numerator / denominator if denominator else 0.0


def evaluate_pairs(
    pairs: list[dict], embed: Callable[[list[str]], list[list[float]]]
) -> dict:
    texts = []
    for pair in pairs:
        texts.extend([pair["left"], pair["right"]])
    started = time.perf_counter()
    vectors = embed(texts)
    latency_ms = round((time.perf_counter() - started) * 1000, 3)
    if len(vectors) != len(texts):
        raise ValueError(f"embedder returned {len(vectors)} vectors for {len(texts)} texts")
    scored = []
    for index, pair in enumerate(pairs):
        scored.append({**pair, "similarity": round(
            cosine(vectors[index * 2], vectors[index * 2 + 1]), 6
        )})
    thresholds = sorted({row["similarity"] for row in scored}, reverse=True)
    thresholds.append(-1.0)
    candidates = []
    positives = sum(bool(row["same_slot"]) for row in scored)
    for threshold in thresholds:
        true_merges = sum(
            row["same_slot"] and row["similarity"] >= threshold for row in scored
        )
        false_merges = sum(
            not row["same_slot"] and row["similarity"] >= threshold for row in scored
        )
        candidates.append({
            "threshold": threshold,
            "true_merges": true_merges,
            "false_merges": false_merges,
            "same_slot_recall": round(true_merges / positives, 4) if positives else 0.0,
        })
    safe = [candidate for candidate in candidates if candidate["false_merges"] == 0]
    best = max(safe, key=lambda candidate: (
        candidate["true_merges"], -candidate["threshold"]
    )) if safe else None
    return {
        "pairs": scored,
        "best_zero_false_merge_threshold": best,
        "embedding_latency_ms": latency_ms,
        "n_texts": len(texts),
    }


def run(fixtures: str, output: str, model: str) -> int:
    payload = json.loads(Path(fixtures).read_text(encoding="utf-8"))
    pairs = payload["embedding_identity_pairs"]
    client = NpuClient(Settings(embedding_model=model))
    result = evaluate_pairs(pairs, lambda texts: client.embed(texts, model=model))
    artifact = {
        "schema_version": 1,
        "experiment": "embedding-slot-identity",
        "repository_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "model": model,
        "fixtures": fixtures,
        "created_at": time.time(),
        **result,
    }
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixtures", default="bench/memory_methods/diagnostic_fixtures.json"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    return run(args.fixtures, args.out, args.model)


if __name__ == "__main__":
    raise SystemExit(main())

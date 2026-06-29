"""Gate 4 — Qwen embedding-identity baseline using a LOCAL HF embedder.

The FLM NPU server returns null embeddings for its generative models, so this
baseline runs a real embedding model (Qwen/Qwen3-Embedding-0.6B) locally via
sentence-transformers and feeds it to the existing ``evaluate_pairs`` harness
(which already accepts any ``embed`` callable). It reports the best zero-false-merge
cosine threshold and same-slot recall over the canonical 6-pair fixture set, and,
separately, over an EXTENDED set of real fragmented slot keys drawn from the
captured-state write-recall audit — the semantic merges string-first
canonicalization could not reach.

Deterministic (temperature-free; encoding is fixed given the model weights).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

from evals.memory_methods.embedding_identity import evaluate_pairs

MODEL = "Qwen/Qwen3-Embedding-0.6B"

# Extended real-key pairs from the captured states (FINDINGS write-recall audit).
# same_slot=True  -> semantically the SAME concept, fragmented across keys
#                    (string-first canonicalization left these split).
# same_slot=False -> genuinely DISTINCT concepts that must never collapse
#                    (the traps: team size vs team-outing headcount; goal vs level).
EXTENDED_PAIRS = [
    # team size (the "5 engineers" concept) fragmented across three keys
    {"id": "team-size-a", "left": "team.size", "right": "team size.size", "same_slot": True},
    {"id": "team-size-b", "left": "team size.size", "right": "group size.number of engineers", "same_slot": True},
    # coding time/day fragmented across four keys
    {"id": "coding-a", "left": "coding exercises.time per day", "right": "coding exercises.duration", "same_slot": True},
    {"id": "coding-b", "left": "coding exercises.time spent per day", "right": "coding exercises.frequency", "same_slot": True},
    # Apex goal value (150) fragmented across keys
    {"id": "apex-goal-a", "left": "goal.level", "right": "level.target level for goal setting", "same_slot": True},
    {"id": "apex-goal-b", "left": "goal.level", "right": "user.goal", "same_slot": True},
    # --- traps that must NOT merge ---
    {"id": "trap-team-outing", "left": "team.size", "right": "team members.count", "same_slot": False},  # 5-engineer team vs 4+Rachel outing headcount (031748ae)
    {"id": "trap-goal-vs-level", "left": "goal.level", "right": "level goal.target level", "same_slot": False},  # current goal (150) vs prior goal (100)
    {"id": "trap-coffee", "left": "morning coffee.maximum quantity", "right": "morning routine.quantity", "same_slot": False},  # coffee limit vs generic morning quantity
]


def load_embedder():
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL)

    def embed(texts):
        # Symmetric slot-key identity: encode both sides the same way (no asymmetric
        # query/document prompt). normalize so cosine == dot, deterministic order.
        vecs = model.encode(list(texts), normalize_embeddings=True,
                            convert_to_numpy=True, show_progress_bar=False)
        return [v.tolist() for v in vecs]

    return embed


def run(fixtures: str, out: str) -> int:
    payload = json.loads(Path(fixtures).read_text(encoding="utf-8"))
    canonical = payload["embedding_identity_pairs"]
    embed = load_embedder()

    sets = {"canonical_fixtures": canonical, "extended_real_keys": EXTENDED_PAIRS}
    results = {}
    for name, pairs in sets.items():
        started = time.perf_counter()
        res = evaluate_pairs(pairs, embed)
        res["wall_ms"] = round((time.perf_counter() - started) * 1000, 1)
        results[name] = res
        best = res["best_zero_false_merge_threshold"]
        print(f"\n=== {name} ({len(pairs)} pairs) ===")
        for row in res["pairs"]:
            print(f"  same={str(row['same_slot']):5s} sim={row['similarity']:.4f}  "
                  f"{row['left']}  <->  {row['right']}")
        if best:
            print(f"  best zero-false-merge: thr={best['threshold']:.4f} "
                  f"same_slot_recall={best['same_slot_recall']} "
                  f"({best['true_merges']} merges, 0 false)")
        else:
            print("  no zero-false-merge threshold recovers any same-slot pair")

    artifact = {
        "schema_version": 1,
        "experiment": "embedding-slot-identity-local",
        "repository_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip(),
        "model": MODEL,
        "backend": "sentence-transformers (local, CPU)",
        "fixtures": fixtures,
        "created_at": time.time(),
        "results": results,
    }
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(f"\nWritten to {out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fixtures", default="bench/memory_methods/diagnostic_fixtures.json")
    ap.add_argument("--out", default="bench/memory_methods/embedding_identity_qwen.json")
    args = ap.parse_args()
    return run(args.fixtures, args.out)


if __name__ == "__main__":
    raise SystemExit(main())

"""Audit the value-gate's numeric_update merges per state (NPU-free).

The instance-aware value-gate (option a) eliminates the textual-distinct false-merge
class, but it CANNOT separate a genuine then->now numeric update (goal 100 -> 150)
from two DISTINCT facts that merely share an attribute word and both carry numbers:
start vs end date, refrigerator vs freezer duration, min vs max of one range, two
different shops' walk distances. This runner dumps every numeric_update merge so the
false-vs-genuine split can be judged by eye or a typed judge — token heuristics
provably mislabel these (FINDINGS: similarity != identity; an attribute-overlap
heuristic files start-date<->end-date as "genuine").

Deterministic; local Qwen embedder only. Use to scrutinize each fresh held-out state
as it lands, rather than trusting the headline "textual_distinct 304 -> 0".
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from rem.memory.tiers import MemoryState
from rem.memory.semantic_identity import (
    FullFactEmbeddingMatcher, resupersede_state, quantity_like,
)
from evals.memory_methods.state_selection import select_state_records

MODEL = "Qwen/Qwen3-Embedding-0.6B"


def load_embedder():
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL)
    return lambda texts: [v.tolist() for v in model.encode(
        list(texts), normalize_embeddings=True, convert_to_numpy=True,
        show_progress_bar=False)]


def numeric_merges(matcher) -> list[dict]:
    out = []
    for m in matcher.merges:
        kv = (m["kept_value"] or "").strip().lower()
        mv = (m["merged_value"] or "").strip().lower()
        if kv != mv and quantity_like(m["kept_value"]) and quantity_like(m["merged_value"]):
            out.append(m)
    return sorted(out, key=lambda m: -m["sim"])


def run(states_dir, manifest, ids, threshold, out, show):
    records = select_state_records(states_dir=states_dir, manifest=manifest, ids=ids)
    if not records:
        print("[numeric-audit] no captured states selected; nothing to do", flush=True)
        return 0
    embed = load_embedder()
    report = {"model": MODEL, "threshold": threshold, "states": {}}
    for rec in records:
        qid = rec["question_id"]
        st = MemoryState.load(rec["state_file"])
        matcher = FullFactEmbeddingMatcher(embed, threshold=threshold, value_aware=True)
        resupersede_state(st, matcher)
        nm = numeric_merges(matcher)
        report["states"][qid] = {
            "numeric_merges": len(nm),
            "safe_dedup_and_numeric_total": len(matcher.merges),
            "blocked_textual": len(matcher.blocked),
            "pairs": [
                {"sim": m["sim"],
                 "merged": f"{m['merged_key']}={m['merged_value']}",
                 "kept": f"{m['kept_key']}={m['kept_value']}"}
                for m in nm
            ],
        }
        print(f"[{qid}] numeric_update merges = {len(nm)}  "
              f"(blocked textual = {len(matcher.blocked)})", flush=True)
        for m in nm[:show]:
            print(f"    sim {m['sim']:.3f}  {m['merged_key']}={m['merged_value']!r} "
                  f"-> {m['kept_key']}={m['kept_value']!r}", flush=True)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(report, indent=2))
    print(f"\nNo auto false/genuine label is emitted on purpose: separating a then->now "
          f"update from a distinct same-attribute numeric fact needs reasoning, not a "
          f"token rule. Read the pairs (or run a typed judge).\nWritten to {out}", flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--states-dir", default="bench/battery/states")
    ap.add_argument("--manifest", default=None,
                    help="frozen development manifest; overrides --states-dir")
    ap.add_argument("--ids", nargs="*", default=None)
    ap.add_argument("--threshold", type=float, default=0.80)
    ap.add_argument("--show", type=int, default=15, help="how many pairs to print per state")
    ap.add_argument("--out", default="bench/memory_methods/numeric_merge_audit.json")
    args = ap.parse_args()
    return run(args.states_dir, args.manifest, args.ids, args.threshold, args.out, args.show)


if __name__ == "__main__":
    raise SystemExit(main())

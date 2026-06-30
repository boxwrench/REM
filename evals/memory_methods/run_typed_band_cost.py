"""Measure the typed-judge band cost on captured states (NPU-free).

The typed-judge supersession (TypedIdentityMatcher) bounds write-time NPU cost by
calling the LLM judge ONLY for candidate pairs whose cosine sim lands in the
ambiguous band [low, high). Clear pairs (>=high SAME, <low DIFFERENT) never call
the judge. This runner quantifies how many judge calls per ingest that would be on
real states, so the write-time cost is known before any NPU judge is wired in.

It replays each state's ledger through resupersede_state with a COUNTING stub
judge (no NPU), under two bracketing policies:
  * different: the judge always says DIFFERENT in band -> nothing merges in band ->
    the most candidate comparisons survive -> an UPPER bound on judge calls.
  * same:      the judge always says SAME in band -> band pairs collapse -> fewer
    later comparisons -> a LOWER bound.
The real cost sits between the two. Also reports cosine-only auto-merges (>=high,
no judge) for context. Deterministic; local Qwen embedder only.
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
    TypedIdentityMatcher, resupersede_state, full_fact_text, share_key_token,
)
from evals.memory_methods.state_selection import select_state_records

MODEL = "Qwen/Qwen3-Embedding-0.6B"


def load_embedder():
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL)
    return lambda texts: [v.tolist() for v in model.encode(
        list(texts), normalize_embeddings=True, convert_to_numpy=True,
        show_progress_bar=False)]


def _counting_judge(verdict: bool):
    calls = {"n": 0}

    def judge(_a, _b):
        calls["n"] += 1
        return verdict

    return judge, calls


def run(states_dir, manifest, ids, low, high, out, prefilter=False):
    records = select_state_records(states_dir=states_dir, manifest=manifest, ids=ids)
    if not records:
        print("[band-cost] no captured states selected; nothing to do", flush=True)
        return 0
    pf = share_key_token if prefilter else None
    embed = load_embedder()

    report = {"model": MODEL, "low": low, "high": high, "states": {}}
    print(f"{'state':12s} {'entries':>8s} {'autosame':>9s} "
          f"{'judge(diff)':>12s} {'judge(same)':>12s}")
    for rec in records:
        qid = rec["question_id"]
        base = MemoryState.load(rec["state_file"])
        per_policy = {}
        for label, verdict in (("different", False), ("same", True)):
            judge, calls = _counting_judge(verdict)
            matcher = TypedIdentityMatcher(embed, judge, low_threshold=low,
                                           high_threshold=high, prefilter=pf)
            _, stats = resupersede_state(base, matcher)
            auto_same = sum(1 for m in matcher.merges if m["sim"] >= high)
            per_policy[label] = {
                "judge_calls": matcher.judge_calls,
                "prefiltered": matcher.prefiltered,
                "auto_same_merges": auto_same,
                "band_merges": sum(1 for j in matcher.judged if j["verdict_same"]),
                "active_after": stats["active_after"],
            }
        entries = len(base.ledger.entries)
        report["states"][qid] = {"entries": entries, **per_policy}
        print(f"{qid:12s} {entries:>8d} "
              f"{per_policy['different']['auto_same_merges']:>9d} "
              f"{per_policy['different']['judge_calls']:>12d} "
              f"{per_policy['same']['judge_calls']:>12d}", flush=True)

    diffs = [s["different"]["judge_calls"] for s in report["states"].values()]
    sames = [s["same"]["judge_calls"] for s in report["states"].values()]
    report["prefilter"] = bool(prefilter)
    report["totals"] = {
        "n_states": len(diffs),
        "judge_calls_upper_total": sum(diffs),
        "judge_calls_lower_total": sum(sames),
        "judge_calls_upper_per_item_avg": round(sum(diffs) / len(diffs), 1),
        "judge_calls_lower_per_item_avg": round(sum(sames) / len(sames), 1),
    }
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(report, indent=2))
    print(f"\nNPU judge calls per ingest — upper(diff) avg "
          f"{report['totals']['judge_calls_upper_per_item_avg']}, "
          f"lower(same) avg {report['totals']['judge_calls_lower_per_item_avg']}")
    print(f"Written to {out}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--states-dir", default="bench/battery/states")
    ap.add_argument("--manifest", default=None,
                    help="frozen development manifest; overrides --states-dir")
    ap.add_argument("--ids", nargs="*", default=None)
    ap.add_argument("--low", type=float, default=0.70)
    ap.add_argument("--high", type=float, default=0.88)
    ap.add_argument("--prefilter", action="store_true",
                    help="apply the share_key_token candidate pre-filter (skip cosine/"
                         "judge for zero-key-overlap pairs); compare judge_calls vs without")
    ap.add_argument("--out", default="bench/memory_methods/typed_band_cost.json")
    args = ap.parse_args()
    return run(args.states_dir, args.manifest, args.ids, args.low, args.high, args.out,
               prefilter=args.prefilter)


if __name__ == "__main__":
    raise SystemExit(main())

"""Gate 4 option (a) — instance-aware supersession vs the plain global-threshold one.

Replays the 5 captured states through embedding supersession twice — the plain
matcher and the value-aware (instance-aware) matcher — and classifies every merge as
safe_dedup / numeric_update / textual_distinct. The value-aware gate blocks
textual_distinct merges (distinct named instances like Poffertjes vs apple pie) while
keeping numeric updates (5->5, one->two, 100->150), so it should remove the unsafe
class without losing the target collapses. NPU-free (local Qwen only).
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


def classify(merges):
    tax = {"safe_dedup": 0, "numeric_update": 0, "textual_distinct": 0}
    samples = {"textual_distinct": []}
    for m in merges:
        kv = (m["kept_value"] or "").strip().lower()
        mv = (m["merged_value"] or "").strip().lower()
        if kv == mv:
            tax["safe_dedup"] += 1
        elif quantity_like(m["kept_value"]) and quantity_like(m["merged_value"]):
            tax["numeric_update"] += 1
        else:
            tax["textual_distinct"] += 1
            if len(samples["textual_distinct"]) < 4:
                samples["textual_distinct"].append(
                    f"{m['merged_key']}={m['merged_value']!r} -> {m['kept_key']}={m['kept_value']!r} (sim {m['sim']})")
    return tax, samples


def team_size_collapsed(state):
    """Did the three '5 engineers' keys collapse to a single active team-size slot?"""
    keys = {"team.size", "team size.size", "group size.number of engineers"}
    actives = [e for e in state.ledger.entries
               if e.slot_key in keys and e.status == "active"]
    stale = [e for e in state.ledger.entries
             if e.slot_key in keys and e.status == "stale"]
    return {"active": [(e.slot_key, e.slot_value) for e in actives],
            "stale": [(e.slot_key, e.slot_value) for e in stale]}


def apex_ordered(state):
    """Is the prior Apex goal 100 stale and 150 active (then->now ordered)?"""
    rows = [(e.source_turn_id, e.status, e.slot_key, e.slot_value)
            for e in state.ledger.entries
            if e.slot_key in {"level goal.target level", "goal.level", "user.goal"}]
    return sorted(rows)


def run(out, states_dir="bench/battery/states", manifest=None, ids=None):
    records = select_state_records(states_dir=states_dir, manifest=manifest, ids=ids)
    if not records:
        print("[instanceaware] no captured states selected; nothing to do", flush=True)
        return 0
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL)
    def embed(t):
        return [v.tolist() for v in model.encode(
            list(t), normalize_embeddings=True, convert_to_numpy=True,
            show_progress_bar=False)]

    report = {"model": MODEL, "threshold": 0.80, "states": {}}
    totals = {"plain": {"safe_dedup": 0, "numeric_update": 0, "textual_distinct": 0},
              "instance_aware": {"safe_dedup": 0, "numeric_update": 0, "textual_distinct": 0}}
    print(f"{'state':10s} {'mode':14s} {'active_after':>12s} {'dedup':>6s} {'update':>7s} {'textual':>8s} {'blocked':>8s}")
    for rec in records:
        qid, path = rec["question_id"], rec["state_file"]
        base_state = MemoryState.load(path)
        entry = {}
        for mode, va in (("plain", False), ("instance_aware", True)):
            matcher = FullFactEmbeddingMatcher(embed, threshold=0.80, value_aware=va)
            new_state, stats = resupersede_state(base_state, matcher)
            tax, samples = classify(matcher.merges)
            for k in totals[mode]:
                totals[mode][k] += tax[k]
            entry[mode] = {
                "active_after": stats["active_after"],
                "active_reduction_pct": stats["active_reduction_pct"],
                "merges": len(matcher.merges), "blocked": len(matcher.blocked),
                "taxonomy": tax, "textual_samples": samples["textual_distinct"],
            }
            if va:  # capture target-merge structure from the safe (instance-aware) state
                entry["team_size_after"] = team_size_collapsed(new_state)
                entry["apex_goal_after"] = apex_ordered(new_state) if qid == "9bbe84a2" else None
            print(f"{qid:10s} {mode:14s} {stats['active_after']:>12d} "
                  f"{tax['safe_dedup']:>6d} {tax['numeric_update']:>7d} "
                  f"{tax['textual_distinct']:>8d} {len(matcher.blocked):>8d}")
        report["states"][qid] = entry

    report["totals"] = totals
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(report, indent=2))
    print(f"\nTOTAL plain:          {totals['plain']}")
    print(f"TOTAL instance_aware: {totals['instance_aware']}")
    # Known diagnostic targets — only print when those items are in the selection.
    if "031748ae" in report["states"]:
        print("\n031748ae team-size (instance-aware):",
              report["states"]["031748ae"].get("team_size_after"))
        print("sample blocked-as-different-instance (031748ae):")
        for s in report["states"]["031748ae"]["plain"]["textual_samples"]:
            print("   ", s)
    if "9bbe84a2" in report["states"]:
        print("9bbe84a2 apex goal  (instance-aware):",
              report["states"]["9bbe84a2"].get("apex_goal_after"))
    print(f"\nWritten to {out}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--states-dir", default="bench/battery/states",
                    help="dir with a flat manifest.json (legacy battery layout)")
    ap.add_argument("--manifest", default=None,
                    help="frozen development manifest ({'items':[...]}); overrides --states-dir")
    ap.add_argument("--ids", nargs="*", default=None,
                    help="optional question_id subset to replay")
    ap.add_argument("--out", default="bench/battery/supersession_instanceaware.json")
    args = ap.parse_args()
    return run(args.out, states_dir=args.states_dir, manifest=args.manifest, ids=args.ids)


if __name__ == "__main__":
    raise SystemExit(main())

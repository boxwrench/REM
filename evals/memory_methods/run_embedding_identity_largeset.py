"""Gate 4 follow-up (2) — larger within-state labeled set for the key-composition sweep.

Confirms whether `full_fact` (natural key + value) keeps its clean zero-false-merge
separation when the pair set grows from 10 to ~50 and is loaded with HARD negatives.

Design choices that keep this honest rather than deck-stacked:
  * Every pair is WITHIN a single captured state — that is how `_apply_supersession`
    actually compares a new entry against existing ones. No cross-user pairs.
  * Labels are auditable: SAME pairs are enumerated from explicit concept CLUSTERS
    (lists of real slot keys), HARD negatives are explicit same-subject / same-`.model`
    traps, and EASY negatives are a seeded sample of cross-concept entries in the same
    state. The resolved (key -> value) for every referenced slot is printed so the
    labels can be checked against the data.
  * Same-slot = same underlying attribute over time (supersession intent), so
    cross-value updates (coffee one->two, goal 100->150) are SAME.

Reports recall / false-merges and the separation margin twice: SAME-vs-HARD only
(the demanding cut) and SAME-vs-ALL. Deterministic. Local Qwen3-Embedding-0.6B.
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import time
from pathlib import Path

from rem.memory.tiers import MemoryState
from evals.memory_methods.embedding_identity import evaluate_pairs
from evals.memory_methods.run_embedding_identity_richkey import (
    KEY_ONLY, VALUE_TOO,
)

MODEL = "Qwen/Qwen3-Embedding-0.6B"
STATES = {
    "031748ae": "bench/battery/states/031748ae_state.json",
    "3ba21379": "bench/battery/states/3ba21379_state.json",
    "9bbe84a2": "bench/battery/states/9bbe84a2_state.json",
    "c6853660": "bench/battery/states/c6853660_state.json",
    "cc5ded98": "bench/battery/states/cc5ded98_state.json",
}

# SAME-slot clusters: same underlying attribute, fragmented across real slot keys.
CLUSTERS = {
    "team_size@031748ae": ("031748ae", [
        "team.size", "team size.size", "group size.number of engineers"]),
    "coding_time@cc5ded98": ("cc5ded98", [
        "coding exercises.frequency", "coding exercises.time per day",
        "coding exercises.time spent per day", "coding exercises.duration"]),
    "mkt_team_size@c6853660": ("c6853660", [
        "client.marketing team size", "marketing team.size",
        "team size.number of salespeople", "marketing team size.size"]),
    "coffee_limit@c6853660": ("c6853660", [
        "morning routine.quantity of coffee cups per morning",
        "morning coffee limit.new limit"]),
    "apex_goal@9bbe84a2": ("9bbe84a2", [
        "level goal.target level", "goal.level", "user.goal"]),
}

# HARD negatives: must NOT merge. (item, keyA, keyB). Same subject token or same
# `.model`/`.size`/`.goal` attribute but genuinely different concept.
HARD_NEG = [
    ("031748ae", "team.size", "team members.count"),          # 5-eng team vs 4+Rachel outing
    ("031748ae", "team.size", "team outing.attendees"),
    ("031748ae", "group size.number of engineers", "team outing.attendees"),
    ("031748ae", "team size.size", "team outing.type"),
    ("c6853660", "morning coffee limit.new limit", "coffee maker.capacity"),
    ("c6853660", "morning routine.quantity of coffee cups per morning", "coffee maker.capacity"),
    ("c6853660", "marketing team.size", "dedicated team.responsibility"),
    ("9bbe84a2", "goal.level", "gail.goal"),
    ("9bbe84a2", "goal.level", "comedian.goal"),
    ("9bbe84a2", "user.goal", "apex legends.appeal"),
    ("3ba21379", "vehicle.model", "project.model"),           # current F-150 vs Mustang scale-model
    ("3ba21379", "vehicle.model", "model car.type"),
    ("3ba21379", "vehicle.model", "rolex watch.model"),
    ("3ba21379", "vehicle.model", "processor.model"),
    ("cc5ded98", "coding exercises.duration", "coding practice.key to improvement"),
    ("cc5ded98", "coding exercises.frequency", "coding buddy.purpose"),
]

EASY_NEG_PER_CLUSTER_ENTRY = 2
SEED = 0


def newest_active_by_key(state: MemoryState, key: str):
    """Return (key, value) for the newest active entry with this slot_key, or None."""
    best = None
    for e in state.ledger.entries:
        if e.status == "active" and e.slot_key == key:
            if best is None or e.source_turn_id > best.source_turn_id:
                best = e
    if best is None:
        return None
    return (best.slot_key, getattr(best, "slot_value", None) or best.text)


def subject(key: str) -> str:
    return key.split(".")[0]


def build_dataset():
    states = {k: MemoryState.load(v) for k, v in STATES.items()}
    resolved = {}   # (item, key) -> (key, value)
    missing = []

    def resolve(item, key):
        rk = (item, key)
        if rk not in resolved:
            r = newest_active_by_key(states[item], key)
            if r is None:
                missing.append(rk)
            resolved[rk] = r
        return resolved[rk]

    pairs = []  # each: {item, left:(k,v), right:(k,v), same_slot, kind, id}

    # SAME — within-cluster
    for cname, (item, keys) in CLUSTERS.items():
        ents = [(k, resolve(item, k)) for k in keys]
        ents = [(k, r) for k, r in ents if r is not None]
        for i in range(len(ents)):
            for j in range(i + 1, len(ents)):
                pairs.append({"item": item, "left": ents[i][1], "right": ents[j][1],
                              "same_slot": True, "kind": "same", "id": f"{cname}:{ents[i][0]}|{ents[j][0]}"})

    # HARD negatives
    for item, ka, kb in HARD_NEG:
        ra, rb = resolve(item, ka), resolve(item, kb)
        if ra and rb:
            pairs.append({"item": item, "left": ra, "right": rb,
                          "same_slot": False, "kind": "hard_neg", "id": f"HARD@{item}:{ka}|{kb}"})

    # EASY negatives — seeded sample of cross-concept entries within the same state
    rng = random.Random(SEED)
    cluster_keys_by_item = {}
    for _, (item, keys) in CLUSTERS.items():
        cluster_keys_by_item.setdefault(item, set()).update(keys)
    for cname, (item, keys) in CLUSTERS.items():
        state = states[item]
        cl_subjects = {subject(k) for k in keys}
        pool = [e for e in state.ledger.entries
                if e.status == "active" and e.slot_key
                and subject(e.slot_key) not in cl_subjects
                and e.slot_key not in cluster_keys_by_item.get(item, set())]
        for k in keys:
            r = resolve(item, k)
            if r is None or not pool:
                continue
            picks = rng.sample(pool, min(EASY_NEG_PER_CLUSTER_ENTRY, len(pool)))
            for p in picks:
                pv = (p.slot_key, getattr(p, "slot_value", None) or p.text)
                pairs.append({"item": item, "left": r, "right": pv,
                              "same_slot": False, "kind": "easy_neg",
                              "id": f"EASY@{item}:{k}|{p.slot_key}@{p.source_turn_id}"})
    return pairs, resolved, missing


def transform(pairs, fn):
    out = []
    for p in pairs:
        lk, lv = p["left"]
        rk, rv = p["right"]
        out.append({"id": p["id"], "same_slot": p["same_slot"],
                    "left": fn(lk, lv), "right": fn(rk, rv)})
    return out


def margin_report(pairs, scored_pairs, neg_kinds):
    """Min same-slot sim vs max negative sim over the chosen negative kinds."""
    same = [s["similarity"] for p, s in zip(pairs, scored_pairs) if p["same_slot"]]
    neg = [s["similarity"] for p, s in zip(pairs, scored_pairs)
           if not p["same_slot"] and p["kind"] in neg_kinds]
    if not same or not neg:
        return None
    # zero-false-merge threshold must exceed the highest negative; recall = same >= that
    thr = max(neg)
    recovered = sum(1 for v in same if v > thr)
    return {"min_same": round(min(same), 4), "max_neg": round(max(neg), 4),
            "margin": round(min(same) - max(neg), 4),
            "clean_recall_above_max_neg": round(recovered / len(same), 4),
            "n_same": len(same), "n_neg": len(neg)}


def run(out):
    pairs, resolved, missing = build_dataset()
    if missing:
        print("WARNING missing slot keys (skipped):")
        for m in missing:
            print("   ", m)
    kinds = {}
    for p in pairs:
        kinds[p["kind"]] = kinds.get(p["kind"], 0) + 1
    print(f"dataset: {len(pairs)} pairs -> {kinds}")

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL)
    def embed(texts):
        return [v.tolist() for v in model.encode(
            list(texts), normalize_embeddings=True, convert_to_numpy=True,
            show_progress_bar=False)]

    strategies = {**KEY_ONLY, **VALUE_TOO}
    results = {}
    print(f"\n{'strategy':14s} {'recall(all)':>11s} {'thr(all)':>9s}   "
          f"{'margin vs HARD':>15s} {'recall>maxHARD':>14s}")
    for name, fn in strategies.items():
        tp = transform(pairs, fn)
        res = evaluate_pairs(tp, embed)
        best = res["best_zero_false_merge_threshold"]
        hard = margin_report(pairs, res["pairs"], {"hard_neg"})
        allm = margin_report(pairs, res["pairs"], {"hard_neg", "easy_neg"})
        results[name] = {"eval": res, "margin_vs_hard": hard, "margin_vs_all": allm}
        recall_all = best["same_slot_recall"] if best else 0.0
        thr_all = f"{best['threshold']:.4f}" if best else "--"
        print(f"{name:14s} {recall_all:11.3f} {thr_all:>9s}   "
              f"{(hard['margin'] if hard else 0):+15.4f} "
              f"{(hard['clean_recall_above_max_neg'] if hard else 0):14.3f}")

    # full_fact detail: which same-slot pairs (if any) fall below the hardest negative
    ff = results["full_fact"]
    print("\nfull_fact — same-slot pairs sorted by similarity (vs max HARD neg "
          f"{ff['margin_vs_hard']['max_neg']}):")
    scored = ff["eval"]["pairs"]
    same_rows = sorted(((s["similarity"], p["id"]) for p, s in zip(pairs, scored)
                        if p["same_slot"]))
    for sim, pid in same_rows:
        flag = "OK " if sim > ff["margin_vs_hard"]["max_neg"] else "MISS"
        print(f"  {flag} {sim:.4f}  {pid}")
    print("\nfull_fact — top hard negatives:")
    hard_rows = sorted(((s["similarity"], p["id"]) for p, s in zip(pairs, scored)
                        if p["kind"] == "hard_neg"), reverse=True)
    for sim, pid in hard_rows[:6]:
        print(f"   {sim:.4f}  {pid}")

    artifact = {
        "schema_version": 1, "experiment": "embedding-slot-identity-largeset",
        "model": MODEL, "backend": "sentence-transformers (local, CPU)",
        "repository_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip(),
        "created_at": time.time(), "seed": SEED,
        "n_pairs": len(pairs), "kind_counts": kinds,
        "resolved_entries": {f"{i}|{k}": v for (i, k), v in resolved.items() if v},
        "pairs": [{k: p[k] for k in ("item", "id", "same_slot", "kind", "left", "right")}
                  for p in pairs],
        "results": {n: {"recall_all": (r["eval"]["best_zero_false_merge_threshold"] or {}).get("same_slot_recall"),
                        "threshold_all": (r["eval"]["best_zero_false_merge_threshold"] or {}).get("threshold"),
                        "margin_vs_hard": r["margin_vs_hard"],
                        "margin_vs_all": r["margin_vs_all"],
                        "pairs": r["eval"]["pairs"]}
                    for n, r in results.items()},
    }
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(f"\nWritten to {out}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="bench/memory_methods/embedding_identity_largeset.json")
    args = ap.parse_args()
    return run(args.out)


if __name__ == "__main__":
    raise SystemExit(main())

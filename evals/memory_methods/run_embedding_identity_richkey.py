"""Gate 4 follow-up — key-composition sweep for embedding slot identity.

Bare slot keys ("team.size") drop the distinguishing signal, so this sweeps richer
key compositions through the same Qwen embedder + ``evaluate_pairs`` harness and
asks which, if any, yields a clean zero-false-merge separation:

  bare_key      "team.size"
  natural_key   "team size"                      (dots/underscores -> spaces)
  subject       "team"                           (text before the first dot)
  full_fact     "team size: 5 engineers"         (natural key + value)   [real set]
  subject_value "team: 5 engineers"              (subject + value)       [real set]

Two pair sets:
  * canonical_fixtures  — the 6 designed key pairs (key-only strategies).
  * real_entries        — 10 pairs built from ACTUAL captured-state entries
    (key + value), labelled by SUPERSESSION intent: the same underlying attribute
    over time is the same slot (even when the value changed, e.g. coffee one->two,
    goal 100->150); genuinely different concepts are traps. This is the behaviour
    the writer needs to order 031748ae's then->now.

The real set is deliberately built to expose the value-inclusion tension: the
031748ae trap (team size "5 engineers" vs outing "4 engineers plus manager Rachel")
has SIMILAR values, so adding the value can RAISE a trap's similarity, while the
cross-value same-slot pairs (one cup vs two cups) DROP when the value is added.

Deterministic. Local Qwen/Qwen3-Embedding-0.6B (sentence-transformers, CPU).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

from evals.memory_methods.embedding_identity import evaluate_pairs

MODEL = "Qwen/Qwen3-Embedding-0.6B"

# Real captured-state entries: (key, value). Sourced from bench/battery/states/*.
E = {
    "team_size_a": ("team.size", "5 engineers"),                                  # 031748ae t74
    "team_size_b": ("team size.size", "five engineers"),                          # 031748ae t67
    "team_size_c": ("group size.number of engineers", "5"),                       # 031748ae t5
    "team_outing_count": ("team members.count", "4 engineers plus manager Rachel"),# 031748ae t12 (outing/4+Rachel)
    "team_outing_attend": ("team outing.attendees", "6"),                         # 031748ae t10
    "coding_freq": ("coding exercises.frequency", "two hours each day"),          # cc5ded98 t134
    "coding_tpd": ("coding exercises.time per day", "Two hours"),                 # cc5ded98 t135
    "coding_tspd": ("coding exercises.time spent per day", "two hours"),          # cc5ded98 t140
    "coding_dur": ("coding exercises.duration", "Two hours a day"),               # cc5ded98 t141
    "coffee_one": ("morning routine.quantity of coffee cups per morning", "one cup"),  # c6853660 t23
    "coffee_two": ("morning coffee limit.new limit", "two cups"),                 # c6853660 t128
    "coffee_maker_cap": ("coffee maker.capacity", "8-cup capacity"),              # c6853660 t129
    "goal_prior": ("level goal.target level", "100"),                            # 9bbe84a2 t61 (prior goal)
    "goal_now": ("goal.level", "150"),                                          # 9bbe84a2 t144 (current goal)
    "budget_level": ("budget.level", "low budget"),                             # cc5ded98 t156 ("level" polysemy)
}

# (left_id, right_id, same_slot). same_slot = same underlying attribute (slot), the
# supersession target — independent of whether the VALUE matches.
REAL_PAIRS = [
    ("team_size_a", "team_size_c", True),   # team size, same value (5)
    ("team_size_a", "team_size_b", True),   # team size, same value
    ("coding_tpd", "coding_dur", True),     # coding time/day, same value
    ("coding_freq", "coding_tspd", True),   # coding time/day, same value
    ("coffee_one", "coffee_two", True),     # coffee cup limit, CROSS-VALUE (one->two)
    ("goal_prior", "goal_now", True),       # Apex level goal, CROSS-VALUE (100->150)
    ("team_size_a", "team_outing_count", False),  # TRAP: team size vs 4+Rachel outing (031748ae)
    ("team_size_a", "team_outing_attend", False), # TRAP: team size vs outing attendees
    ("coffee_two", "coffee_maker_cap", False),    # TRAP: coffee limit vs coffee-maker capacity
    ("goal_now", "budget_level", False),          # TRAP: game-level goal vs budget "level"
]


def comp_bare(key, value):
    return key


def comp_natural(key, value):
    return key.replace(".", " ").replace("_", " ")


def comp_subject(key, value):
    return key.split(".")[0].replace("_", " ")


def comp_full_fact(key, value):
    return f"{comp_natural(key, value)}: {value}"


def comp_subject_value(key, value):
    return f"{comp_subject(key, value)}: {value}"


KEY_ONLY = {"bare_key": comp_bare, "natural_key": comp_natural, "subject": comp_subject}
VALUE_TOO = {"full_fact": comp_full_fact, "subject_value": comp_subject_value}


def load_embedder():
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL)

    def embed(texts):
        vecs = model.encode(list(texts), normalize_embeddings=True,
                            convert_to_numpy=True, show_progress_bar=False)
        return [v.tolist() for v in vecs]

    return embed


def sweep_canonical(pairs, embed):
    out = {}
    for name, fn in KEY_ONLY.items():
        transformed = [
            {"id": p.get("id", ""), "same_slot": p["same_slot"],
             "left": fn(p["left"], None), "right": fn(p["right"], None)}
            for p in pairs
        ]
        out[name] = evaluate_pairs(transformed, embed)
    return out


def sweep_real(embed):
    strategies = {**KEY_ONLY, **VALUE_TOO}
    out = {}
    for name, fn in strategies.items():
        transformed = []
        for lid, rid, same in REAL_PAIRS:
            lk, lv = E[lid]
            rk, rv = E[rid]
            transformed.append({
                "id": f"{lid}|{rid}", "same_slot": same,
                "left": fn(lk, lv), "right": fn(rk, rv),
            })
        out[name] = evaluate_pairs(transformed, embed)
    return out


def summarize(res):
    best = res["best_zero_false_merge_threshold"]
    if best:
        return (f"thr={best['threshold']:.4f} recall={best['same_slot_recall']} "
                f"({best['true_merges']} merges, 0 false)")
    return "no zero-false-merge threshold recovers any same-slot pair"


def run(fixtures, out):
    payload = json.loads(Path(fixtures).read_text(encoding="utf-8"))
    canonical = payload["embedding_identity_pairs"]
    embed = load_embedder()

    canon = sweep_canonical(canonical, embed)
    real = sweep_real(embed)

    print("=== canonical_fixtures (key-only strategies) ===")
    for name, res in canon.items():
        print(f"  {name:14s} {summarize(res)}")
    print("\n=== real_entries (all strategies) ===")
    for name, res in real.items():
        print(f"  {name:14s} {summarize(res)}")
    print("\n--- real_entries per-pair similarity by strategy ---")
    names = list(real)
    header = "same  " + "  ".join(f"{n:>13s}" for n in names) + "   pair"
    print(header)
    n_pairs = len(real[names[0]]["pairs"])
    for i in range(n_pairs):
        row0 = real[names[0]]["pairs"][i]
        sims = "  ".join(f"{real[n]['pairs'][i]['similarity']:13.4f}" for n in names)
        flag = "MERGE" if row0["same_slot"] else "KEEP "
        print(f"{flag} {sims}   {row0['id']}")

    artifact = {
        "schema_version": 1,
        "experiment": "embedding-slot-identity-keycomposition",
        "repository_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip(),
        "model": MODEL, "backend": "sentence-transformers (local, CPU)",
        "created_at": time.time(),
        "canonical_fixtures": canon,
        "real_entries": {"pairs_def": [
            {"left": E[li], "right": E[ri], "same_slot": s} for li, ri, s in REAL_PAIRS
        ], "results": real},
    }
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(f"\nWritten to {out}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fixtures", default="bench/memory_methods/diagnostic_fixtures.json")
    ap.add_argument("--out", default="bench/memory_methods/embedding_identity_keycomp.json")
    args = ap.parse_args()
    return run(args.fixtures, args.out)


if __name__ == "__main__":
    raise SystemExit(main())

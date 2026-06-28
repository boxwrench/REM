"""NPU-free failure-mix analysis over captured MemoryStates.

For each state in the capture manifest, run the Step-0 bounded read path
(RecencySelector via fit_with_selector) and label the item's miss as
size / retrieval-recall / temporal-structure / pass (spec §5). An optional single
brief answer per item (the only NPU) separates temporal-structure from pass.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from rem.config import Settings
from rem.memory.selector import RecencySelector
from rem.memory.tiers import MemoryState
from evals.battery.diagnose_memory import fit_with_selector, gold_in_fitted

GEMMA = "gemma4-it:e2b"

# Curated faithful gold needles per item (spec E5): the value(s) a correct answer
# must contain. Required — a gold needle absent from the fitted slice is a
# retrieval-recall miss. Validated against each item's gold answer and gold turns.
GOLD_NEEDLES = {
    "031748ae": ["4 engineers", "5 engineers"],   # then + now (both gold)
    "3ba21379": ["F-150"],                         # current vehicle
    "cc5ded98": ["two hours"],                     # current coding time/day
    "c6853660": ["one cup", "two cups"],           # answer = "from one cup to two cups"
    "9bbe84a2": ["level 100"],                     # gold IS the prior goal
}

# Structure needles: the contrasting prior / distractor value present in the
# source. NOT required for a correct answer, so they do NOT gate retrieval-recall;
# they are recorded (presence + carrying tier) to diagnose temporal-structure —
# when the gold survives yet the model returns the contrasting value, or both
# then/now values sit in the slice with no order. Empty where the then+now pair is
# already in GOLD_NEEDLES.
STRUCTURE_NEEDLES = {
    "031748ae": [],
    "3ba21379": ["Mustang"],            # competing concurrent model project
    "cc5ded98": ["an hour each day"],   # prior coding time (one hour -> two hours)
    "c6853660": [],
    "9bbe84a2": ["level 150"],          # the updated goal; "previous goal" is gold
}


def needle_tier(state, question, settings, needle) -> str:
    """Which tier of the FITTED state carries the needle."""
    fitted = RecencySelector().select(state, question, settings.read_fit_tokens)
    low = needle.lower()
    for e in fitted.ledger.entries:
        if low in e.text.lower():
            return "slot" if e.slot_key else "free"
    for s in fitted.summaries:
        txt = s.rendered_text if s.rendered_text is not None else s.text
        if low in txt.lower():
            return "summary"
    return "absent"


def label_item(state, question, answer, needles, settings, answerer=None,
               structure_needles=None) -> dict:
    structure_needles = structure_needles or []
    fitted_text, fitted_tokens = fit_with_selector(state, question, settings)
    fits = fitted_tokens <= settings.read_fit_tokens
    hits = gold_in_fitted(fitted_text, needles)
    tiers = {n: needle_tier(state, question, settings, n) for n in needles}

    # Structure needles are recorded, never gating: they diagnose temporal-structure.
    structure_hits = gold_in_fitted(fitted_text, structure_needles)
    structure_tiers = {n: needle_tier(state, question, settings, n)
                       for n in structure_needles}

    brief_answer = None
    answer_contains_gold = None
    answer_contains_structure = None
    if answerer is not None:
        brief_answer = (answerer(fitted_text, question) or "").strip()
        low = brief_answer.lower()
        answer_contains_gold = any(n.lower() in low for n in needles)
        answer_contains_structure = any(n.lower() in low for n in structure_needles)

    if not fits:
        mode = "size"
    elif not all(hits.values()):
        mode = "retrieval-recall"
    elif answerer is None:
        mode = "needs-answer"
    elif answer_contains_gold:
        mode = "pass"
    else:
        mode = "temporal-structure"

    return {
        "fitted_tokens": fitted_tokens, "fits_budget": fits,
        "gold_in_fitted": hits, "needle_tiers": tiers,
        "structure_in_fitted": structure_hits, "structure_tiers": structure_tiers,
        "brief_answer": brief_answer, "answer_contains_gold": answer_contains_gold,
        "answer_contains_structure": answer_contains_structure,
        "failure_mode": mode,
    }


def run(states_dir: str, out: str, settings=None, answerer=None) -> int:
    settings = settings or Settings(summarizer_model=GEMMA)
    sdir = Path(states_dir)
    manifest_path = sdir / "manifest.json"
    if not manifest_path.exists():
        print(f"no manifest at {manifest_path}", file=sys.stderr)
        return 2
    records = json.loads(manifest_path.read_text(encoding="utf-8"))

    rows = []
    counts: dict[str, int] = {}
    for r in records:
        state = MemoryState.load(r["state_file"])
        needles = GOLD_NEEDLES.get(r["question_id"], [])
        structure = STRUCTURE_NEEDLES.get(r["question_id"], [])
        lab = label_item(state, r["question"], r["answer"], needles, settings,
                         answerer=answerer, structure_needles=structure)
        lab["question_id"] = r["question_id"]
        lab["gold_recency"] = r.get("gold_recency")
        rows.append(lab)
        counts[lab["failure_mode"]] = counts.get(lab["failure_mode"], 0) + 1
        print(f"[{r['question_id']}] mode={lab['failure_mode']:18s} "
              f"fitted={lab['fitted_tokens']:6d} fits={lab['fits_budget']} "
              f"gold_tiers={lab['needle_tiers']} "
              f"struct_tiers={lab['structure_tiers']}", flush=True)

    payload = {"states_dir": str(sdir), "n_items": len(rows),
               "mix": counts, "items": rows}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nMIX: {counts}")
    print(f"Written to {out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Failure-mix analysis over captured states")
    ap.add_argument("--states-dir", default="bench/battery/states")
    ap.add_argument("--out", default="bench/battery/mix_report.json")
    ap.add_argument("--answer", action="store_true",
                    help="Take one brief NPU answer per item (separates pass from "
                         "temporal-structure). Off by default = fully NPU-free.")
    args = ap.parse_args()
    answerer = None
    if args.answer:
        from rem.npu_client import NpuClient
        from evals.battery.answerer import answer_question
        npu = NpuClient(Settings(summarizer_model=GEMMA))
        def answerer(ctx, q):
            return answer_question(npu, context=ctx, question=q)
    return run(args.states_dir, args.out, answerer=answerer)


if __name__ == "__main__":
    raise SystemExit(main())

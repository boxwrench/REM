"""Gate 4 a2 follow-up — does typed (LLM-judge) identity separate what embeddings can't?

Embedding similarity (of fact OR attribute) cannot separate same-slot from
different-slot: "number of engineers"~"size" (0.55, same slot) scores below
"likes"~"comments" (0.75, different slots). This probes whether a reasoning judge
(the local gemma) decides SAME/DIFFERENT slot correctly on the decisive pairs —
the typed-claim direction. Illustrative, dev-only (these pairs come from the 5 dev
states; not a held-out benchmark).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from rem.config import Settings
from rem.npu_client import NpuClient

GEMMA = "gemma4-it:e2b"

# (fact_a, fact_b, same_slot_gold). same_slot = should one supersede the other?
PAIRS = [
    # genuine same-slot (cross-attribute / cross-subject fragmentation) -> SAME
    ("team size: 5 engineers", "group size number of engineers: 5", True),
    ("coding exercises duration: Two hours a day", "coding exercises time per day: Two hours", True),
    ("level goal target level: 100", "goal level: 150", True),
    ("morning routine quantity of coffee cups per morning: one cup",
     "morning coffee limit new limit: two cups", True),
    # different-slot (same-subject different attribute, or distinct instance) -> DIFFERENT
    ("linkedin posts likes: 20", "linkedin posts comments: 5", False),
    ("corporate team building package capacity: 20 people",
     "corporate team building package size: 6 people", False),
    ("dessert name: Poffertjes", "dessert name: Dutch apple pie", False),
    ("team size: 5 engineers", "team members count: 4 engineers plus manager Rachel", False),
]

SYSTEM = (
    "You decide whether two stored facts describe the SAME attribute of the SAME "
    "entity (so a newer one should replace the older as an update) or DIFFERENT "
    "attributes/entities (both should be kept). Reply with exactly one word: SAME "
    "or DIFFERENT."
)


def judge(npu, a, b):
    out = npu.chat(
        [{"role": "system", "content": SYSTEM},
         {"role": "user", "content": f"Fact A: {a}\nFact B: {b}\nAnswer:"}],
        model=GEMMA, max_tokens=8).strip().upper()
    return "SAME" if "SAME" in out else ("DIFFERENT" if "DIFFER" in out else out)


def run(out):
    npu = NpuClient(Settings(summarizer_model=GEMMA))
    rows = []
    correct = 0
    for a, b, gold in PAIRS:
        verdict = judge(npu, a, b)
        pred_same = verdict == "SAME"
        ok = pred_same == gold
        correct += ok
        rows.append({"a": a, "b": b, "gold_same": gold, "verdict": verdict, "correct": ok})
        print(f"  [{'OK ' if ok else 'XX '}] gold={'SAME' if gold else 'DIFF':4s} "
              f"pred={verdict:9s}  {a[:40]!r} vs {b[:40]!r}", flush=True)
    acc = round(correct / len(PAIRS), 3)
    print(f"\ntyped-judge accuracy on decisive pairs: {correct}/{len(PAIRS)} = {acc}")
    payload = {"model": GEMMA, "n": len(PAIRS), "correct": correct, "accuracy": acc,
               "pairs": rows,
               "note": "dev-only illustrative; pairs drawn from the 5 dev states"}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(payload, indent=2))
    print(f"Written to {out}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="bench/memory_methods/typed_identity_probe.json")
    args = ap.parse_args()
    return run(args.out)


if __name__ == "__main__":
    raise SystemExit(main())

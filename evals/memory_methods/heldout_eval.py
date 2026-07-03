"""Ad-hoc held-out gate on specific captured states (early generalization read).

Reuses decision_gate's arm context builders (current=LexicalSelector,
sparse=SparseChronologicalSelector safe shipped, oracle=raw gold sessions) and the
fixed gemma answerer. Default builds contexts only (NPU-free, validates plumbing);
--answer generates reps answers per (arm, item) via the NPU. Grade the saved answers
by reading them (no ANTHROPIC_API_KEY here).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from rem.config import Settings
from rem.memory.tiers import MemoryState, count_tokens
from evals.battery.decision_gate import (
    ARMS, GEMMA, arm_context, load_raw_index, real_answerer,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--qids", nargs="+", required=True)
    ap.add_argument("--manifest", default="bench/memory_methods/development_manifest.json")
    ap.add_argument("--states-dir", default="bench/memory_methods/states")
    ap.add_argument("--raw", default="/home/keith/datasets/longmemeval/longmemeval_s")
    ap.add_argument("--out", default="bench/memory_methods/heldout_eval_answers.json")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--answer", action="store_true",
                    help="call the NPU answerer (needs the NPU free). Off = contexts only.")
    ap.add_argument("--read-fit", type=int, default=None,
                    help="override read_fit_tokens (e.g. cap the current arm to the "
                         "flm serving window; only affects the budget-fill current arm).")
    args = ap.parse_args()

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    by_id = {it["question_id"]: it for it in manifest["items"]}
    raw_index = load_raw_index(args.raw)
    settings = Settings(summarizer_model=GEMMA)
    if args.read_fit:
        settings.read_fit_tokens = args.read_fit
    states_dir = Path(args.states_dir)
    answerer = real_answerer(settings) if args.answer else None

    results = []
    for qid in args.qids:
        item = by_id[qid]
        sfile = states_dir / f"{qid}_state.json"
        if not sfile.exists():
            print(f"[{qid}] no state file, skipping", flush=True)
            continue
        state = MemoryState.load(str(sfile))
        raw_entry = raw_index.get(qid, {})
        rec = {"qid": qid, "category": item["category"],
               "question": item["question"], "gold": item["answer"], "arms": {}}
        for arm in ARMS:
            ctx = arm_context(arm, state, item["question"], raw_entry, settings)
            entry = {"ctx_tokens": count_tokens(ctx)}
            if answerer is not None:
                entry["answers"] = [(answerer(ctx, item["question"]) or "").strip()
                                    for _ in range(args.reps)]
            rec["arms"][arm] = entry
            print(f"[{qid}][{arm}] ctx_tokens={entry['ctx_tokens']} "
                  f"answers={len(entry.get('answers', []))}", flush=True)
        results.append(rec)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"answered": args.answer, "results": results},
                                         indent=2), encoding="utf-8")
    print(f"Written to {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

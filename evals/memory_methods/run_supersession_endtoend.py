"""Gate 4 end-to-end — embedding supersession over the 5 states, then the mix.

Replays each captured state through the flagged embedding-matched supersession
(local Qwen full-fact identity, threshold from Settings), then runs the temporal
LexicalSelector + one brief NPU answer per item on the RE-SUPERSEDED state. Answers
the standing question: does collapsing semantically-fragmented slots give the read
path the ordered then->now set, and does 031748ae finally resolve end-to-end?

Phase 1 (NPU-free): resupersede + structural audit (active-entry reduction, every
merge fired, ordered concept views) + selector fit + gold needles. Written to disk
first so the durable measurements survive an answerer failure.
Phase 2 (one brief NPU answer per item): pass vs temporal-structure, vs baseline.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from rem.config import Settings
from rem.memory.selector import LexicalSelector
from rem.memory.tiers import MemoryState
from rem.memory.semantic_identity import (
    FullFactEmbeddingMatcher, resupersede_state, full_fact_text,
)
from evals.battery.diagnose_memory import gold_in_fitted
from evals.battery.mix_report import GOLD_NEEDLES, STRUCTURE_NEEDLES
from evals.battery.mix_report_selector import fit_render_aware, needle_tier

GEMMA = "gemma4-it:e2b"
MODEL = "Qwen/Qwen3-Embedding-0.6B"

# Concept probe per item: regex over slot_key+value to dump the ordered slot history.
CONCEPT = {
    "031748ae": re.compile(r"engineer|team", re.I),
    "9bbe84a2": re.compile(r"\bgoal\b|\blevel\b", re.I),
    "c6853660": re.compile(r"coffee|cup", re.I),
    "cc5ded98": re.compile(r"coding exercise", re.I),
    "3ba21379": re.compile(r"vehicle|model car|\.model", re.I),
}


def concept_view(state, qid):
    rx = CONCEPT.get(qid)
    if rx is None:
        return []
    rows = []
    for e in sorted(state.ledger.entries, key=lambda e: e.source_turn_id):
        if e.slot_key and rx.search(e.slot_key):
            rows.append({"turn": e.source_turn_id, "status": e.status,
                         "slot_key": e.slot_key, "value": e.slot_value,
                         "superseded_by": e.superseded_by_turn_id})
    return rows


def load_embedder():
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL)
    return lambda texts: [v.tolist() for v in model.encode(
        list(texts), normalize_embeddings=True, convert_to_numpy=True,
        show_progress_bar=False)]


def run(states_dir, out, threshold, answer):
    settings = Settings(summarizer_model=GEMMA)
    thr = threshold if threshold is not None else settings.embedding_supersession_threshold
    records = json.loads((Path(states_dir) / "manifest.json").read_text())
    embed = load_embedder()
    selector = LexicalSelector()

    items = []
    for r in records:
        qid = r["question_id"]
        state = MemoryState.load(r["state_file"])
        matcher = FullFactEmbeddingMatcher(embed, threshold=thr)
        new_state, stats = resupersede_state(state, matcher)

        needles = GOLD_NEEDLES.get(qid, [])
        structure = STRUCTURE_NEEDLES.get(qid, [])
        fitted, text, n = fit_render_aware(selector, new_state, r["question"],
                                           settings.read_fit_tokens)
        hits = gold_in_fitted(text, needles)
        tiers = {x: needle_tier(fitted, x) for x in needles}

        item = {
            "question_id": qid, "question": r["question"], "gold_answer": r["answer"],
            "threshold": thr, "supersession_stats": stats,
            "merges_fired": matcher.merges,
            "concept_view_after": concept_view(new_state, qid),
            "fitted_tokens": n, "fits_budget": n <= settings.read_fit_tokens,
            "gold_in_fitted": hits, "needle_tiers": tiers,
            "history_in_fitted_slice": [
                {"status": e.status, "slot_key": e.slot_key, "value": e.slot_value,
                 "turn": e.source_turn_id}
                for e in fitted.ledger.entries
                if CONCEPT.get(qid) and CONCEPT[qid].search(e.slot_key or "")
            ],
            "brief_answer": None, "answer_contains_gold": None, "failure_mode": None,
        }
        items.append(item)
        print(f"[{qid}] active {stats['active_before']}->{stats['active_after']} "
              f"(-{stats['active_reduction_pct']}%)  merges={stats['semantic_merges_fired']}  "
              f"fitted={n} gold={hits}", flush=True)

    payload = {"states_dir": states_dir, "model": MODEL, "threshold": thr,
               "scored": False, "items": items}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(payload, indent=2))
    print(f"[phase1] structural results written to {out}", flush=True)

    if not answer:
        return 0

    from rem.npu_client import NpuClient
    from evals.battery.answerer import answer_question
    npu = NpuClient(settings)
    for item in items:
        # rebuild fitted text deterministically for the answer
        state = MemoryState.load(
            next(r["state_file"] for r in records if r["question_id"] == item["question_id"]))
        matcher = FullFactEmbeddingMatcher(embed, threshold=thr)
        new_state, _ = resupersede_state(state, matcher)
        fitted, text, _ = fit_render_aware(selector, new_state, item["question"],
                                           settings.read_fit_tokens)
        needles = GOLD_NEEDLES.get(item["question_id"], [])
        ans = (answer_question(npu, context=text, question=item["question"]) or "").strip()
        low = ans.lower()
        contains = any(x.lower() in low for x in needles)
        if not item["fits_budget"]:
            mode = "size"
        elif not all(item["gold_in_fitted"].values()):
            mode = "retrieval-recall"
        elif contains:
            mode = "pass"
        else:
            mode = "temporal-structure"
        item.update(brief_answer=ans, answer_contains_gold=contains, failure_mode=mode)
        print(f"[{item['question_id']}] {mode}: {ans[:90]}", flush=True)

    payload["scored"] = True
    payload["mix"] = {}
    for item in items:
        payload["mix"][item["failure_mode"]] = payload["mix"].get(item["failure_mode"], 0) + 1
    Path(out).write_text(json.dumps(payload, indent=2))
    print(f"\nMIX (re-superseded, lexical): {payload['mix']}\nWritten to {out}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--states-dir", default="bench/battery/states")
    ap.add_argument("--out", default="bench/battery/supersession_endtoend.json")
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--answer", action="store_true")
    args = ap.parse_args()
    return run(args.states_dir, args.out, args.threshold, args.answer)


if __name__ == "__main__":
    raise SystemExit(main())

"""Diagnose REM's compacted-memory size and gold survival on one item.

Non-comparative diagnostic. Runs the real REM compaction path on a single
knowledge-update item with a deliberately large assemble window so the memory
renders instead of raising ContextLimitExceeded, then reports:

  * per-tier token breakdown (summaries vs ledger vs verbatim) of the assembled
    prompt, summing exactly to the total;
  * how many summaries / ledger entries the compaction produced;
  * whether the gold survived compaction (three independent checks);
  * whether REM actually answers the question correctly given the full memory.

This separates "write recall works, read path is just too big" (gold present,
answer correct) from "summary-fidelity / write-recall failure" (gold lost).

It does NOT produce a REM-vs-truncation verdict; the window is not token-matched.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from evals.battery.answerer import answer_question
from evals.battery.context_managers import RemContextManager
from evals.battery.judge import judge_answer, make_client as make_judge
from evals.battery.longmemeval_loader import load_knowledge_update
from rem.config import Settings
from rem.memory.facts_ledger import get_extraction_stats, reset_extraction_stats
from rem.memory.tiers import count_tokens
from rem.npu_client import NpuClient

GEMMA = "gemma4-it:e2b"

# Assemble window for the diagnostic only. Large enough to clear the observed
# ~40k overflow so the memory renders; NOT token-matched to the budget.
DIAG_WINDOW_TOKENS = 64000

# The answering model (gemma4-it:e2b) rejects prompts above ~32-40k tokens with
# HTTP 400 "Max length reached!". Fit the answer context under this so the
# answer step tests "is the gold reachable in the compacted tiers?" instead of
# crashing. Headroom left for the question + 256-token answer.
MODEL_FIT_TOKENS = 28000

_HEADERS = [
    "=== SYSTEM ===",
    "=== TASK ===",
    "=== EPISODIC SUMMARIES ===",
    "=== FACTS LEDGER ===",
    "=== SEMANTIC RECALL ===",
    "=== VERBATIM TRANSCRIPT ===",
]


def tier_breakdown(assembled: str) -> dict[str, int]:
    """Slice the assembled prompt on its section headers and count each tier.

    The slices partition the whole string, so the counts sum to the total.
    """
    found = sorted(
        ((assembled.find(h), h) for h in _HEADERS if assembled.find(h) != -1)
    )
    out: dict[str, int] = {}
    for i, (pos, h) in enumerate(found):
        end = found[i + 1][0] if i + 1 < len(found) else len(assembled)
        out[h.strip("= ").lower()] = count_tokens(assembled[pos:end])
    return out


def gold_survival(item, cm: RemContextManager, assembled: str) -> dict:
    """Three independent checks for whether the gold survived into the memory."""
    ctx = assembled.lower()

    # Gather the verbatim gold turns (from the answer sessions).
    gold_turns = [
        turn.get("content", "")
        for sess in item.sessions
        if sess.session_id in item.answer_session_ids
        for turn in sess.turns
    ]

    # 1. The manager's own heuristic (first-6-words of a gold turn present).
    heuristic = cm.evidence_retained(item.answer_session_ids)

    # 2. The gold *answer* string appears (often paraphrased away, so this is a
    #    strong-positive / weak-negative signal).
    answer_present = bool(item.answer) and item.answer.lower() in ctx

    # 3. Per gold turn, does its first salient 8-word run survive verbatim?
    survived = 0
    for txt in gold_turns:
        snippet = " ".join(txt.split()[:8]).lower()
        if snippet and snippet in ctx:
            survived += 1

    # 4. Salient-token scan: distinctive words from the gold answer (numbers and
    #    Capitalized terms), counted in the assembled memory. Survives paraphrase
    #    better than a verbatim snippet match.
    import re
    salient = sorted({
        w for w in re.findall(r"[A-Z][a-zA-Z]+|\d+", item.answer or "")
        if w.isdigit() or len(w) > 1  # keep single-digit facts (e.g. "5 engineers")
    })
    keyword_hits = {w: ctx.count(w.lower()) for w in salient}

    return {
        "n_gold_turns": len(gold_turns),
        "heuristic_retained": heuristic,
        "answer_string_present": answer_present,
        "gold_turns_snippet_survived": survived,
        "salient_keyword_hits": keyword_hits,
        "gold_answer": item.answer,
        "gold_turns_preview": [t[:200] for t in gold_turns[:3]],
    }


def run(data: str, max_gold_recency: float, out: str) -> int:
    items = load_knowledge_update(data, limit=1, max_gold_recency=max_gold_recency)
    if not items:
        print("No matching knowledge-update items.", file=sys.stderr)
        return 2
    it = items[0]
    print(f"item={it.question_id} gold_recency={it.gold_recency:.2f} "
          f"sessions={len(it.sessions)} "
          f"turns={sum(len(s.turns) for s in it.sessions)}", flush=True)

    npu = NpuClient(Settings(summarizer_model=GEMMA))
    cm = RemContextManager(
        client=npu,
        settings=Settings(summarizer_model=GEMMA, max_context_tokens=DIAG_WINDOW_TOKENS),
    )

    reset_extraction_stats()
    t0 = time.time()
    cm.ingest(it.sessions, budget_tokens=1000)
    assembled = cm.assemble()
    ingest_secs = round(time.time() - t0, 1)
    extraction = get_extraction_stats()

    # Persist the compacted state immediately so the 75-min compaction is never
    # lost again: all further analysis can load this NPU-free.
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    state_path = out_path.with_name(out_path.stem + "_state.json")
    cm._state.save(state_path)

    state = cm._state  # diagnostic: inspect the compacted state directly
    total_tokens = count_tokens(assembled)
    breakdown = tier_breakdown(assembled)
    counts = {
        "summaries": len(state.summaries),
        "ledger_entries": len(state.ledger.entries),
        "ledger_active": len(state.ledger.active_entries()),
        "verbatim_turns": len(state.turns),
        "compactions": cm.stats().compactions,
    }
    survival = gold_survival(it, cm, assembled)

    # Write the STRUCTURAL payload before the fragile answer call, so an answer
    # failure can never discard the expensive measurements again.
    payload = {
        "diagnostic": "memory-overflow-breakdown",
        "note": "non-comparative; assemble window is NOT token-matched to budget",
        "item": it.question_id,
        "gold_recency": it.gold_recency,
        "budget_tokens": 1000,
        "assemble_window_tokens": DIAG_WINDOW_TOKENS,
        "model_fit_tokens": MODEL_FIT_TOKENS,
        "ingest_secs": ingest_secs,
        "assembled_total_tokens": total_tokens,
        "exceeds_model_window": total_tokens > MODEL_FIT_TOKENS,
        "tier_token_breakdown": breakdown,
        "memory_counts": counts,
        "extraction": extraction,
        "gold_survival": survival,
        "state_file": str(state_path),
        "question": it.question,
        "timestamp": time.time(),
        # answer fields filled in below (may fail independently)
        "rem_full_answer": None,
        "rem_full_answer_error": None,
        "rem_fitted_answer": None,
        "rem_fitted_judged_correct": None,
        "rem_fitted_judge_reason": None,
    }

    def _write():
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    _write()  # structural data is now durable regardless of what follows

    print(f"\nassembled_total_tokens: {total_tokens}  (ingest {ingest_secs}s)")
    print(f"tier breakdown: {breakdown}")
    print(f"memory counts: {counts}")
    print(f"extraction: {extraction}")
    print(f"gold survival: heuristic={survival['heuristic_retained']} "
          f"answer_present={survival['answer_string_present']} "
          f"gold_turn_snippets={survival['gold_turns_snippet_survived']}"
          f"/{survival['n_gold_turns']}  "
          f"keyword_hits={survival['salient_keyword_hits']}")
    print(f"gold answer: {it.answer}")
    print(f"state saved: {state_path}")

    # Answer attempt 1: full memory. Expected to fail (>model window) — recorded,
    # not fatal.
    try:
        payload["rem_full_answer"] = answer_question(
            npu, context=assembled, question=it.question).strip()
    except Exception as e:  # noqa: BLE001 - record any NPU failure verbatim
        payload["rem_full_answer_error"] = f"{type(e).__name__}: {e}"
    _write()
    print(f"full-memory answer: {payload['rem_full_answer'] or payload['rem_full_answer_error']}")

    # Answer attempt 2: fitted to the model window (keep head: summaries+ledger
    # come first in the assembler order). Tests whether the gold is *reachable*
    # in the compacted tiers given a model-sized read budget.
    fitted = assembled if total_tokens <= MODEL_FIT_TOKENS else assembled[: MODEL_FIT_TOKENS * 4]
    try:
        ans = answer_question(npu, context=fitted, question=it.question).strip()
        judge = make_judge()
        verdict = judge_answer(judge, question=it.question, gold=it.answer, model_answer=ans)
        payload["rem_fitted_answer"] = ans
        payload["rem_fitted_judged_correct"] = verdict.correct
        payload["rem_fitted_judge_reason"] = verdict.reason
    except Exception as e:  # noqa: BLE001
        payload["rem_fitted_judge_reason"] = f"answer/judge failed: {type(e).__name__}: {e}"
    _write()

    print(f"fitted({MODEL_FIT_TOKENS}tok) correct: {payload['rem_fitted_judged_correct']}  "
          f"answer: {(payload['rem_fitted_answer'] or '')[:200]}")
    print(f"Written to {out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Diagnose REM memory size + gold survival")
    ap.add_argument("--data", required=True)
    ap.add_argument("--max-gold-recency", type=float, default=0.33)
    ap.add_argument("--out", default="bench/battery/diag_memory.json")
    args = ap.parse_args()
    return run(args.data, args.max_gold_recency, args.out)


if __name__ == "__main__":
    raise SystemExit(main())

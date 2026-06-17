"""Run the battery spike: truncation vs REM on LongMemEval knowledge-update."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Add project root to path for imports
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from evals.battery.aggregate import aggregate
from evals.battery.answerer import answer_question
from evals.battery.context_managers import RemContextManager, TruncationContextManager
from evals.battery.judge import judge_answer, make_client as make_judge
from evals.battery.longmemeval_loader import load_knowledge_update
from evals.battery.models import ArmRun
from rem.config import Settings
from rem.memory.assembler import ContextLimitExceeded
from rem.npu_client import NpuClient

GEMMA = "gemma4-it:e2b"


def run(data: str, budget: int, limit: int | None, out: str) -> int:
    items = load_knowledge_update(data, limit=limit)
    if not items:
        print("No knowledge-update items found.", file=sys.stderr)
        return 2

    npu = NpuClient(Settings(summarizer_model=GEMMA))
    judge = make_judge()
    runs: list[ArmRun] = []

    for it in items:
        arms = {
            "truncation": TruncationContextManager(),
            "rem": RemContextManager(client=npu,
                                     settings=Settings(summarizer_model=GEMMA,
                                                       compact_trigger_tokens=budget,
                                                       max_context_tokens=budget * 4)),
        }
        for name, cm in arms.items():
            try:
                cm.ingest(it.sessions, budget_tokens=budget)
                ctx = cm.assemble()
            except ContextLimitExceeded as e:
                runs.append(ArmRun(
                    question_id=it.question_id, arm=name,
                    assembled_tokens=0, evidence_retained=False,
                    model_answer="", judged_correct=False,
                    judge_reason=f"context overflow: {e}",
                ))
                continue
            ans = answer_question(npu, context=ctx, question=it.question)
            v = judge_answer(judge, question=it.question, gold=it.answer, model_answer=ans)
            runs.append(ArmRun(
                question_id=it.question_id, arm=name,
                assembled_tokens=cm.stats().assembled_tokens,
                evidence_retained=cm.evidence_retained(it.answer_session_ids),
                model_answer=ans, judged_correct=v.correct, judge_reason=v.reason,
            ))
        print(f"[{it.question_id}] done", flush=True)

    result = aggregate(runs, n_questions=len(items))
    payload = {
        "eval": "battery-spike", "bench": "LongMemEval/knowledge-update",
        "answerer": GEMMA, "judge": "claude-haiku-4-5", "budget_tokens": budget,
        "n_questions": result.n_questions, "valid": result.valid,
        "invalid_reason": result.invalid_reason,
        "arm_accuracy": result.arm_accuracy,
        "arm_evidence_retention": result.arm_evidence_retention,
        "runs": [r.__dict__ for r in result.runs],
        "timestamp": time.time(),
    }
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"\nVALID: {result.valid}  ({result.invalid_reason})")
    print(f"accuracy:  {result.arm_accuracy}")
    print(f"retention: {result.arm_evidence_retention}")
    print(f"Written to {out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="REM battery spike (truncation vs REM)")
    ap.add_argument("--data", required=True, help="Path to LongMemEval JSON")
    ap.add_argument("--budget", type=int, default=8000, help="Context token budget B")
    ap.add_argument("--limit", type=int, default=None, help="Max knowledge-update questions")
    ap.add_argument("--out", default="bench/battery/spike_results.json")
    args = ap.parse_args()
    return run(args.data, args.budget, args.limit, args.out)


if __name__ == "__main__":
    raise SystemExit(main())

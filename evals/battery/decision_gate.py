"""One-day decision gate: does memory QUALITY remain the binding constraint?

Three read arms over the frozen-suite knowledge-update items, same answerer
(gemma4-it:e2b) and same judge (Claude haiku) held constant, so the ONLY thing
that varies is how evidence is selected and serialized:

  * current  — LexicalSelector @ read_fit_tokens. The existing "packed context":
               every candidate gets a positive recency score and the budget is
               filled in score order (selector.py). This is the arm whose 28k
               distractor-fill the external review flagged, and the arm the
               recorded Gate-2 result ("lexical mix == recency mix") used. We also
               record, separately, whether the LITERAL production path (full
               no-selector assemble) overflows max_context_tokens — it does on big
               states, which is itself a finding, not a read-quality result.
  * sparse   — SparseChronologicalSelector: top-k, relevance-floored,
               chronologically rendered. Sparse means sparse; never budget-fill.
  * oracle   — gold answer-sessions ONLY, taken from the RAW LongMemEval file with
               session dates + speaker roles, rendered chronologically. Built from
               raw because our ingestion drops dates/roles/session_id, so this is a
               genuine lossless CEILING, not another lossy path.

Pre-registered decision rule (fixed BEFORE any run, so results can't be
reinterpreted after the fact):
  * Each (arm, item) is answered REPS times (default 3) because the "deterministic"
    backend has varied across runs at temperature zero.
  * An item PASSES an arm iff a MAJORITY of its reps are judged correct.
  * Arm score = fraction of items passing (majority vote). We also report the raw
    per-rep pass-rate and its spread, and break results out PER QUESTION-TYPE.

Reading the matrix (also pre-registered):
  * oracle also fails  OR  all arms ~ equal      -> reader/model/benchmark bound;
                                                    a strategic FREEZE is justified.
  * oracle wins, sparse fails                     -> retrieval/representation deficient.
  * sparse ~ oracle  (>> current)                 -> current PACKING is the problem;
                                                    ship the sparse read path.
  * sparse/oracle expose clean update failures    -> temporal genuinely material;
                                                    continue memory work.

LIMIT: this set is knowledge-update ONLY, so it CANNOT observe stale-leakage /
abstention regressions. Do not generalize a "good enough" read past KU from it.

NPU SAFETY: no NPU (and no Claude) call happens unless --run is passed. The default
is a fully offline dry-run with stub answerer/judge that exercises the whole
pipeline (selection, oracle build, aggregation) so the harness can be validated
while the NPU is busy.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from rem.config import Settings
from rem.memory.assembler import assemble, ContextLimitExceeded
from rem.memory.selector import LexicalSelector, SparseChronologicalSelector
from rem.memory.tiers import MemoryState, count_tokens
from evals.battery.mix_report_selector import fit_render_aware

GEMMA = "gemma4-it:e2b"

# The 6 fresh fixed-extractor KU states (clean held-out) vs the 4 reused dev states
# (PRE-FIX extraction; diagnostic only — degraded facts would confound a read result).
FRESH_KU = {"ce6d2d27", "945e3d21", "6071bd76", "22d2cb42", "dfde3500", "affe2881"}
PREFIX_DIAGNOSTIC = {"3ba21379", "c6853660", "cc5ded98", "9bbe84a2"}


# --------------------------------------------------------------------------- #
# Oracle arm: build a lossless chronological evidence set from RAW LongMemEval.
# --------------------------------------------------------------------------- #
def load_raw_index(raw_path: str) -> dict[str, dict]:
    raw = json.loads(Path(raw_path).read_text(encoding="utf-8"))
    return {e["question_id"]: e for e in raw}


def build_oracle_context(raw_entry: dict) -> str:
    """Gold answer-sessions only, with dates + roles, oldest -> newest."""
    sids = raw_entry["haystack_session_ids"]
    sessions = raw_entry["haystack_sessions"]
    dates = raw_entry.get("haystack_dates") or [""] * len(sids)
    if not (len(sids) == len(sessions) == len(dates)):
        raise ValueError(
            f"raw alignment mismatch: sids={len(sids)} sessions={len(sessions)} "
            f"dates={len(dates)} for {raw_entry.get('question_id')}"
        )
    gold_ids = set(raw_entry.get("answer_session_ids", []))
    gold = [(d, sid, turns) for sid, turns, d in zip(sids, sessions, dates)
            if sid in gold_ids]
    gold.sort(key=lambda x: x[0])  # date strings are fixed-width -> lexical == chrono

    parts = []
    for date, sid, turns in gold:
        lines = [f"=== SESSION {date} ==="]
        for t in turns:
            lines.append(f"{t['role'].upper()}: {t['content']}")
        parts.append("\n".join(lines))
    return "\n\n".join(parts)


# --------------------------------------------------------------------------- #
# Selector arms over captured state.
# --------------------------------------------------------------------------- #
def selector_context(selector, state: MemoryState, question: str,
                     settings: Settings) -> str:
    fitted, text, _ = fit_render_aware(selector, state, question,
                                       settings.read_fit_tokens)
    return text


def full_assemble_overflow(state: MemoryState, question: str,
                           settings: Settings) -> dict:
    """Does the LITERAL serving path (full ledger, no selector) fit max_context?"""
    try:
        text = assemble(state, system="", task=question, settings=settings)
        return {"overflows": False, "tokens": count_tokens(text),
                "cap": settings.max_context_tokens}
    except ContextLimitExceeded as exc:
        return {"overflows": True, "tokens": None,
                "cap": settings.max_context_tokens, "detail": str(exc)}


def arm_context(arm: str, state: MemoryState | None, question: str,
                raw_entry: dict, settings: Settings) -> str:
    if arm == "current":
        return selector_context(LexicalSelector(), state, question, settings)
    if arm == "sparse":
        return selector_context(SparseChronologicalSelector(), state, question, settings)
    if arm == "oracle":
        return build_oracle_context(raw_entry)
    raise ValueError(f"unknown arm {arm}")


# --------------------------------------------------------------------------- #
# Run + aggregation.
# --------------------------------------------------------------------------- #
ARMS = ("current", "sparse", "oracle")


def generate_answers(items, states_dir, raw_index, *, arms, reps, answerer, settings):
    """STAGE 1 (NPU): produce + persist rep answers per (arm, item). No judging.

    Decoupled from judging so the NPU-bound work can run while the box is free and
    the cheap Claude judge can run later (e.g. once ANTHROPIC_API_KEY is available).
    """
    sdir = Path(states_dir)
    results = {a: [] for a in arms}
    overflow = {}

    for it in items:
        qid = it["question_id"]
        question, gold = it["question"], it["answer"]
        raw_entry = raw_index.get(qid, {})
        qtype = raw_entry.get("question_type", it.get("category", "unknown"))
        diagnostic = qid in PREFIX_DIAGNOSTIC

        state = None
        sfile = sdir / f"{qid}_state.json"
        if sfile.exists():
            state = MemoryState.load(str(sfile))
            overflow[qid] = full_assemble_overflow(state, question, settings)

        for arm in arms:
            if arm in ("current", "sparse") and state is None:
                continue
            rec = {"qid": qid, "qtype": qtype, "diagnostic": diagnostic,
                   "question": question, "gold": gold}
            try:
                ctx = arm_context(arm, state, question, raw_entry, settings)
            except Exception as exc:  # noqa: BLE001 - record, don't crash the sweep
                rec.update(error=str(exc), rep_answers=[])
                results[arm].append(rec)
                continue
            rec["ctx_tokens"] = count_tokens(ctx)
            rec["rep_answers"] = [(answerer(ctx, question) or "").strip()
                                  for _ in range(reps)]
            results[arm].append(rec)
            print(f"  [answers][{arm}][{qid}] reps={len(rec['rep_answers'])} "
                  f"ctx_tokens={rec['ctx_tokens']}", flush=True)

    return results, overflow


def attach_judgments(results, *, arms, reps, judge, include_diagnostic):
    """STAGE 2 (Claude): grade persisted answers, compute majority + summary."""
    for arm in arms:
        for rec in results.get(arm, []):
            if "error" in rec or "rep_answers" not in rec:
                continue
            rec["rep_correct"] = [
                bool(judge(question=rec["question"], gold=rec["gold"],
                           model_answer=a))
                for a in rec["rep_answers"]
            ]
            rec["majority_correct"] = sum(rec["rep_correct"]) > (reps / 2)
    return summarize(results, arms, reps, include_diagnostic)


def summarize(results, arms, reps, include_diagnostic):
    summary = {"reps": reps, "arms": {}, "by_qtype": {}}
    for arm in arms:
        rows = [r for r in results[arm]
                if include_diagnostic or not r.get("diagnostic")]
        scored = [r for r in rows if "error" not in r and "rep_correct" in r]
        n = len(scored)
        majorities = [r["majority_correct"] for r in scored]
        rep_rates = [sum(r["rep_correct"]) / reps for r in scored if r["rep_correct"]]
        summary["arms"][arm] = {
            "n_items": n,
            "majority_pass": sum(majorities),
            "majority_pass_rate": round(sum(majorities) / n, 3) if n else None,
            "mean_rep_rate": round(statistics.mean(rep_rates), 3) if rep_rates else None,
            "rep_rate_spread": round(statistics.pstdev(rep_rates), 3)
                if len(rep_rates) > 1 else 0.0,
            "errors": [r["qid"] for r in rows if "error" in r],
        }
        bytype = defaultdict(list)
        for r in scored:
            bytype[r["qtype"]].append(r["majority_correct"])
        summary["by_qtype"][arm] = {
            qt: {"n": len(v), "pass": sum(v)} for qt, v in bytype.items()
        }
    return summary


# --------------------------------------------------------------------------- #
# Stubs (offline) and real (NPU/Claude) answerer + judge.
# --------------------------------------------------------------------------- #
def stub_answerer(item_gold_by_q):
    """Offline 'perfect reader' for plumbing validation."""
    def answerer(ctx, question):
        return item_gold_by_q.get(question, "")
    return answerer


def stub_judge():
    def judge(*, question, gold, model_answer):
        return gold.lower() in (model_answer or "").lower()
    return judge


def real_answerer(settings):
    from rem.npu_client import NpuClient
    from evals.battery.answerer import answer_question
    npu = NpuClient(settings)
    return lambda ctx, question: answer_question(npu, context=ctx, question=question)


def real_judge():
    from evals.battery import judge as judge_mod
    jclient = judge_mod.make_client()

    def judge(*, question, gold, model_answer):
        v = judge_mod.judge_answer(jclient, question=question, gold=gold,
                                   model_answer=model_answer)
        return v.correct
    return judge


def _print_matrix(summary, overflow, arms):
    for arm in arms:
        a = summary["arms"].get(arm)
        if not a:
            continue
        print(f"  {arm:8s} majority_pass={a['majority_pass']}/{a['n_items']} "
              f"rate={a['majority_pass_rate']} mean_rep={a['mean_rep_rate']} "
              f"spread={a['rep_rate_spread']}", flush=True)
    if overflow:
        n_overflow = sum(1 for v in overflow.values() if v["overflows"])
        print(f"  serving-path overflow (full no-selector assemble): "
              f"{n_overflow}/{len(overflow)} states exceed max_context_tokens",
              flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default="bench/memory_methods/development_manifest.json")
    ap.add_argument("--states-dir", default="bench/memory_methods/states")
    ap.add_argument("--raw", default="/home/keith/datasets/longmemeval/longmemeval_s")
    ap.add_argument("--out", default="bench/memory_methods/decision_gate.json")
    ap.add_argument("--answers-file", default="bench/memory_methods/decision_gate_answers.json",
                    help="Where stage 'answers' writes, and stage 'judge' reads.")
    ap.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--items", choices=["fresh", "all"], default="fresh",
                    help="fresh = 6 clean KU states; all = +4 pre-fix diagnostic states")
    ap.add_argument("--run", action="store_true",
                    help="Use real backends. Without it, an offline dry-run validates "
                         "the pipeline end-to-end with stubs.")
    ap.add_argument("--stage", choices=["both", "answers", "judge"], default="both",
                    help="answers = NPU only (persist answers); judge = Claude over a "
                         "saved answers file; both = answer then judge in one pass.")
    args = ap.parse_args()

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    ku = [it for it in manifest["items"] if it["category"] == "knowledge-update"]
    if args.items == "fresh":
        items = [it for it in ku if it["question_id"] in FRESH_KU]
        include_diagnostic = False
    else:
        items = [it for it in ku if it["question_id"] in (FRESH_KU | PREFIX_DIAGNOSTIC)]
        include_diagnostic = True

    raw_index = load_raw_index(args.raw)
    settings = Settings(summarizer_model=GEMMA)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    # --- STAGE: judge (read saved answers, grade with Claude) ---------------- #
    if args.stage == "judge":
        saved = json.loads(Path(args.answers_file).read_text(encoding="utf-8"))
        results, overflow = saved["results"], saved.get("serving_path_overflow", {})
        judge = real_judge() if args.run else stub_judge()
        summary = attach_judgments(results, arms=args.arms, reps=args.reps,
                                   judge=judge, include_diagnostic=include_diagnostic)
        payload = {"mode": "JUDGE", "items": saved.get("items"),
                   "summary": summary, "results": results,
                   "serving_path_overflow": overflow}
        Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[decision_gate] stage=judge run={args.run} items={len(items)}", flush=True)
        _print_matrix(summary, overflow, args.arms)
        print(f"Written to {args.out}")
        return 0

    # --- STAGE: answers / both (generate via NPU or stub) -------------------- #
    answerer = real_answerer(settings) if args.run else \
        stub_answerer({it["question"]: it["answer"] for it in items})
    print(f"[decision_gate] stage={args.stage} run={args.run} items={len(items)} "
          f"arms={args.arms} reps={args.reps}", flush=True)
    results, overflow = generate_answers(
        items, args.states_dir, raw_index, arms=args.arms, reps=args.reps,
        answerer=answerer, settings=settings,
    )

    if args.stage == "answers":
        payload = {"mode": "ANSWERS", "items": [it["question_id"] for it in items],
                   "results": results, "serving_path_overflow": overflow}
        Path(args.answers_file).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"  answers persisted -> {args.answers_file} "
              f"(judge later: --stage judge --run --answers-file {args.answers_file})",
              flush=True)
        _print_matrix({"arms": {}}, overflow, args.arms)
        return 0

    # both
    judge = real_judge() if args.run else stub_judge()
    summary = attach_judgments(results, arms=args.arms, reps=args.reps,
                               judge=judge, include_diagnostic=include_diagnostic)
    payload = {"mode": "BOTH", "items": [it["question_id"] for it in items],
               "summary": summary, "results": results,
               "serving_path_overflow": overflow}
    Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _print_matrix(summary, overflow, args.arms)
    print(f"Written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

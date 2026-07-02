"""Run paired native selectors over the frozen 30-item captured-state suite."""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

from evals.battery.answerer import answer_question
from evals.battery.judge import judge_answer, make_client as make_judge
from evals.memory_methods.artifacts import (
    ItemRun, MemoryMethodArtifact, ModelVersions,
)
from evals.memory_methods.native import NativeSelectorArm
from rem.config import Settings
from rem.memory.selector import (
    LexicalSelector,
    PackedLexicalSelector,
    RecencySelector,
    SparseChronologicalSelector,
)
from rem.memory.tiers import MemoryState
from rem.npu_client import NpuClient

SELECTORS = {
    "recency": RecencySelector,
    "lexical": LexicalSelector,
    "lexical-packed": PackedLexicalSelector,
    "sparse": lambda: SparseChronologicalSelector(prefer_newest=False),
    "safe-sparse": lambda: SparseChronologicalSelector(prefer_newest=False),
    "path-a-candidate": lambda: SparseChronologicalSelector(prefer_newest=True),
}


def _revision() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _p95(values: list[float]) -> float | None:
    values = sorted(values)
    if not values:
        return None
    return values[max(0, int(len(values) * 0.95 + 0.999) - 1)]


def _sources_cover_gold(source_references, gold_groups: list[list[int]]) -> bool | None:
    if not gold_groups:
        return None
    observed = {
        turn_id for reference in source_references for turn_id in reference.turn_ids
    }
    return all(observed.intersection(group) for group in gold_groups)


def run(
    manifest_path: str,
    output_path: str,
    arm_names: list[str],
    budgets: list[int],
    *,
    score: bool = False,
    score_budgets: list[int] | None = None,
    answer_repetitions: int = 1,
    configuration_extra: dict | None = None,
) -> int:
    if answer_repetitions < 1:
        raise ValueError("answer_repetitions must be at least one")
    manifest_bytes = Path(manifest_path).read_bytes()
    manifest = json.loads(manifest_bytes)
    missing_states = [
        item["state_file"] for item in manifest["items"]
        if not Path(item["state_file"]).is_file()
    ]
    if missing_states:
        raise ValueError(
            "capture incomplete; missing state files: " + ", ".join(missing_states)
        )
    states = {
        item["question_id"]: MemoryState.load(item["state_file"])
        for item in manifest["items"]
    }
    arms = {
        name: NativeSelectorArm(
            name,
            SELECTORS[name](),
            lambda namespace, sessions: states[namespace].model_copy(deep=True),
        )
        for name in arm_names
    }
    npu = NpuClient(Settings(summarizer_model="gemma4-it:e2b")) if score else None
    judge = make_judge() if score else None
    runs: list[ItemRun] = []
    for item in manifest["items"]:
        namespace = item["question_id"]
        for arm in arms.values():
            arm.ingest(namespace, [])
            arm.await_ready(namespace, timeout=0)
            full_references = NativeSelectorArm._references(states[namespace])
            write_recall = _sources_cover_gold(
                full_references, item.get("gold_source_turn_groups", [])
            )
            for budget in budgets:
                error = None
                result = arm.recall(namespace, item["question"], budget)
                answer = None
                verdict = None
                model_answers = []
                judge_reasons = []
                should_score = score and (
                    score_budgets is None or budget in score_budgets
                )
                if should_score:
                    try:
                        verdicts = []
                        for _ in range(answer_repetitions):
                            answer = answer_question(
                                npu, context=result.rendered_context,
                                question=item["question"],
                                use_taxonomy=True,
                            )
                            verdict = judge_answer(
                                judge, question=item["question"],
                                gold=item["answer"], model_answer=answer,
                            )
                            model_answers.append(answer)
                            judge_reasons.append(verdict.reason)
                            verdicts.append(verdict.correct)
                        judged_correct = (
                            sum(verdicts) > answer_repetitions / 2
                        )
                    except Exception as exc:  # keep the paired artifact intact
                        error = f"{type(exc).__name__}: {exc}"
                        judged_correct = None
                else:
                    judged_correct = None
                runs.append(ItemRun(
                    question_id=namespace,
                    category=item["category"],
                    arm=arm.name,
                    budget_tokens=budget,
                    memory_tokens=result.token_count,
                    source_references=[asdict(ref) for ref in result.source_references],
                    candidate_count=result.candidate_count,
                    ingest_latency_ms=arm.stats(namespace)["ingest_latency_ms"],
                    read_latency_ms=result.latency_ms,
                    write_recall=write_recall,
                    read_recall=_sources_cover_gold(
                        result.source_references,
                        item.get("gold_source_turn_groups", []),
                    ),
                    judged_correct=judged_correct,
                    model_answers=model_answers,
                    judge_reasons=judge_reasons,
                    context_overflow=result.token_count > budget,
                    provenance_lost=(
                        result.candidate_count > 0 and not result.source_references
                    ),
                    error=error,
                ))
            arm.reset(namespace)
        print(f"[{namespace}] complete", flush=True)

    category_accuracy = {}
    for arm_name in arm_names:
        for category in manifest["selection"]["categories"]:
            judged = [
                run for run in runs
                if run.arm == arm_name and run.category == category
                and run.judged_correct is not None
            ]
            if judged:
                category_accuracy[f"{arm_name}:{category}"] = round(
                    sum(bool(run.judged_correct) for run in judged) / len(judged), 4
                )
    configuration = {
        "arms": arm_names,
        "budgets": budgets,
        "scored": score,
        "scored_budgets": score_budgets if score else [],
        "answer_repetitions": answer_repetitions if score else 0,
        "generated_at": time.time(),
    }
    configuration.update(configuration_extra or {})
    artifact = MemoryMethodArtifact(
        repository_revision=_revision(),
        source_manifest=manifest_path,
        source_dataset_sha256=manifest["source_sha256"],
        models=ModelVersions(),
        configuration=configuration,
        runs=runs,
        category_accuracy=category_accuracy,
        read_latency_p95_ms=_p95([run.read_latency_ms for run in runs]),
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(artifact.model_dump_json(indent=2), encoding="utf-8")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", default="bench/memory_methods/development_manifest.json"
    )
    parser.add_argument(
        "--out", default="bench/memory_methods/native-development.json"
    )
    parser.add_argument(
        "--arms", nargs="+", choices=sorted(SELECTORS), default=list(SELECTORS)
    )
    parser.add_argument("--budgets", nargs="+", type=int, default=[8000, 28000])
    parser.add_argument(
        "--score", action="store_true",
        help="Run the fixed Gemma answerer and Claude judge (otherwise recall-only).",
    )
    parser.add_argument(
        "--score-budgets", nargs="+", type=int,
        help="Only answer/judge these budgets; recall still runs at every budget.",
    )
    parser.add_argument("--answer-repetitions", type=int, default=1)
    args = parser.parse_args()
    return run(
        args.manifest, args.out, args.arms, args.budgets, score=args.score,
        score_budgets=args.score_budgets,
        answer_repetitions=args.answer_repetitions,
    )


if __name__ == "__main__":
    raise SystemExit(main())

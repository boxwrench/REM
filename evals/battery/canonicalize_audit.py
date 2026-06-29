"""Measure post-hoc slot canonicalization over captured REM states.

This audit is NPU-free.  It records the conservative and aggressive string
ceilings without changing the writer or the persisted captures.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from evals.battery.mix_report import GOLD_NEEDLES, STRUCTURE_NEEDLES  # noqa: E402
from evals.battery.write_recall_audit import audit_state, needle_in_full  # noqa: E402
from rem.memory.canonicalize import canonical_slot_key, recanonicalize  # noqa: E402
from rem.memory.tiers import MemoryState  # noqa: E402


def _revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_project_root, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _normalized_value(value: object) -> str:
    return " ".join(str(value).lower().split())


def merge_risk_groups(state: MemoryState, granularity: str) -> list[dict]:
    """Describe groups where canonicalization would collapse distinct values.

    The audit cannot infer whether two values are a legitimate update.  These
    groups are therefore review probes, not automatically labelled false merges.
    Synthetic negative fixtures provide the hard false-merge gate.
    """
    groups: dict[str, list] = defaultdict(list)
    for entry in state.ledger.active_entries():
        if entry.slot_key and entry.slot_value:
            groups[canonical_slot_key(entry.slot_key, granularity)].append(entry)

    probes = []
    for key, entries in sorted(groups.items()):
        values = sorted({_normalized_value(entry.slot_value) for entry in entries})
        if len(entries) >= 2 and len(values) >= 2:
            probes.append({
                "canonical_key": key,
                "slot_keys": sorted({entry.slot_key for entry in entries}),
                "values": values,
                "source_turn_ids": sorted(entry.source_turn_id for entry in entries),
            })
    return probes


def analyze_state(
    state: MemoryState,
    question_id: str,
    granularity: str,
    gold_needles: list[str] | None = None,
    structure_needles: list[str] | None = None,
) -> dict:
    gold_needles = gold_needles or []
    structure_needles = structure_needles or []
    before = audit_state(state, gold_needles, structure_needles)
    transformed = recanonicalize(state, granularity)
    after = audit_state(transformed, gold_needles, structure_needles)
    gold_survival = {
        needle: {
            "before": needle_in_full(state, needle),
            "after": needle_in_full(transformed, needle),
        }
        for needle in gold_needles
    }
    active_reduction = before["ledger_active"] - after["ledger_active"]
    fragmented_reduction = before["fragmented_values"] - after["fragmented_values"]
    return {
        "question_id": question_id,
        "granularity": granularity,
        "before": before,
        "after": after,
        "active_reduction": active_reduction,
        "active_reduction_rate": round(
            active_reduction / before["ledger_active"], 4
        ) if before["ledger_active"] else 0.0,
        "fragmented_values_reduction": fragmented_reduction,
        "fragmented_values_reduction_rate": round(
            fragmented_reduction / before["fragmented_values"], 4
        ) if before["fragmented_values"] else 0.0,
        "gold_survival": gold_survival,
        "all_gold_preserved": all(
            item["before"] == "absent" or item["after"] != "absent"
            for item in gold_survival.values()
        ),
        "merge_risk_groups": merge_risk_groups(state, granularity),
    }


def _summary(rows: list[dict]) -> dict:
    active_before = sum(row["before"]["ledger_active"] for row in rows)
    active_after = sum(row["after"]["ledger_active"] for row in rows)
    frag_before = sum(row["before"]["fragmented_values"] for row in rows)
    frag_after = sum(row["after"]["fragmented_values"] for row in rows)
    return {
        "active_before": active_before,
        "active_after": active_after,
        "active_reduction_rate": round(
            (active_before - active_after) / active_before, 4
        ) if active_before else 0.0,
        "fragmented_values_before": frag_before,
        "fragmented_values_after": frag_after,
        "fragmented_values_reduction_rate": round(
            (frag_before - frag_after) / frag_before, 4
        ) if frag_before else 0.0,
        "superseded_before": sum(row["before"]["superseded"] for row in rows),
        "superseded_after": sum(row["after"]["superseded"] for row in rows),
        "merge_risk_groups": sum(len(row["merge_risk_groups"]) for row in rows),
        "all_gold_preserved": all(row["all_gold_preserved"] for row in rows),
    }


def run(states_dir: str, out: str) -> int:
    states_path = Path(states_dir)
    manifest_path = states_path / "manifest.json"
    if not manifest_path.exists():
        print(f"no manifest at {manifest_path}", file=sys.stderr)
        return 2
    records = json.loads(manifest_path.read_text(encoding="utf-8"))
    by_granularity: dict[str, list[dict]] = {"full": [], "subject": []}
    for record in records:
        state_path = Path(record["state_file"])
        if not state_path.is_absolute():
            state_path = _project_root / state_path
        state = MemoryState.load(state_path)
        question_id = record["question_id"]
        for granularity in by_granularity:
            row = analyze_state(
                state,
                question_id,
                granularity,
                GOLD_NEEDLES.get(question_id, []),
                STRUCTURE_NEEDLES.get(question_id, []),
            )
            by_granularity[granularity].append(row)
            print(
                f"[{question_id} {granularity:7s}] "
                f"active={row['before']['ledger_active']}->{row['after']['ledger_active']} "
                f"fragmented={row['before']['fragmented_values']}"
                f"->{row['after']['fragmented_values']} "
                f"risk_groups={len(row['merge_risk_groups'])} "
                f"gold={row['all_gold_preserved']}",
                flush=True,
            )

    payload = {
        "schema_version": 1,
        "experiment": "post-hoc-string-slot-canonicalization",
        "repository_revision": _revision(),
        "created_at": time.time(),
        "states_manifest": str(manifest_path),
        "n_items": len(records),
        "answer_regression_check": "not_run",
        "granularities": {
            key: {"summary": _summary(rows), "items": rows}
            for key, rows in by_granularity.items()
        },
    }
    output_path = Path(out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Written to {output_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--states-dir", default="bench/battery/states")
    parser.add_argument("--out", default="bench/battery/canonicalize_audit.json")
    args = parser.parse_args()
    return run(args.states_dir, args.out)


if __name__ == "__main__":
    raise SystemExit(main())

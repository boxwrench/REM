"""Path B step 1: NPU-free role-aware post-hoc re-key audit.

This experiment never saves a transformed state and does not participate in
extraction.  It asks whether a conservative, role-aware key compatibility rule
can collapse captured-ledger fragmentation without confusing known negative
roles or named instances.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from evals.battery.write_recall_audit import audit_state  # noqa: E402
from evals.memory_methods.state_selection import select_state_records  # noqa: E402
from rem.memory.canonicalize import _tokens  # noqa: E402
from rem.memory.tiers import MemoryState  # noqa: E402

FRESH_IDS = {
    "ce6d2d27", "945e3d21", "6071bd76", "22d2cb42", "dfde3500", "affe2881",
}

# Evidence is deliberately frozen to the three held-out reads named in the
# kickoff.  Their hashes make the generated artifact reproducible and show
# exactly which human-reviewed false-merge evidence governed the sentinels.
AUDIT_REPORTS = (
    "bench/memory_methods/rem_supersession_heldout_audit.md",
    "bench/memory_methods/rem_supersession_heldout_audit_2026-07-01.md",
    "bench/memory_methods/rem_supersession_heldout_audit_2026-07-02.md",
)

GOLD_NEEDLES = {
    "ce6d2d27": ["Friday"],
    "945e3d21": ["three times a week"],
    "6071bd76": ["5 ounces of water"],
    "22d2cb42": ["Main St"],
    "dfde3500": ["Wednesday"],
    "affe2881": ["32"],
}

_ROLE_DIMENSIONS = {
    "boundary": ({"start", "begin", "opening"}, {"end", "finish", "closing"}),
    "range": ({"min", "minimum", "low", "lower"}, {"max", "maximum", "high", "upper"}),
    "storage": ({"fridge", "refrigerator", "refrigerated"}, {"freezer", "freeze", "frozen"}),
    "exercise": ({"set"}, {"rep", "repetition"}),
}

NEGATIVE_SENTINELS = (
    ("start_end", "event dates.start date", "event dates.end date"),
    ("min_max", "price range.minimum price", "price range.maximum price"),
    ("fridge_freezer", "chicken.refrigerator duration", "chicken.freezer duration"),
    ("sets_reps", "plank.sets", "plank.reps"),
    ("per_instance", "onibus coffee.walk distance", "streamer coffee.walk distance"),
)

POSITIVE_SENTINELS = (
    (
        "coffee_ratio_update",
        "coffee ratio.tablespoon of coffee per ounces of water",
        "coffee brewing.ratio",
    ),
    ("bird_count_update", "bird species.count", "species count.total species count"),
)


def _parts(slot_key: str) -> tuple[set[str], set[str]]:
    subject, separator, attribute = slot_key.rpartition(".")
    if not separator:
        subject, attribute = slot_key, ""
    return _tokens(subject), _tokens(attribute)


def _role_conflict(all_a: set[str], all_b: set[str]) -> bool:
    for left, right in _ROLE_DIMENSIONS.values():
        if (all_a & left and all_b & right) or (all_a & right and all_b & left):
            return True
    return False


def same_role(slot_a: str, slot_b: str) -> bool:
    """Conservative identity candidate: same instance and compatible role.

    Shared subject evidence is mandatory, so two named instances with the same
    attribute remain separate.  A shared non-subject role (including a token
    moved between subject and attribute) or a subset relation handles common
    extractor rephrasings.  Explicit role opposites always veto a match.
    """
    subject_a, attribute_a = _parts(slot_a)
    subject_b, attribute_b = _parts(slot_b)
    all_a, all_b = subject_a | attribute_a, subject_b | attribute_b
    if not all_a or not all_b or _role_conflict(all_a, all_b):
        return False
    shared_subject = subject_a & subject_b
    if not shared_subject:
        return False
    if all_a <= all_b or all_b <= all_a:
        return True
    shared_role = (all_a & all_b) - shared_subject
    if not shared_role:
        return False
    # Distinct leftover names on both subjects are evidence for two instances.
    return not (subject_a - all_b and subject_b - all_a)


def _groups(state: MemoryState) -> list[list[int]]:
    """Return compatible active-entry components without mutating ``state``."""
    entries = state.ledger.entries
    active = [i for i, entry in enumerate(entries) if entry.status == "active" and entry.slot_key]
    parent = {i: i for i in active}
    root_members = {i: {i} for i in active}

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    by_subject_token: dict[str, list[int]] = defaultdict(list)
    compared: set[tuple[int, int]] = set()
    for index in active:
        subject, _ = _parts(entries[index].slot_key or "")
        for token in sorted(subject):
            for other in by_subject_token[token]:
                pair = (other, index)
                if pair in compared:
                    continue
                compared.add(pair)
                if same_role(entries[other].slot_key or "", entries[index].slot_key or ""):
                    left, right = find(other), find(index)
                    # A compatible middle spelling must not bridge two roles or
                    # instances that would be incompatible directly.
                    cross_compatible = all(
                        same_role(
                            entries[left_member].slot_key or "",
                            entries[right_member].slot_key or "",
                        )
                        for left_member in root_members[left]
                        for right_member in root_members[right]
                    )
                    if left != right and cross_compatible:
                        parent[right] = left
                        root_members[left].update(root_members.pop(right))
            by_subject_token[token].append(index)

    grouped: dict[int, list[int]] = defaultdict(list)
    for index in active:
        grouped[find(index)].append(index)
    return [members for members in grouped.values() if len(members) > 1]


def role_aware_rekey(state: MemoryState) -> MemoryState:
    """Return an in-memory audit transform; every fact and history entry remains."""
    out = state.model_copy(deep=True)
    for members in _groups(state):
        source_entries = state.ledger.entries
        # Stable audit-only key.  It is never written into a capture.
        canonical = "audit-role:" + min(source_entries[i].slot_key or "" for i in members)
        ordered = sorted(members, key=lambda i: source_entries[i].source_turn_id)
        newest = ordered[-1]
        for index in ordered:
            entry = out.ledger.entries[index]
            entry.slot_key = canonical
            if index != newest:
                entry.status = "stale"
                entry.superseded_by_turn_id = source_entries[newest].source_turn_id
    out.ledger.rendered_text = None
    return out


def _active_needle(state: MemoryState, needle: str) -> bool:
    low = needle.lower()
    return any(low in entry.text.lower() for entry in state.ledger.active_entries()) or any(
        low in (summary.rendered_text or summary.text).lower() for summary in state.summaries
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def analyze(state: MemoryState, question_id: str) -> dict:
    needles = GOLD_NEEDLES[question_id]
    before = audit_state(state, needles)
    transformed = role_aware_rekey(state)
    after = audit_state(transformed, needles)
    survival = {
        needle: {
            "active_before": _active_needle(state, needle),
            "active_after": _active_needle(transformed, needle),
        }
        for needle in needles
    }
    return {
        "question_id": question_id,
        "before": before,
        "after": after,
        "fragmented_values_reduction": before["fragmented_values"] - after["fragmented_values"],
        "gold_survival": survival,
        "gold_preserved": all(
            not item["active_before"] or item["active_after"] for item in survival.values()
        ),
        "candidate_groups": len(_groups(state)),
    }


def run(manifest: str, out: str) -> int:
    records = [
        record for record in select_state_records(manifest=manifest)
        if record["question_id"] in FRESH_IDS
    ]
    if {record["question_id"] for record in records} != FRESH_IDS:
        print("all six fresh Path B captures are required", file=sys.stderr)
        return 2

    reports = []
    for relative in AUDIT_REPORTS:
        path = _ROOT / relative
        if not path.exists():
            print(f"missing held-out audit report: {relative}", file=sys.stderr)
            return 2
        reports.append({"path": relative, "sha256": _sha256(path)})

    negatives = [
        {"family": family, "left": left, "right": right, "preserved": not same_role(left, right)}
        for family, left, right in NEGATIVE_SENTINELS
    ]
    positives = [
        {"case": case, "left": left, "right": right, "collides": same_role(left, right)}
        for case, left, right in POSITIVE_SENTINELS
    ]
    rows = []
    for record in records:
        row = analyze(MemoryState.load(record["state_file"]), record["question_id"])
        rows.append(row)
        print(
            f"[{row['question_id']}] fragmented={row['before']['fragmented_values']}"
            f"->{row['after']['fragmented_values']} groups={row['candidate_groups']} "
            f"gold={row['gold_preserved']}",
            flush=True,
        )

    before = sum(row["before"]["fragmented_values"] for row in rows)
    after = sum(row["after"]["fragmented_values"] for row in rows)
    rate = (before - after) / before if before else 0.0
    summary = {
        "fragmented_values_before": before,
        "fragmented_values_after": after,
        "fragmentation_reduction_rate": round(rate, 4),
        "fragmentation_target": 0.5,
        "fragmentation_target_met": rate >= 0.5,
        "all_gold_preserved": all(row["gold_preserved"] for row in rows),
        "all_negative_sentinels_preserved": all(item["preserved"] for item in negatives),
        "all_positive_sentinels_collide": all(item["collides"] for item in positives),
    }
    summary["go"] = all((
        summary["fragmentation_target_met"], summary["all_gold_preserved"],
        summary["all_negative_sentinels_preserved"], summary["all_positive_sentinels_collide"],
    ))
    payload = {
        "schema_version": 1,
        "experiment": "path-b-post-hoc-role-aware-rekey",
        "repository_revision": _revision(),
        "created_at": time.time(),
        "manifest": manifest,
        "n_items": len(rows),
        "mutates_persisted_states": False,
        "audit_reports": reports,
        "summary": summary,
        "negative_sentinels": negatives,
        "positive_sentinels": positives,
        "items": rows,
    }
    output = Path(out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Path B decision: {'GO' if summary['go'] else 'NO-GO'}\nWritten to {output}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="bench/memory_methods/development_manifest.json")
    parser.add_argument("--out", default="bench/memory_methods/path_b_role_key_audit.json")
    args = parser.parse_args()
    return run(args.manifest, args.out)


if __name__ == "__main__":
    raise SystemExit(main())

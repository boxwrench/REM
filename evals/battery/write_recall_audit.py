"""NPU-free write-recall audit over captured MemoryStates.

Separates write recall (was the fact extracted into the compacted state at all)
from read recall (did the read path keep it), and quantifies the write-side
quality signal the failure mix pointed at: slot supersession barely fires because
the extractor assigns a fresh slot key for nearly every mention, so the same value
is stored under many distinct keys (fragmentation) and genuine updates never
collapse. Operates on the states captured by capture_states.py; no NPU.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from rem.memory.tiers import MemoryState
from evals.battery.mix_report import GOLD_NEEDLES, STRUCTURE_NEEDLES


def needle_in_full(state, needle: str) -> str:
    """Tier of the FULL (unfitted) state carrying the needle: slot/free/summary/absent.

    This is write recall: presence anywhere in the compacted state, independent of
    what the read path later keeps.
    """
    low = needle.lower()
    for e in state.ledger.entries:
        if low in e.text.lower():
            return "slot" if e.slot_key else "free"
    for s in state.summaries:
        txt = s.rendered_text if s.rendered_text is not None else s.text
        if low in txt.lower():
            return "summary"
    return "absent"


def value_fragmentation(state) -> dict[str, list[str]]:
    """Normalized active slot_value -> the distinct slot keys carrying it (>1 = fragmented).

    A value under multiple keys is a fact the extractor re-keyed instead of
    superseding; it is a lower bound on fragmentation (exact-value match only).
    """
    by_val: dict[str, set] = defaultdict(set)
    for e in state.ledger.active_entries():
        if e.slot_key and e.slot_value:
            v = " ".join(str(e.slot_value).lower().split())
            by_val[v].add(e.slot_key)
    return {v: sorted(keys) for v, keys in by_val.items() if len(keys) > 1}


def audit_state(state, gold_needles, structure_needles=None) -> dict:
    structure_needles = structure_needles or []
    total = len(state.ledger.entries)
    active = len(state.ledger.active_entries())
    stale = total - active
    frag = value_fragmentation(state)
    return {
        "ledger_total": total,
        "ledger_active": active,
        "superseded": stale,
        "supersession_rate": round(stale / total, 4) if total else 0.0,
        "write_recall_gold": {n: needle_in_full(state, n) for n in gold_needles},
        "write_recall_structure": {n: needle_in_full(state, n) for n in structure_needles},
        "fragmented_values": len(frag),
        "fragmentation_examples": dict(
            sorted(frag.items(), key=lambda kv: -len(kv[1]))[:5]),
    }


def run(states_dir: str, out: str) -> int:
    sdir = Path(states_dir)
    manifest_path = sdir / "manifest.json"
    if not manifest_path.exists():
        print(f"no manifest at {manifest_path}", file=sys.stderr)
        return 2
    records = json.loads(manifest_path.read_text(encoding="utf-8"))

    rows = []
    for r in records:
        state = MemoryState.load(r["state_file"])
        a = audit_state(state, GOLD_NEEDLES.get(r["question_id"], []),
                        STRUCTURE_NEEDLES.get(r["question_id"], []))
        a["question_id"] = r["question_id"]
        rows.append(a)
        worst = max((len(k) for k in a["fragmentation_examples"].values()), default=0)
        print(f"[{r['question_id']}] active={a['ledger_active']:4d} "
              f"superseded={a['superseded']:3d} ({a['supersession_rate']:.1%})  "
              f"fragmented_values={a['fragmented_values']:3d} worst_value_in={worst}_keys  "
              f"write_gold={a['write_recall_gold']}", flush=True)

    payload = {"states_dir": str(sdir), "n_items": len(rows), "items": rows}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWritten to {out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="NPU-free write-recall + fragmentation audit")
    ap.add_argument("--states-dir", default="bench/battery/states")
    ap.add_argument("--out", default="bench/battery/write_recall_audit.json")
    args = ap.parse_args()
    return run(args.states_dir, args.out)


if __name__ == "__main__":
    raise SystemExit(main())

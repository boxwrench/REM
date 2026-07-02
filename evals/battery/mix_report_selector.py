"""Selector-parameterized failure-mix over captured MemoryStates (Gate 2).

A variant of ``mix_report.py`` that runs ANY ``MemorySelector`` (recency, lexical,
lexical-packed) over the five captured oldest-gold states, so the query-aware
lexical read path can be measured against the recency baseline on the same items
and the same needle logic.

For each (selector, item):

  * fit the captured state to ``read_fit_tokens`` with a render-aware budget search
    (each selector keeps its own ranking; we only shrink the budget it is given
    until the *rendered* assembly fits, so section scaffolding can never push the
    slice over budget — spec D1/D6);
  * record gold / structure needle presence and the tier that carries each;
  * for the temporal item (031748ae) dump the ordered "engineer view": every
    rendered ledger line mentioning an engineer count, in render order, with its
    slot_key and status — this is what reveals whether query-aware retrieval +
    ordered history actually surfaces a started -> now structure;
  * optionally take ONE brief NPU answer per item (the only paid step) to separate
    ``pass`` from ``temporal-structure``.

NPU-free by default. Pass ``--answer`` for the single brief answer per item.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from rem.config import Settings
from rem.memory.assembler import assemble
from rem.memory.selector import LexicalSelector, PackedLexicalSelector, RecencySelector
from rem.memory.tiers import MemoryState, count_tokens
from evals.battery.diagnose_memory import gold_in_fitted
from evals.battery.mix_report import GOLD_NEEDLES, STRUCTURE_NEEDLES

GEMMA = "gemma4-it:e2b"

SELECTORS = {
    "recency": RecencySelector,
    "lexical": LexicalSelector,
    "lexical-packed": PackedLexicalSelector,
}

# Items whose then/now structure we dump in detail (only the temporal one today).
ENGINEER_RE = re.compile(r"\b(\d+|one|two|three|four|five|six)\s+engineer", re.I)


def fit_render_aware(selector, state, question, budget):
    """Fit ``state`` to ``budget`` rendered tokens using ``selector``'s own ranking.

    The selectors estimate per-item cost without rendering, so the assembled text
    can land just over budget once section headers are added. Rather than trim
    tiers selector-specifically, we shrink the budget handed to the selector and
    re-select until the rendered assembly fits. Each selector therefore keeps its
    own strategy; only the effective budget moves. Returns
    ``(fitted_state, fitted_text, fitted_tokens)``.
    """
    effective = budget
    fitted = selector.select(state, question, effective)
    text = assemble(fitted, system="", task=question)
    n = count_tokens(text)
    guard = 0
    while n > budget and effective > 0 and guard < 12:
        effective -= (n - budget) + 64  # overshoot + small margin
        fitted = selector.select(state, question, max(0, effective))
        text = assemble(fitted, system="", task=question)
        n = count_tokens(text)
        guard += 1
    return fitted, text, n


def needle_tier(fitted, needle) -> str:
    """Which tier of the FITTED state carries the needle (number-aware match)."""
    from evals.battery.needles import present, value_aware_entry
    for e in fitted.ledger.entries:
        if present(needle, e.text) or value_aware_entry(needle, e.slot_key, e.slot_value):
            return "slot" if e.slot_key else "free"
    for s in fitted.summaries:
        txt = s.rendered_text if s.rendered_text is not None else s.text
        if present(needle, txt):
            return "summary"
    return "absent"


def engineer_view(fitted) -> dict:
    """Ordered engineer-count lines in the fitted ledger (render order) + status.

    Surfaces whether the read path presents an ordered started -> now structure or
    just fragmented active facts. Also reports whether include_stale rendering is on
    and whether any engineer line is actually stale (i.e. supersession fired).
    """
    lines = []
    for e in fitted.ledger.entries:
        if ENGINEER_RE.search(e.text):
            lines.append({
                "turn": e.source_turn_id,
                "status": e.status,
                "slot_key": e.slot_key,
                "text": e.text,
            })
    return {
        "include_stale_on_render": fitted.ledger.include_stale_on_render,
        "n_engineer_lines": len(lines),
        "any_stale": any(ln["status"] == "stale" for ln in lines),
        "distinct_slot_keys": sorted({ln["slot_key"] or "" for ln in lines}),
        "lines_in_render_order": lines,
    }


def label_item(selector, state, question, needles, structure, settings, answerer):
    fitted, text, n = fit_render_aware(selector, state, question,
                                       settings.read_fit_tokens)
    fits = n <= settings.read_fit_tokens
    hits = gold_in_fitted(text, needles)
    tiers = {x: needle_tier(fitted, x) for x in needles}
    struct_hits = gold_in_fitted(text, structure)
    struct_tiers = {x: needle_tier(fitted, x) for x in structure}

    brief = answer_contains_gold = answer_contains_struct = None
    if answerer is not None:
        brief = (answerer(text, question) or "").strip()
        low = brief.lower()
        answer_contains_gold = any(x.lower() in low for x in needles)
        answer_contains_struct = any(x.lower() in low for x in structure)

    if not fits:
        mode = "size"
    elif not all(hits.values()):
        mode = "retrieval-recall"
    elif answerer is None:
        mode = "needs-answer"
    elif answer_contains_gold:
        mode = "pass"
    else:
        mode = "temporal-structure"

    out = {
        "fitted_tokens": n, "fits_budget": fits,
        "gold_in_fitted": hits, "needle_tiers": tiers,
        "structure_in_fitted": struct_hits, "structure_tiers": struct_tiers,
        "brief_answer": brief, "answer_contains_gold": answer_contains_gold,
        "answer_contains_structure": answer_contains_struct,
        "failure_mode": mode,
    }
    if ENGINEER_RE.search(question) or any("engineer" in x.lower() for x in needles):
        out["engineer_view"] = engineer_view(fitted)
    return out


def run(states_dir, out, arm_names, settings=None, answerer=None) -> int:
    settings = settings or Settings(summarizer_model=GEMMA)
    sdir = Path(states_dir)
    manifest_path = sdir / "manifest.json"
    if not manifest_path.exists():
        print(f"no manifest at {manifest_path}", file=sys.stderr)
        return 2
    records = json.loads(manifest_path.read_text(encoding="utf-8"))

    arms = {}
    for name in arm_names:
        selector = SELECTORS[name]()
        rows, counts = [], {}
        for r in records:
            state = MemoryState.load(r["state_file"])
            needles = GOLD_NEEDLES.get(r["question_id"], [])
            structure = STRUCTURE_NEEDLES.get(r["question_id"], [])
            lab = label_item(selector, state, r["question"], needles, structure,
                             settings, answerer)
            lab["question_id"] = r["question_id"]
            lab["gold_recency"] = r.get("gold_recency")
            rows.append(lab)
            counts[lab["failure_mode"]] = counts.get(lab["failure_mode"], 0) + 1
            ev = lab.get("engineer_view")
            ev_note = ""
            if ev is not None:
                ev_note = (f" eng_lines={ev['n_engineer_lines']} "
                           f"any_stale={ev['any_stale']} "
                           f"slots={len(ev['distinct_slot_keys'])}")
            print(f"[{name}][{r['question_id']}] mode={lab['failure_mode']:16s} "
                  f"fitted={lab['fitted_tokens']:6d} fits={lab['fits_budget']} "
                  f"gold_tiers={lab['needle_tiers']}{ev_note}", flush=True)
        arms[name] = {"mix": counts, "items": rows}
        print(f"  -> {name} MIX: {counts}\n", flush=True)

    payload = {"states_dir": str(sdir), "n_items": len(records),
               "selectors": arm_names, "scored": answerer is not None, "arms": arms}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Written to {out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--states-dir", default="bench/battery/states")
    ap.add_argument("--out", default="bench/battery/mix_report_selectors.json")
    ap.add_argument("--arms", nargs="+", choices=sorted(SELECTORS),
                    default=["recency", "lexical", "lexical-packed"])
    ap.add_argument("--answer", action="store_true",
                    help="Take one brief NPU answer per item (separates pass from "
                         "temporal-structure). Off by default = fully NPU-free.")
    args = ap.parse_args()
    answerer = None
    if args.answer:
        from rem.npu_client import NpuClient
        from evals.battery.answerer import answer_question
        npu = NpuClient(Settings(summarizer_model=GEMMA))

        def answerer(ctx, q):
            return answer_question(npu, context=ctx, question=q)
    return run(args.states_dir, args.out, args.arms, answerer=answerer)


if __name__ == "__main__":
    raise SystemExit(main())

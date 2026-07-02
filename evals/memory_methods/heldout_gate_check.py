"""Incremental held-out check: run NPU-free per state as the capture lands them.

Path C captures the 20 temporal/multi held-out items one at a time (~75-90 min each).
This reports, for every held-out state that has landed, the NPU-free read-side signal:
  - extraction-recall: is the gold answer present AND active in the ledger (write side)?
  - read serialization: does the SAFE sparse arm surface the gold in its fitted view,
    and does the CURRENT (lexical budget-fill) arm bury it? (the read-side effect)
It does NOT call the answerer (that needs the NPU the capture is using). Once
--max (default 5) held-out states exist, it prints STOP so the capture can be halted
and the definitive answerer gate run on the accumulated states.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from rem.config import Settings
from rem.memory.assembler import assemble
from rem.memory.selector import LexicalSelector, SparseChronologicalSelector
from rem.memory.tiers import MemoryState

HELDOUT_CATEGORIES = {"temporal-reasoning", "multi-session"}
KU_STATES = {"3ba21379", "cc5ded98", "c6853660", "9bbe84a2",
             "ce6d2d27", "945e3d21", "6071bd76", "22d2cb42", "dfde3500", "affe2881"}


def _needles(answer: str) -> list[str]:
    """Salient lowercased tokens/short phrases from the gold answer for a loose
    presence check (temporal golds are phrasey; this is directional, not a judge)."""
    ans = answer.strip().lower()
    toks = [t for t in re.findall(r"[a-z0-9]+", ans) if len(t) > 2]
    # keep the full answer and its content tokens
    return [ans] + toks[:6]


def _present(needles: list[str], hay: str) -> bool:
    hay = hay.lower()
    # require the full-answer phrase OR >=half the content tokens
    if needles and needles[0] in hay:
        return True
    toks = needles[1:]
    if not toks:
        return False
    hit = sum(1 for t in toks if re.search(rf"\b{re.escape(t)}\b", hay))
    return hit >= max(1, len(toks) // 2)


def check_state(item: dict, states_dir: Path, settings: Settings) -> dict:
    qid = item["question_id"]
    st = MemoryState.load(str(states_dir / f"{qid}_state.json"))
    needles = _needles(item["answer"])
    # write side: gold present AND active in the ledger?
    active_hit = any(_present(needles, (e.text or "") + " " + (e.slot_value or ""))
                     for e in st.ledger.entries if e.status == "active")
    q = item["question"]
    surfaced = {}
    for name, sel in (("current", LexicalSelector()),
                      ("sparse", SparseChronologicalSelector())):
        fitted = sel.select(st, q, settings.read_fit_tokens)
        surfaced[name] = _present(needles, assemble(fitted, system="", task=q))
    return {
        "qid": qid, "category": item["category"], "entries": len(st.ledger.entries),
        "gold_extracted_active": active_hit,
        "gold_surfaced_current": surfaced["current"],
        "gold_surfaced_sparse": surfaced["sparse"],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default="bench/memory_methods/development_manifest.json")
    ap.add_argument("--states-dir", default="bench/memory_methods/states")
    ap.add_argument("--max", type=int, default=5)
    args = ap.parse_args()

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    states_dir = Path(args.states_dir)
    settings = Settings()
    heldout = [it for it in manifest["items"]
               if it["category"] in HELDOUT_CATEGORIES
               and (states_dir / f"{it['question_id']}_state.json").exists()]

    print(f"held-out states landed: {len(heldout)}/{args.max} "
          f"(target); {len(heldout)}/20 total temporal+multi")
    for it in heldout:
        r = check_state(it, states_dir, settings)
        print(f"  [{r['qid']}] {r['category']:18} entries={r['entries']:4} "
              f"gold_active={r['gold_extracted_active']!s:5} "
              f"current_surfaces={r['gold_surfaced_current']!s:5} "
              f"sparse_surfaces={r['gold_surfaced_sparse']!s:5}")
    if len(heldout) >= args.max:
        print(f"\nSTOP: {len(heldout)} >= {args.max} held-out states. Halt capture "
              f"(kill $(cat bench/memory_methods/capture_temporal.pid)) and run the "
              f"definitive answerer gate on these {len(heldout)} states.")
    else:
        print(f"\nCONTINUE: waiting for {args.max - len(heldout)} more "
              f"(~75-90 min each).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

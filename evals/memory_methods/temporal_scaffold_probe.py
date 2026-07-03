"""One-off probe: does a reasoning-scaffold answer prompt rescue the 3 held-out
temporal items that the base answerer got 0/3 on? Same evidence (sparse = shipped
read path, and oracle = ceiling), only the system prompt changes. NPU-bound.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from rem.config import Settings
from rem.memory.tiers import MemoryState
from rem.npu_client import NpuClient
from evals.battery.decision_gate import arm_context, load_raw_index, GEMMA

SCAFFOLD = (
    "You answer questions using ONLY the provided conversation memory. "
    "Reason step by step before answering:\n"
    "- If the question asks for the ORDER or sequence of events, find each relevant "
    "event with its date/timestamp and list them earliest-first.\n"
    "- If it asks HOW MANY days or weeks passed between two events, find BOTH dates in "
    "the memory and compute the difference.\n"
    "- If the question names a person or thing that is NOT mentioned anywhere in the "
    "memory, answer that there is not enough information.\n"
    "End with a final line 'ANSWER: <concise answer>'."
)
QIDS = ["gpt4_45189cb4", "gpt4_fe651585_abs", "gpt4_7a0daae1"]


def main() -> int:
    s = Settings(summarizer_model=GEMMA)
    s.read_fit_tokens = 14000  # flm serving window shrank <27k
    npu = NpuClient(s)
    manifest = json.loads(Path("bench/memory_methods/development_manifest.json").read_text())
    by_id = {it["question_id"]: it for it in manifest["items"]}
    raw = load_raw_index("/home/keith/datasets/longmemeval/longmemeval_s")
    out = []
    for qid in QIDS:
        item = by_id[qid]
        st = MemoryState.load(f"bench/memory_methods/states/{qid}_state.json")
        q = item["question"]
        rec = {"qid": qid, "gold": item["answer"], "answers": {}}
        for arm in ["sparse", "oracle"]:
            ctx = arm_context(arm, st, q, raw.get(qid, {}), s)
            msgs = [{"role": "system", "content": SCAFFOLD},
                    {"role": "user", "content": f"=== MEMORY ===\n{ctx}\n\n=== QUESTION ===\n{q}"}]
            ans = npu.chat(msgs, model=GEMMA, max_tokens=512).strip()
            rec["answers"][arm] = ans
            print(f"[{qid}][{arm}] {ans[:200]}", flush=True)
        out.append(rec)
    Path("bench/memory_methods/temporal_scaffold_probe.json").write_text(
        json.dumps(out, indent=2))
    print("written", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

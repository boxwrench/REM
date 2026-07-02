"""Capture per-item compacted MemoryState for the oldest-gold battery items.

Runs the real ~75-min compaction once per item and persists each MemoryState so
the read-path / selector analysis can iterate NPU-free. Idempotent: skips items
whose state file already exists, so the multi-hour ingest is resumable.
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

from evals.battery.context_managers import RemContextManager  # noqa: E402
from evals.battery.longmemeval_loader import load_knowledge_update  # noqa: E402
from rem.config import Settings  # noqa: E402
from rem.memory.facts_ledger import (  # noqa: E402
    get_extraction_stats,
    reset_extraction_stats,
)
from rem.memory.tiers import count_tokens  # noqa: E402

GEMMA = "gemma4-it:e2b"
DIAG_WINDOW_TOKENS = 64000  # match diagnose_memory so all states are comparable


def _meta(it, state_path: Path) -> dict:
    return {
        "question_id": it.question_id,
        "question": it.question,
        "answer": it.answer,
        "answer_session_ids": it.answer_session_ids,
        "gold_recency": it.gold_recency,
        "n_sessions": len(it.sessions),
        "n_turns": sum(len(s.turns) for s in it.sessions),
        "state_file": str(state_path),
    }


def _upsert(records: list[dict], rec: dict) -> list[dict]:
    out = [r for r in records if r["question_id"] != rec["question_id"]]
    out.append(rec)
    out.sort(key=lambda r: r.get("gold_recency", 1.0))
    return out


def capture_item(it, out_dir: Path, make_cm, budget_tokens: int = 1000) -> dict:
    """Ingest one item, save its state, return the manifest record.

    make_cm() is a zero-arg context-manager factory (injected for tests).
    """
    out_dir = Path(out_dir)
    state_path = out_dir / f"{it.question_id}_state.json"
    t0 = time.time()
    reset_extraction_stats()
    cm = make_cm()
    cm.ingest(it.sessions, budget_tokens=budget_tokens)
    extraction = get_extraction_stats()
    ingest_secs = round(time.time() - t0, 1)
    assembled = cm.assemble()
    cm._state.save(state_path)
    rec = _meta(it, state_path)
    rec.update(
        assembled_total_tokens=count_tokens(assembled),
        ingest_secs=ingest_secs,
        captured_at=time.time(),
        extraction=extraction,
    )
    return rec


def run(data: str, out_dir: str, max_gold_recency: float = 0.33,
        limit: int | None = None, budget_tokens: int = 1000, make_cm=None) -> int:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.json"
    records = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else []

    items = load_knowledge_update(data, limit=limit, max_gold_recency=max_gold_recency)
    if not items:
        print("No matching knowledge-update items.", file=sys.stderr)
        return 2
    print(f"selected {len(items)} items; recency="
          f"{[round(it.gold_recency, 3) for it in items]}", flush=True)

    if make_cm is None:
        from rem.npu_client import NpuClient
        npu = NpuClient(Settings(summarizer_model=GEMMA))

        def make_cm():
            return RemContextManager(
                client=npu,
                settings=Settings(summarizer_model=GEMMA,
                                  max_context_tokens=DIAG_WINDOW_TOKENS))

    def _flush():
        manifest_path.write_text(json.dumps(records, indent=2), encoding="utf-8")

    for it in items:
        state_path = out / f"{it.question_id}_state.json"
        if state_path.exists():
            if not any(r["question_id"] == it.question_id for r in records):
                rec = _meta(it, state_path)
                rec.update(assembled_total_tokens=None, ingest_secs=None, captured_at=None)
                records = _upsert(records, rec)
                _flush()
            print(f"[{it.question_id}] state exists, skipping ingest", flush=True)
            continue
        print(f"[{it.question_id}] ingesting "
              f"(recency={it.gold_recency:.3f}, "
              f"{sum(len(s.turns) for s in it.sessions)} turns)…", flush=True)
        rec = capture_item(it, out, make_cm, budget_tokens=budget_tokens)
        records = _upsert(records, rec)
        _flush()
        print(f"[{it.question_id}] saved {rec['assembled_total_tokens']} tok in "
              f"{rec['ingest_secs']}s -> {rec['state_file']}", flush=True)

    print(f"manifest: {manifest_path} ({len(records)} records)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Capture per-item compacted MemoryState")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out-dir", default="bench/battery/states")
    ap.add_argument("--max-gold-recency", type=float, default=0.33)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--budget", type=int, default=1000)
    args = ap.parse_args()
    return run(args.data, args.out_dir, max_gold_recency=args.max_gold_recency,
               limit=args.limit, budget_tokens=args.budget)


if __name__ == "__main__":
    raise SystemExit(main())

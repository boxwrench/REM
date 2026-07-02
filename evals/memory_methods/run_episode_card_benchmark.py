"""Path D paired wall-clock gate for legacy vs one-call episode compaction.

Default mode is a fully offline plumbing replay. Pass ``--run`` only after the
Path C capture releases the NPU. Each repetition uses identical fresh turns and
alternates arm order to reduce thermal/order bias.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from rem.config import Settings
from rem.memory.compactor import compact_once
from rem.memory.tiers import MemoryState, Turn
from rem.npu_client import NpuClient

MODES = ("legacy_two_call", "episode_card")
PROMOTION_RULE = {
    "require_fact_signature_equivalence": True,
    "require_nonempty_summary": True,
    "require_episode_median_faster": True,
    "require_episode_npu_calls": 1,
    "require_legacy_npu_calls": 2,
}

_FACT = {
    "kind": "number",
    "source_turn_id": 3,
    "subject": "coffee brewing",
    "attribute": "ratio",
    "value": "1 tablespoon per 5 ounces of water",
    "is_correction": True,
}
_SUMMARY = "The preferred coffee ratio changed to one tablespoon per five ounces of water."


class _ReplayClient:
    """Offline deterministic responses for harness validation."""

    def __init__(self, mode: str) -> None:
        self.settings = Settings()
        if mode == "legacy_two_call":
            self._responses = [json.dumps([_FACT]), _SUMMARY]
        else:
            self._responses = [json.dumps({"facts": [_FACT], "summary": _SUMMARY})]

    def chat(self, *args, **kwargs) -> str:
        return self._responses.pop(0)


def benchmark_state() -> MemoryState:
    return MemoryState(turns=[
        Turn(
            role="user", content="My French press ratio is one tablespoon per 6 ounces.",
            turn_id=1, tokens=15, session_id="ratio-old", timestamp="2024-01-01",
        ),
        Turn(
            role="assistant", content="Noted.", turn_id=2, tokens=2,
            session_id="ratio-old", timestamp="2024-01-01",
        ),
        Turn(
            role="user", content="Correction: use one tablespoon per 5 ounces now.",
            turn_id=3, tokens=13, session_id="ratio-new", timestamp="2024-02-01",
        ),
        Turn(
            role="user", content="What is my current ratio?", turn_id=4, tokens=7,
            session_id="ratio-new", timestamp="2024-02-01",
        ),
    ])


def _fact_signatures(state: MemoryState) -> list[tuple]:
    return sorted(
        (
            entry.kind,
            entry.source_turn_id,
            entry.slot_key,
            entry.slot_value,
            entry.status,
            entry.session_id,
            entry.timestamp,
        )
        for entry in state.ledger.entries
    )


def _run_one(mode: str, client) -> dict:
    state = benchmark_state()
    settings = Settings(
        keep_recent_turns=1,
        compact_span_turns=3,
        deterministic_fact_capture=False,
        episode_card_consolidation=mode == "episode_card",
    )
    started = time.perf_counter()
    result = compact_once(state, client, settings)
    elapsed = time.perf_counter() - started
    return {
        "mode": mode,
        "compacted": result.compacted,
        "wall_seconds": round(elapsed, 6),
        "npu_elapsed_seconds": result.npu_elapsed_s,
        "npu_calls": result.npu_calls,
        "summary": result.new_summary,
        "fact_signatures": [list(item) for item in _fact_signatures(state)],
    }


def run(output: str, *, repetitions: int = 3, live: bool = False) -> dict:
    rows = []
    for repetition in range(repetitions):
        order = MODES if repetition % 2 == 0 else tuple(reversed(MODES))
        for mode in order:
            client = NpuClient(Settings()) if live else _ReplayClient(mode)
            row = _run_one(mode, client)
            row["repetition"] = repetition
            rows.append(row)
            print(
                f"[{repetition}][{mode}] compacted={row['compacted']} "
                f"calls={row['npu_calls']} wall={row['wall_seconds']:.3f}s",
                flush=True,
            )

    by_mode = {
        mode: [row for row in rows if row["mode"] == mode] for mode in MODES
    }
    medians = {
        mode: round(statistics.median(row["wall_seconds"] for row in mode_rows), 6)
        for mode, mode_rows in by_mode.items()
    }
    paired_equivalence = []
    for repetition in range(repetitions):
        pair = {row["mode"]: row for row in rows if row["repetition"] == repetition}
        paired_equivalence.append(
            pair["legacy_two_call"]["fact_signatures"]
            == pair["episode_card"]["fact_signatures"]
        )
    checks = {
        "all_compacted": all(row["compacted"] for row in rows),
        "fact_signature_equivalence": all(paired_equivalence),
        "nonempty_summaries": all(bool(row["summary"]) for row in rows),
        "call_counts": all(
            row["npu_calls"] == (1 if row["mode"] == "episode_card" else 2)
            for row in rows
        ),
        "episode_median_faster": medians["episode_card"] < medians["legacy_two_call"],
    }
    payload = {
        "schema_version": 1,
        "mode": "LIVE" if live else "OFFLINE_REPLAY",
        "repetitions": repetitions,
        "promotion_rule": PROMOTION_RULE,
        "median_wall_seconds": medians,
        "checks": checks,
        "promote": live and all(checks.values()),
        "runs": rows,
    }
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", default="bench/memory_methods/episode_card_wallclock.json"
    )
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument(
        "--run", action="store_true", help="Use the live local NPU; default is offline."
    )
    args = parser.parse_args()
    payload = run(args.out, repetitions=args.repetitions, live=args.run)
    print(f"Path D promote={payload['promote']} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

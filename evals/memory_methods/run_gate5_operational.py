"""NPU-free Gate 5 checks for REM's persisted memory operations.

The harness exercises public memory APIs on temporary state files. It does not
call a model and does not require a running sidecar server.
"""
from __future__ import annotations

import argparse
import json
import tempfile
from collections.abc import Callable
from pathlib import Path
from unittest.mock import patch

from rem.config import Settings
from rem.memory.compactor import compact_once
from rem.memory.sidecar import MemorySidecar
from rem.memory.tiers import MemoryState, Turn

PASS = "pass"
FAIL = "fail"
NOT_APPLICABLE = "not_applicable"


def _sidecar(vault_dir: Path) -> MemorySidecar:
    """Build a sidecar whose operations cannot schedule model work."""
    return MemorySidecar(
        settings=Settings(vault_dir=str(vault_dir)),
        memory_policy=lambda _state, _settings: False,
        scheduler=lambda *_args: None,
    )


def _restart_persistence(workspace: Path) -> dict:
    vault = workspace / "restart"
    first = _sidecar(vault)
    first_request = {
        "user": "restart-session",
        "messages": [{"role": "user", "content": "Remember cobalt."}],
    }
    _, state_path = first.process_chat_request(first_request)
    first.record_response(state_path, "I will remember cobalt.")

    before_restart = MemoryState.load(state_path)
    second = _sidecar(vault)
    resumed_request = {
        "user": "restart-session",
        "messages": [
            {"role": "user", "content": "Remember cobalt."},
            {"role": "assistant", "content": "I will remember cobalt."},
            {"role": "user", "content": "What should you remember?"},
        ],
    }
    second.process_chat_request(resumed_request)
    after_restart = MemoryState.load(state_path)

    contents = [turn.content for turn in after_restart.turns]
    passed = (
        [turn.content for turn in before_restart.turns]
        == contents[:2]
        and contents == [
            "Remember cobalt.",
            "I will remember cobalt.",
            "What should you remember?",
        ]
        and [turn.turn_id for turn in after_restart.turns] == [1, 2, 3]
        and all(turn.session_id == "restart-session" for turn in after_restart.turns)
    )
    return {
        "status": PASS if passed else FAIL,
        "evidence": {
            "turn_contents": contents,
            "turn_ids": [turn.turn_id for turn in after_restart.turns],
            "session_ids": [turn.session_id for turn in after_restart.turns],
            "fresh_sidecar_instance": True,
        },
    }


def _duplicate_ingest(workspace: Path) -> dict:
    sidecar = _sidecar(workspace / "duplicate")
    request = {
        "user": "duplicate-session",
        "messages": [{"role": "user", "content": "Store this once."}],
    }
    _, state_path = sidecar.process_chat_request(request)
    first = MemoryState.load(state_path)
    sidecar.process_chat_request(request)
    second = MemoryState.load(state_path)

    passed = (
        len(first.turns) == 1
        and second.model_dump() == first.model_dump()
        and second.turns[0].turn_id == 1
    )
    return {
        "status": PASS if passed else FAIL,
        "evidence": {
            "turn_count_after_first_ingest": len(first.turns),
            "turn_count_after_duplicate_ingest": len(second.turns),
            "state_unchanged": second.model_dump() == first.model_dump(),
        },
    }


def _partial_failure_atomicity(workspace: Path) -> dict:
    state_path = workspace / "atomic" / "memory_state.json"
    original = MemoryState(
        turns=[Turn(role="user", content="durable", turn_id=1, tokens=2)]
    )
    original.save(state_path)
    original_bytes = state_path.read_bytes()
    replacement = MemoryState(
        turns=[Turn(role="user", content="replacement", turn_id=2, tokens=3)]
    )

    failure_observed = False
    with patch(
        "rem.memory.tiers.os.replace",
        side_effect=OSError("simulated atomic-swap failure"),
    ):
        try:
            replacement.save(state_path)
        except OSError as exc:
            failure_observed = str(exc) == "simulated atomic-swap failure"

    loaded = MemoryState.load(state_path)
    temp_path = state_path.with_suffix(".tmp")
    passed = (
        failure_observed
        and state_path.read_bytes() == original_bytes
        and loaded.model_dump() == original.model_dump()
        and not temp_path.exists()
    )
    return {
        "status": PASS if passed else FAIL,
        "evidence": {
            "failure_propagated": failure_observed,
            "original_state_preserved": loaded.model_dump() == original.model_dump(),
            "temporary_file_removed": not temp_path.exists(),
        },
    }


class _FailingCompactionClient:
    """NPU-free client that fails at the episode-card call boundary."""

    def __init__(self) -> None:
        self.settings = Settings()
        self.calls = 0

    def chat(self, *_args, **_kwargs) -> str:
        self.calls += 1
        raise RuntimeError("simulated episode-card failure")


def _compaction_failure_rollback(workspace: Path) -> dict:
    state_path = workspace / "compaction" / "memory_state.json"
    original = MemoryState(turns=[
        Turn(role="user", content="alpha", turn_id=1, tokens=5),
        Turn(role="assistant", content="noted", turn_id=2, tokens=5),
        Turn(role="user", content="beta", turn_id=3, tokens=5),
        Turn(role="assistant", content="recent", turn_id=4, tokens=5),
    ])
    original.save(state_path)
    original_bytes = state_path.read_bytes()
    working = MemoryState.load(state_path)
    original_model = working.model_dump()
    client = _FailingCompactionClient()

    result = compact_once(
        working,
        client,
        Settings(
            keep_recent_turns=1,
            compact_span_turns=3,
            episode_card_consolidation=True,
        ),
    )

    serialized_working = working.model_dump_json(indent=2).encode("utf-8")
    persisted = MemoryState.load(state_path)
    passed = (
        result.compacted is False
        and result.turns_compacted == 0
        and result.npu_calls == 1
        and client.calls == 1
        and working.model_dump() == original_model
        and serialized_working == original_bytes
        and persisted.model_dump() == original_model
        and state_path.read_bytes() == original_bytes
    )
    return {
        "status": PASS if passed else FAIL,
        "evidence": {
            "failed_call_count": client.calls,
            "reported_compacted": result.compacted,
            "in_memory_model_preserved": working.model_dump() == original_model,
            "in_memory_bytes_preserved": serialized_working == original_bytes,
            "persisted_model_preserved": persisted.model_dump() == original_model,
            "persisted_bytes_preserved": state_path.read_bytes() == original_bytes,
        },
    }


def _execute(check: Callable[[Path], dict], workspace: Path) -> dict:
    try:
        return check(workspace)
    except Exception as exc:  # pragma: no cover - defensive artifact generation
        return {
            "status": FAIL,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _run_in_workspace(workspace: Path) -> dict:
    workspace.mkdir(parents=True, exist_ok=True)
    checks = {
        "restart_persistence": _execute(_restart_persistence, workspace),
        "duplicate_ingest_idempotency": _execute(_duplicate_ingest, workspace),
        "partial_failure_atomicity": _execute(
            _partial_failure_atomicity, workspace
        ),
        "compaction_failure_rollback": _execute(
            _compaction_failure_rollback, workspace
        ),
        "delayed_indexing": {
            "status": NOT_APPLICABLE,
            "reason": (
                "REM persists MemoryState synchronously and reads it directly; "
                "the API under test has no asynchronous indexing queue."
            ),
            "async_indexing_detected": False,
        },
    }
    applicable = [
        result for result in checks.values() if result["status"] != NOT_APPLICABLE
    ]
    return {
        "schema_version": 1,
        "mode": "NPU_FREE",
        "checks": checks,
        "applicable_checks_pass": all(
            result["status"] == PASS for result in applicable
        ),
        "gate_status": (
            PASS if all(result["status"] == PASS for result in applicable) else FAIL
        ),
    }


def run(output: str | Path, *, workspace: str | Path | None = None) -> dict:
    """Run the operational checks and persist a machine-readable artifact."""
    if workspace is None:
        with tempfile.TemporaryDirectory(prefix="rem-gate5-") as temp_dir:
            payload = _run_in_workspace(Path(temp_dir))
    else:
        payload = _run_in_workspace(Path(workspace))

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", default="bench/memory_methods/gate5_operational.json"
    )
    parser.add_argument(
        "--workspace",
        help="Keep state fixtures in this directory; default uses a temporary one.",
    )
    args = parser.parse_args()
    payload = run(args.out, workspace=args.workspace)
    print(f"Gate 5 operational status={payload['gate_status']} -> {args.out}")
    return 0 if payload["gate_status"] == PASS else 1


if __name__ == "__main__":
    raise SystemExit(main())

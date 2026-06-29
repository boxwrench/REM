"""Select (question_id, question, answer, state_file) records for replay runners.

Accepts either a frozen development manifest (``{"items": [...]}`` from
``freeze_manifest.py``) or a states-dir ``manifest.json`` (the flat list of capture
records written by ``capture_states.py``). Optionally filters by question_id and by
whether the state file exists on disk, so a runner can operate on a partially
captured suite (e.g. while the NPU capture is still mid-flight).
"""
from __future__ import annotations

import json
from pathlib import Path

_FIELDS = ("question_id", "question", "answer", "state_file")


def _normalize(rec: dict) -> dict:
    return {k: rec.get(k) for k in _FIELDS}


def select_state_records(
    states_dir: str | Path | None = None,
    manifest: str | Path | None = None,
    ids: list[str] | set[str] | None = None,
    require_exists: bool = True,
) -> list[dict]:
    """Return normalized capture records, newest selection logic resolved here.

    Exactly one source is used: ``manifest`` if given (a frozen development
    manifest or a flat list), otherwise ``states_dir``/manifest.json. When
    ``require_exists`` is set, records whose ``state_file`` is not yet on disk are
    dropped (and reported), so replay can run on a partially captured suite.
    """
    if manifest:
        payload = json.loads(Path(manifest).read_text(encoding="utf-8"))
        raw = payload["items"] if isinstance(payload, dict) and "items" in payload else payload
    elif states_dir:
        raw = json.loads((Path(states_dir) / "manifest.json").read_text(encoding="utf-8"))
    else:
        raise ValueError("provide either manifest or states_dir")

    recs = [_normalize(r) for r in raw]
    if ids:
        idset = set(ids)
        recs = [r for r in recs if r["question_id"] in idset]
    if require_exists:
        kept, skipped = [], []
        for r in recs:
            present = r["state_file"] and Path(r["state_file"]).exists()
            (kept if present else skipped).append(r)
        if skipped:
            print(
                f"[state-select] skipping {len(skipped)} not-yet-captured: "
                f"{[r['question_id'] for r in skipped]}",
                flush=True,
            )
        recs = kept
    return recs

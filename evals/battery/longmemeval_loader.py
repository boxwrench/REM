"""Load LongMemEval and filter to the knowledge-update subset.

Prerequisite (documented in README): download a LongMemEval JSON, e.g.
`longmemeval_s.json` from the HF dataset `xiaowu0162/longmemeval`, to a local path.
This loader is offline — it reads that file; it does not fetch from the network.
"""
from __future__ import annotations

import json
from pathlib import Path

from evals.battery.models import QAItem, Session

KNOWLEDGE_UPDATE = "knowledge-update"


def load_knowledge_update(path: str | Path, limit: int | None = None) -> list[QAItem]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    items: list[QAItem] = []
    for entry in raw:
        if entry.get("question_type") != KNOWLEDGE_UPDATE:
            continue
        ids = entry["haystack_session_ids"]
        sessions = [
            Session(session_id=sid, turns=turns)
            for sid, turns in zip(ids, entry["haystack_sessions"])
        ]
        items.append(
            QAItem(
                question_id=entry["question_id"],
                question=entry["question"],
                answer=entry["answer"],
                question_type=KNOWLEDGE_UPDATE,
                sessions=sessions,
                answer_session_ids=entry.get("answer_session_ids", []),
            )
        )
    if limit is not None:
        items = items[:limit]
    return items

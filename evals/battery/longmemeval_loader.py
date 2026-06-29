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
MEMORY_METHOD_CATEGORIES = (
    "knowledge-update", "temporal-reasoning", "multi-session",
)


def _gold_recency(haystack_ids: list[str], answer_ids: list[str]) -> float:
    """Normalized position of the latest gold session (0=oldest, 1=newest).

    Items with a low value are the ones where naive truncation drops the gold,
    making the REM-vs-truncation comparison non-trivial. Defaults to 1.0 (treat
    as newest/non-droppable) when the gold cannot be located or there is only
    one session.
    """
    n = len(haystack_ids)
    positions = [haystack_ids.index(a) for a in answer_ids if a in haystack_ids]
    if not positions or n <= 1:
        return 1.0
    return max(positions) / (n - 1)


def load_categories(
    path: str | Path,
    categories: tuple[str, ...] | list[str] | set[str],
    limit: int | None = None,
    max_gold_recency: float | None = None,
) -> list[QAItem]:
    """Load selected LongMemEval categories with stable source ordering.

    When ``max_gold_recency`` is set, keep only items whose latest gold session
    is at or before that normalized position, sorted oldest-gold first — i.e. the
    items where truncation drops the evidence so the battery can be valid.
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    items: list[QAItem] = []
    for entry in raw:
        if entry.get("question_type") not in categories:
            continue
        ids = entry["haystack_session_ids"]
        answer_ids = entry.get("answer_session_ids", [])
        sessions = [
            Session(session_id=sid, turns=turns)
            for sid, turns in zip(ids, entry["haystack_sessions"])
        ]
        items.append(
            QAItem(
                question_id=entry["question_id"],
                question=entry["question"],
                answer=entry["answer"],
                question_type=entry["question_type"],
                sessions=sessions,
                answer_session_ids=answer_ids,
                gold_recency=_gold_recency(ids, answer_ids),
            )
        )
    if max_gold_recency is not None:
        items = [it for it in items if it.gold_recency <= max_gold_recency]
        items.sort(key=lambda it: it.gold_recency)
    if limit is not None:
        items = items[:limit]
    return items


def load_knowledge_update(
    path: str | Path,
    limit: int | None = None,
    max_gold_recency: float | None = None,
) -> list[QAItem]:
    return load_categories(
        path, (KNOWLEDGE_UPDATE,), limit=limit,
        max_gold_recency=max_gold_recency,
    )

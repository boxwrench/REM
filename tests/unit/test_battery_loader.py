import json

import pytest

from evals.battery.longmemeval_loader import load_knowledge_update


def _write_fixture(tmp_path):
    data = [
        {
            "question_id": "ku1",
            "question_type": "knowledge-update",
            "question": "Where does the user work now?",
            "answer": "Acme",
            "haystack_session_ids": ["s1", "s2"],
            "haystack_sessions": [
                [{"role": "user", "content": "I work at Globex"}],
                [{"role": "user", "content": "Update: I now work at Acme"}],
            ],
            "haystack_dates": [
                "2023/05/20 (Sat) 02:21",
                "2023/06/11 (Sun) 14:05",
            ],
            "answer_session_ids": ["s2"],
        },
        {
            "question_id": "ss1",
            "question_type": "single-session-user",
            "question": "irrelevant",
            "answer": "x",
            "haystack_session_ids": ["s9"],
            "haystack_sessions": [[{"role": "user", "content": "hi"}]],
            "answer_session_ids": ["s9"],
        },
    ]
    p = tmp_path / "lme.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


def test_loader_keeps_only_knowledge_update(tmp_path):
    items = load_knowledge_update(_write_fixture(tmp_path))
    assert len(items) == 1
    it = items[0]
    assert it.question_id == "ku1"
    assert it.question_type == "knowledge-update"
    assert len(it.sessions) == 2
    assert it.sessions[1].session_id == "s2"
    assert it.sessions[1].timestamp == "2023/06/11 (Sun) 14:05"
    assert it.sessions[1].turns[0]["content"] == "Update: I now work at Acme"
    assert it.answer_session_ids == ["s2"]


def test_loader_respects_limit(tmp_path):
    items = load_knowledge_update(_write_fixture(tmp_path), limit=0)
    assert items == []


def test_loader_without_haystack_dates_uses_null_timestamp(tmp_path):
    path = _write_fixture(tmp_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    data[0].pop("haystack_dates")
    path.write_text(json.dumps(data), encoding="utf-8")

    item = load_knowledge_update(path)[0]

    assert [session.timestamp for session in item.sessions] == [None, None]


def test_loader_rejects_misaligned_haystack_dates(tmp_path):
    path = _write_fixture(tmp_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    data[0]["haystack_dates"] = data[0]["haystack_dates"][:1]
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="session alignment mismatch"):
        load_knowledge_update(path)


def _write_recency_fixture(tmp_path):
    """Three knowledge-update items whose gold session sits at the oldest,
    middle, and newest position of a 3-session haystack."""
    def item(qid, answer_pos):
        ids = ["a", "b", "c"]
        return {
            "question_id": qid,
            "question_type": "knowledge-update",
            "question": "q",
            "answer": "x",
            "haystack_session_ids": ids,
            "haystack_sessions": [[{"role": "user", "content": s}] for s in ids],
            "answer_session_ids": [ids[answer_pos]],
        }
    data = [item("new", 2), item("old", 0), item("mid", 1)]
    p = tmp_path / "lme_recency.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


def test_loader_computes_gold_recency(tmp_path):
    items = load_knowledge_update(_write_recency_fixture(tmp_path))
    recency = {it.question_id: it.gold_recency for it in items}
    assert recency == {"new": 1.0, "old": 0.0, "mid": 0.5}


def test_loader_filters_and_sorts_by_gold_recency(tmp_path):
    # Keep only items whose latest gold is in the older half, oldest first.
    items = load_knowledge_update(_write_recency_fixture(tmp_path), max_gold_recency=0.5)
    assert [it.question_id for it in items] == ["old", "mid"]  # sorted ascending; "new" excluded

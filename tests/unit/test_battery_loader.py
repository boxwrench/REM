import json

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
    assert it.sessions[1].turns[0]["content"] == "Update: I now work at Acme"
    assert it.answer_session_ids == ["s2"]


def test_loader_respects_limit(tmp_path):
    items = load_knowledge_update(_write_fixture(tmp_path), limit=0)
    assert items == []

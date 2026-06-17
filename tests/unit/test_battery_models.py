from evals.battery.models import Session, QAItem


def test_qaitem_holds_sessions_and_gold():
    item = QAItem(
        question_id="q1",
        question="What is the user's current city?",
        answer="Boston",
        question_type="knowledge-update",
        sessions=[Session(session_id="s1", turns=[{"role": "user", "content": "I live in NYC"}])],
        answer_session_ids=["s1"],
    )
    assert item.question_type == "knowledge-update"
    assert item.sessions[0].session_id == "s1"
    assert item.answer_session_ids == ["s1"]

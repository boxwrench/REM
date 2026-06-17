import json

from evals.battery.judge import judge_answer, JUDGE_MODEL


class _Block:
    type = "text"
    def __init__(self, text): self.text = text


class _Resp:
    def __init__(self, payload): self.content = [_Block(json.dumps(payload))]


class FakeMessages:
    def __init__(self, payload): self._payload = payload; self.kwargs = None
    def create(self, **kwargs):
        self.kwargs = kwargs
        return _Resp(self._payload)


class FakeAnthropic:
    def __init__(self, payload):
        self.messages = FakeMessages(payload)


def test_judge_parses_correct_verdict():
    client = FakeAnthropic({"correct": True, "reason": "matches gold"})
    v = judge_answer(client, question="Where?", gold="Acme", model_answer="Acme")
    assert v.correct is True
    assert "matches" in v.reason
    assert client.messages.kwargs["model"] == JUDGE_MODEL


def test_judge_parses_incorrect_verdict():
    client = FakeAnthropic({"correct": False, "reason": "stale value"})
    v = judge_answer(client, question="Where?", gold="Acme", model_answer="Globex")
    assert v.correct is False

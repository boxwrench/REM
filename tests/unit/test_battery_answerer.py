from evals.battery.answerer import answer_question

GEMMA = "gemma4-it:e2b"


class FakeNpuClient:
    def __init__(self):
        self.calls = []

    def chat(self, messages, *, model=None, max_tokens=None, **kw):
        self.calls.append({"model": model, "messages": messages})
        return "Acme"


def test_answerer_uses_gemma_and_includes_context_and_question():
    fc = FakeNpuClient()
    out = answer_question(fc, context="USER: I now work at Acme",
                          question="Where does the user work?")
    assert out == "Acme"
    call = fc.calls[0]
    assert call["model"] == GEMMA               # NOT the retired 1B default
    joined = " ".join(m["content"] for m in call["messages"])
    assert "Acme" in joined and "Where does the user work?" in joined

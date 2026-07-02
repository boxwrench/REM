from evals.battery.answerer import answer_question, answer_system_prompt
from rem.memory.query import classify_question

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


def test_question_taxonomy_distinguishes_updates_from_plain_counts():
    assert classify_question("Did I switch to more water, or less?") == "change"
    assert classify_question("How many bird species have I seen?") == "current"
    assert classify_question("What was the previous target?") == "previous"
    assert classify_question("What was the target as of 2024?") == "point-in-time"
    assert classify_question("What is the combined total across all trips?") == "aggregation"
    assert classify_question("What is my current doctor's first name?") == "current"


def test_answer_prompt_adds_mode_specific_instruction_and_keeps_legacy_baseline():
    change = answer_system_prompt("Did I switch to more water, or less?")
    current = answer_system_prompt("How many bird species have I seen?")
    legacy = answer_system_prompt("How many?", use_taxonomy=False)
    assert "Question mode: change" in change
    assert "earlier and later" in change
    assert "Question mode: current" in current
    assert "Do not combine successive versions" in current
    assert "Question mode" not in legacy

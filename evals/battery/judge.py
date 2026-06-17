"""Independent LLM judge for answer correctness (Claude, structured output)."""
from __future__ import annotations

import json
from dataclasses import dataclass

JUDGE_MODEL = "claude-haiku-4-5"
JUDGE_MAX_TOKENS = 512

_SYSTEM = (
    "You are a strict grader for a question-answering benchmark. "
    "Given a question, the gold answer, and a model's answer, decide whether the "
    "model's answer is correct: it must convey the same fact as the gold answer "
    "(paraphrase is fine; a stale or contradictory value is incorrect)."
)

_SCHEMA = {
    "type": "object",
    "properties": {
        "correct": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["correct", "reason"],
    "additionalProperties": False,
}


@dataclass
class Verdict:
    correct: bool
    reason: str


def make_client():
    """Construct the default Anthropic client (reads ANTHROPIC_API_KEY from env)."""
    import anthropic
    return anthropic.Anthropic()


def judge_answer(client, *, question: str, gold: str, model_answer: str,
                 model: str = JUDGE_MODEL) -> Verdict:
    prompt = (
        f"Question: {question}\n"
        f"Gold answer: {gold}\n"
        f"Model answer: {model_answer}\n\n"
        "Return your verdict."
    )
    resp = client.messages.create(
        model=model,
        max_tokens=JUDGE_MAX_TOKENS,
        system=_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
        output_config={"format": {"type": "json_schema", "schema": _SCHEMA}},
    )
    text = next(b.text for b in resp.content if b.type == "text")
    data = json.loads(text)
    return Verdict(correct=bool(data["correct"]), reason=str(data.get("reason", "")))

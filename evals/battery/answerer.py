"""Fixed answerer: gemma4-it:e2b on the NPU (held constant across all arms)."""
from __future__ import annotations

ANSWERER_MODEL = "gemma4-it:e2b"
ANSWERER_MAX_TOKENS = 256

_SYSTEM = (
    "You answer questions using ONLY the provided conversation memory. "
    "If the memory contains an updated value, use the most recent one. "
    "Answer concisely with just the fact."
)


def answer_question(client, *, context: str, question: str,
                    model: str = ANSWERER_MODEL) -> str:
    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": f"=== MEMORY ===\n{context}\n\n=== QUESTION ===\n{question}"},
    ]
    return client.chat(messages, model=model, max_tokens=ANSWERER_MAX_TOKENS).strip()

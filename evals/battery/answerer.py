"""Fixed answerer: gemma4-it:e2b on the NPU (held constant across all arms)."""
from __future__ import annotations

from rem.memory.query import question_mode_instruction

ANSWERER_MODEL = "gemma4-it:e2b"
ANSWERER_MAX_TOKENS = 256

_LEGACY_SYSTEM = (
    "You answer questions using ONLY the provided conversation memory. "
    "If the memory contains an updated value, use the most recent one. "
    "Answer concisely with just the fact."
)

def answer_system_prompt(question: str, *, use_taxonomy: bool = True) -> str:
    """Build the answer instruction; the legacy form remains for gate baselines."""
    if not use_taxonomy:
        return _LEGACY_SYSTEM
    return (
        "You answer questions using ONLY the provided conversation memory. "
        f"{question_mode_instruction(question)} "
        "Answer concisely with just the requested fact."
    )


def answer_question(client, *, context: str, question: str,
                    model: str = ANSWERER_MODEL,
                    use_taxonomy: bool = True) -> str:
    messages = [
        {
            "role": "system",
            "content": answer_system_prompt(question, use_taxonomy=use_taxonomy),
        },
        {"role": "user", "content": f"=== MEMORY ===\n{context}\n\n=== QUESTION ===\n{question}"},
    ]
    return client.chat(messages, model=model, max_tokens=ANSWERER_MAX_TOKENS).strip()

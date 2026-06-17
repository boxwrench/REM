"""Data models for the battery spike."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Session:
    """One chat session from the LongMemEval haystack."""
    session_id: str
    turns: list[dict]  # each: {"role": "user"|"assistant", "content": str}


@dataclass
class QAItem:
    """One LongMemEval question with its session haystack and gold answer."""
    question_id: str
    question: str
    answer: str
    question_type: str
    sessions: list[Session]
    answer_session_ids: list[str]


@dataclass
class ArmRun:
    """One arm's result for one question."""
    question_id: str
    arm: str
    assembled_tokens: int
    evidence_retained: bool
    model_answer: str
    judged_correct: bool | None = None
    judge_reason: str = ""


@dataclass
class BatteryResult:
    """Aggregated battery output."""
    arm_accuracy: dict[str, float] = field(default_factory=dict)
    arm_evidence_retention: dict[str, float] = field(default_factory=dict)
    n_questions: int = 0
    runs: list[ArmRun] = field(default_factory=list)
    valid: bool = True
    invalid_reason: str = ""

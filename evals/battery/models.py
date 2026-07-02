"""Data models for the battery spike."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Session:
    """One chat session from the LongMemEval haystack."""
    session_id: str
    turns: list[dict]  # each: {"role": "user"|"assistant", "content": str}
    # LongMemEval's session-level haystack_dates value. Optional because older
    # fixtures and datasets without dates must remain loadable.
    timestamp: str | None = None


@dataclass
class QAItem:
    """One LongMemEval question with its session haystack and gold answer."""
    question_id: str
    question: str
    answer: str
    question_type: str
    sessions: list[Session]
    answer_session_ids: list[str]
    # Normalized position of the latest gold session in the haystack timeline
    # (0.0 = oldest, 1.0 = newest). Low values are the items where naive
    # truncation drops the gold, so REM's compaction can prove its worth.
    gold_recency: float = 1.0


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
    # Per-question fact-extraction telemetry (REM arm only); None for arms with
    # no extraction stage. See rem.memory.facts_ledger.get_extraction_stats.
    extraction: dict | None = None


@dataclass
class BatteryResult:
    """Aggregated battery output."""
    arm_accuracy: dict[str, float] = field(default_factory=dict)
    arm_evidence_retention: dict[str, float] = field(default_factory=dict)
    n_questions: int = 0
    runs: list[ArmRun] = field(default_factory=list)
    valid: bool = True
    invalid_reason: str = ""
    # Per-arm sum of extraction telemetry across questions (arms with no
    # extraction stage are omitted).
    arm_extraction: dict[str, dict[str, int]] = field(default_factory=dict)

"""Deterministic question-mode classification for memory reads."""

from __future__ import annotations

import re
from typing import Literal

QuestionMode = Literal[
    "current", "previous", "change", "aggregation", "point-in-time"
]

_POINT_IN_TIME = re.compile(
    r"\b(?:as of|at the time|on|during|in)\s+"
    r"(?:\d{4}|(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|"
    r"jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|"
    r"nov(?:ember)?|dec(?:ember)?|monday|tuesday|wednesday|thursday|"
    r"friday|saturday|sunday)\b)",
    re.IGNORECASE,
)
_CHANGE = re.compile(
    r"\b(?:change(?:d)?|switch(?:ed)?|increase(?:d)?|decrease(?:d)?|"
    r"more or less|less or more|from .{1,80} to)\b",
    re.IGNORECASE,
)
_PREVIOUS = re.compile(
    r"\b(?:before|earlier|former|initial(?:ly)?|original(?:ly)?|"
    r"previous(?:ly)?|prior|started|used to)\b",
    re.IGNORECASE,
)
_AGGREGATION = re.compile(
    r"\b(?:across all|all sessions|altogether|average|combined|in total|"
    r"overall total|sum(?:med)?|taken together)\b",
    re.IGNORECASE,
)

_MODE_INSTRUCTIONS: dict[QuestionMode, str] = {
    "current": (
        "Return the newest relevant value. Facts may use different labels for the "
        "same attribute; when they conflict, use source-turn order and prefer the "
        "latest observation. Do not combine successive versions."
    ),
    "previous": (
        "Return the relevant value immediately before the newest update, not the "
        "current value. Preserve the stated unit."
    ),
    "change": (
        "Compare the earlier and later observations of the same attribute. Report "
        "the direction of change requested by the question and include the values "
        "when available. Do not answer with the earlier value alone."
    ),
    "aggregation": (
        "Combine distinct relevant observations as requested. Do not treat successive "
        "versions of one attribute as separate contributions."
    ),
    "point-in-time": (
        "Return the value valid at the time named in the question. Later updates are "
        "context, not the answer for that time."
    ),
}


def classify_question(question: str) -> QuestionMode:
    """Classify a question without a model call.

    Priority matters: a dated comparison is point-in-time, and an explicit
    transition is change rather than a generic request for an older value.
    Plain ``how many`` questions remain current-state questions; treating every
    count as aggregation would incorrectly add successive versions of one count.
    """
    if _POINT_IN_TIME.search(question):
        return "point-in-time"
    if _CHANGE.search(question):
        return "change"
    if _PREVIOUS.search(question):
        return "previous"
    if _AGGREGATION.search(question):
        return "aggregation"
    return "current"


def question_mode_instruction(question: str) -> str:
    """Return the shared production/evaluation instruction for one question."""
    mode = classify_question(question)
    return f"Question mode: {mode}. {_MODE_INSTRUCTIONS[mode]}"

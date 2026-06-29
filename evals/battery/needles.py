"""Robust gold-needle matching for the failure-mix labels.

Plain ``needle.lower() in text.lower()`` mislabels two documented cases
(bench/battery/FINDINGS.md): spelled-vs-digit numbers ("five engineers" vs
"5 engineers") and bare-number gold ("level 100" against a slot rendered
"level goal.target level: 100" or an answer phrased "the goal was 100"). This
module normalizes spelled cardinals to digits and matches on a canonicalized,
punctuation-flattened form, and exposes an all-vs-any aggregator so multi-part
questions (031748ae: started=4 AND now=5) require every gold needle.

Deliberately conservative: only whole-token cardinals 0–20 and the tens are
mapped (no "couple"/"few"/"dozen"), and matching stays a contiguous canonical
substring so it does not silently over-match. A fully slot-value-aware match
(number tied to its concept slot) remains the documented follow-up.
"""
from __future__ import annotations

import re

_CARDINALS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
    "eleven": "11", "twelve": "12", "thirteen": "13", "fourteen": "14",
    "fifteen": "15", "sixteen": "16", "seventeen": "17", "eighteen": "18",
    "nineteen": "19", "twenty": "20", "thirty": "30", "forty": "40",
    "fifty": "50", "sixty": "60", "seventy": "70", "eighty": "80", "ninety": "90",
}
_CARDINAL_RX = re.compile(r"\b(" + "|".join(_CARDINALS) + r")\b")


def canonical(text: str | None) -> str:
    """Lowercase, map spelled cardinals to digits, flatten punctuation to spaces.

    "Five engineers"      -> "5 engineers"
    "level goal.target level: 100" -> "level goal target level 100"
    "F-150"               -> "f 150"
    """
    if not text:
        return ""
    low = text.lower()
    low = _CARDINAL_RX.sub(lambda m: _CARDINALS[m.group(1)], low)
    low = re.sub(r"[^a-z0-9]+", " ", low)
    return low.strip()


def present(needle: str, text: str | None) -> bool:
    """True if ``needle`` appears in ``text`` after number/punctuation canonicalization.

    Matches as a contiguous token run, so "5 engineers" hits "five engineers"
    and "level 100" hits "...target level 100", without matching across
    unrelated tokens.
    """
    n = canonical(needle)
    if not n:
        return False
    hay = canonical(text)
    return re.search(r"\b" + re.escape(n) + r"\b", hay) is not None


def value_aware_entry(needle: str, slot_key: str | None, slot_value: str | None) -> bool:
    """Slot-value-aware match for bare-number gold against one ledger entry.

    A number gold like "level 100" is carried by slot_key "level goal.target level",
    slot_value "100" — where the flat string "level 100" never appears contiguously.
    Match when every number token of the needle is in the entry's VALUE and (if the
    needle has non-number words) at least one of them appears in the key or value.
    Returns False for needles with no number token (use ``present`` for those).
    """
    n = canonical(needle)
    if not n:
        return False
    toks = n.split()
    nums = [t for t in toks if t.isdigit()]
    words = [t for t in toks if not t.isdigit()]
    if not nums:
        return False
    val_toks = canonical(slot_value).split()
    if not all(num in val_toks for num in nums):
        return False
    key_val_toks = set(canonical(slot_key).split()) | set(val_toks)
    return (not words) or any(w in key_val_toks for w in words)


def match(needles: list[str], text: str | None) -> dict[str, bool]:
    """Per-needle presence map (canonicalized)."""
    return {x: present(x, text) for x in needles}


def all_present(needles: list[str], text: str | None) -> bool:
    """All gold needles present — the correctness rule for multi-part questions.

    Empty needle list returns False: "no gold defined" is not a pass.
    """
    return bool(needles) and all(present(x, text) for x in needles)


def any_present(needles: list[str], text: str | None) -> bool:
    """Any needle present — for diagnostic (non-gating) structure needles."""
    return any(present(x, text) for x in needles)

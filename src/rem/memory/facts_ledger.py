"""Structured facts ledger extracted verbatim before prose summarization."""

import json
import logging
import re
from typing import Any, TYPE_CHECKING, Literal, TypeAlias
from pydantic import BaseModel, Field, field_validator, model_validator, ValidationError
from rem.npu_client import NpuClient
from rem.memory.prompts import (
    FACT_EXTRACTION_SYSTEM,
    FACT_EXTRACTION_USER_TEMPLATE,
    FACT_EXTRACTION_RETRY_MESSAGE,
    FACT_EXTRACTION_TRUNCATION_RETRY_MESSAGE,
)

if TYPE_CHECKING:
    from rem.memory.tiers import Turn

logger = logging.getLogger("rem.memory.facts_ledger")

SlotObservation: TypeAlias = tuple[int, str]

# ---------------------------------------------------------------------------
# Module-level extraction telemetry.
# Callers reset these at the start of a unit of work (e.g. one battery question)
# and read them afterwards via get_extraction_stats() to see how extraction
# behaved: how often it parsed cleanly, needed repair/retry, salvaged a loop,
# hit truncation, or failed outright. This is what makes a dropped-fact REM miss
# observable instead of hidden inside answer accuracy.
# ---------------------------------------------------------------------------
_EXTRACTION_STAT_KEYS = (
    "attempts",        # extract_facts calls that reached the model
    "strict_parse",    # parsed cleanly on the first strict json.loads
    "repaired",        # recovered via fence-strip/balance/repair
    "retried",         # needed a validator-guided retry
    "retry_success",   # retry produced usable facts
    "loops_detected",  # degenerate/looping output detected
    "loops_salvaged",  # loop salvaged down to usable facts
    "truncations",     # output detected as truncated
    "failures",        # final extraction failure (span kept verbatim)
)


def _fresh_extraction_stats() -> dict[str, int]:
    return {key: 0 for key in _EXTRACTION_STAT_KEYS}


_extraction_stats: dict[str, int] = _fresh_extraction_stats()


def reset_extraction_stats() -> None:
    """Zeroes all extraction telemetry counters."""
    _extraction_stats.update(_fresh_extraction_stats())


def get_extraction_stats() -> dict[str, int]:
    """Returns a snapshot copy of the current extraction telemetry counters."""
    return dict(_extraction_stats)


def _record_extraction_diagnostics(diagnostics: dict) -> None:
    """Folds one robust_extract_json diagnostics dict into the module counters."""
    if diagnostics.get("raw_parse_success"):
        _extraction_stats["strict_parse"] += 1
    if diagnostics.get("repair_success"):
        _extraction_stats["repaired"] += 1
    if diagnostics.get("retried"):
        _extraction_stats["retried"] += 1
    if diagnostics.get("retry_success"):
        _extraction_stats["retry_success"] += 1
    if diagnostics.get("loop_detected"):
        _extraction_stats["loops_detected"] += 1
    if diagnostics.get("loop_salvaged"):
        _extraction_stats["loops_salvaged"] += 1
    if diagnostics.get("truncated"):
        _extraction_stats["truncations"] += 1
    if not diagnostics.get("success"):
        _extraction_stats["failures"] += 1


class FactsExtractionError(Exception):
    """Raised when fact extraction from a span of turns fails after retry."""
    pass


def _canonical_slot_override(text: str) -> tuple[str, str] | None:
    normalized = text.lower()
    if "rate limit" in normalized or "ratelimit" in normalized:
        return "vendor_api", "rate_limit"
    elif "region" in normalized or "replica" in normalized:
        return "replica", "region"
    elif "host" in normalized:
        return "infra", "host"
    elif "port" in normalized:
        return "infra", "port"
    elif "codename" in normalized or "codenamed" in normalized:
        return "infra", "codename"
    elif "threshold" in normalized:
        return "infra", "threshold"
    elif "retained" in normalized or "retention" in normalized:
        return "infra", "retention"
    elif "concurrent" in normalized or "concurrency" in normalized:
        return "task_queue", "concurrency"
    elif "standby site" in normalized or "standby" in normalized:
        return "standby", "site"
    return None

def parse_general_fact(text: str) -> tuple[str | None, str | None, str | None]:
    """Decomposes a fact text into (subject, attribute, value) using general heuristics."""
    normalized = text.lower().strip()
    if normalized.endswith("."):
        normalized = normalized[:-1].strip()

    # Clean prefixes at the start to prevent relation-split conflicts
    prefixes = [
        "plan to place the",
        "plan to place",
        "correct",
        "real",
        "the",
        "a",
        "please",
        "remember",
        "correction on",
        "correction",
        "actually",
        "schedule the",
        "configure the",
        "we decided to",
        "make sure the",
        "make sure",
        "assign",
    ]
    for p in prefixes:
        if normalized.startswith(p + " "):
            normalized = normalized[len(p) + 1:].strip()
            break

    # Also clean colon prefixes
    if normalized.startswith("correction:"):
        normalized = normalized[len("correction:"):].strip()
    for p in prefixes:
        if normalized.startswith(p + " "):
            normalized = normalized[len(p) + 1:].strip()
            break

    # 0. Keyword/concept-based overrides for standard slot mapping
    # This guarantees consistent slot keys for known attributes across different phrasings.
    subject = None
    attribute = None
    value = None

    override = _canonical_slot_override(text)
    if override:
        subject, attribute = override

    if subject and attribute:
        legacy_val = infer_slot_value(f"{subject}.{attribute}", text)
        if legacy_val:
            return subject, attribute, legacy_val

    # If no legacy value matched or not a standard keyword, split by relation
    copulas = [
        "is configured for",
        "was configured for",
        "is configured to",
        "was configured to",
        "configured for",
        "configured to",
        "is set to",
        "was set to",
        "changed to",
        "retained for",
        "must live in",
        "live in",
        "listen on",
        "set to",
        "is",
        "was",
        "be",
    ]
    prepositions = ["to", "for", "in", "with", "as"]
    
    parts = None
    for r in copulas:
        pattern = r"\b" + re.escape(r) + r"\b"
        p = re.split(pattern, normalized, maxsplit=1)
        if len(p) == 2:
            parts = p
            break
    else:
        for r in prepositions:
            pattern = r"\b" + re.escape(r) + r"\b"
            p = re.split(pattern, normalized, maxsplit=1)
            if len(p) == 2:
                parts = p
                break

    if subject and attribute and parts:
        right = parts[1].strip()
        # Clean value prefixes/suffixes
        prefixes_to_strip = ["the", "a", "to", "port", "host", "region", "threshold", "codename", "site"]
        for p in prefixes_to_strip:
            if right.startswith(p + " "):
                right = right[len(p) + 1:].strip()
                break
        right_suffixes = ["now", "after all", "instead", "exactly", "daily"]
        for rs in right_suffixes:
            if right.endswith(" " + rs):
                right = right[:-len(rs) - 1].strip()
        value = right
        return subject, attribute, value

    # If no keyword matched, perform the generic subject-attribute-value extraction
    instead_match = re.search(r"use\s+([\w-]+)\s+instead\s+of\s+([\w-]+)", normalized)
    if instead_match:
        val = instead_match.group(1)
        subj = "infra"
        for kw in ["telemetry", "database", "db", "engine"]:
            if kw in normalized:
                subj = kw
                break
        return subj, "engine", val

    # Assign ... as ... case
    if parts and parts[0] and " as " in normalized:
        val = parts[0].strip()
        left = parts[1].strip()
        left_parts = left.split()
        if len(left_parts) > 1:
            attr = left_parts[-1]
            subj = " ".join(left_parts[:-1])
        else:
            attr = left_parts[0]
            subj = "infra"
        return subj, attr, val

    # Migrate to ... for ... case
    migrate_match = re.search(r"(?:migrate|go|move)\s+to\s+(\S+)\s+for\s+(.*)", normalized)
    if migrate_match:
        val = migrate_match.group(1).strip()
        left = migrate_match.group(2).strip()
        left_parts = left.split()
        if len(left_parts) > 1:
            attr = left_parts[-1]
            subj = " ".join(left_parts[:-1])
        else:
            attr = left_parts[0]
            subj = "infra"
        return subj, attr, val

    # Generic relation split
    if parts:
        left, right = parts[0].strip(), parts[1].strip()
        # Clean suffixes of left
        suffixes = ["must", "should", "will", "was", "is", "to", "to run"]
        for s in suffixes:
            if left.endswith(" " + s):
                left = left[:-len(s) - 1].strip()
                break

        # Check right prefixes
        right_prefixes = ["port", "host", "region", "threshold", "codename", "retention", "limit", "rate limit", "site"]
        for rp in right_prefixes:
            if right.startswith(rp + " "):
                attr = rp
                right = right[len(rp) + 1:].strip()
                break
        else:
            attr = None

        if not attr:
            if not left:
                return None, None, None
            left_parts = left.split()
            if len(left_parts) > 1:
                attr = left_parts[-1]
                subj = " ".join(left_parts[:-1])
            else:
                attr = left_parts[0]
                subj = "infra"
        else:
            subj = left if left else "infra"

        # Map attribute/subject
        attr_map = {
            "ratelimit": "rate_limit",
            "rate limit": "rate_limit",
            "retained": "retention",
            "retention": "retention",
            "codename": "codename",
            "codenamed": "codename",
            "limit": "rate_limit" if "rate" in subj else "limit",
        }
        attr = attr_map.get(attr, attr)
        if attr == "limit" and subj.endswith(" rate"):
            subj = subj[:-5].strip()
            attr = "rate_limit"
        elif attr == "limit" and subj == "rate":
            subj = "infra"
            attr = "rate_limit"

        subj_map = {
            "replica": "replica",
            "staging gateway": "infra",
            "cold-storage": "infra",
            "telemetry": "infra",
            "anomaly alert": "infra",
            "raw logs": "infra",
            "vendor api": "vendor_api",
        }
        subj = subj_map.get(subj, subj)

        right_suffixes = ["now", "after all", "instead", "exactly", "daily"]
        for rs in right_suffixes:
            if right.endswith(" " + rs):
                right = right[:-len(rs) - 1].strip()

        return subj, attr, right

    return None, None, None


class FactEntry(BaseModel):
    """Represents a single fact entry in the ledger."""
    kind: Literal["entity", "number", "decision", "quote"]
    text: str
    source_turn_id: int
    status: Literal["active", "stale"] = "active"
    slot_key: str | None = None
    slot_value: str | None = None
    subject: str | None = None
    attribute: str | None = None
    superseded_by_turn_id: int | None = None
    # Model-emitted structured fields (prompt schema v2).
    # "value" carries the model-extracted value string directly;
    # "is_correction" signals the model believes this supersedes an older fact.
    # These fields are accepted from JSON and used to seed slot_value; they
    # are NOT written back to the prompt or used as a hard constraint.
    value: str | None = None
    is_correction: bool = False

    @field_validator("kind", mode="before")
    @classmethod
    def normalize_kind(cls, v: str) -> str:
        if isinstance(v, str):
            v_lower = v.strip().lower()
            if v_lower not in ("entity", "number", "decision", "quote"):
                return "entity"
            return v_lower
        return v

    @model_validator(mode="after")
    def infer_state_slot(self) -> "FactEntry":
        """Infers a current-state slot, preferring canonical identity.

        Priority order:
        1. Model-emitted value seeds slot_value
        2. Heuristic extraction
        3. Canonical identity overrides model labels
        4. Normalized model labels for general facts
        """
        if self.value and not self.slot_value:
            self.slot_value = self.value

        if not self.subject or not self.attribute or not self.slot_value:
            subj, attr, val = parse_general_fact(self.text)
            if subj and attr and val:
                if not self.subject:
                    self.subject = subj
                if not self.attribute:
                    self.attribute = attr
                if not self.slot_value:
                    self.slot_value = val

        canonical_key = infer_slot_key(self.text)
        override = _canonical_slot_override(self.text)
        if not canonical_key and override:
            canonical_key = f"{override[0]}.{override[1]}"
            
        if canonical_key:
            self.slot_key = canonical_key
            if not self.slot_value:
                self.slot_value = infer_slot_value(self.slot_key, self.text)
        elif not self.slot_key:
            if self.subject and self.attribute:
                raw_key = f"{self.subject}.{self.attribute}"
                normalized_key = raw_key.lower()
                import string
                punct = string.punctuation.replace('.', '')
                normalized_key = normalized_key.translate(str.maketrans(punct, ' '*len(punct)))
                generics = {"the", "our", "a", "preferred", "current", "new", "now", "originally", "default"}
                words = [w for w in normalized_key.split() if w not in generics]
                normalized_key = " ".join(words).strip()
                if normalized_key:
                    self.slot_key = normalized_key

        return self


def infer_slot_key(text: str) -> str | None:
    """Returns a coarse state slot for facts that should supersede older values."""
    normalized = text.lower()
    if "rate limit" in normalized or "ratelimit" in normalized:
        return "vendor_api.rate_limit"
    if (
        ("region" in normalized or "replica" in normalized)
        and re.search(r"\b[a-z]{2}-[a-z]+-\d\b", normalized)
    ):
        return "replica.region"
    if "host" in normalized:
        return "infra.host"
    if "port" in normalized:
        return "infra.port"
    if "codename" in normalized or "codenamed" in normalized:
        return "infra.codename"
    if "instead of" in normalized and (
        "telemetry" in normalized or "sqlite" in normalized or "duckdb" in normalized
    ):
        return "infra.engine"
    if "threshold" in normalized:
        return "infra.threshold"
    if "retained" in normalized or "retention" in normalized:
        return "infra.retention"
    return None


def infer_slot_value(slot_key: str, text: str) -> str | None:
    """Extracts the comparable value for a known state slot."""
    if slot_key == "vendor_api.rate_limit":
        match = re.search(
            r"\b\d[\d,]*\s+requests?\s+per\s+minute\b",
            text,
            flags=re.IGNORECASE,
        )
        if match:
            return match.group(0)
    if slot_key == "replica.region":
        match = re.search(r"\b[a-z]{2}-[a-z]+-\d\b", text, flags=re.IGNORECASE)
        if match:
            return match.group(0)
    if slot_key == "infra.host":
        match = re.search(
            r"\bhost\s+(?:is|set\s+to|to)\s+([a-zA-Z0-9_-]+)\b",
            text,
            flags=re.IGNORECASE,
        )
        if match:
            return match.group(1)
        # Fallback if no separator is used (e.g. "host vega-archive-02")
        match = re.search(r"\bhost\s+([a-zA-Z0-9_-]+)\b", text, flags=re.IGNORECASE)
        if match and match.group(1).lower() not in ("is", "to", "set"):
            return match.group(1)
    if slot_key == "infra.port":
        match = re.search(
            r"\bport\s+(?:to\s+|is\s+)?(\d+)\b",
            text,
            flags=re.IGNORECASE,
        )
        if match:
            return match.group(1)
    if slot_key == "infra.codename":
        match = re.search(
            r"\b(?:codenamed?|codename\s+(?:is|to|tracked\s+as))\s+"
            r"([A-Za-z0-9_-]+)\b",
            text,
            flags=re.IGNORECASE,
        )
        if match:
            return match.group(1)
    if slot_key == "infra.engine":
        match = re.search(
            r"\buse\s+([a-zA-Z0-9_-]+)\s+instead\s+of\s+"
            r"([a-zA-Z0-9_-]+)\b",
            text,
            flags=re.IGNORECASE,
        )
        if match:
            return match.group(1)
    if slot_key == "infra.threshold":
        match = re.search(
            r"\bthreshold\s+(?:is|to|set\s+to)\s+(\d+(?:\.\d+)?)\b",
            text,
            flags=re.IGNORECASE,
        )
        if match:
            return match.group(1)
    if slot_key == "infra.retention":
        match = re.search(
            r"\b(?:retained|retention)\s+(?:for\s+)?([\w\s-]+)\b",
            text,
            flags=re.IGNORECASE,
        )
        if match:
            val = match.group(1).strip()
            if val.lower().endswith(" exactly"):
                val = val[:-8].strip()
            return val
    return None


class FactsLedger(BaseModel):
    """Manages the list of fact entries."""
    entries: list[FactEntry] = Field(default_factory=list)
    duplicate_active_suppressions: int = 0
    rendered_text: str | None = Field(default=None, repr=False)
    max_stale_entries: int = 10

    def add(self, entry: FactEntry) -> None:
        """Adds a fact entry to the ledger and marks older slot values stale."""
        # Deduplicate by normalized text:
        norm_new = self._normalize(entry.text)
        for existing in self.entries:
            if self._normalize(existing.text) == norm_new:
                # Keep the newer turn ID, keep active status if either was active
                if entry.source_turn_id > existing.source_turn_id:
                    existing.source_turn_id = entry.source_turn_id
                if entry.status == "active":
                    existing.status = "active"
                    existing.superseded_by_turn_id = None
                self.rendered_text = None
                return

        self._apply_supersession(entry)
        self.entries.append(entry)
        self._evict_excess_stale()
        self.rendered_text = None

    @staticmethod
    def _normalize(text: str) -> str:
        """Normalizes text for comparison by lowercasing and stripping whitespace."""
        return text.strip().lower()

    def _apply_supersession(self, new_entry: FactEntry) -> None:
        """Marks older active entries in the same inferred slot as stale.

        Re-extraction guard: if the new entry's slot_value matches any
        currently-stale value for the same slot, the new entry is carrying
        re-extracted stale data (the model read an old mention in passing).
        The active correction must win regardless of source_turn_id ordering;
        the new entry is immediately marked stale.
        """
        if not new_entry.slot_key or not new_entry.slot_value:
            return

        # Re-extraction guard: new entry re-surfaces a value that is
        # already recorded as stale for this slot → mark it stale immediately.
        for existing in self.entries:
            if existing.status != "stale":
                continue
            if existing.slot_key != new_entry.slot_key:
                continue
            if existing.slot_value == new_entry.slot_value:
                # The new entry carries a value that has already been superseded.
                # Find the active winner to record the supersession link.
                winner_turn = existing.superseded_by_turn_id
                if winner_turn is None:
                    # Defensive: find the active entry for this slot.
                    for e2 in self.entries:
                        if (e2.status == "active" and
                                e2.slot_key == new_entry.slot_key and
                                e2.slot_value != new_entry.slot_value):
                            winner_turn = e2.source_turn_id
                            break
                new_entry.status = "stale"
                new_entry.superseded_by_turn_id = winner_turn
                return

        # Normal latest-wins: mark older active entries stale.
        for existing in self.entries:
            if existing.status != "active":
                continue
            if existing.slot_key != new_entry.slot_key:
                continue
            if existing.slot_value == new_entry.slot_value:
                continue
            if existing.source_turn_id <= new_entry.source_turn_id:
                existing.status = "stale"
                existing.superseded_by_turn_id = new_entry.source_turn_id
            else:
                new_entry.status = "stale"
                new_entry.superseded_by_turn_id = existing.source_turn_id

    def _evict_excess_stale(self) -> None:
        """Evicts the oldest stale entries if they exceed max_stale_entries."""
        stale_indices = [
            i for i, entry in enumerate(self.entries)
            if entry.status == "stale"
        ]
        if len(stale_indices) > self.max_stale_entries:
            # Sort stale entries by source_turn_id (ascending) to find the oldest
            stale_indices.sort(key=lambda idx: self.entries[idx].source_turn_id)
            # Find how many we need to evict
            to_evict_count = len(stale_indices) - self.max_stale_entries
            indices_to_remove = set(stale_indices[:to_evict_count])
            self.entries = [
                entry for i, entry in enumerate(self.entries)
                if i not in indices_to_remove
            ]

    def merge(self, other: "FactsLedger") -> None:
        """Merges another facts ledger into this one.

        Deduplicates entries by normalized text and marks older values in the same
        inferred state slot as stale.
        """
        self.rendered_text = None
        existing_normalized = {self._normalize(entry.text): entry for entry in self.entries}
        for entry in other.entries:
            norm = self._normalize(entry.text)
            if norm in existing_normalized:
                existing = existing_normalized[norm]
                if entry.source_turn_id > existing.source_turn_id:
                    existing.source_turn_id = entry.source_turn_id
                if entry.status == "active":
                    existing.status = "active"
                    existing.superseded_by_turn_id = None
            else:
                self.add(entry)
                existing_normalized[norm] = entry

    def active_entries(self) -> list[FactEntry]:
        """Returns entries that should be rendered as current state."""
        raw_active = [entry for entry in self.entries if entry.status == "active"]

        # Group active entries by slot_key
        by_slot: dict[str, list[FactEntry]] = {}
        for entry in raw_active:
            if entry.slot_key:
                by_slot.setdefault(entry.slot_key, []).append(entry)

        # For each slot, keep only the newest one (highest source_turn_id)
        # If there are duplicates, suppress the older ones
        suppressed_count = 0
        final_active = []
        for entry in raw_active:
            if not entry.slot_key:
                final_active.append(entry)
                continue

            # Find the newest entry for this slot
            entries_for_slot = by_slot[entry.slot_key]
            newest = max(entries_for_slot, key=lambda e: e.source_turn_id)
            if entry == newest:
                final_active.append(entry)
            else:
                suppressed_count += 1

        # Update diagnostic counter
        self.duplicate_active_suppressions = suppressed_count
        return final_active

    def stale_entries(self) -> list[FactEntry]:
        """Returns retained stale entries for audit/debugging."""
        return [entry for entry in self.entries if entry.status == "stale"]

    def render(
        self,
        include_stale: bool = False,
        suppress_slots: (
            dict[str, list[SlotObservation]] | list[str] | set[str] | None
        ) = None,
        quarantine: dict[str, set[str]] | None = None,
    ) -> str:
        """Renders the ledger entries as a formatted string block.

        By default, renders only active facts. Stale entries are retained for audit
        but are not active memory. Active facts can be suppressed if their slot has
        a newer, different value in recent verbatim turns.
        """
        entries = self.entries if include_stale else self.active_entries()
        if suppress_slots:
            if isinstance(suppress_slots, dict):
                entries = [
                    entry
                    for entry in entries
                    if not _is_suppressed_by_recent_value(entry, suppress_slots)
                ]
            else:
                suppress_set = set(suppress_slots)
                entries = [
                    entry
                    for entry in entries
                    if not entry.slot_key or entry.slot_key not in suppress_set
                ]
        if quarantine:
            entries = [
                entry
                for entry in entries
                if not (entry.slot_key and entry.slot_value and entry.slot_value in quarantine.get(entry.slot_key, set()))
            ]
        if not entries:
            return ""
        lines = ["Facts Ledger:"]
        for entry in entries:
            status = "" if entry.status == "active" else " stale"
            lines.append(
                f"- [{entry.kind}{status}] {entry.text} (Turn {entry.source_turn_id})"
            )
        return "\n".join(lines)


def _is_suppressed_by_recent_value(
    entry: FactEntry,
    recent_values: dict[str, list[SlotObservation]],
) -> bool:
    """Returns true when a newer verbatim turn carries a different slot value."""
    if not entry.slot_key or not entry.slot_value:
        return False
    for turn_id, slot_value in recent_values.get(entry.slot_key, []):
        if turn_id > entry.source_turn_id and slot_value != entry.slot_value:
            return True
    return False


STOPWORDS: set[str] = {
    "the", "a", "an", "and", "or", "but", "if", "then", "else",
    "is", "are", "was", "were", "be", "been", "being",
    "to", "of", "in", "on", "at", "by", "for", "with", "about",
    "against", "between", "into", "through", "during", "before",
    "after", "above", "below", "to", "from", "up", "down", "in",
    "out", "on", "off", "over", "under", "again", "further",
    "then", "once", "here", "there", "when", "where", "why",
    "how", "all", "any", "both", "each", "few", "more", "most",
    "other", "some", "such", "no", "nor", "not", "only", "own",
    "same", "so", "than", "too", "very", "s", "t", "can", "will",
    "just", "don", "should", "now", "i", "me", "my", "myself",
    "we", "our", "ours", "ourselves", "you", "your", "yours",
    "yourself", "yourselves", "he", "him", "his", "himself",
    "she", "her", "hers", "herself", "it", "its", "itself",
    "they", "them", "their", "theirs", "themselves", "what",
    "which", "who", "whom", "this", "that", "these", "those",
    "am", "please", "remember", "actually", "correct", "correction"
}


def map_fact_to_turn(text: str, turns: list["Turn"]) -> int | None:
    """Confidently maps a fact text to exactly one source turn from the extraction span.
    
    Returns the turn ID if mapped confidently, otherwise None.
    """
    normalized_fact = text.lower().strip()
    
    # Clean punctuation
    fact_clean = re.sub(r'[^\w\s]', ' ', normalized_fact)
    fact_words = [w for w in fact_clean.split() if w and w not in STOPWORDS]
    
    if not fact_words:
        return None
        
    matching_turns = []
    for turn in turns:
        turn_content_norm = turn.content.lower()
        # Check direct substring first
        if normalized_fact in turn_content_norm or re.sub(r'[^\w\s]', ' ', turn_content_norm).replace(" ", "") in fact_clean.replace(" ", ""):
            matching_turns.append(turn.turn_id)
            continue
        
        # Check word overlap: all non-stopwords of the fact must be present in the turn
        turn_clean = re.sub(r'[^\w\s]', ' ', turn_content_norm)
        turn_words = set(turn_clean.split())
        if all(w in turn_words for w in fact_words):
            matching_turns.append(turn.turn_id)
            
    # Deduplicate matching_turns
    matching_turns = list(set(matching_turns))
    
    if len(matching_turns) == 1:
        return matching_turns[0]
        
    return None


def clean_and_check_truncation(response_text: str) -> tuple[str, bool]:
    """Cleans the response text to isolate the JSON block, and determines if it is truncated.
    
    Returns (cleaned_text, is_truncated).
    """
    text = response_text.strip()
    if not text:
        return "", False

    # Strip markdown fences
    if text.startswith("```"):
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline:].strip()
        if text.endswith("```"):
            text = text[:-3].strip()
    text = text.strip()

    first_brace = text.find("{")
    first_bracket = text.find("[")
    if first_brace == -1 and first_bracket == -1:
        return text, True  # No JSON start found, must be truncated/broken

    start_idx = min(idx for idx in (first_brace, first_bracket) if idx != -1)

    json_candidate = text[start_idx:]
    
    if json_candidate.startswith("["):
        last_bracket = json_candidate.rfind("]")
        if last_bracket == -1:
            return json_candidate, True
        
        json_slice = json_candidate[:last_bracket + 1]
        
        # Check if there is any unfinished object starting at the end of the candidate
        after_last = json_candidate[last_bracket + 1:].strip()
        if any(c in after_last for c in ('{', '"', ':')):
            return json_candidate, True

        # Check for unclosed braces inside the sliced array
        if json_slice.count("{") > json_slice.count("}"):
            return json_slice, True

        return json_slice, False

    elif json_candidate.startswith("{"):
        last_brace = json_candidate.rfind("}")
        if last_brace == -1:
            return json_candidate, True
        
        json_slice = json_candidate[:last_brace + 1]
        
        if json_slice.count("{") > json_slice.count("}"):
            return json_slice, True
            
        return json_slice, False

    return json_candidate, True


_ALLOWED_FACT_KEYS = {
    "kind", "source_turn_id", "subject", "attribute", "value",
    "is_correction", "text", "status", "slot_key", "slot_value",
    "superseded_by_turn_id",
}


def _recover_malformed_keys(item: dict) -> dict:
    """Salvage a json_repair-mangled fact dict by keeping its valid core.

    The small extraction model emits malformed JSON (unescaped quotes in values,
    ``value=`` instead of ``"value":``); after repair this leaves dicts with
    mangled extra keys. Rather than rejecting the whole fact:
      1. keep all already-allowed keys,
      2. remap an unknown key to an allowed field if it normalizes to one
         (e.g. ``value=`` -> ``value``) and that field is not already set,
      3. drop the remaining unknown keys.
    The missing-text guard downstream still drops entries with no recoverable core.
    """
    cleaned = {k: v for k, v in item.items() if k in _ALLOWED_FACT_KEYS}
    for k, v in item.items():
        if k in _ALLOWED_FACT_KEYS or not isinstance(k, str):
            continue
        norm = k.strip(" \t\n\"'=:.,!}{")
        if norm in _ALLOWED_FACT_KEYS and norm not in cleaned:
            cleaned[norm] = v
    return cleaned


def validate_and_repair_items(parsed: Any, turns: list["Turn"]) -> list[FactEntry]:
    """Validates the parsed JSON list of facts, repairing missing source_turn_id if confident.

    The model is no longer required to emit ``text``; if it is
    absent but ``subject``/``attribute``/``value`` are present the text is
    derived deterministically as ``"{subject} {attribute}: {value}"`` so that
    ``render()`` still produces a value-containing string and needle survival
    is maintained.

    Raises ValueError on validation/repair failures.
    """
    if not isinstance(parsed, list):
        if isinstance(parsed, dict):
            parsed = [parsed]
        else:
            raise ValueError("Parsed JSON is not a list or dictionary")

    valid_turn_ids = {t.turn_id for t in turns}
    fact_entries = []
    last_error = None

    for item in parsed:
        try:
            if not isinstance(item, dict):
                raise ValueError(f"Fact entry item must be a dictionary, got: {type(item)}")

            # Strip/remap json_repair artifact keys and keep the valid core
            # rather than rejecting the whole fact (see _recover_malformed_keys).
            item = _recover_malformed_keys(item)

            # Safe coercion of subject/attribute/value to string defensively
            for k in ("subject", "attribute", "value"):
                if k in item:
                    v = item[k]
                    if v is None:
                        item[k] = None
                    elif isinstance(v, bool):
                        item[k] = "True" if v else "False"
                    else:
                        item[k] = str(v)

            # Text is now optional; derive it from subject/attribute/value if absent.
            if "text" not in item or not item.get("text"):
                subj = (item.get("subject") or "").strip()
                attr = (item.get("attribute") or "").strip()
                val = (item.get("value") or "").strip()
                # Recover a json_repair artifact: when the model drops the comma
                # between attribute and value (`"attribute":"value":"X"`),
                # json_repair merges the tail into the attribute string as
                # `value":"X`, leaving no value field. A literal `":"` inside an
                # attribute is never natural language, so split it back into
                # (attribute, value) rather than dropping a recoverable fact.
                if not val and '":"' in attr:
                    recovered_attr, _, recovered_val = attr.partition('":"')
                    recovered_attr = recovered_attr.strip().strip('"').strip()
                    recovered_val = recovered_val.strip().strip('"').strip()
                    if recovered_attr and recovered_val:
                        attr = recovered_attr
                        val = recovered_val
                        item["attribute"] = attr
                        item["value"] = val
                if subj and attr and val:
                    item["text"] = f"{subj} {attr}: {val}"
                else:
                    raise ValueError(
                        "Fact entry is missing 'text' and cannot derive it "
                        f"(subject={item.get('subject')!r}, attribute={item.get('attribute')!r}, "
                        f"value={item.get('value')!r})"
                    )

            if "kind" not in item:
                item["kind"] = "entity"

            # Validate/repair source_turn_id
            source_turn_id = item.get("source_turn_id")
            try:
                if source_turn_id is not None:
                    source_turn_id = int(source_turn_id)
            except (TypeError, ValueError):
                source_turn_id = None

            if source_turn_id is None:
                repaired_id = map_fact_to_turn(item["text"], turns)
                if repaired_id is not None:
                    item["source_turn_id"] = repaired_id
                else:
                    raise ValueError(
                        f"Fact '{item['text']}' has missing source_turn_id "
                        f"and could not be confidently mapped to a single source turn. "
                        f"Available turn IDs: {sorted(list(valid_turn_ids))}"
                    )
            else:
                item["source_turn_id"] = source_turn_id

            # Validate with FactEntry
            fact_entries.append(FactEntry.model_validate(item))
        except (ValueError, ValidationError) as e:
            logger.warning(f"Skipping malformed fact entry {item} due to error: {e}")
            last_error = e
            continue

    if not fact_entries and parsed:
        if last_error is not None:
            raise ValueError(str(last_error))
        else:
            raise ValueError("No valid fact entries collected from parsed list")

    return fact_entries



def clean_json_text(text: str) -> str:
    """Defensively cleans and extracts raw JSON from an LLM response string."""
    from rem.memory.robust_extract import strip_markdown_fences, find_balanced_json, coerce_sibling_objects_to_list
    from json_repair import repair_json

    stripped = strip_markdown_fences(text)
    isolated, _ = find_balanced_json(stripped)
    isolated = coerce_sibling_objects_to_list(isolated)

    try:
        json.loads(isolated)
        return isolated
    except Exception:
        pass

    repaired = repair_json(isolated)
    return repaired


def extract_deterministic_facts(turns: list["Turn"]) -> list[FactEntry]:
    """Helper to deterministically extract infrastructure facts from a list of turns."""
    entries = []
    for turn in turns:
        content = turn.content
        sentences = re.split(r"(?<=[.!?])\s+", content)
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            slot_key = infer_slot_key(sentence)
            if slot_key:
                slot_value = infer_slot_value(slot_key, sentence)
                if slot_value:
                    if slot_key in ("infra.host", "infra.codename", "replica.region"):
                        kind = "entity"
                    elif slot_key in ("infra.port", "infra.threshold", "infra.retention", "vendor_api.rate_limit"):
                        kind = "number"
                    elif slot_key == "infra.engine":
                        kind = "decision"
                    else:
                        kind = "entity"

                    entry = FactEntry(
                        kind=kind,
                        text=sentence,
                        source_turn_id=turn.turn_id,
                        slot_key=slot_key,
                        slot_value=slot_value,
                    )
                    entries.append(entry)
    return entries


def extract_facts(
    turns: list["Turn"],
    client: NpuClient,
    deterministic_fact_capture: bool = True,
) -> FactsLedger:
    """Extracts facts from a list of turns using the NpuClient.

    Tries once, and on failure retries once with a schema-enforcement reminder.
    Raises FactsExtractionError if both attempts fail.
    """
    if not turns:
        return FactsLedger()

    # Format the turns into a clean text block
    conversation_lines = []
    for turn in turns:
        conversation_lines.append(
            f"Turn {turn.turn_id} - {turn.role.upper()}: {turn.content}"
        )
    conversation_text = "\n".join(conversation_lines)

    user_prompt = FACT_EXTRACTION_USER_TEMPLATE.format(
        conversation_text=conversation_text
    )

    messages = [
        {"role": "system", "content": FACT_EXTRACTION_SYSTEM},
        {"role": "user", "content": user_prompt},
    ]

    _extraction_stats["attempts"] += 1
    response_text = ""
    try:
        response_text = client.chat(messages, max_tokens=client.settings.npu_max_tokens)
    except Exception as exc:
        raise FactsExtractionError(f"First fact extraction API call failed: {exc}") from exc

    from rem.memory.robust_extract import robust_extract_json
    fact_entries, diagnostics = robust_extract_json(response_text, turns, client, messages)

    _record_extraction_diagnostics(diagnostics)

    if not diagnostics["success"]:
        error_msg = diagnostics.get("error") or "Unknown extraction pipeline failure"
        if diagnostics.get("truncated"):
            error_prefix = f"Failed to extract facts after retry (due to truncation): {error_msg}"
        else:
            error_prefix = f"Failed to extract facts after retry: {error_msg}"
        raise FactsExtractionError(
            f"{error_prefix}. Raw first response: {repr(response_text)}"
        )

    ledger = FactsLedger(entries=fact_entries)
    if deterministic_fact_capture:
        ledger.merge(FactsLedger(entries=extract_deterministic_facts(turns)))
    return ledger

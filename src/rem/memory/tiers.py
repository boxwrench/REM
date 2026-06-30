"""Data models and serialization for REM memory tiers."""

import json
import os
from pathlib import Path
from time import time as get_time
from pydantic import BaseModel, Field
from rem.memory.facts_ledger import FactsLedger


def count_tokens(text: str) -> int:
    """Counts tokens in a string using a len(text) // 4 heuristic.

    This fallback heuristic is used because a real tokenizer is not
    active or available during offline/standalone runtime.
    """
    if not text:
        return 0
    return max(1, len(text) // 4)


class Turn(BaseModel):
    """Represents a single chat turn in the verbatim memory tier."""
    role: str
    content: str
    turn_id: int
    tokens: int
    # Provenance (additive, nullable; backward-compatible with pre-existing states
    # serialized without these fields). session_id ties a turn to its conversation;
    # timestamp is an ISO-8601 string. Both default None so old captures still load.
    # Needed for temporal reasoning and an honest "freeze with known limits": the
    # observation store must be able to express WHEN and WHO, which it could not before.
    session_id: str | None = None
    timestamp: str | None = None


class SpanSummary(BaseModel):
    """Represents a prose summary covering a span of consolidated turns."""
    covers_turn_ids: list[int]
    text: str
    tokens: int
    created_at: float = Field(default_factory=get_time)
    rendered_text: str | None = None



class MemoryState(BaseModel):
    """Holds the overall state of the volatile memory tiers.

    Supports atomic persistence and schema version gating.
    """
    schema_version: int = 1
    turns: list[Turn] = []
    summaries: list[SpanSummary] = []
    ledger: FactsLedger = Field(default_factory=FactsLedger)

    def save(self, path: str | Path) -> None:
        """Atomically saves the MemoryState to a file.

        Uses a temp file write followed by a rename swap (os.replace).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(".tmp")
        try:
            # Serialize using Pydantic's JSON export
            with open(tmp_path, "w", encoding="utf-8") as f:
                f.write(self.model_dump_json(indent=2))
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, path)
        except Exception as e:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except Exception:
                    pass
            raise e

    @classmethod
    def load(cls, path: str | Path) -> "MemoryState":
        """Loads a MemoryState from a file.

        Refuses to load if the file's schema_version is higher than supported.
        """
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        file_version = data.get("schema_version", 1)
        current_version = 1
        if file_version > current_version:
            raise ValueError(
                f"Cannot load state: schema version {file_version} "
                f"is higher than supported version {current_version}."
            )

        return cls.model_validate(data)

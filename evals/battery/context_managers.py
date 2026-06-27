"""Context managers: Truncation (control) and REM arms."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from evals.battery.models import Session
from rem.config import Settings
from rem.memory.assembler import assemble
from rem.memory.compactor import compact_once, should_compact
from rem.memory.tiers import MemoryState, Turn, count_tokens


@dataclass
class ContextStats:
    assembled_tokens: int
    retained_session_ids: set[str] = field(default_factory=set)
    compactions: int = 0


class ContextManager(Protocol):
    def ingest(self, sessions: list[Session], budget_tokens: int) -> None: ...
    def assemble(self) -> str: ...
    def stats(self) -> ContextStats: ...
    def evidence_retained(self, answer_session_ids: list[str]) -> bool: ...


def _flatten(sessions: list[Session]) -> list[tuple[str, dict]]:
    """Return (session_id, turn) pairs in chronological order."""
    out: list[tuple[str, dict]] = []
    for s in sessions:
        for turn in s.turns:
            out.append((s.session_id, turn))
    return out


def _render(turn: dict) -> str:
    return f"{turn.get('role', 'user').upper()}: {turn.get('content', '')}"


class TruncationContextManager:
    """Control: keep the most-recent turns that fit in the budget."""

    def __init__(self) -> None:
        self._kept: list[tuple[str, dict]] = []
        self._assembled = ""

    def ingest(self, sessions: list[Session], budget_tokens: int) -> None:
        flat = _flatten(sessions)
        kept_rev: list[tuple[str, dict]] = []
        total = 0
        for sid, turn in reversed(flat):  # newest first
            t = count_tokens(_render(turn))
            if total + t > budget_tokens:
                break
            kept_rev.append((sid, turn))
            total += t
        self._kept = list(reversed(kept_rev))
        self._assembled = "\n".join(_render(turn) for _, turn in self._kept)

    def assemble(self) -> str:
        return self._assembled

    def stats(self) -> ContextStats:
        return ContextStats(
            assembled_tokens=count_tokens(self._assembled),
            retained_session_ids={sid for sid, _ in self._kept},
        )

    def evidence_retained(self, answer_session_ids: list[str]) -> bool:
        kept = {sid for sid, _ in self._kept}
        return any(sid in kept for sid in answer_session_ids)


REM_SYSTEM = "You are a helpful assistant with long-term memory."
REM_TASK = "Answer the user's question using the conversation memory."

# REM's assembled memory (summaries + facts ledger + recent window) is rendered
# in full by the assembler: the ledger and summaries grow with conversation
# length and are NOT bounded. 16k is the reserved memory region from the
# architecture spec (§3: 32k window, 16k for memory), used here as the assemble
# ceiling instead of budget*4.
#
# WARNING — this does NOT fix the overflow on long-haystack knowledge-update
# items. Measured compacted memory there is 37k–58k tokens (see
# bench/battery/FINDINGS.md and bench/battery/diag_031748ae_w64k.json), which
# exceeds both this 16k ceiling AND the answering model's own context window
# (~32–40k; it returns HTTP 400 "Max length reached!"). REM therefore still
# raises ContextLimitExceeded and cannot answer on these items. Making REM
# answer requires a bounded read path (retrieval/eviction that fits memory to
# the model window), not a larger ceiling. Tracked as the next architecture gate.
REM_MEMORY_WINDOW_TOKENS = 16000


class RemContextManager:
    """REM arm: run the real compaction loop over foreign chat turns."""

    def __init__(self, client, settings: Settings | None = None) -> None:
        self._client = client
        self._settings = settings or Settings(summarizer_model="gemma4-it:e2b")
        self._state = MemoryState()
        self._compactions = 0
        self._assembled = ""
        self._gold_texts: list[tuple[str, str]] = []  # (session_id, turn text)

    def ingest(self, sessions: list[Session], budget_tokens: int) -> None:
        """Single-call: construct a new RemContextManager per question."""
        if self._state.turns:
            raise RuntimeError("RemContextManager.ingest is single-call; build a new instance per question")
        s = self._settings.model_copy()
        s.compact_trigger_tokens = budget_tokens
        # Assemble within the REM memory window, not budget*4: compacted memory
        # (summaries + ledger) grows with conversation length and overflows
        # budget*4 on long-haystack items. See REM_MEMORY_WINDOW_TOKENS.
        if s.max_context_tokens < REM_MEMORY_WINDOW_TOKENS:
            s.max_context_tokens = REM_MEMORY_WINDOW_TOKENS
        turn_id = 0
        for sess in sessions:
            for turn in sess.turns:
                text = _render(turn)
                turn_id += 1
                self._gold_texts.append((sess.session_id, turn.get("content", "")))
                self._state.turns.append(
                    Turn(role=turn.get("role", "user"), content=text,
                         turn_id=turn_id, tokens=count_tokens(text))
                )
                while should_compact(self._state, s):
                    res = compact_once(self._state, self._client, s)
                    if not res.compacted:
                        break
                    self._compactions += 1
        self._assembled = assemble(self._state, REM_SYSTEM, REM_TASK, settings=s)

    def assemble(self) -> str:
        return self._assembled

    def stats(self) -> ContextStats:
        return ContextStats(
            assembled_tokens=count_tokens(self._assembled),
            retained_session_ids=set(),  # REM doesn't preserve session ids; see evidence_retained
            compactions=self._compactions,
        )

    def evidence_retained(self, answer_session_ids: list[str]) -> bool:
        gold = [txt for sid, txt in self._gold_texts if sid in answer_session_ids]
        ctx = self._assembled.lower()
        # heuristic: a gold turn's first salient token-run survived into the context
        for txt in gold:
            snippet = " ".join(txt.split()[:6]).lower()
            if snippet and snippet in ctx:
                return True
        return False

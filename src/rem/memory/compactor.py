"""Core memory compaction job that consolidates oldest turns into summaries and facts."""

import argparse
import logging
import sys
from pathlib import Path
from pydantic import BaseModel
from filelock import FileLock, Timeout

from rem.config import Settings
from rem.npu_client import NpuClient
from rem.memory.tiers import MemoryState, SpanSummary, count_tokens
from rem.memory.facts_ledger import extract_facts, FactsExtractionError
from rem.memory.prompts import FACT_COMPACTION_SYSTEM, FACT_COMPACTION_USER_TEMPLATE

logger = logging.getLogger("rem.memory.compactor")


class CompactionResult(BaseModel):
    """Result of a single compaction step."""
    compacted: bool
    turns_compacted: int
    new_summary: str | None = None
    new_facts_count: int = 0


def should_compact(state: MemoryState, settings: Settings | None = None) -> bool:
    """Checks if the total token count of verbatim turns exceeds the threshold."""
    settings = settings or Settings()
    total_verbatim_tokens = sum(turn.tokens for turn in state.turns)
    return total_verbatim_tokens > settings.compact_trigger_tokens


def compact_once(
    state: MemoryState, client: NpuClient, settings: Settings | None = None
) -> CompactionResult:
    """Selects the oldest span of turns, extracts facts, summarizes them, and swaps.

    Verbatim turns inside the protected recent window are preserved.
    Atomicity: State mutation occurs only after NPU calls succeed.
    """
    settings = settings or Settings()

    # Calculate candidates beyond the protected recent window
    candidates_count = len(state.turns) - settings.keep_recent_turns
    if candidates_count <= 0:
        logger.info("No candidate turns available for compaction beyond the recent window.")
        return CompactionResult(compacted=False, turns_compacted=0)

    # Select the oldest span of turns to compact
    span_size = min(settings.compact_span_turns, candidates_count)
    span_turns = state.turns[:span_size]

    # Step 1: Extract facts from the oldest span
    try:
        new_ledger = extract_facts(
            span_turns,
            client,
            deterministic_fact_capture=settings.deterministic_fact_capture,
        )
    except FactsExtractionError as exc:
        # Fall back to keeping the span verbatim, log warning, do not compact
        logger.warning(
            f"Fact extraction failed: {exc}. Keeping span verbatim as fallback."
        )
        return CompactionResult(compacted=False, turns_compacted=0)

    # Step 2: Summarize the oldest span (including the extracted facts for consistency)
    conversation_lines = []
    for turn in span_turns:
        conversation_lines.append(
            f"Turn {turn.turn_id} - {turn.role.upper()}: {turn.content}"
        )
    conversation_text = "\n".join(conversation_lines)

    system_prompt = FACT_COMPACTION_SYSTEM.format(
        rendered_ledger=new_ledger.render()
    )
    user_prompt = FACT_COMPACTION_USER_TEMPLATE.format(
        conversation_text=conversation_text
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    try:
        summary_text = client.chat(
            messages, model=settings.summarizer_model, max_tokens=settings.npu_max_tokens
        )
    except Exception as exc:
        logger.warning(
            f"Compaction summarization failed: {exc}. Keeping span verbatim."
        )
        return CompactionResult(compacted=False, turns_compacted=0)

    # Step 3: Swap span with summary and merge ledger atomically
    # We only mutate here, ensuring all-or-nothing completion
    state.ledger.merge(new_ledger)
    state.ledger.rendered_text = None

    # Reset rendered_text cache for all summaries to force recompute at next assembly
    for s in state.summaries:
        s.rendered_text = None

    covers_turn_ids = [turn.turn_id for turn in span_turns]
    summary_tokens = count_tokens(summary_text)
    new_summary = SpanSummary(
        covers_turn_ids=covers_turn_ids,
        text=summary_text,
        tokens=summary_tokens,
    )
    state.summaries.append(new_summary)

    # Remove the compacted oldest turns from verbatim tier
    state.turns = state.turns[span_size:]

    logger.info(
        f"Compacted {span_size} turns into 1 summary containing {len(new_ledger.entries)} facts."
    )
    return CompactionResult(
        compacted=True,
        turns_compacted=span_size,
        new_summary=summary_text,
        new_facts_count=len(new_ledger.entries),
    )


def state_lock_path(state_path_obj: Path) -> Path:
    """Path of the short-lived lock guarding state load-mutate-save sections.

    Held briefly by both the foreground sidecar and the compactor's snapshot /
    merge steps — never during the slow NPU work — so foreground turns are never
    blocked on compaction.
    """
    return state_path_obj.with_suffix(".state.lock")


def _fold_in_concurrent_turns(
    result: MemoryState, latest: MemoryState, compacted_turn_ids: set[int]
) -> None:
    """Append any turns the foreground added since the compaction snapshot.

    Compaction only ever removes the *oldest* turns and the foreground only ever
    *appends*, so a turn present on disk that was neither compacted away nor
    already in the result is a concurrent foreground turn — graft it onto the
    end (turn ids are monotonic, so order is preserved)."""
    existing_ids = {t.turn_id for t in result.turns}
    for turn in latest.turns:
        if turn.turn_id not in compacted_turn_ids and turn.turn_id not in existing_ids:
            result.turns.append(turn)


def run_background(
    state_path: str, client: NpuClient, settings: Settings | None = None
) -> None:
    """Continuously compacts state until the trigger clears.

    Locking model (prevents the lost-update race with the foreground path):
    - A non-blocking *compaction* lock serializes compactions — if one is already
      running, this call returns immediately rather than queueing.
    - The slow NPU work runs with **no** state lock held, so foreground requests
      are never blocked on it.
    - The snapshot load and the final save each take a short *state* lock; the
      save re-reads the latest on-disk state and folds in any turns the foreground
      appended during compaction, so its result never clobbers newer turns.
    """
    settings = settings or Settings()
    state_path_obj = Path(state_path)
    compaction_lock = FileLock(state_path_obj.with_suffix(".lock"))
    state_lock = FileLock(state_lock_path(state_path_obj))

    try:
        compaction_lock.acquire(timeout=0)
    except Timeout:
        logger.info("Compaction already in progress for %s; skipping.", state_path)
        return

    try:
        with state_lock:
            if state_path_obj.exists():
                state = MemoryState.load(state_path_obj)
            else:
                state = MemoryState()

        snapshot_turn_ids = {t.turn_id for t in state.turns}

        any_compacted = False
        while should_compact(state, settings):
            res = compact_once(state, client, settings)
            if not res.compacted:
                # Break loop to prevent infinite retry if extraction/summarize fails
                break
            any_compacted = True

        if any_compacted:
            compacted_turn_ids = snapshot_turn_ids - {t.turn_id for t in state.turns}
            with state_lock:
                latest = (
                    MemoryState.load(state_path_obj)
                    if state_path_obj.exists()
                    else MemoryState()
                )
                _fold_in_concurrent_turns(state, latest, compacted_turn_ids)
                state.save(state_path_obj)
    finally:
        compaction_lock.release()


def main() -> None:
    """Entry point for the compactor CLI."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )

    parser = argparse.ArgumentParser(description="REM Memory Compactor Background Job")
    parser.add_argument(
        "--state",
        type=str,
        help="Path to memory state JSON file",
    )
    args = parser.parse_args()

    settings = Settings()
    state_path = args.state or str(
        Path(settings.vault_dir) / "memory_state.json"
    )

    client = NpuClient(settings)
    try:
        run_background(state_path, client, settings)
    except Exception as e:
        logger.error(f"Failed to run compactor background job: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

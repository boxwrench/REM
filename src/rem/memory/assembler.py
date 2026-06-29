"""Context assembler that builds the prompt for the LLM in a stability-first order.

Order of prompt sections:
1. System
2. Task
3. Episodic summaries (oldest-first)
4. Facts ledger (in full)
5. Semantic recall block (if present)
6. Recent verbatim turns
"""

import re
from rem.config import Settings
from rem.memory.facts_ledger import SlotObservation, infer_slot_key, infer_slot_value
from rem.memory.tiers import MemoryState, count_tokens


class ContextLimitExceeded(Exception):
    """Raised when the assembled context exceeds the maximum allowed tokens."""
    pass


def recent_slot_values(state: MemoryState) -> dict[str, list[SlotObservation]]:
    """Returns known current-state slot values observed in recent verbatim turns."""
    values: dict[str, list[SlotObservation]] = {}
    for turn in state.turns:
        slot_key = infer_slot_key(turn.content)
        if not slot_key:
            continue
        slot_value = infer_slot_value(slot_key, turn.content)
        if not slot_value:
            continue
        values.setdefault(slot_key, []).append((turn.turn_id, slot_value))
    return values


def assemble(
    state: MemoryState,
    system: str,
    task: str,
    semantic_block: str = "",
    settings: Settings | None = None,
) -> str:
    """Assembles the final prompt string in stability-first order.

    Order: System + Task + Episodic Summaries + Facts Ledger + Semantic Recall + Verbatim turns.
    Checks the assembled context length against settings.max_context_tokens and raises
    ContextLimitExceeded if the limit is violated.
    """
    settings = settings or Settings()

    parts = []

    # 1. System Prompt
    parts.append(f"=== SYSTEM ===\n{system}")

    # 2. Task
    parts.append(f"=== TASK ===\n{task}")

    # Compute slot quarantine
    recent_slots = recent_slot_values(state)
    quarantine = get_quarantined_stale_values(state)
    if state.ledger.include_stale_on_render:
        # A temporal selector intentionally requested ordered history. Applying
        # the current-state quarantine here would silently remove that evidence.
        quarantine = {}
    all_quarantined_vals = {val for vals in quarantine.values() for val in vals}

    # 3. Episodic Summaries (ordered oldest-first, matching insertion order in state.summaries)
    if state.summaries:
        sorted_summaries = sorted(
            state.summaries,
            key=lambda s: min(s.covers_turn_ids) if s.covers_turn_ids else 0
        )
        summary_lines = []
        for s in sorted_summaries:
            if s.rendered_text is None:
                s.rendered_text = filter_summary_text(s.text, all_quarantined_vals)
            if s.rendered_text.strip():
                summary_lines.append(f"- {s.rendered_text}")
        if summary_lines:
            parts.append("=== EPISODIC SUMMARIES ===\n" + "\n".join(summary_lines))

    # 4. Facts Ledger (always rendered in full, never truncated/omitted)
    if state.ledger.rendered_text is None:
        state.ledger.rendered_text = state.ledger.render(
            include_stale=state.ledger.include_stale_on_render,
            suppress_slots=recent_slots,
            quarantine=quarantine,
        )
    if state.ledger.rendered_text:
        parts.append(f"=== FACTS LEDGER ===\n{state.ledger.rendered_text}")

    # 5. Semantic Recall Block (from Path B)
    if semantic_block:
        parts.append(f"=== SEMANTIC RECALL ===\n{semantic_block}")

    # 6. Recent Verbatim Turns
    if state.turns:
        verbatim_lines = []
        for turn in state.turns:
            verbatim_lines.append(f"{turn.role.upper()}: {turn.content}")
        parts.append("=== VERBATIM TRANSCRIPT ===\n" + "\n".join(verbatim_lines))

    prompt = "\n\n".join(parts)

    # Hard cap check
    tokens = count_tokens(prompt)
    if tokens > settings.max_context_tokens:
        raise ContextLimitExceeded(
            f"Assembled context tokens ({tokens}) exceeds maximum limit ({settings.max_context_tokens})."
        )

    return prompt


def assemble_messages(
    state: MemoryState,
    system: str,
    task: str,
    semantic_block: str = "",
    settings: Settings | None = None,
) -> list[dict]:
    """Assembles the final prompt as a list of chat messages for chat APIs.

    The system prompt, task, summaries, ledger, and semantic recall are combined
    into the first system message. The recent verbatim turns are appended as
    separate user/assistant messages.
    """
    settings = settings or Settings()

    parts = []
    parts.append(f"=== SYSTEM ===\n{system}")
    parts.append(f"=== TASK ===\n{task}")

    # Compute slot quarantine
    recent_slots = recent_slot_values(state)
    quarantine = get_quarantined_stale_values(state)
    if state.ledger.include_stale_on_render:
        quarantine = {}
    all_quarantined_vals = {val for vals in quarantine.values() for val in vals}

    if state.summaries:
        sorted_summaries = sorted(
            state.summaries,
            key=lambda s: min(s.covers_turn_ids) if s.covers_turn_ids else 0
        )
        summary_lines = []
        for s in sorted_summaries:
            if s.rendered_text is None:
                s.rendered_text = filter_summary_text(s.text, all_quarantined_vals)
            if s.rendered_text.strip():
                summary_lines.append(f"- {s.rendered_text}")
        if summary_lines:
            parts.append("=== EPISODIC SUMMARIES ===\n" + "\n".join(summary_lines))

    if state.ledger.rendered_text is None:
        state.ledger.rendered_text = state.ledger.render(
            include_stale=state.ledger.include_stale_on_render,
            suppress_slots=recent_slots,
            quarantine=quarantine,
        )
    if state.ledger.rendered_text:
        parts.append(f"=== FACTS LEDGER ===\n{state.ledger.rendered_text}")

    if semantic_block:
        parts.append(f"=== SEMANTIC RECALL ===\n{semantic_block}")

    system_content = "\n\n".join(parts)

    messages = [{"role": "system", "content": system_content}]
    for turn in state.turns:
        messages.append({"role": turn.role, "content": turn.content})

    # Hard cap check on total text content of the messages
    all_text = "\n\n".join(msg["content"] for msg in messages)
    tokens = count_tokens(all_text)
    if tokens > settings.max_context_tokens:
        raise ContextLimitExceeded(
            f"Assembled messages tokens ({tokens}) exceeds maximum limit ({settings.max_context_tokens})."
        )

    return messages


def assembled_tokens(
    state: MemoryState,
    system: str,
    task: str,
    semantic_block: str = "",
    settings: Settings | None = None,
) -> int:
    """Counts the tokens in the assembled context string.

    Raises ContextLimitExceeded if the limit is violated.
    """
    prompt = assemble(state, system, task, semantic_block, settings)
    return count_tokens(prompt)


def get_quarantined_stale_values(state: MemoryState) -> dict[str, set[str]]:
    """Returns a dict mapping slot_key to a set of stale values that are quarantined.

    A value is quarantined for a slot_key if there is a newer active value for the same
    slot_key (from either active ledger entries or recent verbatim turns) with a turn ID
    greater than the stale value's source_turn_id.
    """
    # 1. Collect all active observations for each slot key
    # Each observation is a tuple: (turn_id, slot_value)
    active_obs: dict[str, list[tuple[int, str]]] = {}

    # Active observations from ledger:
    for entry in state.ledger.active_entries():
        if entry.slot_key and entry.slot_value:
            active_obs.setdefault(entry.slot_key, []).append((entry.source_turn_id, entry.slot_value))

    # Active observations from recent verbatim turns:
    recent = recent_slot_values(state)
    for slot_key, obs in recent.items():
        for turn_id, val in obs:
            active_obs.setdefault(slot_key, []).append((turn_id, val))

    # Find the newest active value for each slot key
    newest_active: dict[str, tuple[int, str]] = {}
    for slot_key, obs_list in active_obs.items():
        if obs_list:
            # Newest is the one with the maximum turn_id
            newest_active[slot_key] = max(obs_list, key=lambda x: x[0])

    # 2. Identify stale values to quarantine
    # A stale value is any slot value in the ledger that is older than the newest active value
    # for that slot key, and has a different value.
    quarantine: dict[str, set[str]] = {}
    for entry in state.ledger.entries:
        if not entry.slot_key or not entry.slot_value:
            continue
        slot_key = entry.slot_key
        if slot_key in newest_active:
            newest_turn_id, newest_val = newest_active[slot_key]
            # If there is a newer active value for this slot key, and the entry's value is different,
            # and the entry's turn is older than that newer active turn:
            if newest_turn_id > entry.source_turn_id and entry.slot_value != newest_val:
                quarantine.setdefault(slot_key, set()).add(entry.slot_value)

    for summary in state.summaries:
        if not summary.covers_turn_ids:
            continue
        summary_latest_turn = max(summary.covers_turn_ids)
        for slot_key, (newest_turn_id, newest_val) in newest_active.items():
            if summary_latest_turn >= newest_turn_id:
                continue
            for sentence in split_summary_sentences(summary.text):
                summary_val = infer_slot_value(slot_key, sentence)
                if summary_val and summary_val != newest_val:
                    quarantine.setdefault(slot_key, set()).add(summary_val)

    return quarantine


def contains_stale_value(text: str, stale_values: set[str]) -> bool:
    """Returns True if the text contains any of the stale values, enforcing word boundaries."""
    for val in stale_values:
        start = 0
        while True:
            pos = text.find(val, start)
            if pos == -1:
                break
            # check characters before and after
            char_before = text[pos - 1] if pos > 0 else ' '
            char_after = text[pos + len(val)] if pos + len(val) < len(text) else ' '

            # If both characters before and after are non-alphanumeric/non-underscore, it's a match!
            if not char_before.isalnum() and char_before != '_' and not char_after.isalnum() and char_after != '_':
                return True
            start = pos + 1
    return False


def filter_summary_text(text: str, quarantined_vals: set[str]) -> str:
    """Removes sentences from the summary text that contain quarantined stale values."""
    if not quarantined_vals:
        return text
    # Split by lines to preserve structural line breaks
    lines = text.split('\n')
    filtered_lines = []
    for line in lines:
        filtered_sentences = []
        for sentence in split_summary_sentences(line):
            if not contains_stale_value(sentence, quarantined_vals):
                filtered_sentences.append(sentence)
        if filtered_sentences:
            filtered_lines.append(" ".join(filtered_sentences))
    return "\n".join(filtered_lines)


def split_summary_sentences(text: str) -> list[str]:
    """Splits summary text into conservative sentence chunks."""
    return [part for part in re.split(r"(?<=[.!?])\s+", text) if part]

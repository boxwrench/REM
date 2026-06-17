"""Unit tests for the context assembler component."""

import pytest
from rem.config import Settings
from rem.memory.tiers import MemoryState, Turn, SpanSummary, count_tokens
from rem.memory.facts_ledger import FactsLedger, FactEntry
from rem.memory.assembler import (
    assemble,
    assemble_messages,
    assembled_tokens,
    ContextLimitExceeded,
)


def test_assembler_ordering_and_content():
    """Asserts that prompt segments are deterministic, stable, and ordered correctly."""
    turns = [
        Turn(role="user", content="Turn 1 content", turn_id=1, tokens=5),
        Turn(role="assistant", content="Turn 2 content", turn_id=2, tokens=5),
    ]
    summaries = [
        SpanSummary(covers_turn_ids=[1], text="Summary 1 text", tokens=5),
        SpanSummary(covers_turn_ids=[2], text="Summary 2 text", tokens=5),
    ]
    ledger = FactsLedger()
    ledger.add(FactEntry(kind="decision", text="Important decision", source_turn_id=1))

    state = MemoryState(turns=turns, summaries=summaries, ledger=ledger)

    system = "System prompt here"
    task = "Task instruction here"
    semantic = "Semantic memory block"

    prompt = assemble(state, system, task, semantic)

    # Assert section markers exist and appear in correct stability-first order:
    # System -> Task -> Summaries -> Ledger -> Semantic -> Verbatim
    idx_system = prompt.index("=== SYSTEM ===")
    idx_task = prompt.index("=== TASK ===")
    idx_summaries = prompt.index("=== EPISODIC SUMMARIES ===")
    idx_ledger = prompt.index("=== FACTS LEDGER ===")
    idx_semantic = prompt.index("=== SEMANTIC RECALL ===")
    idx_verbatim = prompt.index("=== VERBATIM TRANSCRIPT ===")

    assert idx_system < idx_task
    assert idx_task < idx_summaries
    assert idx_summaries < idx_ledger
    assert idx_ledger < idx_semantic
    assert idx_semantic < idx_verbatim

    # Verify content in sections
    assert "System prompt here" in prompt
    assert "Task instruction here" in prompt
    assert "- Summary 1 text" in prompt
    assert "- Summary 2 text" in prompt
    assert "Important decision" in prompt
    assert "Semantic memory block" in prompt
    assert "USER: Turn 1 content" in prompt
    assert "ASSISTANT: Turn 2 content" in prompt


def test_assembler_empty_optional_sections():
    """Asserts that optional blocks are omitted when they are empty."""
    state = MemoryState(turns=[], summaries=[], ledger=FactsLedger())

    prompt = assemble(state, "System", "Task", "")

    assert "=== SYSTEM ===" in prompt
    assert "=== TASK ===" in prompt
    assert "=== EPISODIC SUMMARIES ===" not in prompt
    assert "=== FACTS LEDGER ===" not in prompt
    assert "=== SEMANTIC RECALL ===" not in prompt
    assert "=== VERBATIM TRANSCRIPT ===" not in prompt


def test_assembler_messages_variant():
    """Asserts that assemble_messages structures messages correctly for chat APIs."""
    turns = [
        Turn(role="user", content="Turn 1 content", turn_id=1, tokens=5),
        Turn(role="assistant", content="Turn 2 content", turn_id=2, tokens=5),
    ]
    summaries = [
        SpanSummary(covers_turn_ids=[1], text="Summary 1 text", tokens=5),
    ]
    ledger = FactsLedger()
    ledger.add(FactEntry(kind="decision", text="Important decision", source_turn_id=1))

    state = MemoryState(turns=turns, summaries=summaries, ledger=ledger)

    messages = assemble_messages(state, "System", "Task", "Semantic")

    # Expecting: 1 system message, 2 history turns
    assert len(messages) == 3
    assert messages[0]["role"] == "system"
    assert "=== SYSTEM ===" in messages[0]["content"]
    assert "=== TASK ===" in messages[0]["content"]
    assert "=== EPISODIC SUMMARIES ===" in messages[0]["content"]
    assert "=== FACTS LEDGER ===" in messages[0]["content"]
    assert "=== SEMANTIC RECALL ===" in messages[0]["content"]
    assert "=== VERBATIM TRANSCRIPT ===" not in messages[0]["content"]

    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "Turn 1 content"

    assert messages[2]["role"] == "assistant"
    assert messages[2]["content"] == "Turn 2 content"


def test_assembler_hard_cap_limit():
    """Asserts that ContextLimitExceeded is raised when the context exceeds max_context_tokens."""
    state = MemoryState(
        turns=[Turn(role="user", content="Long content " * 10, turn_id=1, tokens=20)]
    )
    settings = Settings(max_context_tokens=10)  # Very low threshold

    # assemble raises exception
    with pytest.raises(ContextLimitExceeded) as exc_info:
        assemble(state, "System", "Task", "", settings)
    assert "exceeds maximum limit" in str(exc_info.value)

    # assemble_messages raises exception
    with pytest.raises(ContextLimitExceeded):
        assemble_messages(state, "System", "Task", "", settings)

    # assembled_tokens raises exception
    with pytest.raises(ContextLimitExceeded):
        assembled_tokens(state, "System", "Task", "", settings)


def test_assembled_tokens_calculation():
    """Asserts that assembled_tokens counts the tokens of the fully assembled string correctly."""
    state = MemoryState(
        turns=[Turn(role="user", content="Turn content", turn_id=1, tokens=5)]
    )

    prompt = assemble(state, "System", "Task", "Semantic")
    expected_tokens = count_tokens(prompt)

    assert assembled_tokens(state, "System", "Task", "Semantic") == expected_tokens


def test_assembler_suppresses_old_region_in_ledger():
    """Asserts that if recent verbatim turns contain a newer region, the older ledger region is suppressed."""
    # Old region in ledger
    ledger = FactsLedger()
    ledger.add(FactEntry(kind="entity", text="Plan to place the replica in the us-west-2 region.", source_turn_id=10))

    # New region in recent verbatim turns
    turns = [
        Turn(role="user", content="Correction: legal says the replica must live in eu-central-1 after all.", turn_id=20, tokens=10)
    ]

    state = MemoryState(turns=turns, ledger=ledger)
    prompt = assemble(state, "System", "Task")
    messages = assemble_messages(state, "System", "Task")

    assert "eu-central-1" in prompt
    assert "us-west-2" not in prompt
    assert "eu-central-1" in messages[1]["content"]
    assert "us-west-2" not in messages[0]["content"]


def test_assembler_suppresses_old_ratelimit_in_ledger():
    """Asserts that if recent verbatim turns contain a newer rate limit, the older ledger rate limit is suppressed."""
    # Old rate limit in ledger
    ledger = FactsLedger()
    ledger.add(FactEntry(kind="number", text="The vendor API rate limit is 1,200 requests per minute.", source_turn_id=10))

    # New rate limit in recent verbatim turns
    turns = [
        Turn(role="user", content="Correction: the real rate limit is 950 requests per minute.", turn_id=20, tokens=10)
    ]

    state = MemoryState(turns=turns, ledger=ledger)
    prompt = assemble(state, "System", "Task")
    messages = assemble_messages(state, "System", "Task")

    assert "950 requests per minute" in prompt
    assert "1,200 requests per minute" not in prompt
    assert "950 requests per minute" in messages[1]["content"]
    assert "1,200 requests per minute" not in messages[0]["content"]


def test_assembler_keeps_ledger_for_same_value_reaffirmation():
    """Asserts that matching recent slot values do not suppress active ledger state."""
    ledger = FactsLedger()
    ledger.add(
        FactEntry(
            kind="number",
            text="The vendor API rate limit is 950 requests per minute.",
            source_turn_id=10,
        )
    )
    turns = [
        Turn(
            role="user",
            content="Please remember the vendor API rate limit is 950 requests per minute.",
            turn_id=20,
            tokens=10,
        )
    ]

    state = MemoryState(turns=turns, ledger=ledger)
    prompt = assemble(state, "System", "Task")
    messages = assemble_messages(state, "System", "Task")

    assert "The vendor API rate limit is 950 requests per minute." in prompt
    assert "The vendor API rate limit is 950 requests per minute." in messages[0]["content"]


def test_assembler_keeps_newer_ledger_against_older_recent_slot_value():
    """Asserts that older verbatim slot values do not suppress newer ledger facts."""
    ledger = FactsLedger()
    ledger.add(
        FactEntry(
            kind="entity",
            text="Correction: legal says the replica must live in eu-central-1 after all.",
            source_turn_id=20,
        )
    )
    turns = [
        Turn(
            role="user",
            content="Earlier note: the replica was planned for us-west-2.",
            turn_id=10,
            tokens=10,
        )
    ]

    state = MemoryState(turns=turns, ledger=ledger)
    prompt = assemble(state, "System", "Task")
    messages = assemble_messages(state, "System", "Task")

    assert "eu-central-1" in prompt
    assert "eu-central-1" in messages[0]["content"]


def test_quarantine_old_region_in_summary_without_stale_ledger_entry():
    """Asserts summary-only old region text is suppressed when a newer region exists."""
    from rem.memory.tiers import SpanSummary
    summaries = [
        SpanSummary(
            covers_turn_ids=[1],
            text="The replica region was us-west-2. The network setup is complete.",
            tokens=5,
        )
    ]
    turns = [
        Turn(role="user", content="Correction: please move to eu-central-1.", turn_id=20, tokens=10)
    ]
    ledger = FactsLedger()
    ledger.add(FactEntry(kind="entity", text="Correct replica region is eu-central-1.", slot_key="replica.region", slot_value="eu-central-1", source_turn_id=20))

    state = MemoryState(turns=turns, summaries=summaries, ledger=ledger)
    prompt = assemble(state, "System", "Task")

    assert "eu-central-1" in prompt
    assert "us-west-2" not in prompt
    assert "The network setup is complete." in prompt


def test_quarantine_old_ratelimit_in_summary():
    """Asserts that an old ratelimit in a summary is not rendered if there is a newer ratelimit."""
    from rem.memory.tiers import SpanSummary
    summaries = [SpanSummary(covers_turn_ids=[1], text="Rate limit is 1,200 requests per minute. Threshold is 0.5.", tokens=5)]

    ledger = FactsLedger()
    ledger.add(FactEntry(kind="number", text="Correct limit is 950 requests per minute.", slot_key="vendor_api.rate_limit", slot_value="950 requests per minute", source_turn_id=20))
    ledger.add(FactEntry(kind="number", text="Old limit was 1,200 requests per minute.", slot_key="vendor_api.rate_limit", slot_value="1,200 requests per minute", source_turn_id=10, status="stale"))

    state = MemoryState(turns=[], summaries=summaries, ledger=ledger)
    prompt = assemble(state, "System", "Task")

    assert "950 requests per minute" in prompt
    assert "1,200 requests per minute" not in prompt
    assert "Threshold is 0.5." in prompt


def test_duplicate_active_ledger_entries_only_newest_renders():
    """Asserts that duplicate active ledger entries for the same slot only render the newest."""
    e1 = FactEntry(kind="number", text="Set port to 13306", slot_key="infra.port", slot_value="13306", source_turn_id=10, status="active")
    e2 = FactEntry(kind="number", text="Set port to 14306", slot_key="infra.port", slot_value="14306", source_turn_id=20, status="active")

    ledger = FactsLedger(entries=[e1, e2])

    rendered = ledger.render()
    assert "14306" in rendered
    assert "13306" not in rendered
    assert ledger.duplicate_active_suppressions == 1


def test_non_compaction_append_only_prefix_behavior():
    """Asserts that adding turns on non-compaction turns does not rewrite summary text (append-only prefix stability)."""
    from rem.memory.tiers import SpanSummary
    summaries = [
        SpanSummary(
            covers_turn_ids=[1],
            text="The replica region was us-west-2. The network setup is complete.",
            tokens=5,
        )
    ]
    ledger = FactsLedger()
    ledger.add(FactEntry(kind="entity", text="Correct replica region is us-west-2.", slot_key="replica.region", slot_value="us-west-2", source_turn_id=10))

    turns = [
        Turn(role="user", content="Checking status.", turn_id=11, tokens=5)
    ]

    state = MemoryState(turns=turns, summaries=summaries, ledger=ledger)
    prompt1 = assemble(state, "System", "Task")

    # Add a turn with a correcting value (e.g., region = eu-central-1)
    # This turn is added on a non-compaction turn.
    # Note: containing "replica region" ensures it matches slot key for active slot matching.
    state.turns.append(
        Turn(role="user", content="Correction: please move to replica region eu-central-1.", turn_id=12, tokens=10)
    )

    prompt2 = assemble(state, "System", "Task")

    # Assert that prompt2 starts with prompt1 (append-only)
    assert prompt2.startswith(prompt1)
    # The old value 'us-west-2' should still be in both prompts because no compaction has occurred yet
    assert "us-west-2" in prompt1
    assert "us-west-2" in prompt2

    # Now simulate a compaction (which resets rendered_text caches)
    for s in state.summaries:
        s.rendered_text = None
    state.ledger.rendered_text = None
    # Also add the new correction to the ledger
    state.ledger.add(FactEntry(kind="entity", text="Correct replica region is eu-central-1.", slot_key="replica.region", slot_value="eu-central-1", source_turn_id=12))

    prompt3 = assemble(state, "System", "Task")

    # Now the old region is quarantined and filtered out, so it should not be present
    assert "us-west-2" not in prompt3
    assert "eu-central-1" in prompt3


def test_correcting_turns_do_not_mutate_rendered_ledger_between_compactions():
    """Asserts that correcting turns matching slot keys do not mutate the rendered facts ledger between compactions."""
    ledger = FactsLedger()
    ledger.add(FactEntry(kind="entity", text="Correct replica region is us-west-2.", slot_key="replica.region", slot_value="us-west-2", source_turn_id=10))

    turns = [
        Turn(role="user", content="Checking status.", turn_id=11, tokens=5)
    ]

    state = MemoryState(turns=turns, ledger=ledger)
    prompt1 = assemble(state, "System", "Task")

    # Add a correcting turn that matches "replica.region"
    state.turns.append(
        Turn(role="user", content="Correction: please use replica region eu-central-1 instead.", turn_id=12, tokens=10)
    )

    prompt2 = assemble(state, "System", "Task")

    # The ledger sections should be identical, indicating that prompt2 is prefix-stable.
    # The active ledger rendered section has been cached and should not have been updated.
    assert "us-west-2" in prompt1
    assert "us-west-2" in prompt2
    assert "eu-central-1" not in prompt2.split("=== VERBATIM TRANSCRIPT ===")[0]
    assert prompt2.startswith(prompt1)



def test_deterministic_summary_ordering():
    """Asserts that summaries are sorted oldest-first based on covers_turn_ids when rendered."""
    from rem.memory.tiers import SpanSummary
    s1 = SpanSummary(covers_turn_ids=[10, 11], text="Summary for turn 10-11", tokens=5)
    s2 = SpanSummary(covers_turn_ids=[5, 6], text="Summary for turn 5-6", tokens=5)
    s3 = SpanSummary(covers_turn_ids=[20], text="Summary for turn 20", tokens=5)

    # Put them in out-of-order order in state.summaries
    state = MemoryState(turns=[], summaries=[s3, s1, s2], ledger=FactsLedger())
    prompt = assemble(state, "System", "Task")

    # They should be rendered in order: s2 (turns 5-6), s1 (turns 10-11), s3 (turn 20)
    expected_order = [
        "- Summary for turn 5-6",
        "- Summary for turn 10-11",
        "- Summary for turn 20"
    ]
    for line in expected_order:
        assert line in prompt

    idx2 = prompt.index("Summary for turn 5-6")
    idx1 = prompt.index("Summary for turn 10-11")
    idx3 = prompt.index("Summary for turn 20")

    assert idx2 < idx1 < idx3


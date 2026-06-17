"""Unit tests for facts ledger extraction and utilities."""

import pytest
import httpx
from rem.npu_client import NpuClient
from rem.memory.tiers import Turn
from rem.memory.facts_ledger import (
    FactEntry,
    FactsLedger,
    extract_facts,
    clean_json_text,
    FactsExtractionError,
)


def test_clean_json_text():
    """Asserts that clean_json_text strips markdown fences, conversational text, and converts sequences of objects."""
    assert clean_json_text("  [1, 2]  ") == "[1, 2]"
    assert clean_json_text("```json\n[3, 4]\n```") == "[3, 4]"
    assert clean_json_text("```\n{\"key\": \"val\"}\n```") == "{\"key\": \"val\"}"
    
    # Surrounding conversational text
    assert clean_json_text("Here is JSON: [1, 2] end of message") == "[1, 2]"
    assert clean_json_text("Some text {\"key\": \"val\"} other text") == "{\"key\": \"val\"}"
    
    # Sequences of objects converted to list
    multiple_objects = (
        "{\n  \"a\": 1\n}\n\n{\n  \"b\": 2\n}"
    )
    expected_list = "[{\n  \"a\": 1\n},{\n  \"b\": 2\n}]"
    assert clean_json_text(multiple_objects) == expected_list



def test_extract_facts_success(mock_npu):
    """Asserts successful fact extraction on the first attempt."""
    client = NpuClient()
    turns = [Turn(role="user", content="Deploy to NPU", turn_id=1, tokens=3)]

    # Mock JSON response
    mock_npu.post("/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": '[{"kind": "decision", "text": "Deploy to NPU", "source_turn_id": 1}]'
                        }
                    }
                ]
            },
        )
    )

    ledger = extract_facts(turns, client)
    assert len(ledger.entries) == 1
    assert ledger.entries[0].kind == "decision"
    assert ledger.entries[0].text == "Deploy to NPU"
    assert ledger.entries[0].source_turn_id == 1


def test_extract_facts_sibling_json_objects(mock_npu):
    """Regression: some small models (e.g. llama3.2:1b) emit list elements
    as sibling JSON objects separated by whitespace, not a single array. The
    caller must run clean_json_text before json.loads so this does not raise
    JSONDecodeError ("Extra data") and silently fail every compaction. Guards
    against re-dropping the clean_json_text call from extract_facts."""
    client = NpuClient()
    turns = [
        Turn(role="user", content="The checklist reflects this", turn_id=3, tokens=4),
        Turn(role="user", content="Revisit the capacity estimate", turn_id=5, tokens=4),
    ]

    sibling_objects = (
        '{"kind": "entity", "text": "The checklist reflects this", "source_turn_id": 3}\n\n'
        '{"kind": "decision", "text": "Revisit the capacity estimate", "source_turn_id": 5}'
    )
    mock_npu.post("/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={"choices": [{"message": {"content": sibling_objects}}]},
        )
    )

    # Must not raise; both sibling objects must be coerced into entries.
    ledger = extract_facts(turns, client, deterministic_fact_capture=False)
    assert mock_npu.calls.call_count == 1, "should parse on first attempt, not fall to retry"
    assert len(ledger.entries) == 2
    assert {e.text for e in ledger.entries} == {
        "The checklist reflects this",
        "Revisit the capacity estimate",
    }


def test_extract_facts_retry_success(mock_npu):
    """Asserts fact extraction succeeds on the second attempt after first fails."""
    client = NpuClient()
    turns = [Turn(role="user", content="Testing retry", turn_id=2, tokens=3)]

    # First returns bad JSON, second returns valid JSON list
    mock_npu.post("/v1/chat/completions").mock(
        side_effect=[
            httpx.Response(200, json={"choices": [{"message": {"content": "Bad JSON response text"}}]}),
            httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "content": '[{"kind": "entity", "text": "Strix Box", "source_turn_id": 2}]'
                            }
                        }
                    ]
                },
            ),
        ]
    )

    ledger = extract_facts(turns, client)
    assert len(ledger.entries) == 1
    assert ledger.entries[0].text == "Strix Box"
    assert mock_npu.calls.call_count == 2


def test_extract_facts_failure_raises_error(mock_npu):
    """Asserts that both failures result in a FactsExtractionError."""
    client = NpuClient()
    turns = [Turn(role="user", content="Testing failure", turn_id=3, tokens=3)]

    # Both returns bad JSON
    mock_npu.post("/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={"choices": [{"message": {"content": "Not JSON at all"}}]})
    )

    with pytest.raises(FactsExtractionError) as exc_info:
        extract_facts(turns, client)

    assert "Failed to extract facts after retry" in str(exc_info.value)
    assert mock_npu.calls.call_count == 2


def test_facts_ledger_merge_deduplication():
    """Asserts that merging ledgers deduplicates entries by normalized text."""
    ledger1 = FactsLedger()
    ledger1.add(FactEntry(kind="decision", text="Deploy to NPU", source_turn_id=1))
    ledger1.add(FactEntry(kind="entity", text="Strix Halo", source_turn_id=2))

    ledger2 = FactsLedger()
    # "deploy to npu" is a duplicate when normalized (case and whitespace)
    ledger2.add(FactEntry(kind="decision", text=" deploy to NPU ", source_turn_id=3))
    # "Zen 5" is a new fact
    ledger2.add(FactEntry(kind="entity", text="Zen 5", source_turn_id=4))

    ledger1.merge(ledger2)
    
    assert len(ledger1.entries) == 3
    # Check that we kept the first one and added Zen 5, skipping the duplicate
    texts = [e.text for e in ledger1.entries]
    assert "Deploy to NPU" in texts
    assert "Strix Halo" in texts
    assert "Zen 5" in texts
    assert " deploy to NPU " not in texts


def test_facts_ledger_supersedes_old_ratelimit_value():
    """Asserts that corrected rate-limit facts do not both render as active."""
    ledger = FactsLedger()
    ledger.add(
        FactEntry(
            kind="number",
            text="The vendor API rate limit is 1,200 requests per minute.",
            source_turn_id=33,
        )
    )
    ledger.merge(
        FactsLedger(
            entries=[
                FactEntry(
                    kind="number",
                    text=(
                        "Correction on the vendor API: the real rate limit is "
                        "950 requests per minute."
                    ),
                    source_turn_id=241,
                )
            ]
        )
    )

    active_rendered = ledger.render()
    assert "950 requests per minute" in active_rendered
    assert "1,200 requests per minute" not in active_rendered

    stale_entries = ledger.stale_entries()
    assert len(stale_entries) == 1
    assert stale_entries[0].slot_key == "vendor_api.rate_limit"
    assert stale_entries[0].slot_value == "1,200 requests per minute"
    assert stale_entries[0].superseded_by_turn_id == 241
    assert "1,200 requests per minute" in ledger.render(include_stale=True)


def test_facts_ledger_supersedes_old_region_value():
    """Asserts that corrected cloud regions are treated as current-state facts."""
    ledger = FactsLedger(
        entries=[
            FactEntry(
                kind="entity",
                text="Plan to place the replica in the us-west-2 region.",
                source_turn_id=61,
            )
        ]
    )
    ledger.merge(
        FactsLedger(
            entries=[
                FactEntry(
                    kind="entity",
                    text=(
                        "Correction: legal says the replica must live in "
                        "eu-central-1 after all."
                    ),
                    source_turn_id=321,
                )
            ]
        )
    )

    active_rendered = ledger.render()
    assert "eu-central-1" in active_rendered
    assert "us-west-2" not in active_rendered
    assert [entry.slot_value for entry in ledger.stale_entries()] == ["us-west-2"]


def test_facts_ledger_render():
    """Asserts formatting of the rendered bulleted string."""
    ledger = FactsLedger()
    assert ledger.render() == ""

    ledger.add(FactEntry(kind="decision", text="Enable IOMMU", source_turn_id=5))
    ledger.add(FactEntry(kind="number", text="50 TOPS", source_turn_id=6))

    expected = (
        "Facts Ledger:\n"
        "- [decision] Enable IOMMU (Turn 5)\n"
        "- [number] 50 TOPS (Turn 6)"
    )
    assert ledger.render() == expected


def test_fact_entry_normalization():
    """Asserts that unrecognized or differently cased fact kinds are normalized to valid kinds."""
    # Unrecognized kind "text" is mapped to "entity"
    entry1 = FactEntry(kind="text", text="some generic text", source_turn_id=1)
    assert entry1.kind == "entity"

    # Unrecognized kind "fact" is mapped to "entity"
    entry2 = FactEntry(kind="fact", text="some fact", source_turn_id=2)
    assert entry2.kind == "entity"

    # Differently cased valid kind "DECISION" is normalized to "decision"
    entry3 = FactEntry(kind="DECISION", text="decision made", source_turn_id=3)
    assert entry3.kind == "decision"

    # Valid kinds are left unchanged
    entry4 = FactEntry(kind="number", text="42", source_turn_id=4)
    assert entry4.kind == "number"


def test_deterministic_extraction_and_supersession():
    """Asserts that high-precision slot inference/extraction works for the 8 A7a fact shapes."""
    from rem.memory.facts_ledger import extract_deterministic_facts

    turns = [
        Turn(
            role="user",
            content=(
                "Remember: the staging gateway must listen on port 47713. "
                "The cold-storage host is vega-archive-02; all dumps go there."
            ),
            turn_id=1,
            tokens=20,
        ),
        Turn(
            role="user",
            content=(
                "Internally this whole effort is codenamed BRAMBLE. "
                "Decision: we will use DuckDB instead of SQLite for the telemetry store."
            ),
            turn_id=2,
            tokens=20,
        ),
        Turn(
            role="user",
            content=(
                "Set the anomaly alert threshold to 0.83 and do not tune it further. "
                "Compliance requires raw logs be retained for ninety-one days exactly."
            ),
            turn_id=3,
            tokens=20,
        ),
        # Supersessions
        Turn(
            role="user",
            content=(
                "Correction on the vendor API: the real rate limit is "
                "950 requests per minute."
            ),
            turn_id=4,
            tokens=20,
        ),
        Turn(
            role="user",
            content="Actually, the host is vega-archive-03 now.",
            turn_id=5,
            tokens=20,
        ),
        Turn(
            role="user",
            content="We also changed the listen port to 47715 instead.",
            turn_id=6,
            tokens=20,
        ),
        Turn(
            role="user",
            content="Plan to place the replica in the us-west-2 region.",
            turn_id=7,
            tokens=20,
        ),
        Turn(
            role="user",
            content="Correction: legal says the replica must live in eu-central-1 after all.",
            turn_id=8,
            tokens=20,
        ),
    ]

    entries = extract_deterministic_facts(turns)

    # Check that we extracted entries
    ledger = FactsLedger()
    for entry in entries:
        ledger.add(entry)

    # Check active entries
    active = {entry.slot_key: entry.slot_value for entry in ledger.active_entries()}

    assert active["infra.host"] == "vega-archive-03"
    assert active["infra.port"] == "47715"
    assert active["infra.codename"] == "BRAMBLE"
    assert active["infra.engine"] == "DuckDB"
    assert active["infra.threshold"] == "0.83"
    assert active["infra.retention"] == "ninety-one days"
    assert active["vendor_api.rate_limit"] == "950 requests per minute"
    assert active["replica.region"] == "eu-central-1"

    # Check stale entries
    stale = {entry.slot_key: entry.slot_value for entry in ledger.stale_entries()}
    assert stale["infra.host"] == "vega-archive-02"
    assert stale["infra.port"] == "47713"
    assert stale["replica.region"] == "us-west-2"

    # Check turn ids and provenance are preserved
    port_entries = [e for e in ledger.entries if e.slot_key == "infra.port"]
    # 47713 should be turn 1, 47715 should be turn 6
    p1 = next(e for e in port_entries if e.slot_value == "47713")
    p2 = next(e for e in port_entries if e.slot_value == "47715")
    assert p1.source_turn_id == 1
    assert p2.source_turn_id == 6


def test_held_out_facts_general_representation():
    """Asserts that held-out facts get general subjects and attributes without A7a-specific regexes."""
    # We instantiate FactEntry for a held-out version needle
    entry = FactEntry(kind="number", text="Make sure the deployed software version is set to 2.4.9.", source_turn_id=1)
    assert entry.subject == "deployed software"
    assert entry.attribute == "version"
    assert entry.slot_key == "deployed software.version"
    assert entry.slot_value == "2.4.9"


def test_held_out_supersession():
    """Asserts that supersession works generally for the new general subject/attribute shape."""
    ledger = FactsLedger()
    # Old statement
    ledger.add(FactEntry(kind="number", text="We originally configured the system for 8 concurrent jobs.", source_turn_id=1))
    # New statement
    ledger.add(FactEntry(kind="number", text="Correction: the task queue is configured for 12 concurrent jobs now.", source_turn_id=2))

    active = ledger.active_entries()
    stale = ledger.stale_entries()

    assert len(active) == 1
    assert active[0].slot_value == "12 concurrent jobs"
    assert len(stale) == 1
    assert stale[0].slot_value == "8 concurrent jobs"


def test_bounded_ledger_growth():
    """Asserts that ledger size remains bounded under repeated corrections and duplicates."""
    ledger = FactsLedger(max_stale_entries=3)
    
    # 1. Test duplicate additions do not grow ledger
    for _ in range(5):
        ledger.add(FactEntry(kind="entity", text="The deployment target machine is polaris-node-09.", source_turn_id=1))
    assert len(ledger.entries) == 1

    # 2. Test stale growth limit
    # We generate multiple corrections for the same slot to create stale entries
    for i in range(10):
        ledger.add(FactEntry(kind="number", text=f"Make sure the deployed software version is set to 2.4.{i}.", source_turn_id=i+2))

    # We should have exactly 2 active entries + at most 3 stale entries
    assert len(ledger.active_entries()) == 2
    assert len(ledger.stale_entries()) <= 3
    assert len(ledger.entries) <= 5


def test_ledger_cache_invalidation():
    """Asserts that cache invalidation of rendered_text holds on any mutation."""
    ledger = FactsLedger()
    ledger.add(FactEntry(kind="entity", text="cold-storage host is vega-archive-02", source_turn_id=1))
    
    # Set rendered_text manually to simulate cache
    ledger.rendered_text = "Cached Rendering"
    
    # Mutate via add
    ledger.add(FactEntry(kind="entity", text="staging gateway host is vega-archive-03", source_turn_id=2))
    assert ledger.rendered_text is None

    # Set cache again
    ledger.rendered_text = "Cached Rendering 2"
    # Mutate via merge
    other = FactsLedger()
    other.add(FactEntry(kind="entity", text="replica region is us-west-2", source_turn_id=3))
    ledger.merge(other)
    assert ledger.rendered_text is None


def test_missing_source_turn_id_repaired(mock_npu):
    """Asserts that a missing source_turn_id is repaired when the fact can be confidently mapped to one turn."""
    client = NpuClient()
    turns = [
        Turn(role="user", content="polaris-node-09 is the main deployment target.", turn_id=5, tokens=10),
        Turn(role="user", content="The listen port is 8080.", turn_id=6, tokens=10),
    ]

    mock_npu.post("/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": '[{"kind": "entity", "text": "polaris-node-09 is the main target"}]'
                        }
                    }
                ]
            },
        )
    )

    ledger = extract_facts(turns, client, deterministic_fact_capture=False)
    assert len(ledger.entries) == 1
    assert ledger.entries[0].source_turn_id == 5
    assert ledger.entries[0].text == "polaris-node-09 is the main target"


def test_missing_source_turn_id_rejected_ambiguous(mock_npu):
    """Asserts that a missing source_turn_id is rejected when the mapping to turns is ambiguous."""
    client = NpuClient()
    turns = [
        Turn(role="user", content="We set version to 1.0.", turn_id=5, tokens=10),
        Turn(role="user", content="Correction: we set version to 2.0.", turn_id=6, tokens=10),
    ]

    mock_npu.post("/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": '[{"kind": "number", "text": "version set"}]'
                        }
                    }
                ]
            },
        )
    )

    with pytest.raises(FactsExtractionError) as exc_info:
        extract_facts(turns, client, deterministic_fact_capture=False)

    assert "could not be confidently mapped to a single source turn" in str(exc_info.value)


def test_truncation_detection_raises_error(mock_npu):
    """Asserts that a truncated JSON output is detected and raises FactsExtractionError."""
    client = NpuClient()
    turns = [
        Turn(role="user", content="polaris-node-09 is the target.", turn_id=5, tokens=10),
    ]

    # Model returns truncated JSON on both tries
    mock_npu.post("/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": '[{"kind": "number"'
                        }
                    }
                ]
            },
        )
    )

    with pytest.raises(FactsExtractionError) as exc_info:
        extract_facts(turns, client, deterministic_fact_capture=False)

    assert "due to truncation" in str(exc_info.value)


def test_retry_path_succeeds_with_correction(mock_npu):
    """Asserts that if the first attempt is truncated, a retry with a larger budget succeeds."""
    client = NpuClient()
    turns = [
        Turn(role="user", content="polaris-node-09 is the target.", turn_id=5, tokens=10),
    ]

    # First attempt: truncated response
    # Second attempt: complete response
    mock_npu.post("/v1/chat/completions").mock(
        side_effect=[
            httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "content": '[{"kind": "number"'
                            }
                        }
                    ]
                },
            ),
            httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "content": '[{"kind": "entity", "text": "polaris-node-09", "source_turn_id": 5}]'
                            }
                        }
                    ]
                },
            ),
        ]
    )

    ledger = extract_facts(turns, client, deterministic_fact_capture=False)
    assert len(ledger.entries) == 1
    assert ledger.entries[0].text == "polaris-node-09"
    assert ledger.entries[0].source_turn_id == 5
    assert mock_npu.calls.call_count == 2


# ---------------------------------------------------------------------------
# Regression tests — re-extraction guard + model-emitted general fields
# ---------------------------------------------------------------------------

def test_dq3_re_extraction_guard_general_supersession():
    """RED→GREEN: a re-surfaced OLD value must LOSE to the active one.

    The failure mode: the model re-extracts stale data from a turn that
    mentions an old value in passing, AFTER a correction has already been ledgered.
    The re-extracted old entry must be marked stale immediately by the
    re-extraction guard — it must NOT appear in active_entries() or render().

    Guard rule: if (subject, attribute, slot_value) is already in stale_entries,
    a new entry with that same triple must be rejected as stale regardless of its
    source_turn_id. The active entry (correction) always wins.
    """
    ledger = FactsLedger()

    # Turn 16: old value — model emits subject/attribute explicitly
    ledger.add(FactEntry(
        kind="number",
        text="We originally configured the system for 8 concurrent jobs.",
        source_turn_id=16,
        subject="task_queue",
        attribute="concurrency",
        slot_value="8 concurrent jobs",
    ))

    # Turn 120: correction supersedes turn 16
    ledger.add(FactEntry(
        kind="number",
        text="Correction: the task queue is configured for 12 concurrent jobs now.",
        source_turn_id=120,
        subject="task_queue",
        attribute="concurrency",
        slot_value="12 concurrent jobs",
    ))

    # Verify supersession fired: 12 active, 8 stale
    assert len(ledger.active_entries()) == 1
    assert ledger.active_entries()[0].slot_value == "12 concurrent jobs"
    assert len(ledger.stale_entries()) == 1
    assert ledger.stale_entries()[0].slot_value == "8 concurrent jobs"

    # Turn 200: model re-extracts the old value from a turn mentioning it in passing.
    # source_turn_id=200 is numerically newer than 120, but the VALUE is already stale.
    # The re-extraction guard must detect this and keep "8 concurrent jobs" as stale.
    ledger.add(FactEntry(
        kind="number",
        text="The system had 8 concurrent jobs configured previously.",
        source_turn_id=200,
        subject="task_queue",
        attribute="concurrency",
        slot_value="8 concurrent jobs",
    ))

    # Fix: old value must still be stale; 12 concurrent jobs still active
    active = ledger.active_entries()
    assert len(active) == 1, (
        f"Expected 1 active entry (12 concurrent jobs), got {len(active)}: "
        f"{[e.slot_value for e in active]}"
    )
    assert active[0].slot_value == "12 concurrent jobs", (
        f"Expected '12 concurrent jobs' as active, got '{active[0].slot_value}'"
    )
    rendered = ledger.render()
    assert "12 concurrent jobs" in rendered
    assert "8 concurrent jobs" not in rendered, (
        "Re-extracted stale value must NOT appear in rendered active ledger"
    )


def test_dq3_model_emitted_fields_pass_through(mock_npu):
    """Model-emitted subject/attribute/value fields drive supersession end-to-end.

    The updated extraction schema asks the model to emit subject, attribute, value,
    is_correction. This test verifies those fields are accepted by extract_facts and
    that FactEntry uses them to set slot_key/slot_value, triggering correct supersession.
    """
    import json as _json
    client = NpuClient()
    turns = [
        Turn(role="user",
             content="The preferred notification channel is Slack.",
             turn_id=10, tokens=8),
        Turn(role="user",
             content="Actually, use email for notifications instead.",
             turn_id=20, tokens=8),
    ]

    # Model response in the extended schema (subject/attribute/value explicitly emitted)
    model_response = _json.dumps([
        {
            "kind": "decision",
            "text": "Preferred notification channel is Slack.",
            "source_turn_id": 10,
            "subject": "notification",
            "attribute": "channel",
            "value": "Slack",
        },
        {
            "kind": "decision",
            "text": "Notification channel changed to email.",
            "source_turn_id": 20,
            "subject": "notification",
            "attribute": "channel",
            "value": "email",
            "is_correction": True,
        },
    ])

    mock_npu.post("/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={"choices": [{"message": {"content": model_response}}]},
        )
    )

    ledger = extract_facts(turns, client, deterministic_fact_capture=False)
    active = ledger.active_entries()
    # email must win over Slack in the rendered output
    assert len(active) == 1, (
        f"Expected 1 active entry, got {len(active)}: {[e.slot_value for e in active]}"
    )
    assert active[0].slot_value == "email", (
        f"Expected 'email' as active, got '{active[0].slot_value}'"
    )
    rendered = ledger.render()
    assert "email" in rendered
    assert "Slack" not in rendered, (
        "Old notification channel must not appear in rendered active ledger"
    )

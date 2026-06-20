"""Regression coverage for the JSON robustness pipeline (robust_extract.py).

These tests pin the behavior of `robust_extract_json` against the malformed
outputs the small NPU extraction model produces in practice, per the
implementation roadmap (item 3):

  - markdown-fenced JSON
  - truncated arrays or objects
  - sibling JSON objects without an enclosing list
  - repeated/looping objects
  - partly salvageable responses with at least one valid fact
  - unrecoverable responses that should fail cleanly without compaction

We test at the `robust_extract_json` entry point (not the helper level) so the
returned `diagnostics` contract is covered alongside the parsing logic. That
diagnostics dict is what the battery artifact will later surface (roadmap item 2).
"""

import httpx
import pytest

from rem.memory.robust_extract import robust_extract_json
from rem.memory.tiers import Turn
from rem.npu_client import NpuClient


def _turn(turn_id: int, content: str = "placeholder content") -> Turn:
    return Turn(role="user", content=content, turn_id=turn_id, tokens=4)


# The retry stage of the pipeline calls client.chat(); for the no-retry cases a
# plain NpuClient never touches the network. Cases that exercise retry use the
# mock_npu fixture to control the second response.
_NO_RETRY_MESSAGES: list[dict[str, str]] = [
    {"role": "system", "content": "extract facts"},
    {"role": "user", "content": "conversation"},
]


def test_markdown_fenced_json_is_stripped_and_parsed():
    """A ```json fenced array must parse on the fence-strip path, not fall to retry."""
    turns = [_turn(1, "Deploy the service to the NPU")]
    raw = (
        "```json\n"
        '[{"kind": "decision", "text": "Deploy the service to the NPU", "source_turn_id": 1}]\n'
        "```"
    )

    facts, diag = robust_extract_json(raw, turns, NpuClient(), _NO_RETRY_MESSAGES)

    assert diag["success"] is True
    assert diag["fence_stripped"] is True
    assert diag["retried"] is False
    assert len(facts) == 1
    assert facts[0].text == "Deploy the service to the NPU"


def test_truncated_array_salvages_complete_prefix_and_flags_truncation():
    """A response cut off mid-second-object must keep the first complete fact and
    record truncated=True so the miss is observable rather than silent."""
    turns = [_turn(5, "polaris-node-09 is the deployment target")]
    raw = (
        '[{"kind": "entity", "text": "polaris-node-09 is the deployment target", '
        '"source_turn_id": 5}, {"kind": "number"'
    )

    facts, diag = robust_extract_json(raw, turns, NpuClient(), _NO_RETRY_MESSAGES)

    assert diag["truncated"] is True
    assert diag["success"] is True
    assert len(facts) == 1
    assert facts[0].text == "polaris-node-09 is the deployment target"


def test_sibling_objects_without_list_are_coerced():
    """Two top-level objects separated by whitespace (no enclosing []) must both
    be recovered without falling to retry."""
    turns = [_turn(3, "The checklist reflects this"), _turn(5, "Revisit the capacity estimate")]
    raw = (
        '{"kind": "entity", "text": "The checklist reflects this", "source_turn_id": 3}\n\n'
        '{"kind": "decision", "text": "Revisit the capacity estimate", "source_turn_id": 5}'
    )

    facts, diag = robust_extract_json(raw, turns, NpuClient(), _NO_RETRY_MESSAGES)

    assert diag["success"] is True
    assert diag["retried"] is False
    assert {f.text for f in facts} == {
        "The checklist reflects this",
        "Revisit the capacity estimate",
    }


def test_repeated_looping_objects_are_detected_and_collapsed():
    """A degenerate response that repeats the same object many times must be
    flagged as a loop and salvaged down to a single fact."""
    turns = [_turn(7, "The anomaly alert threshold is 0.83")]
    one = (
        '{"kind": "number", "text": "The anomaly alert threshold is 0.83", '
        '"source_turn_id": 7}'
    )
    raw = "[" + ", ".join([one] * 6) + "]"

    facts, diag = robust_extract_json(raw, turns, NpuClient(), _NO_RETRY_MESSAGES)

    assert diag["loop_detected"] is True
    assert diag["loop_salvaged"] is True
    assert diag["success"] is True
    assert len(facts) == 1
    assert facts[0].text == "The anomaly alert threshold is 0.83"


def test_partly_salvageable_keeps_valid_fact_drops_garbage():
    """A list mixing one valid fact with non-fact garbage objects must yield the
    valid fact and silently drop the garbage."""
    turns = [_turn(2, "Use DuckDB for the telemetry store")]
    raw = (
        "["
        '{"kind": "decision", "text": "Use DuckDB for the telemetry store", "source_turn_id": 2},'
        '{"not_a_fact": true, "random": 99},'
        '{"kind": "number"}'
        "]"
    )

    facts, diag = robust_extract_json(raw, turns, NpuClient(), _NO_RETRY_MESSAGES)

    assert diag["success"] is True
    assert len(facts) == 1
    assert facts[0].text == "Use DuckDB for the telemetry store"


def test_unrecoverable_response_fails_cleanly(mock_npu):
    """Prose with no JSON, where the retry also returns prose, must fail cleanly:
    success=False and no facts, so the caller skips compaction rather than
    fabricating or crashing."""
    turns = [_turn(3, "Testing failure")]
    raw = "I'm sorry, I cannot produce that. Here is a paragraph instead."

    # Retry (stage 6) returns more prose.
    mock_npu.post("/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={"choices": [{"message": {"content": "Still just prose, no JSON here."}}]},
        )
    )

    facts, diag = robust_extract_json(raw, turns, NpuClient(), _NO_RETRY_MESSAGES)

    assert diag["success"] is False
    assert facts == []
    assert diag["error"]

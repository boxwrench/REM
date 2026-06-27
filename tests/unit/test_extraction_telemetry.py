"""Extraction telemetry must be observable, not write-only.

Roadmap item 2: a REM miss caused by dropped facts should be visible as an
extraction failure, not hidden inside answer accuracy. These tests pin the
public accessors (`reset_extraction_stats` / `get_extraction_stats`) and the
diagnostic taxonomy they expose (strict parse, repair, retry, salvage,
truncation, loop detection, final failure).
"""

import httpx
import pytest

from rem.npu_client import NpuClient
from rem.memory.tiers import Turn
from rem.memory.facts_ledger import (
    extract_facts,
    FactsExtractionError,
    reset_extraction_stats,
    get_extraction_stats,
)


def _mock_content(mock_npu, content, *more):
    """Mock the chat endpoint to return `content`, then each of `more` in turn."""
    responses = [
        httpx.Response(200, json={"choices": [{"message": {"content": c}}]})
        for c in (content, *more)
    ]
    if len(responses) == 1:
        mock_npu.post("/v1/chat/completions").mock(return_value=responses[0])
    else:
        mock_npu.post("/v1/chat/completions").mock(side_effect=responses)


def test_reset_zeroes_all_counters():
    reset_extraction_stats()
    stats = get_extraction_stats()
    assert stats["attempts"] == 0
    assert stats["failures"] == 0
    assert stats["truncations"] == 0


def test_clean_first_attempt_counts_strict_parse(mock_npu):
    reset_extraction_stats()
    client = NpuClient()
    turns = [Turn(role="user", content="Deploy to NPU", turn_id=1, tokens=3)]
    _mock_content(
        mock_npu,
        '[{"kind": "decision", "text": "Deploy to NPU", "source_turn_id": 1}]',
    )

    extract_facts(turns, client, deterministic_fact_capture=False)

    stats = get_extraction_stats()
    assert stats["attempts"] == 1
    assert stats["strict_parse"] == 1
    assert stats["failures"] == 0
    assert stats["truncations"] == 0


def test_unrecoverable_response_counts_failure(mock_npu):
    reset_extraction_stats()
    client = NpuClient()
    turns = [Turn(role="user", content="Testing failure", turn_id=1, tokens=3)]
    _mock_content(mock_npu, "Not JSON at all", "Still not JSON at all")

    with pytest.raises(FactsExtractionError):
        extract_facts(turns, client, deterministic_fact_capture=False)

    stats = get_extraction_stats()
    assert stats["attempts"] == 1
    assert stats["retried"] == 1
    assert stats["failures"] == 1


def test_initial_api_failure_counts_as_failure(mock_npu):
    # When the model endpoint rejects the request outright (e.g. HTTP 400
    # "Max length reached!"), client.chat() raises before any JSON pipeline runs.
    # That is still an extraction failure and must be counted, not hidden as
    # attempts=1/failures=0.
    reset_extraction_stats()
    client = NpuClient()
    turns = [Turn(role="user", content="anything", turn_id=1, tokens=2)]
    mock_npu.post("/v1/chat/completions").mock(
        return_value=httpx.Response(400, json={"error": {"message": "Max length reached!"}})
    )

    with pytest.raises(FactsExtractionError):
        extract_facts(turns, client, deterministic_fact_capture=False)

    stats = get_extraction_stats()
    assert stats["attempts"] == 1
    assert stats["failures"] == 1


def test_truncated_failure_counts_truncation(mock_npu):
    reset_extraction_stats()
    client = NpuClient()
    turns = [Turn(role="user", content="polaris-node-09 is the target", turn_id=5, tokens=10)]
    # Truncated JSON on both attempts -> detected as truncation, ultimately fails.
    _mock_content(mock_npu, '[{"kind": "number"', '[{"kind": "number"')

    with pytest.raises(FactsExtractionError):
        extract_facts(turns, client, deterministic_fact_capture=False)

    stats = get_extraction_stats()
    assert stats["truncations"] >= 1
    assert stats["failures"] == 1

"""Unit tests for memory tier data models and persistence."""

import json
import os
import pytest
from rem.memory.tiers import Turn, SpanSummary, MemoryState, count_tokens


def test_count_tokens_monotonicity():
    """Asserts token count monotonicity: longer text >= tokens, and base cases."""
    assert count_tokens("") == 0
    assert count_tokens("a") == 1
    assert count_tokens("abcd") == 1
    assert count_tokens("abcde") == 1
    assert count_tokens("abcdefgh") == 2

    # Test monotonicity across lengths 0 to 100
    last_tokens = 0
    for length in range(1, 100):
        text = "x" * length
        tokens = count_tokens(text)
        assert tokens >= last_tokens
        last_tokens = tokens


def test_save_load_roundtrip(tmp_path):
    """Asserts that MemoryState can be saved and loaded with exact equivalence."""
    state_file = tmp_path / "memory_state.json"

    # Construct a state with mock data
    turns = [
        Turn(role="user", content="Hello NPU", turn_id=1, tokens=2),
        Turn(role="assistant", content="Hello User", turn_id=2, tokens=2),
    ]
    summaries = [
        SpanSummary(covers_turn_ids=[1, 2], text="Initial greeting.", tokens=3, created_at=123.45)
    ]
    state = MemoryState(turns=turns, summaries=summaries)

    # Save to disk
    state.save(state_file)
    assert state_file.exists()

    # Load from disk
    loaded_state = MemoryState.load(state_file)

    # Assert equality
    assert loaded_state.schema_version == state.schema_version
    assert len(loaded_state.turns) == len(state.turns)
    assert loaded_state.turns[0].content == "Hello NPU"
    assert loaded_state.turns[1].role == "assistant"
    assert len(loaded_state.summaries) == len(state.summaries)
    assert loaded_state.summaries[0].text == "Initial greeting."
    assert loaded_state.summaries[0].covers_turn_ids == [1, 2]


def test_schema_version_refusal(tmp_path):
    """Asserts that load refuses to read files with a higher schema version."""
    state_file = tmp_path / "bad_version_state.json"

    # Write a mock JSON file with a higher schema version
    bad_data = {
        "schema_version": 2,  # Current supported is 1
        "turns": [],
        "summaries": [],
        "ledger": {"entries": []}
    }
    with open(state_file, "w") as f:
        json.dump(bad_data, f)

    with pytest.raises(ValueError) as exc_info:
        MemoryState.load(state_file)

    assert "schema version 2 is higher than supported version 1" in str(exc_info.value)


def test_atomic_save_failure_keeps_original(tmp_path, monkeypatch):
    """Asserts that a failure during save does not corrupt the existing file."""
    state_file = tmp_path / "atomic_state.json"

    # Save initial valid state
    initial_state = MemoryState(
        turns=[Turn(role="user", content="Original Content", turn_id=1, tokens=3)]
    )
    initial_state.save(state_file)
    assert state_file.exists()

    # Read original raw content for comparison
    with open(state_file, "r") as f:
        original_json = f.read()

    # Create a new state we try to save
    new_state = MemoryState(
        turns=[Turn(role="user", content="New Corrupting Content", turn_id=2, tokens=4)]
    )

    # Mock os.replace to simulate a system failure (e.g. permission error / disk full)
    def mock_replace(src, dst):
        raise OSError("Simulated system swap failure")

    monkeypatch.setattr(os, "replace", mock_replace)

    # Save should fail with the OSError
    with pytest.raises(OSError, match="Simulated system swap failure"):
        new_state.save(state_file)

    # Verify that the original file is completely untouched
    with open(state_file, "r") as f:
        current_json = f.read()
    assert current_json == original_json

    # Verify that no leftover temporary file exists
    assert not state_file.with_suffix(".tmp").exists()

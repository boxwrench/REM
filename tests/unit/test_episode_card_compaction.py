"""NPU-free replay coverage for Path D one-call episode cards."""

import json

import httpx

from rem.config import Settings
from rem.memory.compactor import compact_once, parse_episode_card
from rem.memory.tiers import MemoryState, Turn
from rem.npu_client import NpuClient


FACT = {
    "kind": "entity",
    "source_turn_id": 2,
    "subject": "launch",
    "attribute": "city",
    "value": "Seattle",
    "is_correction": False,
}
SUMMARY = "The launch planning selected Seattle."


def _turns() -> list[Turn]:
    return [
        Turn(
            role="user",
            content="Initial context",
            turn_id=1,
            tokens=10,
            session_id="s1",
            timestamp="2024-01-01",
        ),
        Turn(
            role="user",
            content="The launch city is Seattle",
            turn_id=2,
            tokens=10,
            session_id="s1",
            timestamp="2024-01-02",
        ),
        Turn(
            role="assistant",
            content="Noted",
            turn_id=3,
            tokens=10,
            session_id="s2",
            timestamp="2024-01-03",
        ),
        Turn(role="user", content="Recent", turn_id=4, tokens=10),
    ]


def _response(content: str) -> httpx.Response:
    return httpx.Response(
        200, json={"choices": [{"message": {"content": content}}]}
    )


def test_setting_is_default_off():
    assert Settings().episode_card_consolidation is False


def test_episode_card_parser_repairs_fences_and_derives_fact_text():
    raw = "```json\n" + json.dumps({"facts": [FACT], "summary": SUMMARY}) + "\n```"
    ledger, summary = parse_episode_card(
        raw, _turns()[:3], deterministic_fact_capture=False
    )
    assert summary == SUMMARY
    assert len(ledger.entries) == 1
    assert ledger.entries[0].text == "launch city: Seattle"
    assert ledger.entries[0].slot_key == "launch.city"


def test_one_call_replay_matches_legacy_ledger_and_summary(mock_npu):
    client = NpuClient()
    legacy_state = MemoryState(turns=_turns())
    episode_state = MemoryState(turns=_turns())
    mock_npu.post("/v1/chat/completions").mock(side_effect=[
        _response(json.dumps([FACT])),
        _response(SUMMARY),
        _response(json.dumps({"facts": [FACT], "summary": SUMMARY})),
    ])

    legacy = compact_once(
        legacy_state,
        client,
        Settings(
            keep_recent_turns=1,
            compact_span_turns=3,
            deterministic_fact_capture=False,
        ),
    )
    episode = compact_once(
        episode_state,
        client,
        Settings(
            keep_recent_turns=1,
            compact_span_turns=3,
            deterministic_fact_capture=False,
            episode_card_consolidation=True,
        ),
    )

    assert legacy.compacted and episode.compacted
    assert legacy.compaction_mode == "legacy_two_call"
    assert episode.compaction_mode == "episode_card"
    assert legacy.npu_calls == 2
    assert episode.npu_calls == 1
    assert legacy.npu_elapsed_s >= 0
    assert episode.npu_elapsed_s >= 0
    assert legacy.new_summary == episode.new_summary == SUMMARY
    assert legacy_state.ledger.model_dump() == episode_state.ledger.model_dump()
    legacy_summary = legacy_state.summaries[0].model_dump(exclude={"created_at"})
    episode_summary = episode_state.summaries[0].model_dump(exclude={"created_at"})
    assert legacy_summary == episode_summary


def test_bad_episode_card_is_atomic_and_reports_one_call(mock_npu):
    client = NpuClient()
    state = MemoryState(turns=_turns())
    original = state.model_dump()
    mock_npu.post("/v1/chat/completions").mock(
        return_value=_response('{"facts": [], "summary": ""}')
    )

    result = compact_once(
        state,
        client,
        Settings(
            keep_recent_turns=1,
            compact_span_turns=3,
            episode_card_consolidation=True,
        ),
    )

    assert result.compacted is False
    assert result.npu_calls == 1
    assert result.compaction_mode == "episode_card"
    assert state.model_dump() == original

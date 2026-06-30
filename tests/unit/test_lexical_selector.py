"""Query-aware native selectors and budget packing."""

from rem.memory.assembler import assemble
from rem.memory.facts_ledger import FactEntry, FactsLedger
from rem.memory.selector import (
    LexicalSelector,
    PackedLexicalSelector,
    SparseChronologicalSelector,
    SPARSE_TOP_K,
)
from rem.memory.tiers import MemoryState, SpanSummary, count_tokens


def _fact(text, turn, *, key=None, value=None, status="active"):
    return FactEntry(
        kind="entity", text=text, source_turn_id=turn, status=status,
        slot_key=key, slot_value=value,
    )


def _retrieval_state():
    entries = [
        _fact("The vehicle model is a Ford F-150 pickup truck", 2,
              key="vehicle.model", value="Ford F-150"),
    ]
    entries.extend(
        _fact((f"Recent unrelated gardening note {i} " * 8), 100 + i)
        for i in range(30)
    )
    summaries = [
        SpanSummary(covers_turn_ids=[200 + i], text=f"new cooking summary {i} " * 10,
                    tokens=30)
        for i in range(20)
    ]
    return MemoryState(summaries=summaries, ledger=FactsLedger(entries=entries))


def test_lexical_selector_retrieves_old_query_match_under_tight_budget():
    state = _retrieval_state()
    selected = LexicalSelector().select(
        state, "Which vehicle model am I working on?", 800
    )
    rendered = assemble(selected, system="", task="Which vehicle model?")
    assert "Ford F-150" in rendered
    assert count_tokens(rendered) <= 800


def test_packed_selector_enforces_budget_and_is_deterministic():
    state = _retrieval_state()
    query = "What vehicle model am I working on?"
    first = PackedLexicalSelector().select(state, query, 800)
    second = PackedLexicalSelector().select(state, query, 800)
    first_text = assemble(first, system="", task=query)
    second_text = assemble(second, system="", task=query)
    assert first_text == second_text
    assert count_tokens(first_text) <= 800


def test_sparse_selector_floors_distractors_and_does_not_fill_budget():
    """Sparse keeps the genuine match, drops pure-recency distractors, stays bounded."""
    state = _retrieval_state()
    query = "Which vehicle model am I working on?"
    big_budget = 28000
    sparse = SparseChronologicalSelector().select(state, query, big_budget)
    lexical = LexicalSelector().select(state, query, big_budget)
    s_text = assemble(sparse, system="", task=query)
    l_text = assemble(lexical, system="", task=query)
    # the real query match survives
    assert "Ford F-150" in s_text
    # top-k cap holds and the 30 gardening / 20 cooking distractors are floored out
    assert len(sparse.ledger.entries) + len(sparse.summaries) <= SPARSE_TOP_K
    # lexical fills the budget; sparse does not
    assert len(sparse.ledger.entries) < len(lexical.ledger.entries)
    assert count_tokens(s_text) < count_tokens(l_text)


def test_sparse_selector_renders_chronologically():
    """Survivors are emitted oldest -> newest, not in score order."""
    entries = [
        _fact("project status update alpha", 5, key="proj.a", value="alpha"),
        _fact("project status update gamma", 30, key="proj.c", value="gamma"),
        _fact("project status update beta", 12, key="proj.b", value="beta"),
    ]
    state = MemoryState(ledger=FactsLedger(entries=entries))
    selected = SparseChronologicalSelector().select(
        state, "project status update", 28000
    )
    turn_order = [e.source_turn_id for e in selected.ledger.entries]
    assert turn_order == sorted(turn_order)
    assert turn_order == [5, 12, 30]


def test_packed_selector_deduplicates_fact_text_and_preserves_source_reference():
    state = MemoryState(ledger=FactsLedger(entries=[
        _fact("favorite camera is Sony", 3, key="camera.model", value="Sony"),
        _fact(" favorite   camera is Sony ", 9, key="camera.model", value="Sony"),
    ]))
    selected = PackedLexicalSelector().select(state, "favorite camera", 1000)
    assert len(selected.ledger.entries) == 1
    assert selected.ledger.entries[0].source_turn_id == 9
    assert selected.ledger.entries[0].slot_key == "camera.model"


def test_temporal_query_can_render_stale_history_with_provenance():
    state = MemoryState(ledger=FactsLedger(entries=[
        _fact("previous goal was level 100", 10, key="game.goal", value="100",
              status="stale"),
        _fact("current goal is level 150", 20, key="game.goal", value="150"),
    ]))
    selected = PackedLexicalSelector().select(
        state, "What was my previous goal before I updated it?", 1200
    )
    rendered = assemble(selected, system="", task="previous goal?")
    assert "level 100" in rendered
    assert "[entity stale]" in rendered
    assert "Turn 10" in rendered


def test_current_query_filters_stale_history():
    state = MemoryState(ledger=FactsLedger(entries=[
        _fact("previous goal was level 100", 10, key="game.goal", value="100",
              status="stale"),
        _fact("current goal is level 150", 20, key="game.goal", value="150"),
    ]))
    selected = PackedLexicalSelector().select(state, "What is my current goal?", 1200)
    rendered = assemble(selected, system="", task="current goal?")
    assert "level 150" in rendered
    assert "level 100" not in rendered


def test_distinct_similar_slots_are_both_preserved():
    state = MemoryState(ledger=FactsLedger(entries=[
        _fact("camera model is Sony", 1, key="camera.model", value="Sony"),
        _fact("camera capacity is 128 GB", 2, key="camera.capacity", value="128 GB"),
    ]))
    selected = PackedLexicalSelector().select(state, "camera model and capacity", 1200)
    assert {entry.slot_key for entry in selected.ledger.entries} == {
        "camera.model", "camera.capacity"
    }

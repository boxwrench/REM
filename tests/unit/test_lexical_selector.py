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


def test_sparse_current_query_promotes_newest_fragmented_slot_family():
    state = MemoryState(
        summaries=[
            SpanSummary(
                covers_turn_ids=[179],
                text="The user celebrated seeing 27 bird species.",
                tokens=10,
            ),
        ],
        ledger=FactsLedger(entries=[
            _fact("bird species count: 27", 178, key="bird species.count", value="27"),
            _fact(
                "species count total species count: 32", 275,
                key="species count.total species count", value="32",
            ),
        ]),
    )
    selected = SparseChronologicalSelector(top_k=1, prefer_newest=True).select(
        state, "How many different species of birds have I seen?", 1200
    )
    values = [entry.slot_value for entry in selected.ledger.entries]
    assert values == ["32"]
    assert selected.summaries == []
    assert selected.ledger.entries[0].text.startswith("LATEST CURRENT OBSERVATION:")


def test_sparse_change_query_preserves_ordered_fragmented_values():
    state = MemoryState(ledger=FactsLedger(entries=[
        _fact(
            "coffee ratio is 1 tablespoon per 6 ounces of water", 13,
            key="coffee ratio.tablespoon per ounces water", value="6 ounces",
        ),
        _fact(
            "coffee brewing ratio is 1 tablespoon per 5 ounces of water", 209,
            key="coffee brewing.ratio", value="5 ounces",
        ),
    ]))
    selected = SparseChronologicalSelector(prefer_newest=True).select(
        state, "For my coffee, did I switch to more water per tablespoon, or less?",
        1200,
    )
    assert [entry.slot_value for entry in selected.ledger.entries] == [
        "6 ounces", "5 ounces"
    ]
    assert selected.ledger.entries[-1].text.startswith("UPDATE SEQUENCE:")
    assert "earlier value was 6 ounces" in selected.ledger.entries[-1].text
    assert "latest value is 5 ounces" in selected.ledger.entries[-1].text


def test_sparse_newest_preference_keeps_distinct_role_slots():
    state = MemoryState(ledger=FactsLedger(entries=[
        _fact("conference start date is May 1", 10,
              key="conference.start date", value="May 1"),
        _fact("conference end date is May 3", 11,
              key="conference.end date", value="May 3"),
        _fact("refrigerated broth lasts 4 days", 12,
              key="broth.refrigerator shelf life", value="4 days"),
        _fact("frozen broth lasts 3 months", 13,
              key="broth.freezer shelf life", value="3 months"),
    ]))
    selected = SparseChronologicalSelector(prefer_newest=True).select(
        state, "What are the conference dates and broth shelf life?", 1600
    )
    assert {entry.slot_value for entry in selected.ledger.entries} == {
        "May 1", "May 3", "4 days", "3 months"
    }


def test_sparse_newest_preference_does_not_merge_generic_or_distinct_attributes():
    state = MemoryState(ledger=FactsLedger(entries=[
        _fact("emotional symptom is anxiety", 10,
              key="emotional changes.symptom", value="anxiety"),
        _fact("physical symptom is tenderness", 11,
              key="physical changes.symptom", value="tenderness"),
        _fact("guitar service timing is Tuesday", 12,
              key="acoustic guitar service.timing", value="Tuesday"),
        _fact("guitar service location is Main St", 13,
              key="acoustic guitar service.location", value="Main St"),
    ]))
    selected = SparseChronologicalSelector(prefer_newest=True).select(
        state, "What are the emotional and physical symptoms and guitar service details?",
        1600,
    )
    assert {entry.slot_value for entry in selected.ledger.entries} == {
        "anxiety", "tenderness", "Tuesday", "Main St"
    }


def test_experimental_newest_keeps_instance_side_time_and_camera_roles_distinct():
    entries = [
        _fact("morning dosage is 5 mg", 1, key="medication.morning dosage", value="5 mg"),
        _fact("evening dosage is 10 mg", 2, key="medication.evening dosage", value="10 mg"),
        _fact("exercise 1 repetitions is 8", 3, key="exercise 1.repetitions", value="8"),
        _fact("exercise 2 repetitions is 12", 4, key="exercise 2.repetitions", value="12"),
        _fact("left arm pressure is 120", 5, key="blood pressure.left arm", value="120"),
        _fact("right arm pressure is 125", 6, key="blood pressure.right arm", value="125"),
        _fact("front camera resolution is 12 MP", 7,
              key="phone camera.front resolution", value="12 MP"),
        _fact("rear camera resolution is 48 MP", 8,
              key="phone camera.rear resolution", value="48 MP"),
    ]
    state = MemoryState(ledger=FactsLedger(entries=entries))
    selected = SparseChronologicalSelector(prefer_newest=True).select(
        state,
        "What are the medication dosages, exercise repetitions, arm blood pressure, "
        "and phone camera resolutions?",
        2400,
    )
    assert {entry.slot_value for entry in selected.ledger.entries} == {
        "5 mg", "10 mg", "8", "12", "120", "125", "12 MP", "48 MP"
    }


def test_previous_mode_annotates_penultimate_not_first_value():
    state = MemoryState(ledger=FactsLedger(entries=[
        _fact("project target was 10", 1, key="project.target", value="10"),
        _fact("project target was 20", 2, key="project.target", value="20"),
        _fact("project target is 30", 3, key="project.target", value="30"),
    ]))
    selected = SparseChronologicalSelector(prefer_newest=True).select(
        state, "What was the previous project target?", 1200
    )
    rendered = assemble(selected, system="", task="previous project target")
    assert "PREVIOUS SEQUENCE: earlier value was 20" in rendered
    assert "latest value is 30" in rendered


def test_sparse_zero_overlap_returns_no_compacted_distractors():
    state = _retrieval_state()
    selected = SparseChronologicalSelector().select(
        state, "What is the zephyr warranty?", 1200
    )
    assert selected.ledger.entries == []
    assert selected.summaries == []

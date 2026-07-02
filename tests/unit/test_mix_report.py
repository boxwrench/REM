"""mix_report labels each item's read-path miss NPU-free."""
from rem.config import Settings
from rem.memory.facts_ledger import FactEntry, FactsLedger
from rem.memory.tiers import MemoryState, Turn
from evals.battery.mix_report import label_item


def _slot_gold():
    entries = [
        FactEntry(kind="number", text="team size is 5 engineers", source_turn_id=74,
                  status="active", slot_key="team.size", slot_value="5 engineers"),
        FactEntry(kind="number", text="outing had 4 engineers", source_turn_id=12,
                  status="active", slot_key="team_members.count", slot_value="4 engineers"),
    ]
    return MemoryState(turns=[Turn(role="user", content="now?", turn_id=900, tokens=2)],
                       summaries=[], ledger=FactsLedger(entries=entries))


def test_label_retrieval_recall_when_needle_absent():
    settings = Settings(read_fit_tokens=4000)
    # gold "5 engineers" present; "9 engineers" never in memory -> absent -> recall miss
    out = label_item(_slot_gold(), "headcount?", "5 engineers",
                     ["5 engineers", "9 engineers"], settings)
    assert out["fits_budget"] is True
    assert out["gold_in_fitted"]["9 engineers"] is False
    assert out["failure_mode"] == "retrieval-recall"


def test_label_needs_answer_when_all_present_no_answerer():
    settings = Settings(read_fit_tokens=4000)
    out = label_item(_slot_gold(), "headcount?", "5 engineers",
                     ["4 engineers", "5 engineers"], settings)
    assert out["failure_mode"] == "needs-answer"
    assert out["needle_tiers"]["5 engineers"] == "slot"


def test_label_caps_current_slot_tier_when_it_exceeds_budget():
    # The revised selector contract is a hard cap: newest slots win when the
    # current-state tier itself is larger than the budget.
    entries = [FactEntry(kind="number", text=f"metric {i} is forty two units here",
                         source_turn_id=i, status="active",
                         slot_key=f"m.{i}", slot_value=str(i)) for i in range(200)]
    entries.append(FactEntry(kind="number", text="team size is 5 engineers",
                             source_turn_id=999, status="active",
                             slot_key="team.size", slot_value="5 engineers"))
    state = MemoryState(turns=[], summaries=[], ledger=FactsLedger(entries=entries))
    settings = Settings(read_fit_tokens=800)
    out = label_item(state, "headcount?", "5 engineers", ["5 engineers"], settings)
    assert out["fits_budget"] is True
    assert out["gold_in_fitted"]["5 engineers"] is True


def test_label_pass_requires_all_multipart_needles():
    # Corrected multi-part semantics (FINDINGS): a pass needs EVERY gold needle.
    settings = Settings(read_fit_tokens=4000)
    good = label_item(
        _slot_gold(), "headcount?", "5 engineers", ["4 engineers", "5 engineers"],
        settings,
        answerer=lambda ctx, q: "you started with 4 engineers and now lead 5 engineers")
    assert good["failure_mode"] == "pass"
    # The now-only answer omits 'started=4' -> temporal-structure, not a pass.
    # This is exactly the 031748ae any()-substring artifact the fix removes.
    now_only = label_item(_slot_gold(), "headcount?", "5 engineers",
                          ["4 engineers", "5 engineers"], settings,
                          answerer=lambda ctx, q: "you lead 5 engineers")
    assert now_only["failure_mode"] == "temporal-structure"
    bad = label_item(_slot_gold(), "headcount?", "5 engineers",
                     ["4 engineers", "5 engineers"], settings,
                     answerer=lambda ctx, q: "the memory does not say")
    assert bad["failure_mode"] == "temporal-structure"


def test_structure_needles_are_recorded_but_not_gating():
    # A present gold + a structure needle that is ABSENT from the slice must NOT
    # flip the label to retrieval-recall; structure needles only diagnose.
    settings = Settings(read_fit_tokens=4000)
    out = label_item(_slot_gold(), "headcount?", "5 engineers",
                     ["5 engineers"], settings,
                     structure_needles=["9 engineers"])
    assert out["failure_mode"] == "needs-answer"           # gold present -> not recall
    assert out["structure_in_fitted"]["9 engineers"] is False
    assert out["structure_tiers"]["9 engineers"] == "absent"


def test_structure_needle_present_records_tier():
    # The contrasting value sitting in the slice is recorded with its carrying tier.
    settings = Settings(read_fit_tokens=4000)
    out = label_item(_slot_gold(), "headcount?", "5 engineers",
                     ["5 engineers"], settings,
                     structure_needles=["4 engineers"])
    assert out["structure_in_fitted"]["4 engineers"] is True
    assert out["structure_tiers"]["4 engineers"] == "slot"
    # answerer that returns the contrasting (wrong) value is flagged
    out2 = label_item(_slot_gold(), "headcount?", "5 engineers",
                      ["5 engineers"], settings, structure_needles=["4 engineers"],
                      answerer=lambda ctx, q: "you lead 4 engineers")
    assert out2["answer_contains_structure"] is True

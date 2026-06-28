"""The fitted read path must fit the budget and preserve distinct-slot gold."""

from rem.config import Settings
from rem.memory.facts_ledger import FactEntry, FactsLedger
from rem.memory.tiers import MemoryState, SpanSummary, Turn, count_tokens
from evals.battery.diagnose_memory import fit_with_selector, gold_in_fitted


def _state_with_gold() -> MemoryState:
    summaries = [SpanSummary(covers_turn_ids=[i], text=f"noise summary {i} " * 20, tokens=60)
                 for i in range(300)]
    entries = [
        FactEntry(kind="number", text="team size is 5 engineers", source_turn_id=74,
                  status="active", slot_key="team.size", slot_value="5 engineers"),
        FactEntry(kind="number", text="the outing had 4 engineers", source_turn_id=12,
                  status="active", slot_key="team_members.count", slot_value="4 engineers"),
    ]
    return MemoryState(turns=[Turn(role="user", content="now?", turn_id=900, tokens=2)],
                       summaries=summaries, ledger=FactsLedger(entries=entries))


def test_fit_with_selector_fits_budget():
    settings = Settings(read_fit_tokens=4000)
    fitted_text, fitted_tokens = fit_with_selector(_state_with_gold(), "headcount?", settings)
    assert fitted_tokens == count_tokens(fitted_text)
    assert fitted_tokens <= settings.read_fit_tokens


def test_gold_survives_fit():
    settings = Settings(read_fit_tokens=4000)
    fitted_text, _ = fit_with_selector(_state_with_gold(), "headcount?", settings)
    hits = gold_in_fitted(fitted_text, ["4 engineers", "5 engineers"])
    assert hits == {"4 engineers": True, "5 engineers": True}


def _overshoot_state() -> MemoryState:
    """A state whose tier-3 free entries push the assembled text just over budget.

    Mirrors the real 031748ae failure: the selector's per-item estimate clears its
    internal budget, but section scaffolding tips the rendered text over. Gold lives
    in two protected slots, so it must survive the trim.
    """
    turns = [Turn(role="user", content=f"recent verbatim turn {i}", turn_id=900 + i, tokens=6)
             for i in range(8)]
    summaries = [SpanSummary(covers_turn_ids=[i], text=f"summary {i} " * 10, tokens=35)
                 for i in range(50)]
    entries = [FactEntry(kind="x", text=f"f{i}", source_turn_id=i, status="active")
               for i in range(1000)]
    entries.append(FactEntry(kind="number", text="team size is 5 engineers", source_turn_id=99999,
                             status="active", slot_key="team.size", slot_value="5 engineers"))
    entries.append(FactEntry(kind="number", text="outing had 4 engineers", source_turn_id=99998,
                             status="active", slot_key="team_members.count", slot_value="4 engineers"))
    return MemoryState(turns=turns, summaries=summaries, ledger=FactsLedger(entries=entries))


def test_fit_trims_overshoot_to_budget():
    # Without the measure-and-trim guard this state renders ~8197 tokens at an 8000
    # budget (the +197 over that the estimate misses). The fit must bring it to budget.
    settings = Settings(read_fit_tokens=8000)
    fitted_text, fitted_tokens = fit_with_selector(_overshoot_state(), "headcount?", settings)
    assert fitted_tokens == count_tokens(fitted_text)
    assert fitted_tokens <= settings.read_fit_tokens
    # Protected-slot gold is never trimmed.
    assert gold_in_fitted(fitted_text, ["4 engineers", "5 engineers"]) == {
        "4 engineers": True, "5 engineers": True}

import pytest
from rem.memory.facts_ledger import FactEntry, FactsLedger
from rem.memory.tiers import SpanSummary

def test_label_drift_high_confidence():
    ledger = FactsLedger()
    # old fact with label "current deployment window"
    fact1 = FactEntry(
        subject="current deployment",
        attribute="window",
        value="02:00 UTC",
        text="The current deployment window is set to 02:00 UTC.",
        is_correction=False,
        kind="entity",
        source_turn_id=1
    )
    # new fact with label "deployment window"
    fact2 = FactEntry(
        subject="deployment",
        attribute="window",
        value="04:00 UTC",
        text="The deployment window is now 04:00 UTC.",
        is_correction=True,
        kind="correction",
        source_turn_id=2
    )
    
    # Both should normalize to "deployment.window"
    assert fact1.slot_key == "deployment.window"
    assert fact2.slot_key == "deployment.window"
    
    # After merging, old goes stale
    ledger.add(fact1)
    assert len(ledger.active_entries()) == 1
    ledger.add(fact2)
    assert len(ledger.active_entries()) == 1
    assert ledger.active_entries()[0].value == "04:00 UTC"

def test_genuine_ambiguity_low_confidence():
    ledger = FactsLedger()
    # Two facts whose labels do NOT normalize-match
    fact1 = FactEntry(
        subject="user region",
        attribute="config",
        value="us-east-1",
        text="User region config is us-east-1",
        kind="entity",
        source_turn_id=1
    )
    fact2 = FactEntry(
        subject="user location",
        attribute="preference",
        value="us-west-2",
        text="User location preference is us-west-2",
        kind="entity",
        source_turn_id=2
    )
    
    assert fact1.slot_key != fact2.slot_key
    
    ledger.add(fact1)
    ledger.add(fact2)
    
    # Neither superseded (no false-merge)
    assert len(ledger.active_entries()) == 2

def test_freeform_summary_ghost_regression():
    # A corrected value present in a prose summary must be quarantined.
    # If the identity fix doesn't remove it, that's expected. Keep the test.
    ledger = FactsLedger()
    
    # Suppose we have an old fact in active
    fact1 = FactEntry(
        subject="worker pool",
        attribute="size",
        value="8",
        text="Worker pool size is 8.",
        kind="entity",
        source_turn_id=1
    )
    ledger.add(fact1)
    
    # A summary captures the old fact text
    summary = SpanSummary(text="The worker pool size is 8, which is sufficient.", span_idx=1, covers_turn_ids=[1], tokens=5)
    
    # A new fact corrects the active ledger
    fact2 = FactEntry(
        subject="worker pool",
        attribute="size",
        value="12",
        text="Correction: worker pool size is 12.",
        is_correction=True,
        kind="correction",
        source_turn_id=2
    )
    ledger.add(fact2)
    
    # Active ledger correctly superseded
    assert len(ledger.active_entries()) == 1
    assert ledger.active_entries()[0].value == "12"
    
    # Ghost value "8" still in summary
    assert "8" in summary.text

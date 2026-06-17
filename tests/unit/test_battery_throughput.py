"""Unit tests for the throughput probe aggregation math (no NPU required)."""
from evals.battery.throughput_probe import CompactionSample, ProbeResult


def test_drain_rate_is_absorbed_over_wall():
    r = ProbeResult(samples=[
        CompactionSample(absorbed_tokens=1000, wall_s=2.0, summary_tokens=200, facts=5, compacted=True),
        CompactionSample(absorbed_tokens=500, wall_s=0.5, summary_tokens=100, facts=2, compacted=True),
    ])
    # (1000 + 500) / (2.0 + 0.5) = 600 tok/s
    assert r.drain_tok_per_s() == 600.0


def test_empty_probe_does_not_divide_by_zero():
    assert ProbeResult().drain_tok_per_s() == 0.0


def test_fallback_calls_excluded_from_absorbed_but_counted():
    r = ProbeResult(samples=[
        CompactionSample(absorbed_tokens=900, wall_s=1.0, summary_tokens=150, facts=4, compacted=True),
        CompactionSample(absorbed_tokens=0, wall_s=0.5, summary_tokens=0, facts=0, compacted=False),
    ])
    s = r.summary()
    assert s["n_compaction_calls"] == 2
    assert s["n_fallback"] == 1
    assert s["n_compacted"] == 1
    assert s["tokens_absorbed_total"] == 900
    # drain still divides by total wall incl. the wasted fallback time: 900/1.5
    assert s["drain_tok_per_s"] == 600.0


def test_stability_gate_pass_and_fail():
    r = ProbeResult(samples=[
        CompactionSample(absorbed_tokens=250, wall_s=10.0, summary_tokens=50, facts=1, compacted=True),
    ])
    # drain = 25 tok/s: keeps up with 10tps, falls behind 30 and 60
    gate = r.summary()["stability_gate"]
    assert gate["slow_agent_10tps"]["keeps_up"] is True
    assert gate["typical_agent_30tps"]["keeps_up"] is False
    assert gate["fast_agent_60tps"]["keeps_up"] is False

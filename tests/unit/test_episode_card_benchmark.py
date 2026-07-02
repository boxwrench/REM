from evals.memory_methods.run_episode_card_benchmark import run


def test_episode_card_wallclock_harness_offline(tmp_path):
    output = tmp_path / "wallclock.json"
    payload = run(str(output), repetitions=2, live=False)
    assert payload["mode"] == "OFFLINE_REPLAY"
    assert payload["promote"] is False
    assert payload["checks"]["all_compacted"] is True
    assert payload["checks"]["fact_signature_equivalence"] is True
    assert payload["checks"]["call_counts"] is True
    assert len(payload["runs"]) == 4
    assert output.exists()

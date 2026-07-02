from evals.battery.decision_gate import (
    ARMS,
    FRESH_KU,
    PATH_A_PROMOTION_RULE,
    _path_a_promotion,
    merge_base_answers,
)


def _row(qid, correct):
    return {
        "qid": qid,
        "diagnostic": False,
        "majority_correct": correct,
    }


def test_path_a_candidate_is_fourth_arm_and_rule_is_pre_registered():
    assert ARMS == ("current", "sparse", "oracle", "candidate")
    assert PATH_A_PROMOTION_RULE["minimum_candidate_pass"] == 5
    assert PATH_A_PROMOTION_RULE["require_no_regression_vs_sparse"] is True


def test_path_a_promotion_requires_five_and_no_sparse_regression():
    qids = sorted(FRESH_KU)
    results = {
        "sparse": [_row(qid, index < 4) for index, qid in enumerate(qids)],
        "candidate": [_row(qid, index < 5) for index, qid in enumerate(qids)],
    }
    decision = _path_a_promotion(results, include_diagnostic=False)
    assert decision["ship_on_dev"] is True
    assert decision["candidate_pass"] == 5
    assert decision["regressions_vs_sparse"] == []

    results["candidate"][0]["majority_correct"] = False
    decision = _path_a_promotion(results, include_diagnostic=False)
    assert decision["ship_on_dev"] is False
    assert decision["regressions_vs_sparse"] == [qids[0]]

    results["candidate"] = results["candidate"][:-1]
    decision = _path_a_promotion(results, include_diagnostic=False)
    assert decision["evaluated"] is False


def test_merge_base_answers_reuses_only_missing_arms(tmp_path):
    base = tmp_path / "base.json"
    base.write_text(
        '{"items":["q0"],"run_context":{"reps":1},"results":{'
        '"sparse":[{"qid":"q0","question":"q","gold":"g",'
        '"rep_answers":["a"]}],"oracle":[{"qid":"q0","question":"q",'
        '"gold":"g","rep_answers":["a"]}]},'
        '"serving_path_overflow":{"q0":{}}}',
        encoding="utf-8",
    )
    results, overflow, reused = merge_base_answers(
        {"candidate": [{"qid": "q0", "question": "q", "gold": "g"}]},
        {},
        base,
        expected_items=["q0"],
        expected_reps=1,
        expected_run_context={"reps": 1},
    )
    assert list(results) == ["candidate", "sparse", "oracle"]
    assert overflow == {"q0": {}}
    assert reused == ["sparse", "oracle"]

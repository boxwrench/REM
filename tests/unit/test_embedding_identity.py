import math

from evals.memory_methods.embedding_identity import cosine, evaluate_pairs


def test_cosine_similarity():
    assert cosine([1, 0], [1, 0]) == 1
    assert math.isclose(cosine([1, 0], [0, 1]), 0)


def test_identity_audit_selects_zero_false_merge_threshold():
    pairs = [
        {"id": "same-a", "left": "a", "right": "b", "same_slot": True},
        {"id": "same-b", "left": "c", "right": "d", "same_slot": True},
        {"id": "different", "left": "e", "right": "f", "same_slot": False},
    ]
    vectors = {
        "a": [1.0, 0.0], "b": [0.95, 0.05],
        "c": [0.8, 0.2], "d": [0.7, 0.3],
        "e": [1.0, 0.0], "f": [0.0, 1.0],
    }
    result = evaluate_pairs(pairs, lambda texts: [vectors[text] for text in texts])
    best = result["best_zero_false_merge_threshold"]
    assert best["true_merges"] == 2
    assert best["false_merges"] == 0
    assert best["same_slot_recall"] == 1


def test_identity_audit_rejects_bad_embedder_cardinality():
    pairs = [{"id": "same", "left": "a", "right": "b", "same_slot": True}]
    try:
        evaluate_pairs(pairs, lambda texts: [[1.0]])
    except ValueError as exc:
        assert "vectors" in str(exc)
    else:
        raise AssertionError("expected cardinality validation")

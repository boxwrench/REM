"""String-first slot-key canonicalization (NPU-free)."""

import pytest

from rem.memory.canonicalize import _tokens, canonical_slot_key


def test_merges_trivial_key_variants_under_full():
    assert canonical_slot_key("team.size", "full") == canonical_slot_key(
        "team size.size", "full"
    )


def test_does_not_merge_semantically_distinct_keys_under_full():
    assert canonical_slot_key("team.size", "full") != canonical_slot_key(
        "group size.number of engineers", "full"
    )


def test_subject_granularity_merges_attributes_of_one_subject():
    assert canonical_slot_key(
        "coding exercises.frequency", "subject"
    ) == canonical_slot_key("coding exercises.duration", "subject")


def test_singularizes_and_drops_stopwords():
    assert _tokens("number of engineers") == {"engineer"}
    assert _tokens("cups") == {"cup"}
    assert canonical_slot_key("morning.cups") == canonical_slot_key("morning.cup")


def test_deterministic_token_order():
    assert canonical_slot_key("size.team") == canonical_slot_key("team.size")


def test_rejects_unknown_granularity():
    with pytest.raises(ValueError, match="granularity"):
        canonical_slot_key("team.size", "semantic")

"""Needle-matching methodology (evals.battery.needles).

Each test pins a case FINDINGS.md flagged as a substring-needle mislabel.
"""
from evals.battery.needles import (
    canonical, present, value_aware_entry, match, all_present, any_present,
)


def test_canonical_maps_spelled_cardinals_and_flattens_punctuation():
    assert canonical("Five engineers") == "5 engineers"
    assert canonical("level goal.target level: 100") == "level goal target level 100"
    assert canonical("F-150") == "f 150"
    assert canonical(None) == ""


def test_spelled_equals_digit_both_directions():
    # "5 engineers" gold must hit a "five engineers" rendering and vice versa.
    assert present("5 engineers", "the team has five engineers")
    assert present("five engineers", "team size: 5 engineers")
    assert present("two cups", "raised the limit to 2 cups")
    assert present("one cup", "from 1 cup of coffee")


def test_vehicle_model_with_hyphen_still_matches():
    assert present("F-150", "current vehicle is a Ford F-150 pickup")


def test_bare_number_needle_hits_rendered_slot_text():
    # "level 100" appears contiguously inside the flattened slot rendering.
    assert present("level 100", "facts: level goal.target level: 100 (active)")


def test_value_aware_entry_matches_number_gold_in_slot_value():
    # The documented 9bbe84a2 artifact: gold 100 lives as a slot value, and the
    # flat string "level 100" need not appear — slot-value awareness catches it.
    assert value_aware_entry("level 100", "goal.level", "100")
    assert value_aware_entry("level 100", "level goal.target level", "100")
    # number present but concept word absent from key+value -> no match
    assert not value_aware_entry("level 100", "shoe.size", "100")
    # different number -> no match
    assert not value_aware_entry("level 100", "level goal.target level", "150")
    # a needle with no number token is out of scope for the value-aware path
    assert not value_aware_entry("Mustang", "model.name", "Ford Mustang")


def test_no_false_match_on_unrelated_text():
    assert not present("5 engineers", "we hired 5 designers and 2 engineers")
    assert not present("two cups", "two bowls and a cup")


def test_all_present_requires_every_needle_for_multipart():
    # 031748ae: started=4 AND now=5 — both required for a correct answer.
    both = "I started with 5 engineers and now lead 4 engineers"  # both contiguous
    assert all_present(["4 engineers", "5 engineers"], both)
    only_now = "I lead 5 engineers."
    assert not all_present(["4 engineers", "5 engineers"], only_now)
    # any_present is the diagnostic (non-gating) rule for structure needles
    assert any_present(["4 engineers", "5 engineers"], only_now)


def test_all_present_empty_needles_is_not_a_pass():
    assert all_present([], "anything") is False


def test_match_returns_per_needle_map():
    m = match(["4 engineers", "5 engineers"], "four engineers at first, now five engineers")
    assert m == {"4 engineers": True, "5 engineers": True}

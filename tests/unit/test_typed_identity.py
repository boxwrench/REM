"""TypedIdentityMatcher: cosine fast-path + LLM judge only in the ambiguous band.

The judge is stubbed and embeddings are injected, so these are fully NPU-free and
pin the cost-bounding behavior (clear cases never call the judge).
"""
import pytest

from rem.memory.facts_ledger import FactEntry
from rem.memory.semantic_identity import (
    TypedIdentityMatcher, FullFactEmbeddingMatcher, full_fact_text,
    make_gemma_slot_judge, share_key_token,
)


def _boom_embed(texts):
    raise AssertionError("embed must not be called for a pre-filtered pair")


def _entry(key, value, turn=1):
    return FactEntry(kind="number", text=f"{key} {value}", source_turn_id=turn,
                     status="active", slot_key=key, slot_value=value)


def _embed_from(mapping):
    def embed(texts):
        return [mapping[t] for t in texts]
    return embed


def _pair(key_a, val_a, key_b, val_b, vec_a, vec_b):
    a, b = _entry(key_a, val_a), _entry(key_b, val_b, turn=2)
    mapping = {full_fact_text(a): vec_a, full_fact_text(b): vec_b}
    return a, b, _embed_from(mapping)


def test_clear_same_skips_judge_and_merges():
    # sim 1.0 >= high -> SAME by cosine alone; judge must NOT be called.
    a, b, embed = _pair("team.size", "5 engineers", "group size.count", "5",
                        [1.0, 0.0], [1.0, 0.0])
    calls = []
    m = TypedIdentityMatcher(embed, lambda x, y: calls.append(1) or True,
                             low_threshold=0.70, high_threshold=0.88)
    assert m.same_slot(a, b) is True
    assert m.judge_calls == 0 and calls == []
    assert len(m.merges) == 1


def test_clear_different_skips_judge():
    # sim 0.0 < low -> DIFFERENT by cosine alone; judge must NOT be called.
    a, b, embed = _pair("team.size", "5 engineers", "dessert.name", "apple pie",
                        [1.0, 0.0], [0.0, 1.0])
    m = TypedIdentityMatcher(embed, lambda x, y: True, 0.70, 0.88)
    assert m.same_slot(a, b) is False
    assert m.judge_calls == 0 and m.merges == []


def test_band_calls_judge_and_merges_when_same():
    # sim 0.8 in [0.70, 0.88) -> judge decides; SAME -> merge.
    a, b, embed = _pair("team.size", "5 engineers",
                        "group size.number of engineers", "5",
                        [1.0, 0.0], [0.8, 0.6])
    m = TypedIdentityMatcher(embed, lambda x, y: True, 0.70, 0.88)
    assert m.same_slot(a, b) is True
    assert m.judge_calls == 1
    assert m.judged and abs(m.judged[0]["sim"] - 0.8) < 1e-6
    assert len(m.merges) == 1


def test_band_judge_different_blocks_merge():
    # The case cosine cannot separate: likes vs comments sit in the band; judge says
    # DIFFERENT -> no merge (this is the 'similarity != identity' win).
    a, b, embed = _pair("posts.likes", "20", "posts.comments", "5",
                        [1.0, 0.0], [0.8, 0.6])
    m = TypedIdentityMatcher(embed, lambda x, y: False, 0.70, 0.88)
    assert m.same_slot(a, b) is False
    assert m.judge_calls == 1 and m.merges == []


def test_value_aware_blocks_distinct_named_values_even_if_judge_says_same():
    a, b, embed = _pair("dessert.name", "Poffertjes", "dessert.name", "apple pie",
                        [1.0, 0.0], [0.8, 0.6])
    m = TypedIdentityMatcher(embed, lambda x, y: True, 0.70, 0.88, value_aware=True)
    assert m.same_slot(a, b) is False
    assert len(m.blocked) == 1 and m.merges == []


def test_invalid_threshold_order_raises():
    with pytest.raises(ValueError):
        TypedIdentityMatcher(lambda t: [], lambda a, b: True,
                             low_threshold=0.9, high_threshold=0.8)


def test_share_key_token_prefilter_predicate():
    # genuine cross-subject fragmentation still shares a token ("size") -> eligible
    assert share_key_token(_entry("team.size", "5 engineers"),
                           _entry("group size.number of engineers", "5"))
    # same key -> eligible (value-gate/judge separates by value downstream)
    assert share_key_token(_entry("dessert.name", "Poffertjes"),
                           _entry("dessert.name", "apple pie"))
    assert share_key_token(_entry("posts.likes", "20"), _entry("posts.comments", "5"))
    # zero shared key token -> NOT eligible
    assert not share_key_token(_entry("camera.model", "Sony A7"),
                               _entry("linkedin posts.likes", "20"))
    # function words don't count as shared content
    assert not share_key_token(_entry("a.of", "x"), _entry("the.for", "y"))


def test_typed_prefilter_skips_cosine_and_judge_for_unrelated_keys():
    a, b = _entry("camera.model", "Sony A7"), _entry("linkedin posts.likes", "20")
    m = TypedIdentityMatcher(_boom_embed, lambda x, y: True,
                             prefilter=share_key_token)
    assert m.same_slot(a, b) is False      # blocked before embed/judge
    assert m.prefiltered == 1
    assert m.judge_calls == 0 and m.merges == []


def test_typed_prefilter_lets_related_band_pair_through_to_judge():
    a, b, embed = _pair("team.size", "5 engineers",
                        "group size.number of engineers", "5",
                        [1.0, 0.0], [0.8, 0.6])  # share "size" -> eligible; band sim
    m = TypedIdentityMatcher(embed, lambda x, y: True, prefilter=share_key_token)
    assert m.same_slot(a, b) is True
    assert m.prefiltered == 0 and m.judge_calls == 1 and len(m.merges) == 1


def test_embedding_matcher_prefilter_blocks_and_passes():
    # blocked pair: no shared key token -> embed never called
    blk = FullFactEmbeddingMatcher(_boom_embed, threshold=0.8, prefilter=share_key_token)
    assert blk.same_slot(_entry("camera.model", "X"),
                         _entry("dessert.name", "Y")) is False
    assert blk.prefiltered == 1 and blk.merges == []
    # eligible pair (shares team/size) at high cosine -> merges
    a, b = _entry("team.size", "5 engineers"), _entry("team size.size", "5 engineers")
    ta, tb = full_fact_text(a), full_fact_text(b)
    ok = FullFactEmbeddingMatcher(_embed_from({ta: [1.0, 0.0], tb: [1.0, 0.0]}),
                                  threshold=0.8, prefilter=share_key_token)
    assert ok.same_slot(a, b) is True
    assert ok.prefiltered == 0 and len(ok.merges) == 1


def test_make_gemma_slot_judge_parses_verdict_without_npu():
    class FakeNpu:
        def __init__(self, reply): self.reply = reply
        def chat(self, *a, **k): return self.reply

    assert make_gemma_slot_judge(FakeNpu("SAME"))("x", "y") is True
    assert make_gemma_slot_judge(FakeNpu("DIFFERENT"))("x", "y") is False
    # tolerant of a short sentence around the verdict word
    assert make_gemma_slot_judge(FakeNpu("Answer: SAME"))("x", "y") is True

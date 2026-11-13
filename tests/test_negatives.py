"""NegCLIP-style swap generation (needs the spaCy model en_core_web_sm)."""
import random

import pytest

spacy = pytest.importorskip("spacy")
try:
    spacy.load("en_core_web_sm")
except OSError:
    pytest.skip("spaCy model en_core_web_sm is not installed", allow_module_level=True)

from compbind.negatives import (ALL_KINDS, Swap, candidate_swaps, generate_candidates,  # noqa: E402
                                get_nlp, normalize_caption, one_per_kind)


def swaps_of(caption):
    return generate_candidates([caption])[0][1]


def test_adjective_swap_keeps_spacing_and_capital():
    texts = {s.text: s for s in swaps_of("A black dog chases a white cat.")}
    assert "A white dog chases a black cat." in texts
    assert texts["A white dog chases a black cat."].kind == "JJ"


def test_noun_swap_moves_the_capital_letter():
    texts = {s.text for s in swaps_of("Horse eating the grass.")}
    assert "Grass eating the horse." in texts


def test_implausible_noun_swap_is_generated():
    texts = {s.text for s in swaps_of("the horse is eating the grass")}
    assert "the grass is eating the horse" in texts


def test_only_same_tag_words_are_swapped():
    for swap in swaps_of("a young man riding a brown horse on a sandy beach"):
        assert swap.kind in ALL_KINDS
        assert swap.text != "a young man riding a brown horse on a sandy beach"


def test_coordination_is_flagged_symmetric():
    swaps = [s for s in swaps_of("a dog and a cat on a sofa") if s.text == "a cat and a dog on a sofa"]
    assert swaps and swaps[0].symmetric


def test_subject_object_swap_is_not_symmetric():
    swaps = [s for s in swaps_of("the horse is eating the grass") if s.text == "the grass is eating the horse"]
    assert swaps and not swaps[0].symmetric


def test_no_duplicates_and_never_the_original():
    caption = "a red car and a red bus near a red house"
    texts = [s.text.lower() for s in swaps_of(caption)]
    assert len(texts) == len(set(texts))
    assert caption not in texts


def test_noun_phrase_swap_needs_three_tokens():
    doc = get_nlp()("a big dog chases the small cat")
    kinds = {s.kind for s in candidate_swaps(doc)}
    assert "NP" in kinds                                       # "a big dog" <-> "the small cat"
    doc = get_nlp()("a dog chases the cat")
    assert "NP" not in {s.kind for s in candidate_swaps(doc)}  # two-token phrases are left to NN swaps


def test_one_per_kind():
    swaps = [Swap("NN", (), (), "a", False), Swap("NN", (), (), "b", False), Swap("JJ", (), (), "c", False)]
    chosen = one_per_kind(swaps, random.Random(0))
    assert sorted(s.kind for s in chosen) == ["JJ", "NN"]


def test_normalize_caption():
    assert normalize_caption("  a   dog \n runs ") == "a dog runs"

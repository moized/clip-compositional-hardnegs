"""Hard negative captions made by swapping two elements of a caption (NegCLIP style).

NegCLIP (Yuksekgonul et al., ICLR 2023, App. C.1) builds negatives by swapping two
elements of the same type with spaCy. The authors did not release the generation
code, only the resulting training file (`temp_data/train_neg_clip.tsv` in their
repository). The rules below were inferred from that file:

* single-word swaps happen only between two words with the same fine-grained tag,
  and only for the tags NN, NNS, JJ, RB and VBG;
* noun-phrase swaps happen between two noun chunks of at least three tokens each;
* each caption gets at most one negative per type (so at most six).

`scripts/03_build_training_data.py --check-against-negclip` measures how many of
the authors' released negatives this module reproduces.

Unlike the released file, a swap here keeps the caption's original spacing and
sentence-initial capital letter, so that a caption and its negative differ only
by the swap. That matters when a language model compares their fluency.
"""
from __future__ import annotations

import itertools
import random
from dataclasses import dataclass
from typing import Iterable, Iterator, Sequence

# Fine-grained Penn Treebank tags that NegCLIP swaps (inferred, see module docstring).
SWAP_TAGS: tuple = ("NN", "NNS", "JJ", "RB", "VBG")
NOUN_PHRASE = "NP"
ALL_KINDS: tuple = SWAP_TAGS + (NOUN_PHRASE,)
MIN_CHUNK_TOKENS = 3

# Words that link two phrases without giving them different roles. Swapping the two
# phrases around one of these usually keeps the meaning ("a dog next to a cat" /
# "a cat next to a dog"), so the swapped caption is not a real negative.
SYMMETRIC_CONNECTORS = frozenset({
    "and", "or", ",", ", and", ", or", "&", "and a", "and an", "and the",
    "with", "along with", "together with", "alongside",
    "next to", "beside", "besides", "near", "nearby", "by", "close to", "adjacent to",
    "across from", "opposite",
})

_nlp = None


def get_nlp():
    """The spaCy pipeline used everywhere (tagger + parser; NER and lemmatiser off)."""
    global _nlp
    if _nlp is None:
        import spacy

        try:
            _nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
        except OSError as err:  # model not installed
            raise RuntimeError(
                "spaCy model missing. Run:  python -m spacy download en_core_web_sm"
            ) from err
    return _nlp


def normalize_caption(caption: str) -> str:
    """Collapse runs of whitespace and strip the ends."""
    return " ".join(str(caption).split())


@dataclass(frozen=True)
class Swap:
    """One candidate negative caption."""

    kind: str          # "NN", "NNS", "JJ", "RB", "VBG" or "NP"
    span_a: tuple      # (start, end) token indices of the first element
    span_b: tuple      # (start, end) token indices of the second element
    text: str          # the caption after the swap
    symmetric: bool    # True if the swap probably keeps the meaning (see is_symmetric)


def _swapped_text(doc, span_a: tuple, span_b: tuple) -> str:
    """Exchange two non-overlapping token spans, keeping spacing and sentence case."""
    (a0, a1), (b0, b1) = span_a, span_b
    text_a, text_b = doc[a0:a1].text, doc[b0:b1].text
    space_a, space_b = doc[a1 - 1].whitespace_, doc[b1 - 1].whitespace_
    starts_sentence = a0 == 0 and text_a[:1].isupper()
    is_name = doc[a0].tag_ in ("NNP", "NNPS") or doc[a0].text == "I"
    if starts_sentence and not is_name:
        text_b = text_b[:1].upper() + text_b[1:]
        text_a = text_a[:1].lower() + text_a[1:]
    return (
        doc[:a0].text_with_ws + text_b + space_a
        + doc[a1:b0].text_with_ws + text_a + space_b
        + doc[b1:].text_with_ws
    )


def _conj_root(token):
    """Follow `conj` links up to the first member of a coordination."""
    seen = 0
    while token.dep_ == "conj" and token.head.i != token.i and seen < 50:
        token = token.head
        seen += 1
    return token


def _chunk_of(doc, token):
    for chunk in doc.noun_chunks:
        if chunk.start <= token.i < chunk.end:
            return chunk
    return doc[token.i:token.i + 1]


def is_symmetric(doc, span_a: tuple, span_b: tuple) -> bool:
    """Heuristic: does swapping these two elements probably keep the meaning?

    True when the two elements are
    (1) members of one coordination ("pots and pans", "a sink, tub and toilet"),
    (2) two modifiers of the same word ("a big red ball"), or
    (3) two phrases joined only by a symmetric connector ("a dog next to a cat").

    This is a heuristic on top of an automatic parse, so it misses cases and has
    false alarms. `scripts/03_build_training_data.py` exports a sample for a manual
    audit; report the audited error rate, do not assume the flag is right.
    """
    root_a, root_b = doc[span_a[0]:span_a[1]].root, doc[span_b[0]:span_b[1]].root
    # (1) same coordination
    if (root_a.dep_ == "conj" or root_b.dep_ == "conj") and _conj_root(root_a).i == _conj_root(root_b).i:
        return True
    # (2) two modifiers of the same head word
    modifier = ("amod", "advmod", "compound", "nummod")
    if root_a.dep_ in modifier and root_b.dep_ in modifier and root_a.head.i == root_b.head.i:
        return True
    # (3) only a symmetric connector stands between the two phrases
    if span_a[1] - span_a[0] == 1 and root_a.tag_ in ("NN", "NNS"):
        phrase_a, phrase_b = _chunk_of(doc, root_a), _chunk_of(doc, root_b)
        end_a, start_b = phrase_a.end, phrase_b.start
    else:
        end_a, start_b = span_a[1], span_b[0]
    if end_a <= start_b:
        between = doc[end_a:start_b].text.lower().strip()
        if between in SYMMETRIC_CONNECTORS:
            return True
    return False


def candidate_swaps(doc) -> list:
    """Every swap the NegCLIP rules allow for one parsed caption (duplicates removed)."""
    original = doc.text.lower()
    seen, swaps = {original}, []

    def add(kind: str, span_a: tuple, span_b: tuple) -> None:
        if doc[span_a[0]:span_a[1]].text.lower() == doc[span_b[0]:span_b[1]].text.lower():
            return  # swapping identical words changes nothing
        text = _swapped_text(doc, span_a, span_b)
        key = text.lower()
        if key in seen:
            return
        seen.add(key)
        swaps.append(Swap(kind, span_a, span_b, text, is_symmetric(doc, span_a, span_b)))

    for tag in SWAP_TAGS:
        positions = [token.i for token in doc if token.tag_ == tag]
        for i, j in itertools.combinations(positions, 2):
            add(tag, (i, i + 1), (j, j + 1))
    chunks = [chunk for chunk in doc.noun_chunks if len(chunk) >= MIN_CHUNK_TOKENS]
    for chunk_a, chunk_b in itertools.combinations(chunks, 2):
        if chunk_a.end <= chunk_b.start:
            add(NOUN_PHRASE, (chunk_a.start, chunk_a.end), (chunk_b.start, chunk_b.end))
    return swaps


def one_per_kind(swaps: Sequence, rng: random.Random) -> list:
    """NegCLIP's rule: keep one randomly chosen swap for each type present."""
    by_kind: dict = {}
    for swap in swaps:
        by_kind.setdefault(swap.kind, []).append(swap)
    return [rng.choice(by_kind[kind]) for kind in ALL_KINDS if kind in by_kind]


def parse_captions(captions: Iterable, batch_size: int = 1000, n_process: int = 1) -> Iterator:
    """Parse captions with spaCy (after whitespace normalisation)."""
    nlp = get_nlp()
    texts = (normalize_caption(c) for c in captions)
    return nlp.pipe(texts, batch_size=batch_size, n_process=n_process)


def generate_candidates(captions: Sequence, batch_size: int = 1000, n_process: int = 1) -> list:
    """For each caption return (normalised caption, list of candidate swaps)."""
    out = []
    for doc in parse_captions(captions, batch_size=batch_size, n_process=n_process):
        out.append((doc.text, candidate_swaps(doc)))
    return out

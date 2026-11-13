"""Word-order perturbations for the COCO-Order task.

Adapted from `dataset_zoo/perturbations.py` and `dataset_zoo/retrieval.py` in
https://github.com/mertyg/vision-language-models-are-bows (MIT licence,
Copyright (c) 2023 Mert Yuksekgonul), which in turn follows O'Connor & Andreas
(2021). The four perturbations and `pre_caption` behave as in the original; the
only change is that the random numbers come from generators owned by the object,
so a seed fully determines the perturbed captions.
"""
from __future__ import annotations

import random
import re

import numpy as np

NOUN_TAGS = ("NN", "NNS", "NNP", "NNPS")
ADJECTIVE_TAGS = ("JJ", "JJR", "JJS")


def pre_caption(caption: str, max_words: int = 50) -> str:
    """Lower-case, drop some punctuation, squeeze spaces, truncate (BLIP convention)."""
    caption = re.sub(r"([.!\"()*#:;~])", " ", caption.lower())
    caption = re.sub(r"\s{2,}", " ", caption)
    caption = caption.rstrip("\n").strip(" ")
    words = caption.split(" ")
    if len(words) > max_words:
        caption = " ".join(words[:max_words])
    return caption


def ensure_nltk_tokenizer() -> None:
    """Make sure NLTK's sentence tokenizer data is available (needed by word_tokenize)."""
    import nltk

    try:
        nltk.word_tokenize("a test.")
        return
    except LookupError:
        pass
    for resource in ("punkt", "punkt_tab"):
        try:
            nltk.download(resource, quiet=True)
        except Exception:
            pass
    nltk.word_tokenize("a test.")  # raises a clear LookupError if still missing


class TextShuffler:
    """The four ARO word-order perturbations."""

    def __init__(self, seed: int = 0):
        from .negatives import get_nlp

        self.nlp = get_nlp()
        self.np_rng = np.random.RandomState(seed)
        self.py_rng = random.Random(seed)
        ensure_nltk_tokenizer()

    def _shuffle_positions(self, tokens: list, positions: list) -> None:
        values = [tokens[i] for i in positions]
        for i, j in zip(positions, self.np_rng.permutation(len(positions))):
            tokens[i] = values[j]

    def shuffle_nouns_and_adj(self, text: str) -> str:
        """Shuffle the nouns among themselves and the adjectives among themselves."""
        doc = self.nlp(text)
        tokens = [token.text for token in doc]
        self._shuffle_positions(tokens, [i for i, t in enumerate(doc) if t.tag_ in NOUN_TAGS])
        self._shuffle_positions(tokens, [i for i, t in enumerate(doc) if t.tag_ in ADJECTIVE_TAGS])
        return " ".join(tokens)

    def shuffle_allbut_nouns_and_adj(self, text: str) -> str:
        """Shuffle every word that is neither a noun nor an adjective."""
        doc = self.nlp(text)
        tokens = [token.text for token in doc]
        keep = NOUN_TAGS + ADJECTIVE_TAGS
        self._shuffle_positions(tokens, [i for i, t in enumerate(doc) if t.tag_ not in keep])
        return " ".join(tokens)

    @staticmethod
    def _trigrams(tokens: list) -> list:
        return [tokens[i:i + 3] for i in range(0, len(tokens), 3)]

    def shuffle_within_trigrams(self, text: str) -> str:
        """Shuffle the words inside each consecutive group of three."""
        import nltk

        trigrams = self._trigrams(nltk.word_tokenize(text))
        for trigram in trigrams:
            self.py_rng.shuffle(trigram)
        return " ".join(" ".join(trigram) for trigram in trigrams)

    def shuffle_trigrams(self, text: str) -> str:
        """Shuffle the order of the groups of three, keeping each group intact."""
        import nltk

        trigrams = self._trigrams(nltk.word_tokenize(text))
        self.py_rng.shuffle(trigrams)
        return " ".join(" ".join(trigram) for trigram in trigrams)

    def all_perturbations(self, text: str) -> list:
        """The four perturbed versions, in the order used by ARO's COCO_Order."""
        return [
            self.shuffle_nouns_and_adj(text),
            self.shuffle_allbut_nouns_and_adj(text),
            self.shuffle_within_trigrams(text),
            self.shuffle_trigrams(text),
        ]

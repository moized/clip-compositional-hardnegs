"""Text-only ("blind") scorers: they rate a caption without ever seeing the image.

Two scorers are provided:

* `Gpt2Scorer` - log-probability of the caption under GPT-2. Used as the fluency
  measure for training negatives and as a blind baseline on the benchmarks.
* `GrammarScorer` - probability that the sentence is grammatically acceptable,
  from a DistilBERT model fine-tuned on CoLA. This is the "Grammar model" of
  SugarCrepe (Hsieh et al., NeurIPS 2023; `text_model_eval.py` in their repo),
  re-implemented with batching.

A blind baseline picks, for each test item, the caption with the highest score.
If it beats chance, the benchmark can be partly solved without looking at the image.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch


def _batches_by_length(texts: Sequence, batch_size: int):
    """Yield index batches, longest texts first, so padding is small."""
    order = sorted(range(len(texts)), key=lambda i: -len(texts[i]))
    for start in range(0, len(order), batch_size):
        yield order[start:start + batch_size]


class Gpt2Scorer:
    """Mean and total log-probability (in nats) of each text under GPT-2."""

    name = "gpt2"

    def __init__(self, device: str = "cpu", model_name: str = "gpt2", batch_size: int = 128):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = device
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device).eval()

    @torch.no_grad()
    def score(self, texts: Sequence, progress: bool = False) -> dict:
        """Return {'mean': array, 'sum': array, 'n_tokens': array}, one value per text.

        A beginning-of-text token is placed before every text so that the first
        real token also gets a probability.
        """
        n = len(texts)
        total = np.zeros(n, dtype=np.float64)
        count = np.zeros(n, dtype=np.int64)
        batches = list(_batches_by_length(texts, self.batch_size))
        if progress:
            from tqdm import tqdm

            batches = tqdm(batches, desc="GPT-2 scoring")
        bos = self.tokenizer.bos_token_id
        for idx in batches:
            enc = self.tokenizer([texts[i] for i in idx], return_tensors="pt", padding=True)
            ids = enc["input_ids"].to(self.device)
            mask = enc["attention_mask"].to(self.device)
            start = torch.full((ids.shape[0], 1), bos, dtype=ids.dtype, device=self.device)
            inputs = torch.cat([start, ids], dim=1)
            input_mask = torch.cat([torch.ones_like(start), mask], dim=1)
            logits = self.model(input_ids=inputs, attention_mask=input_mask).logits[:, :-1]
            log_probs = torch.log_softmax(logits.float(), dim=-1)
            token_lp = log_probs.gather(-1, ids.unsqueeze(-1)).squeeze(-1) * mask
            total[idx] = token_lp.sum(dim=1).double().cpu().numpy()
            count[idx] = mask.sum(dim=1).cpu().numpy()
        count = np.maximum(count, 1)
        return {"mean": total / count, "sum": total, "n_tokens": count}


class GrammarScorer:
    """P(sentence is grammatically acceptable) from a CoLA classifier."""

    name = "grammar"

    def __init__(self, device: str = "cpu",
                 model_name: str = "textattack/distilbert-base-uncased-CoLA",
                 batch_size: int = 256):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.device = device
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device).eval()

    @torch.no_grad()
    def score(self, texts: Sequence, progress: bool = False) -> dict:
        """Return {'prob': array} with the probability of class 1 ("acceptable")."""
        prob = np.zeros(len(texts), dtype=np.float64)
        batches = list(_batches_by_length(texts, self.batch_size))
        if progress:
            from tqdm import tqdm

            batches = tqdm(batches, desc="Grammar scoring")
        for idx in batches:
            enc = self.tokenizer([texts[i] for i in idx], return_tensors="pt",
                                 padding=True, truncation=True, max_length=128).to(self.device)
            logits = self.model(**enc).logits.float()
            prob[idx] = torch.softmax(logits, dim=-1)[:, 1].double().cpu().numpy()
        return {"prob": prob}


def score_unique(scorer, texts: Sequence, key: str, progress: bool = False) -> np.ndarray:
    """Score each distinct text once and return one value per input text."""
    unique = list(dict.fromkeys(texts))
    values = scorer.score(unique, progress=progress)[key]
    lookup = dict(zip(unique, values))
    return np.array([lookup[t] for t in texts], dtype=np.float64)

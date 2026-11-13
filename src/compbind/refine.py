"""Fluency-matched hard negatives.

NegCLIP's swapped negatives are usually less fluent than the true caption, so a
language model can tell them apart without the image. SugarCrepe (Hsieh et al.,
NeurIPS 2023, Algorithm 1; `adversarial_refine.py` in their repository, MIT
licence) removes that shortcut from an *evaluation* set by subsampling it until
the fluency gap between true and false captions is distributed symmetrically
around zero. This module applies the same idea to *training* negatives.

Fluency gap of a (caption, negative) pair:
    gap = mean GPT-2 log-probability of the caption - that of the negative.
gap > 0 means the negative is less fluent, so a text-only model would pick the
true caption; gap < 0 means the negative reads better than the true caption.

`build_matched_pairs` selects one (caption, negative) pair per image such that,
in every gap-size bin, there are exactly as many pairs with gap > 0 as with
gap < 0. A text-only model that picks the more fluent caption is then right on
exactly half of the selected pairs.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd


def gap_bins(gap: np.ndarray, bin_width: float) -> np.ndarray:
    """Bin index of |gap|."""
    return np.floor(np.abs(np.asarray(gap, dtype=np.float64)) / bin_width).astype(np.int64)


def blind_accuracy(gap: np.ndarray) -> float:
    """Accuracy of 'pick the more fluent caption' on these pairs (ties count as wrong)."""
    gap = np.asarray(gap, dtype=np.float64)
    return float((gap > 0).mean()) if len(gap) else float("nan")


def build_matched_pairs(candidates: pd.DataFrame, bin_width: float = 0.02, max_abs_gap: float = 1.0,
                        seed: int = 0) -> pd.DataFrame:
    """Select one fluency-matched (caption, negative) pair per image.

    `candidates` needs the columns image_id, gap and symmetric (one row per candidate
    negative; other columns are carried through). Symmetric swaps, exact ties and
    gaps larger than `max_abs_gap` are never selected.

    Steps:
    1. Negatives that read better than their caption (gap < 0) are rare, so every
       image that has one contributes one such pair, chosen at random.
    2. For each of those pairs, a pair with gap > 0 in the same |gap| bin is drawn
       from an image not used yet.
    3. If a bin runs out of unused images, surplus gap < 0 pairs of that bin are
       dropped, so both sides stay equal in every bin.

    Returns the selected rows with the extra columns `gap_bin` and `side`
    ("negative_more_fluent" or "caption_more_fluent").
    """
    rng = np.random.RandomState(seed)
    pool = candidates[(~candidates["symmetric"].astype(bool))
                      & (candidates["gap"] != 0)
                      & (candidates["gap"].abs() <= max_abs_gap)].copy()
    pool["gap_bin"] = gap_bins(pool["gap"].values, bin_width)
    pool = pool.iloc[rng.permutation(len(pool))]                     # random order = random choice

    below = pool[pool["gap"] < 0].drop_duplicates("image_id")        # step 1: one per image
    used = set(below["image_id"])
    above = pool[(pool["gap"] > 0) & (~pool["image_id"].isin(used))]

    above_by_bin = defaultdict(list)
    for index, image_id, gap_bin in zip(above.index, above["image_id"].values, above["gap_bin"].values):
        above_by_bin[gap_bin].append((index, image_id))

    keep_below, keep_above = [], []
    below_by_bin = below.groupby("gap_bin").groups                   # bin -> row labels
    for gap_bin in sorted(below_by_bin):                             # scarce small gaps first
        wanted = list(below_by_bin[gap_bin])
        matched = []
        for index, image_id in above_by_bin.get(gap_bin, []):
            if len(matched) == len(wanted):
                break
            if image_id not in used:
                used.add(image_id)
                matched.append(index)
        keep_above.extend(matched)
        keep_below.extend(wanted[:len(matched)])                     # step 3: drop the surplus

    selected = pd.concat([pool.loc[keep_below], pool.loc[keep_above]])
    selected["side"] = np.where(selected["gap"] < 0, "negative_more_fluent", "caption_more_fluent")
    return selected.sort_values("image_id").reset_index(drop=True)


def describe_pool(gap: np.ndarray, kinds=None, symmetric=None) -> dict:
    """Summary statistics of a set of (caption, negative) pairs."""
    gap = np.asarray(gap, dtype=np.float64)
    out = {
        "n_pairs": int(len(gap)),
        "blind_accuracy_gpt2": blind_accuracy(gap),
        "gap_mean": float(gap.mean()) if len(gap) else float("nan"),
        "gap_median": float(np.median(gap)) if len(gap) else float("nan"),
        "gap_quantiles_05_25_75_95": [float(q) for q in np.quantile(gap, [0.05, 0.25, 0.75, 0.95])]
        if len(gap) else [],
    }
    if kinds is not None:
        counts = pd.Series(list(kinds)).value_counts(normalize=True)
        out["kind_share"] = {k: float(v) for k, v in counts.items()}
    if symmetric is not None:
        out["symmetric_share"] = float(np.mean(np.asarray(symmetric, dtype=bool))) if len(gap) else float("nan")
    return out

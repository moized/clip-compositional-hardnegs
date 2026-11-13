"""Uncertainty estimates for accuracies.

Test items that share an image are not independent (VG-Relation has about four
items per image, COCO-Order has five captions per image). All intervals here are
therefore *cluster* bootstraps: whole images are resampled with replacement, and
every item of a drawn image comes along.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd


def _cluster_weights(n_clusters: int, n_boot: int, seed: int) -> np.ndarray:
    """[n_boot, n_clusters] counts: how often each cluster is drawn in each replicate."""
    rng = np.random.RandomState(seed)
    probabilities = np.full(n_clusters, 1.0 / n_clusters)
    return rng.multinomial(n_clusters, probabilities, size=n_boot).astype(np.float64)


def _codes(values: Sequence) -> tuple:
    codes, uniques = pd.factorize(np.asarray(values))
    return codes, len(uniques)


def _interval(samples: np.ndarray, alpha: float) -> tuple:
    low, high = np.nanquantile(samples, [alpha / 2, 1 - alpha / 2])
    return float(low), float(high)


def _bootstrap_accuracy(correct: np.ndarray, clusters: Optional[Sequence], n_boot: int, seed: int) -> np.ndarray:
    correct = np.asarray(correct, dtype=np.float64)
    if clusters is None:
        clusters = np.arange(len(correct))
    codes, n_clusters = _codes(clusters)
    sums = np.bincount(codes, weights=correct, minlength=n_clusters)
    counts = np.bincount(codes, minlength=n_clusters).astype(np.float64)
    weights = _cluster_weights(n_clusters, n_boot, seed)
    return (weights @ sums) / (weights @ counts)


_EMPTY = {"value": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": 0}


def accuracy_ci(correct: Sequence, clusters: Optional[Sequence] = None, n_boot: int = 2000,
                seed: int = 0, alpha: float = 0.05) -> dict:
    """Accuracy with a (1 - alpha) percentile bootstrap interval."""
    correct = np.asarray(correct, dtype=np.float64)
    if len(correct) == 0:
        return dict(_EMPTY)
    low, high = _interval(_bootstrap_accuracy(correct, clusters, n_boot, seed), alpha)
    return {"value": float(correct.mean()), "ci_low": low, "ci_high": high, "n": int(len(correct))}


def _group_matrices(correct: np.ndarray, groups: Sequence, clusters: Sequence) -> tuple:
    group_codes, n_groups = _codes(groups)
    cluster_codes, n_clusters = _codes(clusters)
    sums = np.zeros((n_groups, n_clusters))
    counts = np.zeros((n_groups, n_clusters))
    np.add.at(sums, (group_codes, cluster_codes), correct)
    np.add.at(counts, (group_codes, cluster_codes), 1.0)
    return sums, counts, n_clusters


def _bootstrap_macro(correct: np.ndarray, groups: Sequence, clusters: Sequence,
                     n_boot: int, seed: int) -> np.ndarray:
    sums, counts, n_clusters = _group_matrices(np.asarray(correct, dtype=np.float64), groups, clusters)
    weights = _cluster_weights(n_clusters, n_boot, seed)
    numerator, denominator = weights @ sums.T, weights @ counts.T      # [n_boot, n_groups]
    with np.errstate(divide="ignore", invalid="ignore"):
        per_group = numerator / denominator                           # NaN if a group was not drawn
    return np.nanmean(per_group, axis=1)


def macro_accuracy_ci(correct: Sequence, groups: Sequence, clusters: Sequence, n_boot: int = 2000,
                      seed: int = 0, alpha: float = 0.05) -> dict:
    """Mean of per-group accuracies with a cluster-bootstrap interval."""
    if len(correct) == 0:
        return dict(_EMPTY, n_groups=0)
    frame = pd.DataFrame({"correct": np.asarray(correct, dtype=np.float64), "group": np.asarray(groups)})
    value = float(frame.groupby("group")["correct"].mean().mean())
    low, high = _interval(_bootstrap_macro(frame["correct"].values, groups, clusters, n_boot, seed), alpha)
    return {"value": value, "ci_low": low, "ci_high": high, "n": int(len(frame)),
            "n_groups": int(frame["group"].nunique())}


def paired_difference(correct_a: Sequence, correct_b: Sequence, clusters: Optional[Sequence] = None,
                      groups: Optional[Sequence] = None, n_boot: int = 2000, seed: int = 0,
                      alpha: float = 0.05) -> dict:
    """Accuracy of A minus accuracy of B on the same items, with a bootstrap interval.

    The same resampled images are used for both models, so the interval reflects
    the difference itself, not the two separate uncertainties. With `groups`, the
    difference is between macro accuracies. `p_value` is two-sided.
    """
    a = np.asarray(correct_a, dtype=np.float64)
    b = np.asarray(correct_b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("Both models must be scored on the same items.")
    if clusters is None:
        clusters = np.arange(len(a))
    if groups is None:
        samples = (_bootstrap_accuracy(a, clusters, n_boot, seed)
                   - _bootstrap_accuracy(b, clusters, n_boot, seed))
        value = float(a.mean() - b.mean())
    else:
        samples = (_bootstrap_macro(a, groups, clusters, n_boot, seed)
                   - _bootstrap_macro(b, groups, clusters, n_boot, seed))
        frame = pd.DataFrame({"a": a, "b": b, "group": np.asarray(groups)})
        means = frame.groupby("group")[["a", "b"]].mean()
        value = float((means["a"] - means["b"]).mean())
    low, high = _interval(samples, alpha)
    p_value = float(min(1.0, 2 * min((samples <= 0).mean(), (samples >= 0).mean())))
    return {"difference": value, "ci_low": low, "ci_high": high, "p_value": p_value, "n": int(len(a))}


def mean_std(values: Sequence) -> dict:
    """Mean and sample standard deviation across seeds."""
    values = np.asarray(list(values), dtype=np.float64)
    std = float(values.std(ddof=1)) if len(values) > 1 else float("nan")
    return {"mean": float(values.mean()), "std": std, "n": int(len(values))}

"""Cluster-bootstrap intervals and paired differences."""
import numpy as np
import pytest

from compbind.stats import accuracy_ci, macro_accuracy_ci, mean_std, paired_difference


def test_accuracy_interval_contains_the_value():
    rng = np.random.RandomState(0)
    correct = (rng.rand(500) < 0.7).astype(int)
    out = accuracy_ci(correct, n_boot=500)
    assert out["value"] == pytest.approx(correct.mean())
    assert out["ci_low"] < out["value"] < out["ci_high"]
    assert out["n"] == 500


def test_clustered_items_give_a_wider_interval():
    rng = np.random.RandomState(0)
    per_image = (rng.rand(100) < 0.5).astype(int)
    correct = np.repeat(per_image, 10)                 # 10 perfectly correlated items per image
    clusters = np.repeat(np.arange(100), 10)
    independent = accuracy_ci(correct, n_boot=1000)
    clustered = accuracy_ci(correct, clusters, n_boot=1000)
    width = lambda d: d["ci_high"] - d["ci_low"]  # noqa: E731
    assert width(clustered) > 2 * width(independent)


def test_macro_accuracy_is_the_mean_of_group_means():
    correct = np.array([1, 1, 1, 1, 0, 0])
    groups = np.array(["a", "a", "a", "a", "b", "b"])
    out = macro_accuracy_ci(correct, groups, np.arange(6), n_boot=200)
    assert out["value"] == pytest.approx(0.5)          # (1.0 + 0.0) / 2, not 4/6
    assert out["n_groups"] == 2


def test_paired_difference_of_identical_models_is_zero():
    correct = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 1])
    out = paired_difference(correct, correct, n_boot=300)
    assert out["difference"] == 0 and out["ci_low"] == 0 and out["ci_high"] == 0


def test_paired_difference_detects_a_consistent_gain():
    rng = np.random.RandomState(1)
    base = (rng.rand(2000) < 0.5).astype(int)
    better = np.where(rng.rand(2000) < 0.2, 1, base)   # fixes 20% of the items, never breaks one
    out = paired_difference(better, base, n_boot=500)
    assert out["difference"] > 0.05 and out["ci_low"] > 0 and out["p_value"] < 0.01


def test_paired_difference_with_groups_is_macro():
    a = np.array([1, 1, 1, 1, 1, 1])
    b = np.array([1, 1, 1, 1, 0, 0])
    groups = np.array(["x", "x", "x", "x", "y", "y"])
    out = paired_difference(a, b, clusters=np.arange(6), groups=groups, n_boot=200)
    assert out["difference"] == pytest.approx(0.5)     # group x: 0, group y: +1


def test_empty_input_does_not_crash():
    assert np.isnan(accuracy_ci([])["value"])
    assert macro_accuracy_ci([], [], [])["n_groups"] == 0


def test_mean_std():
    out = mean_std([0.5, 0.7])
    assert out["mean"] == pytest.approx(0.6) and out["std"] == pytest.approx(np.std([0.5, 0.7], ddof=1))

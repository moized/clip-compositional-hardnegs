"""Fluency-matched selection: balanced in every gap bin, one pair per image."""
import numpy as np
import pandas as pd

from compbind import refine
from compbind.traindata import split_balanced


def synthetic_candidates(n_images=400, seed=0):
    rng = np.random.RandomState(seed)
    rows = []
    for image_id in range(n_images):
        for caption in range(3):
            for _ in range(4):
                rows.append({"image_id": image_id, "caption_index": image_id * 3 + caption,
                             "gap": rng.normal(0.6, 0.5), "symmetric": rng.rand() < 0.15,
                             "kind": "NN", "text": f"negative {len(rows)}"})
    return pd.DataFrame(rows)


def test_matched_pairs_are_balanced_in_every_bin():
    matched = refine.build_matched_pairs(synthetic_candidates(), bin_width=0.05, max_abs_gap=1.0, seed=0)
    assert len(matched) > 50
    assert matched["image_id"].is_unique                       # one pair per image
    assert not matched["symmetric"].any()                      # symmetric swaps are never selected
    assert (matched["gap"].abs() <= 1.0).all() and (matched["gap"] != 0).all()
    for _, rows in matched.groupby("gap_bin"):
        assert (rows["gap"] < 0).sum() == (rows["gap"] > 0).sum()
    assert refine.blind_accuracy(matched["gap"].values) == 0.5


def test_selection_is_deterministic_for_a_seed():
    candidates = synthetic_candidates()
    first = refine.build_matched_pairs(candidates, seed=3)
    second = refine.build_matched_pairs(candidates, seed=3)
    assert first["text"].tolist() == second["text"].tolist()


def test_blind_accuracy_counts_ties_as_wrong():
    assert refine.blind_accuracy(np.array([1.0, -1.0, 0.0, 2.0])) == 0.5


def test_split_keeps_the_balance():
    matched = refine.build_matched_pairs(synthetic_candidates(800), bin_width=0.05, seed=0)
    split = split_balanced(matched, n_val=40, max_train=None, seed=0)
    assert set(split.unique()) <= {"train", "val", "unused"}
    for name in ("train", "val"):
        part = matched[split.values == name]
        assert len(part) > 0
        assert (part["gap"] < 0).sum() == (part["gap"] > 0).sum()
    assert not set(matched[split.values == "train"]["image_id"]) & set(matched[split.values == "val"]["image_id"])

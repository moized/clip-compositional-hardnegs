"""Build the fine-tuning data: leak-free COCO pairs with three kinds of negatives.

For every selected (image, caption) pair the table holds
* `neg_swap`        NegCLIP-style negatives: one random swap per type (all candidates);
* `neg_swap_nosym`  the same, but without swaps flagged as meaning-preserving;
* `neg_fluent`      one fluency-matched negative (see refine.py).
All training conditions use the same images and captions; only the negative differs.

Images come from COCO train2014 only, minus the COCO images that appear in ARO's
Visual Genome test sets, so no test image of any benchmark is seen in training.
"""
from __future__ import annotations

import json
import random

import numpy as np
import pandas as pd

from . import overlap, sugarcrepe
from .coco import coco_id, load_karpathy
from .negatives import ALL_KINDS, candidate_swaps, normalize_caption, one_per_kind, parse_captions
from .paths import DataPaths


def leak_free_pool(paths: DataPaths, max_images: int | None = None, seed: int = 0) -> tuple:
    """Captions of COCO train2014 images that no benchmark uses.

    Returns (DataFrame[image_id, image_file, caption], statistics dict).
    """
    train = load_karpathy(paths.karpathy, "train")
    rows = [(coco_id(a["image"]), a["image"], a["caption"]) for a in train
            if a["image"].startswith("train2014/")]
    pool = pd.DataFrame(rows, columns=["image_id", "image_file", "caption"])
    stats = {"karpathy_train_captions": len(train),
             "train2014_captions": len(pool),
             "train2014_images": int(pool["image_id"].nunique())}
    aro_ids = overlap.aro_coco_ids(paths.aro, paths.vg_image_data)
    in_aro = pool["image_id"].isin(aro_ids)
    stats["images_removed_because_in_aro_test"] = int(pool.loc[in_aro, "image_id"].nunique())
    pool = pool[~in_aro]
    pool["caption"] = pool["caption"].map(normalize_caption)
    pool = pool[pool["caption"].str.len() > 0].drop_duplicates(["image_id", "caption"])
    if max_images is not None:
        ids = sorted(pool["image_id"].unique())
        keep = set(random.Random(seed).sample(ids, min(max_images, len(ids))))
        pool = pool[pool["image_id"].isin(keep)]
    pool = pool.sort_values(["image_id", "caption"]).reset_index(drop=True)
    stats["pool_images"] = int(pool["image_id"].nunique())
    stats["pool_captions"] = int(len(pool))
    return pool, stats


def leak_report(image_ids, paths: DataPaths) -> dict:
    """How many of `image_ids` appear in each test set (all must be zero)."""
    ids = set(int(i) for i in image_ids)
    test = {coco_id(a["image"]) for a in load_karpathy(paths.karpathy, "test")}
    val = {coco_id(a["image"]) for a in load_karpathy(paths.karpathy, "val")}
    report = {
        "overlap_with_sugarcrepe_images": len(ids & set(sugarcrepe.image_ids(paths.sugarcrepe))),
        "overlap_with_karpathy_test_images": len(ids & test),
        "overlap_with_karpathy_val_images": len(ids & val),
        "overlap_with_aro_vg_images": len(ids & overlap.aro_coco_ids(paths.aro, paths.vg_image_data)),
    }
    report["leak_free"] = all(v == 0 for v in report.values())
    return report


def build_candidates(pool: pd.DataFrame, n_process: int = 1, progress: bool = True) -> pd.DataFrame:
    """All candidate swaps for every caption of the pool (one row per candidate)."""
    caption_index, kinds, symmetric, texts = [], [], [], []
    docs = parse_captions(pool["caption"].tolist(), n_process=n_process)
    if progress:
        from tqdm import tqdm

        docs = tqdm(docs, total=len(pool), desc="Generating swaps")
    for index, doc in enumerate(docs):
        for swap in candidate_swaps(doc):
            caption_index.append(index)
            kinds.append(swap.kind)
            symmetric.append(swap.symmetric)
            texts.append(swap.text)
    candidates = pd.DataFrame({"caption_index": caption_index, "kind": kinds,
                               "symmetric": symmetric, "text": texts})
    candidates["image_id"] = pool["image_id"].values[candidates["caption_index"].values]
    return candidates


def add_fluency_gap(pool: pd.DataFrame, candidates: pd.DataFrame, scorer) -> None:
    """Add GPT-2 mean log-probabilities and the gap (caption minus negative), in place."""
    pool["lp"] = scorer.score(pool["caption"].tolist(), progress=True)["mean"]
    candidates["lp"] = scorer.score(candidates["text"].tolist(), progress=True)["mean"]
    candidates["gap"] = pool["lp"].values[candidates["caption_index"].values] - candidates["lp"].values


def negclip_style_lists(candidates: pd.DataFrame, caption_indices, seed: int = 0) -> tuple:
    """For each caption: NegCLIP-style negatives (one per swap type), with and without
    the swaps flagged symmetric. Returns two dicts caption_index -> list of texts."""
    from .negatives import Swap

    rng = random.Random(seed)
    wanted = set(int(i) for i in caption_indices)
    part = candidates[candidates["caption_index"].isin(wanted)]
    everything, without_symmetric = {}, {}
    for caption_index, group in part.groupby("caption_index", sort=True):
        swaps = [Swap(k, (), (), t, bool(s)) for k, t, s in
                 zip(group["kind"], group["text"], group["symmetric"])]
        everything[int(caption_index)] = [s.text for s in one_per_kind(swaps, rng)]
        kept = [s for s in swaps if not s.symmetric]
        without_symmetric[int(caption_index)] = [s.text for s in one_per_kind(kept, rng)]
    return everything, without_symmetric


def split_balanced(matched: pd.DataFrame, n_val: int, max_train: int | None, seed: int = 0) -> pd.Series:
    """Assign "train" / "val" / "unused" so that every set keeps the same number of
    pairs on both sides of zero in every gap bin."""
    rng = np.random.RandomState(seed)
    assignment = pd.Series("unused", index=matched.index)
    total = len(matched)
    val_fraction = min(1.0, n_val / max(total, 1))
    train_fraction = 1.0 if max_train is None else min(1.0, max_train / max(total - n_val, 1))
    for _, bin_rows in matched.groupby("gap_bin"):
        below = bin_rows.index[bin_rows["gap"] < 0].values
        above = bin_rows.index[bin_rows["gap"] > 0].values
        k = min(len(below), len(above))
        below, above = rng.permutation(below)[:k], rng.permutation(above)[:k]
        n_v = int(rng.binomial(k, val_fraction))
        n_t = int(rng.binomial(k - n_v, train_fraction))
        for side in (below, above):
            assignment.loc[side[:n_v]] = "val"
            assignment.loc[side[n_v:n_v + n_t]] = "train"
    return assignment


def assemble_pairs(pool: pd.DataFrame, candidates: pd.DataFrame, matched: pd.DataFrame,
                   n_val: int, max_train: int | None, seed: int = 0) -> pd.DataFrame:
    """Final table: one row per selected image with its caption and the three negative sets."""
    matched = matched.copy()
    matched["split"] = split_balanced(matched, n_val, max_train, seed).values
    matched = matched[matched["split"] != "unused"]
    everything, without_symmetric = negclip_style_lists(candidates, matched["caption_index"], seed)
    rows = pool.iloc[matched["caption_index"].values]
    table = pd.DataFrame({
        "image_id": rows["image_id"].values,
        "image_file": rows["image_file"].values,
        "caption": rows["caption"].values,
        "neg_fluent": matched["text"].values,
        "gap_fluent": matched["gap"].values.astype(np.float32),
        "kind_fluent": matched["kind"].values,
        "neg_swap": [json.dumps(everything[int(i)]) for i in matched["caption_index"]],
        "neg_swap_nosym": [json.dumps(without_symmetric[int(i)]) for i in matched["caption_index"]],
        "caption_index": matched["caption_index"].values,
        "split": matched["split"].values,
    })
    return table.sort_values(["split", "image_id"]).reset_index(drop=True)


def pool_gaps(table: pd.DataFrame, candidates: pd.DataFrame, column: str) -> pd.DataFrame:
    """Gap, kind and symmetric flag of every negative listed in `column` of the pair table."""
    lookup = candidates.set_index(["caption_index", "text"])[["gap", "kind", "symmetric"]]
    lookup = lookup[~lookup.index.duplicated()]
    keys = [(int(ci), text) for ci, texts in zip(table["caption_index"], table[column])
            for text in json.loads(texts)]
    return lookup.loc[keys].reset_index()


def kind_counts(candidates: pd.DataFrame) -> dict:
    counts = candidates["kind"].value_counts()
    return {k: int(counts.get(k, 0)) for k in ALL_KINDS}

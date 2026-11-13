"""One entry point for the four benchmarks: load items and summarise results."""
from __future__ import annotations

import hashlib
import os
import random
from typing import Optional

import numpy as np
import pandas as pd

from . import aro, sugarcrepe
from .coco import CocoImages, coco_id, load_karpathy
from .paths import DataPaths
from .items import Item
from .stats import accuracy_ci, macro_accuracy_ci
from .utils import ensure_dir, load_json, save_json

BENCHMARKS = ("vg_relation", "vg_attribution", "coco_order", "sugarcrepe")
CHANCE = {"vg_relation": 0.5, "vg_attribution": 0.5, "coco_order": 0.2, "sugarcrepe": 0.5}


def _coco_order_items(paths: DataPaths, images: CocoImages, seed: int) -> list:
    """COCO-Order items for one seed; the shuffled captions are cached on disk so that
    every script (and every model) sees exactly the same candidates."""
    cache = os.path.join(paths.cache, f"coco_order_seed{seed}.json")
    if os.path.isfile(cache):
        rows = load_json(cache)
        return [Item(item_id=r["item_id"], image_path=images.path(int(r["image_id"])),
                     captions=tuple(r["captions"]), label=0, group="coco_order",
                     image_id=str(r["image_id"])) for r in rows]
    items = aro.load_coco_order(load_karpathy(paths.karpathy, "test"), images, seed=seed)
    ensure_dir(paths.cache)
    save_json([{"item_id": i.item_id, "image_id": i.image_id, "captions": list(i.captions)}
               for i in items], cache)
    return items


def get_items(name: str, paths: DataPaths, images: CocoImages, seed: int = 0,
              limit: Optional[int] = None) -> list:
    """Test items of one benchmark. `seed` only matters for coco_order.
    `limit` keeps a fixed random subset (for quick tests only)."""
    if name == "vg_relation":
        items = aro.load_vg_relation(paths.aro)
    elif name == "vg_attribution":
        items = aro.load_vg_attribution(paths.aro)
    elif name == "coco_order":
        items = _coco_order_items(paths, images, seed)
    elif name == "sugarcrepe":
        items = sugarcrepe.load_items(paths.sugarcrepe, images)
    else:
        raise ValueError(f"Unknown benchmark {name!r}; choose from {BENCHMARKS}")
    if limit is not None and limit < len(items):
        items = random.Random(0).sample(items, limit)
    return items


def items_fingerprint(items: list) -> str:
    """Short hash of the item ids, captions and labels.

    Two result files can only be compared item by item if they were computed on
    the same test items; the hash makes that checkable (COCO-Order captions, for
    example, depend on the shuffle seed and on the spaCy version).
    """
    digest = hashlib.sha1()
    for item in items:
        digest.update(item.item_id.encode("utf-8") + b"\x00")
        for caption in item.captions:
            digest.update(caption.encode("utf-8") + b"\x00")
        digest.update(str(item.label).encode("utf-8") + b"\x01")
    return digest.hexdigest()[:16]


def needed_coco_ids(paths: DataPaths) -> dict:
    """COCO ids (with download folder) needed by SugarCrepe and COCO-Order."""
    from .coco import ids_with_folder

    wanted = dict(sugarcrepe.image_ids(paths.sugarcrepe))
    wanted.update(ids_with_folder(a["image"] for a in load_karpathy(paths.karpathy, "test")))
    return wanted


def summarize(name: str, table: pd.DataFrame, column: str = "correct", n_boot: int = 2000) -> dict:
    """Headline metrics with cluster-bootstrap 95% intervals (images are the clusters).

    `headline` is the number the papers report: macro accuracy for the two Visual
    Genome tasks, plain accuracy for COCO-Order, mean of the seven subset accuracies
    for SugarCrepe. `headline_ci` is the same value with its interval.
    """
    out: dict = {"benchmark": name, "n_items": int(len(table)), "chance": CHANCE[name]}
    clusters = table["image_id"].values
    if name in ("vg_relation", "vg_attribution"):
        keep = aro.relation_groups(table) if name == "vg_relation" else aro.attribution_groups(table)
        kept = table[keep]
        out["macro_accuracy"] = macro_accuracy_ci(kept[column].values, kept["group"].values,
                                                 kept["image_id"].values, n_boot=n_boot)
        out["micro_accuracy"] = accuracy_ci(table[column].values, clusters, n_boot=n_boot)
        out["micro_accuracy_kept_groups"] = accuracy_ci(kept[column].values, kept["image_id"].values,
                                                       n_boot=n_boot)
        out["n_groups"] = int(kept["group"].nunique())
        out["headline_ci"] = out["macro_accuracy"]
    elif name == "coco_order":
        out["accuracy"] = accuracy_ci(table[column].values, clusters, n_boot=n_boot)
        out["headline_ci"] = out["accuracy"]
    elif name == "sugarcrepe":
        subsets = {}
        for subset in sugarcrepe.SUBSETS:
            part = table[table["group"] == subset]
            if len(part):
                subsets[subset] = accuracy_ci(part[column].values, part["image_id"].values, n_boot=n_boot)
        out["subsets"] = subsets
        categories = {}
        for category in ("replace", "swap", "add"):
            names = [s for s in sugarcrepe.SUBSETS if sugarcrepe.CATEGORY[s] == category and s in subsets]
            if not names:
                continue
            part = table[table["group"].isin(names)]
            categories[category] = {
                # SugarCrepe paper (Table 5): all items of the category pooled
                "pooled": accuracy_ci(part[column].values, part["image_id"].values, n_boot=n_boot),
                # later papers (e.g. C2LIP, TripletCLIP): plain mean of the subset accuracies
                "mean_of_subsets": float(np.mean([subsets[s]["value"] for s in names])),
            }
        out["categories"] = categories
        out["overall_pooled"] = accuracy_ci(table[column].values, clusters, n_boot=n_boot)
        # mean of the subset accuracies = macro accuracy with the subsets as groups
        out["overall_mean_of_subsets"] = macro_accuracy_ci(
            table[column].values, table["group"].values, clusters, n_boot=n_boot)
        out["headline_ci"] = out["overall_mean_of_subsets"]
    out["headline"] = out["headline_ci"]["value"]
    return out

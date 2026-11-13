"""ARO benchmark (Yuksekgonul et al., ICLR 2023): VG-Relation, VG-Attribution, COCO-Order.

Adapted from `dataset_zoo/aro_datasets.py` and the notebook
"Replicate ARO! VG-Relation, VG-Attribution" in
https://github.com/mertyg/vision-language-models-are-bows (MIT licence,
Copyright (c) 2023 Mert Yuksekgonul). The test cases, the candidate order and the
metrics follow the original code:

* VG-Relation / VG-Attribution: candidates are [false caption, true caption];
  the image is cropped to the box around the two objects.
  - Relation score = mean over relations of per-relation accuracy, after removing
    the relations in `SYMMETRIC_RELATIONS` ("macro accuracy").
  - Attribution score = mean over attribute pairs with at least 25 cases.
* COCO-Order: candidates are [true caption, 4 shuffled versions]; chance is 20%.

Both the paper's macro accuracy and the plain per-item ("micro") accuracy are
returned, because follow-up papers do not all use the same one.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from .coco import CocoImages, coco_id
from .perturb import TextShuffler, pre_caption
from .items import Item
from .utils import load_json

# Google Drive file ids used by the original repository.
GDRIVE_IDS = {
    "visual_genome_relation.json": "1kX2iCHEv0CADL8dSO1nMdW-V0NqIAiP3",
    "visual_genome_attribution.json": "13tWvOrNOLHxl3Rm9cR3geAdHx2qR3-Tw",
    "vgr_vga_images.zip": "1qaPlrwhGNMrR3a11iopZUT_GPP_LrgP9",
}
NEGCLIP_GDRIVE_ID = "1ooVVPxB-tvptgmHlIMMFGV3Cg-IrhbRZ"
MIN_ATTRIBUTE_COUNT = 25

# Relations the ARO authors drop before averaging (copied from their notebook).
SYMMETRIC_RELATIONS = frozenset([
    'adjusting', 'attached to', 'between', 'bigger than', 'biting', 'boarding', 'brushing',
    'chewing', 'cleaning', 'climbing', 'close to', 'coming from', 'coming out of', 'contain',
    'crossing', 'dragging', 'draped over', 'drinking', 'drinking from', 'driving',
    'driving down', 'driving on', 'eating from', 'eating in', 'enclosing', 'exiting', 'facing',
    'filled with', 'floating in', 'floating on', 'flying', 'flying above', 'flying in',
    'flying over', 'flying through', 'full of', 'going down', 'going into', 'going through',
    'grazing in', 'growing in', 'growing on', 'guiding', 'hanging from', 'hanging in',
    'hanging off', 'hanging over', 'higher than', 'holding onto', 'hugging', 'in between',
    'jumping off', 'jumping on', 'jumping over', 'kept in', 'larger than', 'leading',
    'leaning over', 'leaving', 'licking', 'longer than', 'looking in', 'looking into',
    'looking out', 'looking over', 'looking through', 'lying next to', 'lying on top of',
    'making', 'mixed with', 'mounted on', 'moving', 'on the back of', 'on the edge of',
    'on the front of', 'on the other side of', 'opening', 'painted on', 'parked at',
    'parked beside', 'parked by', 'parked in', 'parked in front of', 'parked near',
    'parked next to', 'perched on', 'petting', 'piled on', 'playing', 'playing in',
    'playing on', 'playing with', 'pouring', 'reaching for', 'reading', 'reflected on',
    'riding on', 'running in', 'running on', 'running through', 'seen through',
    'sitting behind', 'sitting beside', 'sitting by', 'sitting in front of', 'sitting near',
    'sitting next to', 'sitting under', 'skiing down', 'skiing on', 'sleeping in',
    'sleeping on', 'smiling at', 'sniffing', 'splashing', 'sprinkled on', 'stacked on',
    'standing against', 'standing around', 'standing behind', 'standing beside',
    'standing in front of', 'standing near', 'standing next to', 'staring at', 'stuck in',
    'surrounding', 'swimming in', 'swinging', 'talking to', 'topped with', 'touching',
    'traveling down', 'traveling on', 'tying', 'typing on', 'underneath', 'wading in',
    'waiting for', 'walking across', 'walking by', 'walking down', 'walking next to',
    'walking through', 'working in', 'working on', 'worn on', 'wrapped around', 'wrapped in',
    'by', 'of', 'near', 'next to', 'with', 'beside', 'on the side of', 'around',
])


def _vg_items(aro_dir: str, filename: str, group_of, prefix: str) -> list:
    image_dir = os.path.join(aro_dir, "images")
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"{image_dir} not found. Run scripts/00_download_data.py first.")
    items = []
    for index, case in enumerate(load_json(os.path.join(aro_dir, filename))):
        items.append(Item(
            item_id=f"{prefix}/{index}",
            image_path=os.path.join(image_dir, case["image_path"]),
            captions=(case["false_caption"], case["true_caption"]),
            label=1,
            group=group_of(case),
            image_id=str(case["image_id"]),
            box=(case["bbox_x"], case["bbox_y"], case["bbox_w"], case["bbox_h"]),
        ))
    return items


def load_vg_relation(aro_dir: str) -> list:
    """VG-Relation test items; `group` is the relation name."""
    return _vg_items(aro_dir, "visual_genome_relation.json", lambda c: c["relation_name"], "vgr")


def load_vg_attribution(aro_dir: str) -> list:
    """VG-Attribution test items; `group` is the attribute pair, e.g. 'open_crouched'."""
    return _vg_items(aro_dir, "visual_genome_attribution.json",
                     lambda c: f"{c['attributes'][0]}_{c['attributes'][1]}", "vga")


def vg_image_ids(aro_dir: str) -> set:
    """Visual Genome image ids (as strings) used by VG-Relation and VG-Attribution."""
    ids = set()
    for filename in ("visual_genome_relation.json", "visual_genome_attribution.json"):
        ids.update(str(case["image_id"]) for case in load_json(os.path.join(aro_dir, filename)))
    return ids


def load_coco_order(karpathy_test: list, images: CocoImages, seed: int = 0,
                    max_words: int = 30) -> list:
    """COCO-Order test items built from the Karpathy test split (5 captions per image).

    The shuffles are random, so results depend on `seed`; report several seeds.
    """
    shuffler = TextShuffler(seed=seed)
    items = []
    for annotation in karpathy_test:
        image_id = coco_id(annotation["image"])
        path = images.path(image_id)
        for caption_index, caption in enumerate(annotation["caption"]):
            options = [pre_caption(caption, max_words)]
            options += [pre_caption(p, max_words) for p in shuffler.all_perturbations(caption)]
            items.append(Item(
                item_id=f"coco_order/{image_id}/{caption_index}",
                image_path=path,
                captions=tuple(options),
                label=0,
                group="coco_order",
                image_id=str(image_id),
            ))
    return items


def relation_groups(table: pd.DataFrame) -> pd.Series:
    """True for rows whose relation counts toward the ARO relation score."""
    return ~table["group"].isin(SYMMETRIC_RELATIONS)


def attribution_groups(table: pd.DataFrame) -> pd.Series:
    """True for rows whose attribute pair has at least 25 test cases."""
    counts = table["group"].map(table["group"].value_counts())
    return counts >= MIN_ATTRIBUTE_COUNT


def macro_accuracy(table: pd.DataFrame, keep: pd.Series, column: str = "correct") -> float:
    """Mean of per-group accuracies over the kept rows."""
    kept = table[keep]
    return float(kept.groupby("group")[column].mean().mean())


def summarize(name: str, table: pd.DataFrame, column: str = "correct") -> dict:
    """Headline numbers for one ARO task from its per-item table."""
    out = {"n_items": int(len(table)), "micro_accuracy": float(table[column].mean())}
    if name == "vg_relation":
        keep = relation_groups(table)
    elif name == "vg_attribution":
        keep = attribution_groups(table)
    else:
        out["accuracy"] = out["micro_accuracy"]
        return out
    out["macro_accuracy"] = macro_accuracy(table, keep, column)
    out["n_groups"] = int(table[keep]["group"].nunique())
    out["n_items_in_macro"] = int(keep.sum())
    out["micro_accuracy_kept_groups"] = float(table[keep][column].mean())
    out["accuracy"] = out["macro_accuracy"]
    return out


def per_group_table(table: pd.DataFrame) -> pd.DataFrame:
    """Accuracy and count per relation / attribute pair (like ARO's Table 2)."""
    grouped = table.groupby("group")["correct"].agg(["mean", "size"]).reset_index()
    grouped.columns = ["group", "accuracy", "count"]
    return grouped.sort_values("count", ascending=False).reset_index(drop=True)

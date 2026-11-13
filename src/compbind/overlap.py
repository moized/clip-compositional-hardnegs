"""Which test images were seen in training? Helpers for split-overlap checks.

Two facts make overlap possible:
* Visual Genome (the source of ARO's VG-Relation and VG-Attribution) reuses many
  COCO images; `image_data.json` from Visual Genome gives the COCO id of each.
* The released NegCLIP checkpoint was fine-tuned on the Karpathy *train* split,
  which contains images from COCO val2014. SugarCrepe's images (val2017) are a
  subset of val2014.

The authors' training file (`temp_data/train_neg_clip.tsv` in the ARO repository)
lists exactly which images and captions NegCLIP was trained on.
"""
from __future__ import annotations

import csv
import os
from functools import lru_cache

from .coco import coco_id
from .utils import load_json


def _caption_key(caption: str) -> str:
    return " ".join(str(caption).lower().split())


@lru_cache(maxsize=4)
def negclip_training_set(tsv_path: str) -> tuple:
    """(set of COCO image ids, set of normalised captions) used to train released NegCLIP."""
    csv.field_size_limit(10 ** 9)
    ids, captions = set(), set()
    with open(tsv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            ids.add(coco_id(row["filepath"]))
            captions.add(_caption_key(row["title"]))
    return frozenset(ids), frozenset(captions)


@lru_cache(maxsize=2)
def vg_to_coco(vg_image_data_path: str) -> dict:
    """Visual Genome image id (str) -> COCO image id (int), for VG images taken from COCO."""
    mapping = {}
    for record in load_json(vg_image_data_path):
        vg_id = record.get("image_id", record.get("id"))
        if record.get("coco_id") is not None:
            mapping[str(vg_id)] = int(record["coco_id"])
    return mapping


def aro_coco_ids(aro_dir: str, vg_image_data_path: str) -> set:
    """COCO ids of the images used by VG-Relation and VG-Attribution."""
    from .aro import vg_image_ids

    mapping = vg_to_coco(vg_image_data_path)
    return {mapping[v] for v in vg_image_ids(aro_dir) if v in mapping}


def item_coco_id(benchmark: str, image_id: str, vg_map: dict):
    """COCO id behind a test item's image (None if the image is not from COCO)."""
    if benchmark in ("vg_relation", "vg_attribution"):
        return vg_map.get(str(image_id))
    return int(image_id)


def have_overlap_files(paths) -> bool:
    return os.path.isfile(paths.negclip_train_tsv) and os.path.isfile(paths.vg_image_data)

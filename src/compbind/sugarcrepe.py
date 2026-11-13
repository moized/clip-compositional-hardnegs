"""SugarCrepe benchmark (Hsieh et al., NeurIPS 2023).

Data and protocol follow https://github.com/RAIVNLab/sugar-crepe (MIT licence):
each example is a COCO val2017 image, its true caption and one fluent hard
negative; the model is correct when the true caption scores strictly higher.
Here the negative is listed first, so a tie counts as an error, as in the
original `main_eval.py`.
"""
from __future__ import annotations

import os

from .coco import CocoImages, coco_id
from .items import Item
from .utils import load_json

SUBSETS = ("replace_obj", "replace_att", "replace_rel", "swap_obj", "swap_att", "add_obj", "add_att")
CATEGORY = {"replace_obj": "replace", "replace_att": "replace", "replace_rel": "replace",
            "swap_obj": "swap", "swap_att": "swap", "add_obj": "add", "add_att": "add"}


def load_raw(data_dir: str) -> dict:
    """{subset: {example_id: {"filename", "caption", "negative_caption"}}}."""
    return {name: load_json(os.path.join(data_dir, f"{name}.json")) for name in SUBSETS}


def image_ids(data_dir: str) -> dict:
    """COCO ids of all SugarCrepe images, mapped to their download folder (val2017)."""
    ids = {}
    for examples in load_raw(data_dir).values():
        for example in examples.values():
            ids[coco_id(example["filename"])] = "val2017"
    return ids


def load_items(data_dir: str, images: CocoImages, subsets=SUBSETS) -> list:
    """All SugarCrepe test items; `group` is the subset name."""
    raw = load_raw(data_dir)
    items = []
    for name in subsets:
        for example_id, example in raw[name].items():
            image_id = coco_id(example["filename"])
            items.append(Item(
                item_id=f"{name}/{example_id}",
                image_path=images.path(image_id),
                captions=(example["negative_caption"], example["caption"]),
                label=1,
                group=name,
                image_id=str(image_id),
            ))
    return items

"""Which test images were seen in training? Helpers for split-overlap checks.

Two facts make overlap possible:
* Visual Genome (the source of ARO's VG-Relation and VG-Attribution) reuses many
  COCO images; Visual Genome's `image_data.json` gives the COCO id of each.
* The released NegCLIP checkpoint was fine-tuned on the Karpathy *train* split,
  which contains images from COCO val2014. SugarCrepe's images (val2017) are a
  subset of val2014.

The authors' training file (`temp_data/train_neg_clip.tsv` in the ARO repository)
lists exactly which images and captions NegCLIP was trained on.

Only the COCO ids of ARO's own images are needed. They ship with this repository in
`resources/aro_vg_to_coco.csv`, taken from Visual Genome's `image_data.json`
(Krishna et al., 2017; CC BY 4.0), so the experiments do not depend on the Visual
Genome server being up. When the full `image_data.json` has been downloaded, it is
used instead, and `scripts/00_download_data.py` checks that the two agree.
"""
from __future__ import annotations

import csv
import os
from functools import lru_cache

from .coco import coco_id
from .utils import load_json

RESOURCE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources", "aro_vg_to_coco.csv")


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


def _have_full_file(vg_image_data_path: str | None) -> bool:
    return bool(vg_image_data_path) and os.path.isfile(vg_image_data_path)


@lru_cache(maxsize=4)
def read_bundled_mapping(path: str = RESOURCE) -> tuple:
    """(VG image id -> COCO id for the images taken from COCO, set of every VG id listed)."""
    mapping, listed = {}, set()
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            listed.add(row["vg_image_id"])
            if row["coco_id"]:
                mapping[row["vg_image_id"]] = int(row["coco_id"])
    return mapping, frozenset(listed)


@lru_cache(maxsize=2)
def vg_to_coco(vg_image_data_path: str | None) -> dict:
    """Visual Genome image id (str) -> COCO image id (int), for VG images taken from COCO.

    Reads Visual Genome's image_data.json when it exists; otherwise the bundled
    mapping, which covers ARO's images (all that this project needs).
    """
    if not _have_full_file(vg_image_data_path):
        return read_bundled_mapping()[0]
    mapping = {}
    for record in load_json(vg_image_data_path):
        vg_id = record.get("image_id", record.get("id"))
        if record.get("coco_id") is not None:
            mapping[str(vg_id)] = int(record["coco_id"])
    return mapping


def mapping_source(vg_image_data_path: str | None) -> str:
    """Where the VG -> COCO ids come from in this run (recorded with the results)."""
    if _have_full_file(vg_image_data_path):
        return "Visual Genome image_data.json"
    return "src/compbind/resources/aro_vg_to_coco.csv"


def aro_coco_ids(aro_dir: str, vg_image_data_path: str | None, bundled_path: str = RESOURCE) -> set:
    """COCO ids of the images used by VG-Relation and VG-Attribution."""
    from .aro import vg_image_ids

    ids = vg_image_ids(aro_dir)
    if _have_full_file(vg_image_data_path):
        mapping = vg_to_coco(vg_image_data_path)
    else:
        mapping, listed = read_bundled_mapping(bundled_path)
        missing = ids - listed
        if missing:
            raise RuntimeError(
                f"{len(missing)} images of ARO's test files are not listed in {bundled_path}, so the "
                "ARO files differ from those the list was made for. Download Visual Genome's "
                "image_data.json (scripts/00_download_data.py --only vg_meta) and run again.")
    return {mapping[v] for v in ids if v in mapping}


def mapping_rows(aro_dir: str, vg_image_data_path: str) -> list:
    """[(VG id, COCO id or "")] for every image of ARO's two Visual Genome test files."""
    from .aro import vg_image_ids

    mapping = vg_to_coco(vg_image_data_path)
    return [(v, str(mapping.get(v, ""))) for v in sorted(vg_image_ids(aro_dir), key=int)]


def write_bundled_mapping(aro_dir: str, vg_image_data_path: str, path: str = RESOURCE) -> int:
    """(Re)create the bundled mapping from the full image_data.json. Returns the number of rows."""
    rows = mapping_rows(aro_dir, vg_image_data_path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["vg_image_id", "coco_id"])
        writer.writerows(rows)
    read_bundled_mapping.cache_clear()
    return len(rows)


def compare_with_bundled(aro_dir: str, vg_image_data_path: str, path: str = RESOURCE) -> dict:
    """Does the bundled mapping agree with the full image_data.json for ARO's images?"""
    expected = dict(mapping_rows(aro_dir, vg_image_data_path))
    mapping, listed = read_bundled_mapping(path)
    bundled = {v: str(mapping.get(v, "")) for v in listed}
    differences = sorted(v for v in set(expected) | set(bundled) if expected.get(v) != bundled.get(v))
    return {"aro_images": len(expected), "bundled_rows": len(bundled), "differences": len(differences),
            "agree": not differences, "examples": differences[:5]}


def item_coco_id(benchmark: str, image_id: str, vg_map: dict):
    """COCO id behind a test item's image (None if the image is not from COCO)."""
    if benchmark in ("vg_relation", "vg_attribution"):
        return vg_map.get(str(image_id))
    return int(image_id)


def have_overlap_files(paths) -> bool:
    return os.path.isfile(paths.negclip_train_tsv) and (
        os.path.isfile(paths.vg_image_data) or os.path.isfile(RESOURCE))

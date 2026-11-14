"""COCO ids of ARO's Visual Genome images: the bundled list and its fallback logic."""
import csv
import json
import os

import pytest

from compbind import overlap


def test_bundled_list_is_well_formed():
    with open(overlap.RESOURCE, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows and list(rows[0]) == ["vg_image_id", "coco_id"]
    ids = [r["vg_image_id"] for r in rows]
    assert len(ids) == len(set(ids)) == 6603             # every image of ARO's two Visual Genome files
    coco = [int(r["coco_id"]) for r in rows if r["coco_id"]]
    assert len(coco) == 3225 and all(c > 0 for c in coco)


def test_bundled_list_is_used_when_image_data_is_missing(tmp_path):
    mapping = overlap.vg_to_coco(str(tmp_path / "missing" / "image_data.json"))
    assert mapping == overlap.read_bundled_mapping()[0]
    assert overlap.mapping_source(None).endswith("aro_vg_to_coco.csv")


def write_aro(folder, relation_ids, attribution_ids):
    os.makedirs(folder, exist_ok=True)
    for name, ids in (("visual_genome_relation.json", relation_ids),
                      ("visual_genome_attribution.json", attribution_ids)):
        with open(os.path.join(folder, name), "w") as f:
            json.dump([{"image_id": i} for i in ids], f)


def write_bundled(path, rows):
    with open(path, "w", newline="") as f:
        csv.writer(f).writerows([("vg_image_id", "coco_id")] + rows)


def test_aro_coco_ids_from_the_bundled_list(tmp_path):
    aro = str(tmp_path / "aro")
    write_aro(aro, [1, 2], [2, 3])
    bundled = str(tmp_path / "list.csv")
    write_bundled(bundled, [("1", "101"), ("2", ""), ("3", "103")])
    assert overlap.aro_coco_ids(aro, None, bundled_path=bundled) == {101, 103}
    overlap.read_bundled_mapping.cache_clear()


def test_unknown_aro_image_is_an_error_not_a_silent_gap(tmp_path):
    aro = str(tmp_path / "aro")
    write_aro(aro, [1, 2], [4])                            # image 4 is not in the list
    bundled = str(tmp_path / "list.csv")
    write_bundled(bundled, [("1", "101"), ("2", "")])
    with pytest.raises(RuntimeError, match="not listed"):
        overlap.aro_coco_ids(aro, None, bundled_path=bundled)
    overlap.read_bundled_mapping.cache_clear()

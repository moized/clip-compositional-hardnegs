"""Finding COCO images by id under both naming schemes."""
import os

import pytest

from compbind.coco import CocoImages, coco_id, discover_image_dirs, folder_of, ids_with_folder


def touch(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"x")


def test_coco_id_from_both_naming_schemes():
    assert coco_id("val2014/COCO_val2014_000000391895.jpg") == 391895
    assert coco_id("000000397133.jpg") == 397133
    with pytest.raises(ValueError):
        coco_id("not_a_coco_file.jpg")


def test_images_are_found_in_2014_and_2017_folders(tmp_path):
    touch(tmp_path / "a" / "coco2017" / "train2017" / "000000000009.jpg")
    touch(tmp_path / "b" / "val2014" / "COCO_val2014_000000000042.jpg")
    images = CocoImages([str(tmp_path)])
    assert len(images.dirs) == 2
    assert images.find(9).endswith("000000000009.jpg")
    assert images.find(42).endswith("COCO_val2014_000000000042.jpg")
    assert images.find(7) is None
    with pytest.raises(FileNotFoundError):
        images.path(7)


def test_discover_ignores_missing_roots(tmp_path):
    assert discover_image_dirs([str(tmp_path / "missing"), ""]) == []


def test_ensure_reports_present_images_without_downloading(tmp_path):
    touch(tmp_path / "val2017" / "000000000001.jpg")
    images = CocoImages([str(tmp_path)])
    assert images.ensure({1: "val2017"}, progress=False) == {"present": 1, "downloaded": 0}
    with pytest.raises(FileNotFoundError):
        images.ensure({2: "val2017"}, progress=False)       # missing and no download folder given


def test_karpathy_path_helpers():
    assert folder_of("val2014/COCO_val2014_000000391895.jpg") == "val2014"
    assert ids_with_folder(["train2014/COCO_train2014_000000000009.jpg"]) == {9: "train2014"}

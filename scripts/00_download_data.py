#!/usr/bin/env python
"""Download everything the experiments need (skips files that already exist).

    python scripts/00_download_data.py --data-dir data --coco-roots /kaggle/input

What is fetched:
  sugarcrepe  7 JSON files from the SugarCrepe repository (pinned commit)
  aro         VG-Relation / VG-Attribution annotations and images (Google Drive links
              from the ARO repository), plus the authors' released NegCLIP training file
  negclip     the released NegCLIP checkpoint (1.8 GB, Google Drive)
  karpathy    COCO Karpathy split files
  vg_meta     Visual Genome image metadata (maps VG images to COCO ids)
  coco_eval   the COCO images used by SugarCrepe and COCO-Order (about 6,400 files)

Training images are fetched later by scripts/03_build_training_data.py, once the
training subset is known.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
import zipfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from compbind import aro, sugarcrepe  # noqa: E402
from compbind.benchmarks import needed_coco_ids  # noqa: E402
from compbind.coco import CocoImages, download_file, load_karpathy  # noqa: E402
from compbind.paths import (ARO_COMMIT, ARO_REPO, SUGARCREPE_COMMIT, SUGARCREPE_REPO,  # noqa: E402
                            VG_IMAGE_DATA_URL, DataPaths, add_common_args)
from compbind.utils import ensure_dir, run_record, save_json  # noqa: E402

STEPS = ("sugarcrepe", "aro", "negclip", "karpathy", "vg_meta", "coco_eval")
RAW = "https://raw.githubusercontent.com/{repo}/{commit}/{path}"


def gdrive(file_id: str, target: str) -> None:
    """Download a Google Drive file with gdown (Python API, then command line)."""
    if os.path.isfile(target) and os.path.getsize(target) > 0:
        return
    ensure_dir(os.path.dirname(os.path.abspath(target)))
    tmp = target + ".part"
    try:
        import gdown

        gdown.download(id=file_id, output=tmp, quiet=False)
    except Exception as err:
        print(f"gdown Python API failed ({err}); trying the command line.")
        subprocess.call([sys.executable, "-m", "gdown", file_id, "-O", tmp])
    if not os.path.isfile(tmp) or os.path.getsize(tmp) == 0:
        raise IOError(
            f"Could not download Google Drive file {file_id}. Google sometimes blocks popular "
            f"files for a few hours. Open https://drive.google.com/uc?id={file_id} in a browser, "
            f"download it by hand and save it as {target}."
        )
    os.replace(tmp, target)


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    add_common_args(parser)
    parser.add_argument("--only", nargs="*", default=list(STEPS), choices=STEPS,
                        help="Steps to run (default: all).")
    parser.add_argument("--download-workers", type=int, default=16)
    args = parser.parse_args()
    paths = DataPaths(ensure_dir(args.data_dir))
    manifest = {"run": run_record(args), "files": {},
                "upstream": {ARO_REPO: ARO_COMMIT, SUGARCREPE_REPO: SUGARCREPE_COMMIT}}

    if "sugarcrepe" in args.only:
        for name in sugarcrepe.SUBSETS:
            download_file(RAW.format(repo=SUGARCREPE_REPO, commit=SUGARCREPE_COMMIT,
                                     path=f"data/{name}.json"),
                          os.path.join(paths.sugarcrepe, f"{name}.json"))
        print("SugarCrepe: ok")

    if "aro" in args.only:
        for filename in ("visual_genome_relation.json", "visual_genome_attribution.json"):
            gdrive(aro.GDRIVE_IDS[filename], os.path.join(paths.aro, filename))
        if not os.path.isdir(os.path.join(paths.aro, "images")):
            archive = os.path.join(paths.aro, "vgr_vga_images.zip")
            gdrive(aro.GDRIVE_IDS["vgr_vga_images.zip"], archive)
            with zipfile.ZipFile(archive) as z:
                z.extractall(paths.aro)
            os.remove(archive)
        download_file(RAW.format(repo=ARO_REPO, commit=ARO_COMMIT, path="temp_data/train_neg_clip.tsv"),
                      paths.negclip_train_tsv)
        download_file(RAW.format(repo=ARO_REPO, commit=ARO_COMMIT, path="temp_data/valid_neg_clip.tsv"),
                      paths.negclip_valid_tsv)
        print("ARO: ok,", len(os.listdir(os.path.join(paths.aro, "images"))), "images")

    if "negclip" in args.only:
        gdrive(aro.NEGCLIP_GDRIVE_ID, paths.negclip_checkpoint)
        print("NegCLIP checkpoint: ok")

    if "karpathy" in args.only:
        for split in ("train", "val", "test"):
            load_karpathy(paths.karpathy, split)
        print("Karpathy splits: ok")

    if "vg_meta" in args.only:
        if not os.path.isfile(paths.vg_image_data):
            archive = paths.vg_image_data + ".zip"
            download_file(VG_IMAGE_DATA_URL, archive)
            with zipfile.ZipFile(archive) as z:
                z.extractall(os.path.dirname(paths.vg_image_data))
            os.remove(archive)
        print("Visual Genome metadata: ok")

    if "coco_eval" in args.only:
        images = CocoImages(args.coco_roots, download_dir=paths.coco_download)
        print("COCO image folders found:", images.dirs or "none (images will be downloaded)")
        wanted = needed_coco_ids(paths)
        stats = images.ensure(wanted, workers=args.download_workers)
        print(f"COCO evaluation images: {stats['present']} already present, "
              f"{stats['downloaded']} downloaded")
        manifest["coco_eval_images"] = {"needed": len(wanted), **stats, "folders": images.dirs}

    for label, path in [
        ("vg_relation", os.path.join(paths.aro, "visual_genome_relation.json")),
        ("vg_attribution", os.path.join(paths.aro, "visual_genome_attribution.json")),
        ("negclip_train_tsv", paths.negclip_train_tsv),
        ("negclip_checkpoint", paths.negclip_checkpoint),
        ("karpathy_test", os.path.join(paths.karpathy, "coco_karpathy_test.json")),
        ("karpathy_train", os.path.join(paths.karpathy, "coco_karpathy_train.json")),
        ("vg_image_data", paths.vg_image_data),
    ] + [(f"sugarcrepe_{n}", os.path.join(paths.sugarcrepe, f"{n}.json")) for n in sugarcrepe.SUBSETS]:
        if os.path.isfile(path):
            manifest["files"][label] = {"path": os.path.relpath(path, paths.root),
                                        "bytes": os.path.getsize(path), "sha256": sha256(path)}
    save_json(manifest, os.path.join(paths.root, "data_manifest.json"))
    print("Wrote", os.path.join(paths.root, "data_manifest.json"))


if __name__ == "__main__":
    main()

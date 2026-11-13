"""Where every data file lives, relative to one data directory."""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

# Upstream revisions this project was developed against (for reproducibility).
ARO_REPO = "mertyg/vision-language-models-are-bows"
ARO_COMMIT = "81fdd20dc5b6f434aa5a4a971717bc80d1cedb0d"
SUGARCREPE_REPO = "RAIVNLab/sugar-crepe"
SUGARCREPE_COMMIT = "0047054b243992f0fad63d6f64f7544862daf846"
VG_IMAGE_DATA_URL = "https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/image_data.json.zip"


@dataclass(frozen=True)
class DataPaths:
    root: str

    @property
    def sugarcrepe(self) -> str:
        return os.path.join(self.root, "sugarcrepe")

    @property
    def aro(self) -> str:
        return os.path.join(self.root, "aro")

    @property
    def negclip_train_tsv(self) -> str:
        return os.path.join(self.root, "aro", "negclip_train.tsv")

    @property
    def negclip_valid_tsv(self) -> str:
        return os.path.join(self.root, "aro", "negclip_valid.tsv")

    @property
    def negclip_checkpoint(self) -> str:
        return os.path.join(self.root, "ckpt", "negclip.pth")

    @property
    def karpathy(self) -> str:
        return os.path.join(self.root, "karpathy")

    @property
    def vg_image_data(self) -> str:
        return os.path.join(self.root, "vg", "image_data.json")

    @property
    def coco_download(self) -> str:
        return os.path.join(self.root, "coco")

    @property
    def training(self) -> str:
        return os.path.join(self.root, "training")

    @property
    def cache(self) -> str:
        return os.path.join(self.root, "cache")


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Arguments shared by all scripts."""
    parser.add_argument("--data-dir", default=os.environ.get("COMPBIND_DATA", "data"),
                        help="Folder for downloaded data (default: ./data or $COMPBIND_DATA).")
    parser.add_argument("--coco-roots", nargs="*", default=[],
                        help="Folders that may already contain COCO images, e.g. /kaggle/input. "
                             "Images not found there are downloaded.")
    parser.add_argument("--device", default=None, help="cuda, cuda:1 or cpu (default: auto).")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4, help="Image-loading worker processes.")
    return parser


def pick_device(requested: str | None) -> str:
    if requested:
        return requested
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"

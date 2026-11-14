"""COCO helpers: find or download images by id, and load the Karpathy splits.

COCO images have two naming schemes. The 2014 release uses
`train2014/COCO_train2014_000000522418.jpg`; the 2017 release uses
`train2017/000000522418.jpg`. Both releases contain the same pictures, split
differently, so an image is identified here by its integer id and looked up
under either name. That lets the code run on whatever COCO copy is available
(for example a Kaggle dataset) and download only the files that are missing.

Split facts used by this project (checked in `scripts/03_build_training_data.py`):
* val2017 (5,000 images, used by SugarCrepe) is a subset of val2014;
* the Karpathy test and val splits are also subsets of val2014;
* train2014 shares no image with val2014.
So training on train2014 images only keeps every COCO-based test set unseen.
"""
from __future__ import annotations

import os
import re
import shutil
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterable, Sequence

from .utils import ensure_dir, load_json

KARPATHY_URL = "https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_{split}.json"
COCO_IMAGE_URL = "http://images.cocodataset.org/{folder}/{name}"
IMAGE_FOLDERS = ("train2014", "val2014", "train2017", "val2017")

_ID_PATTERN = re.compile(r"(\d{12})\.jpg$")


def coco_id(path_or_name: str) -> int:
    """Integer COCO image id from a file name of either naming scheme."""
    match = _ID_PATTERN.search(str(path_or_name))
    if match is None:
        raise ValueError(f"No 12-digit COCO id in {path_or_name!r}")
    return int(match.group(1))


def candidate_names(image_id: int) -> tuple:
    return (
        f"{image_id:012d}.jpg",
        f"COCO_train2014_{image_id:012d}.jpg",
        f"COCO_val2014_{image_id:012d}.jpg",
    )


def discover_image_dirs(roots: Sequence, max_depth: int = 5) -> list:
    """Find folders named train2014 / val2014 / train2017 / val2017 under `roots`.

    Image folders are not descended into (they hold >100k files each).
    """
    found = []
    for root in roots:
        if not root or not os.path.isdir(root):
            continue
        stack = [(os.path.abspath(root), 0)]
        while stack:
            current, depth = stack.pop()
            try:
                entries = [e for e in os.scandir(current) if e.is_dir(follow_symlinks=True)]
            except OSError:
                continue
            for entry in entries:
                if entry.name in IMAGE_FOLDERS:
                    found.append(entry.path)
                    nested = os.path.join(entry.path, entry.name)   # e.g. train2017/train2017/*.jpg
                    if os.path.isdir(nested):
                        found.append(nested)
                elif depth < max_depth and not entry.name.startswith("."):
                    stack.append((entry.path, depth + 1))
    return sorted(set(found))


class CocoImages:
    """Locate COCO images by id across several folders; download missing ones."""

    def __init__(self, roots: Sequence = (), download_dir: str | None = None):
        self.download_dir = download_dir
        search_roots = list(roots) + ([download_dir] if download_dir else [])
        self.dirs = discover_image_dirs(search_roots)
        self._cache: dict = {}
        self._lock = threading.Lock()

    def find(self, image_id: int) -> str | None:
        """Path of the image file, or None if it is not on disk."""
        image_id = int(image_id)
        if image_id in self._cache:
            return self._cache[image_id]
        for folder in self.dirs:
            for name in candidate_names(image_id):
                path = os.path.join(folder, name)
                if os.path.isfile(path):
                    self._cache[image_id] = path
                    return path
        return None

    def path(self, image_id: int) -> str:
        found = self.find(image_id)
        if found is None:
            raise FileNotFoundError(
                f"COCO image {image_id} not found in {self.dirs or 'any folder'}. "
                "Run scripts/00_download_data.py or pass --coco-roots."
            )
        return found

    def _download_one(self, image_id: int, folder: str, retries: int = 4) -> str:
        """Download one image from the official COCO server into download_dir."""
        if folder.endswith("2014"):
            name = f"COCO_{folder}_{image_id:012d}.jpg"
        else:
            name = f"{image_id:012d}.jpg"
        target_dir = ensure_dir(os.path.join(self.download_dir, folder))
        target = os.path.join(target_dir, name)
        if os.path.isfile(target) and os.path.getsize(target) > 0:
            return target
        url = COCO_IMAGE_URL.format(folder=folder, name=name)
        last_error = None
        for attempt in range(retries):
            try:
                tmp = f"{target}.part{threading.get_ident()}"
                with urllib.request.urlopen(url, timeout=60) as response, open(tmp, "wb") as out:
                    out.write(response.read())
                if os.path.getsize(tmp) == 0:
                    raise IOError("empty download")
                os.replace(tmp, target)
                return target
            except Exception as err:  # network hiccup: wait and retry
                last_error = err
                time.sleep(2 ** attempt)
        raise IOError(f"Could not download {url}: {last_error}")

    def ensure(self, ids_to_folder: dict, workers: int = 16, progress: bool = True) -> dict:
        """Make sure every image is on disk.

        `ids_to_folder` maps image id -> the official folder it can be downloaded
        from ("train2014", "val2014" or "val2017"). Returns {"present": n, "downloaded": n}.
        """
        missing = {i: f for i, f in ids_to_folder.items() if self.find(i) is None}
        stats = {"present": len(ids_to_folder) - len(missing), "downloaded": 0}
        if not missing:
            return stats
        if not self.download_dir:
            raise FileNotFoundError(
                f"{len(missing)} COCO images are missing and no download_dir was given."
            )
        for folder in set(missing.values()):
            path = ensure_dir(os.path.join(self.download_dir, folder))
            if path not in self.dirs:
                self.dirs.append(path)
        iterator = None
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(self._download_one, i, f): i for i, f in missing.items()}
            iterator = as_completed(futures)
            if progress:
                from tqdm import tqdm

                iterator = tqdm(iterator, total=len(futures), desc="Downloading COCO images")
            for future in iterator:
                image_id = futures[future]
                self._cache[image_id] = future.result()
                stats["downloaded"] += 1
        return stats


def small_copy_path(folder: str, image_id: int) -> str:
    return os.path.join(folder, f"{int(image_id):012d}.jpg")


def _make_small_copy(source: str, target: str, shorter_side: int) -> None:
    from PIL import Image

    with Image.open(source) as image:
        image = image.convert("RGB")
        scale = shorter_side / min(image.size)
        if scale < 1:
            image = image.resize((round(image.width * scale), round(image.height * scale)), Image.BICUBIC)
        tmp = f"{target}.part{threading.get_ident()}"
        image.save(tmp, format="JPEG", quality=95, subsampling=0)
    os.replace(tmp, target)


def make_small_copies(sources: dict, folder: str, shorter_side: int = 256, workers: int | None = None,
                      progress: bool = True) -> dict:
    """Write a reduced JPEG copy (shorter side `shorter_side` pixels) of every image.

    `sources` maps image id -> path of the original. Training reads every image once per
    epoch, and decoding full-size JPEGs is what limits the speed on a machine with few
    CPU cores. The random crop used in training is then taken from the reduced copy; the
    model's input is 224 pixels, so little is lost. Existing copies are kept.
    Returns {"present": n, "made": n}.
    """
    ensure_dir(folder)
    jobs = [(path, small_copy_path(folder, image_id)) for image_id, path in sources.items()]
    jobs = [(source, target) for source, target in jobs if not os.path.isfile(target)]
    stats = {"present": len(sources) - len(jobs), "made": 0}
    if not jobs:
        return stats
    with ThreadPoolExecutor(max_workers=workers or os.cpu_count() or 4) as pool:   # Pillow releases the GIL
        futures = [pool.submit(_make_small_copy, source, target, shorter_side) for source, target in jobs]
        iterator = as_completed(futures)
        if progress:
            from tqdm import tqdm

            iterator = tqdm(iterator, total=len(futures), desc="Making reduced image copies")
        for future in iterator:
            future.result()
            stats["made"] += 1
    return stats


def download_file(url: str, target: str, retries: int = 4, timeout: float = 60) -> str:
    """Download `url` to `target` unless it already exists."""
    if os.path.isfile(target) and os.path.getsize(target) > 0:
        return target
    ensure_dir(os.path.dirname(os.path.abspath(target)))
    last_error = None
    for attempt in range(retries):
        try:
            tmp = target + ".part"
            # the timeout applies to each network operation, so a slow but working download is fine
            with urllib.request.urlopen(url, timeout=timeout) as response, open(tmp, "wb") as out:
                shutil.copyfileobj(response, out, length=1 << 20)
            os.replace(tmp, target)
            return target
        except Exception as err:
            last_error = err
            time.sleep(2 ** attempt)
    raise IOError(f"Could not download {url}: {last_error}")


def load_karpathy(data_dir: str, split: str) -> list:
    """Load (downloading if needed) a Karpathy split file in the BLIP format.

    train: list of {"caption": str, "image": "train2014/COCO_train2014_....jpg", "image_id": ...}
    val/test: list of {"image": "val2014/COCO_val2014_....jpg", "caption": [str, ...]}
    """
    path = os.path.join(data_dir, f"coco_karpathy_{split}.json")
    download_file(KARPATHY_URL.format(split=split), path)
    return load_json(path)


def folder_of(karpathy_image_path: str) -> str:
    """'val2014/COCO_val2014_000000391895.jpg' -> 'val2014'."""
    return karpathy_image_path.split("/")[0]


def ids_with_folder(image_paths: Iterable) -> dict:
    """Map COCO id -> official download folder for Karpathy-style relative paths."""
    return {coco_id(p): folder_of(p) for p in image_paths}

"""Downstream checks: zero-shot classification and COCO retrieval.

These measure what fine-tuning costs. ARO (Table 6) reports that NegCLIP keeps
CLIP's zero-shot accuracy; "No Hard Negatives Required" (CVPR 2026, Table A6)
re-measured with one tool and found a clear drop. To be comparable with the
latter, zero-shot classification follows the same tool, CLIP_benchmark
(https://github.com/LAION-AI/CLIP_benchmark): its datasets, class names and prompt
templates. The classifier and the accuracy are computed here, mirroring
`clip_benchmark/metrics/zeroshot_classification.py` (whose own accuracy function
does not run with recent NumPy versions):

1. for every class, fill each prompt template with the class name, encode the
   prompts, L2-normalise, average, and L2-normalise again -> one vector per class;
2. an image is assigned to the class whose vector is most similar to its embedding.

COCO retrieval follows the usual protocol on the Karpathy test split (5,000 images,
25,000 captions), as in ARO's `COCO_Retrieval`.
"""
from __future__ import annotations

import csv
import json
import os
import random
import re

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .coco import CocoImages, coco_id
from .perturb import pre_caption
from .scoring import load_image

IMAGENET_FIRST, IMAGENET_LAST = "n01440764", "n15075141"     # first and last class id (sorted)


def packaged(name: str, kind: str) -> list:
    """Class names ("classnames") or prompt templates ("zeroshot_classification_templates")
    that CLIP_benchmark ships for a dataset."""
    try:
        import clip_benchmark
    except ImportError as err:
        raise RuntimeError("Install CLIP_benchmark:  pip install clip-benchmark") from err
    path = os.path.join(os.path.dirname(clip_benchmark.__file__), "datasets", f"en_{kind}.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)[name]


def imagenet_files(root: str, n_classes: int = 1000) -> list:
    """[(image path, class index)] of the ImageNet-1k validation set.

    Two folder layouts are recognised:
    * the Kaggle dataset "imagenet-object-localization-challenge": LOC_synset_mapping.txt,
      LOC_val_solution.csv and ILSVRC/Data/CLS-LOC/val/*.JPEG;
    * one folder per class: <root>/val/<class id>/*.JPEG (or the class folders directly in <root>).
    Class indices follow the sorted class ids, the order of CLIP_benchmark's class names.
    """
    mapping, solution = os.path.join(root, "LOC_synset_mapping.txt"), os.path.join(root, "LOC_val_solution.csv")
    if os.path.isfile(mapping) and os.path.isfile(solution):
        with open(mapping, "r", encoding="utf-8") as f:
            class_ids = [line.split()[0] for line in f if line.strip()]
        index = {class_id: i for i, class_id in enumerate(class_ids)}
        folder = os.path.join(root, "ILSVRC", "Data", "CLS-LOC", "val")
        with open(solution, "r", encoding="utf-8", newline="") as f:
            files = [(os.path.join(folder, row["ImageId"] + ".JPEG"), index[row["PredictionString"].split()[0]])
                     for row in csv.DictReader(f)]
    else:
        files, class_ids = [], []
        for folder in (os.path.join(root, "val"), root):
            if os.path.isdir(folder):
                class_ids = sorted(d for d in os.listdir(folder) if re.fullmatch(r"n\d{8}", d))
            if class_ids:
                for i, class_id in enumerate(class_ids):
                    for name in sorted(os.listdir(os.path.join(folder, class_id))):
                        files.append((os.path.join(folder, class_id, name), i))
                break
    if not files:
        raise FileNotFoundError(f"No ImageNet validation images found under {root} (see the docstring "
                                "of compbind.downstream.imagenet_files for the two supported layouts).")
    if len(class_ids) != n_classes or class_ids != sorted(class_ids):
        raise ValueError(f"Expected {n_classes} sorted class ids under {root}, found {len(class_ids)}.")
    if n_classes == 1000 and (class_ids[0], class_ids[-1]) != (IMAGENET_FIRST, IMAGENET_LAST):
        raise ValueError("The class ids are not those of ImageNet-1k (ILSVRC 2012).")
    return sorted(files)


class _Files(Dataset):
    def __init__(self, files: list, transform):
        self.files, self.transform = files, transform

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int):
        path, label = self.files[index]
        return self.transform(load_image(path)), label


def build_zero_shot_dataset(name: str, root: str, transform, limit: int | None = None, seed: int = 0):
    """Return (dataset, class names, templates) for a zero-shot test set."""
    if name == "imagenet1k":
        files = imagenet_files(root)
        data, classes = _Files(files, transform), packaged("imagenet1k", "classnames")
        templates = packaged("imagenet1k", "zeroshot_classification_templates")
    else:
        from clip_benchmark.datasets.builder import build_dataset

        data = build_dataset(dataset_name=name, root=root, transform=transform, split="test", download=True)
        classes, templates = data.classes, data.templates
    if limit is not None and limit < len(data):       # a fixed random subset (for a faster estimate)
        from torch.utils.data import Subset

        data = Subset(data, sorted(random.Random(seed).sample(range(len(data)), limit)))
    return data, list(classes), templates


@torch.no_grad()
def class_vectors(bundle, classes: list, templates, amp: bool = False) -> torch.Tensor:
    """One L2-normalised text vector per class, [n_classes, d]."""
    vectors = []
    for name in classes:
        prompts = templates[name] if isinstance(templates, dict) else [t.format(c=name) for t in templates]
        tokens = bundle.tokenizer(prompts).to(bundle.device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
            features = bundle.model.encode_text(tokens)
        mean = torch.nn.functional.normalize(features.float(), dim=-1).mean(dim=0)
        vectors.append(mean / mean.norm())
    return torch.stack(vectors)


@torch.no_grad()
def zero_shot(bundle, dataset: str, root: str, batch_size: int = 256, workers: int = 4,
              amp: bool = False, limit: int | None = None) -> dict:
    """Zero-shot top-1 / top-5 accuracy on cifar10, cifar100, imagenet1k or another CLIP_benchmark set."""
    amp = amp and str(bundle.device).startswith("cuda")
    bundle.model.eval()
    data, classes, templates = build_zero_shot_dataset(dataset, root, bundle.preprocess_eval, limit)
    classifier = class_vectors(bundle, classes, templates, amp)                 # [C, d]
    loader = DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=workers)
    top1 = top5 = total = 0
    for images, labels in loader:
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
            features = bundle.model.encode_image(images.to(bundle.device))
        scores = torch.nn.functional.normalize(features.float(), dim=-1) @ classifier.t()
        best = scores.topk(min(5, len(classes)), dim=1).indices.cpu()
        labels = torch.as_tensor(labels).view(-1, 1)
        top1 += int((best[:, :1] == labels).sum())
        top5 += int((best == labels).any(dim=1).sum())
        total += len(labels)
    return {"acc1": top1 / total, "acc5": top5 / total, "n_images": total, "n_classes": len(classes),
            "n_templates": len(templates) if isinstance(templates, list) else None,
            "subset": limit if limit is not None and limit <= total else None}


def _recall_at(ranks: np.ndarray, k: int) -> float:
    return float((ranks < k).mean())


def coco_retrieval(embedder, karpathy_test: list, images: CocoImages, max_words: int = 30) -> dict:
    """Recall@1/5/10 for image->text and text->image on the Karpathy test split."""
    image_keys, texts, text_to_image, image_to_texts = [], [], [], []
    for image_index, annotation in enumerate(karpathy_test):
        image_keys.append((images.path(coco_id(annotation["image"])), None))
        own = []
        for caption in annotation["caption"]:
            own.append(len(texts))
            texts.append(pre_caption(caption, max_words))
            text_to_image.append(image_index)
        image_to_texts.append(own)
    image_emb = torch.from_numpy(embedder.images(image_keys))
    text_emb = torch.from_numpy(embedder.texts(texts))
    sims = image_emb @ text_emb.t()                                   # [n_images, n_texts]

    # image -> text: rank of the best-ranked true caption
    order = sims.argsort(dim=1, descending=True).numpy()
    position = np.empty_like(order)
    rows = np.arange(order.shape[0])[:, None]
    position[rows, order] = np.arange(order.shape[1])[None, :]
    i2t_rank = np.array([position[i, own].min() for i, own in enumerate(image_to_texts)])

    # text -> image: rank of the caption's image
    target = torch.tensor(text_to_image)
    sims_t = sims.t()                                                 # [n_texts, n_images]
    true_scores = sims_t[torch.arange(len(texts)), target]
    t2i_rank = (sims_t > true_scores[:, None]).sum(dim=1).numpy()

    return {
        "text_r1": _recall_at(i2t_rank, 1), "text_r5": _recall_at(i2t_rank, 5),
        "text_r10": _recall_at(i2t_rank, 10),
        "image_r1": _recall_at(t2i_rank, 1), "image_r5": _recall_at(t2i_rank, 5),
        "image_r10": _recall_at(t2i_rank, 10),
        "n_images": len(image_keys), "n_texts": len(texts),
    }

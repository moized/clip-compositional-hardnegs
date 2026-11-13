"""Score "pick the right caption" test items with a CLIP model.

A test item is one image plus several candidate captions, exactly one of which is
correct. The model is right when the correct caption has the highest cosine
similarity to the image. Ties go to the first candidate, as `np.argmax` does in
the ARO code; SugarCrepe's rule (`positive > negative`) is the same thing when
the negative caption is listed first.

Every benchmark in this project (VG-Relation, VG-Attribution, COCO-Order,
SugarCrepe) is expressed as a list of `Item`s and scored by `score_items`, which
returns one row per item so that slices, confidence intervals and paired
comparisons can be computed later without running the model again.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .items import Item  # noqa: F401  (re-exported: scoring.Item)


def load_image(path: str, box: Optional[tuple] = None) -> Image.Image:
    image = Image.open(path).convert("RGB")
    if box is not None:
        x, y, w, h = box
        image = image.crop((x, y, x + w, y + h))
    return image


class _ImageKeys(Dataset):
    def __init__(self, keys: Sequence, transform):
        self.keys = list(keys)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, index: int):
        path, box = self.keys[index]
        return self.transform(load_image(path, box))


class Embedder:
    """Encodes images and texts with one CLIP model and remembers image embeddings."""

    def __init__(self, bundle, batch_size: int = 256, num_workers: int = 4, amp: bool = False,
                 progress: bool = True):
        self.bundle = bundle
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.amp = amp and str(bundle.device).startswith("cuda")
        self.progress = progress
        self._image_cache: dict = {}
        self._short_text_ok: Optional[bool] = None   # decided on first use

    def _autocast(self):
        return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.amp)

    @torch.no_grad()
    def images(self, keys: Sequence) -> np.ndarray:
        """L2-normalised embeddings for (path, box) keys, as float32 [n, d]."""
        missing = [k for k in dict.fromkeys(keys) if k not in self._image_cache]
        if missing:
            loader = DataLoader(_ImageKeys(missing, self.bundle.preprocess_eval),
                                batch_size=self.batch_size, shuffle=False,
                                num_workers=self.num_workers)
            if self.progress:
                from tqdm import tqdm

                loader = tqdm(loader, desc="Encoding images")
            position = 0
            for batch in loader:
                with self._autocast():
                    features = self.bundle.model.encode_image(batch.to(self.bundle.device))
                features = torch.nn.functional.normalize(features.float(), dim=-1).cpu().numpy()
                for row in features:
                    self._image_cache[missing[position]] = row
                    position += 1
        return np.stack([self._image_cache[k] for k in keys]).astype(np.float32)

    def _encode_text_short(self, tokens: torch.Tensor) -> torch.Tensor:
        """Text features computed on the used part of the 77-token window only.

        CLIP pads every caption to 77 tokens, but its text encoder attends only to
        earlier positions and reads the output at the end-of-text token, so the
        padding after that token cannot change the result. Dropping it gives the
        same features several times faster. `_check_short_text_path` verifies this
        on the loaded model before the shortcut is used.
        """
        model = self.bundle.model
        length = int((tokens != 0).sum(dim=1).max())
        tokens = tokens[:, :length]
        x = model.token_embedding(tokens)
        x = x + model.positional_embedding[:length].to(x.dtype)
        x = model.transformer(x, attn_mask=model.attn_mask[:length, :length])
        x = model.ln_final(x)
        x = x[torch.arange(x.shape[0], device=x.device), tokens.argmax(dim=-1)]
        projection = model.text_projection
        return projection(x) if isinstance(projection, torch.nn.Linear) else x @ projection

    def _check_short_text_path(self) -> bool:
        """True if the shortcut reproduces `encode_text` on this model (else fall back)."""
        probe = ["a photo of a cat", "the horse is eating the grass",
                 "a man in a red shirt standing next to a blue car on a sunny day"]
        try:
            model = self.bundle.model
            if getattr(model, "text_pool_type", "argmax") != "argmax":
                return False
            tokens = self.bundle.tokenizer(probe).to(self.bundle.device)
            full = model.encode_text(tokens).float()
            short = self._encode_text_short(tokens).float()
            return bool((full - short).abs().max() < 1e-3 * full.abs().max())
        except Exception:
            return False

    @torch.no_grad()
    def texts(self, texts: Sequence) -> np.ndarray:
        """L2-normalised embeddings for texts, as float32 [n, d]."""
        if self._short_text_ok is None:
            self._short_text_ok = self._check_short_text_path()
        out = []
        starts = range(0, len(texts), self.batch_size)
        if self.progress:
            from tqdm import tqdm

            starts = tqdm(starts, desc="Encoding texts")
        for start in starts:
            tokens = self.bundle.tokenizer(list(texts[start:start + self.batch_size]))
            tokens = tokens.to(self.bundle.device)
            with self._autocast():
                if self._short_text_ok:
                    features = self._encode_text_short(tokens)
                else:
                    features = self.bundle.model.encode_text(tokens)
            out.append(torch.nn.functional.normalize(features.float(), dim=-1).cpu().numpy())
        return np.concatenate(out).astype(np.float32)


def shuffled_image_index(image_ids: Sequence, seed: int = 0) -> np.ndarray:
    """For each item, the index of another item whose source image is different.

    Items are grouped by source image, the groups are put in random order, and
    every item takes the image of the item half the list away. Used for the
    shuffled-image control.
    """
    image_ids = np.asarray(image_ids)
    n = len(image_ids)
    unique, inverse = np.unique(image_ids, return_inverse=True)
    if len(unique) < 2:
        raise ValueError("Need at least two different images for the shuffled-image control.")
    rng = np.random.RandomState(seed)
    rank = rng.permutation(len(unique))[inverse]            # random position of each item's image
    order = np.lexsort((np.arange(n), rank))                # items sorted by image, images shuffled
    partner = np.empty(n, dtype=np.int64)
    partner[order] = order[(np.arange(n) + n // 2) % n]
    clash = image_ids[partner] == image_ids
    for i in np.where(clash)[0]:                            # rare: one image holds half the items
        for j in order:
            if image_ids[j] != image_ids[i]:
                partner[i] = j
                break
    return partner


def score_items(embedder: Embedder, items: Sequence, shuffle_seed: int = 0) -> pd.DataFrame:
    """Score items and return one row per item.

    Columns: item_id, group, image_id, label, pred, correct, margin, s0..s{K-1},
    and pred_shuffled / correct_shuffled for the shuffled-image control (the same
    captions scored against a different image).
    `margin` is the true caption's score minus the best wrong caption's score.
    """
    if not items:
        raise ValueError("No items to score.")
    k = len(items[0].captions)
    if any(len(item.captions) != k for item in items):
        raise ValueError("All items of one benchmark must have the same number of captions.")
    image_keys = [(item.image_path, item.box) for item in items]
    image_emb = embedder.images(image_keys)                                   # [n, d]
    all_texts = [caption for item in items for caption in item.captions]
    unique_texts = list(dict.fromkeys(all_texts))
    text_emb = embedder.texts(unique_texts)
    lookup = {text: i for i, text in enumerate(unique_texts)}
    text_index = np.array([lookup[t] for t in all_texts]).reshape(len(items), k)
    option_emb = text_emb[text_index]                                         # [n, k, d]
    scores = np.einsum("nd,nkd->nk", image_emb, option_emb)
    labels = np.array([item.label for item in items])
    pred = scores.argmax(axis=1)

    wrong = scores.copy()
    wrong[np.arange(len(items)), labels] = -np.inf
    margin = scores[np.arange(len(items)), labels] - wrong.max(axis=1)

    image_ids = [item.image_id or item.image_path for item in items]
    partner = shuffled_image_index(image_ids, seed=shuffle_seed)
    scores_shuffled = np.einsum("nd,nkd->nk", image_emb[partner], option_emb)
    pred_shuffled = scores_shuffled.argmax(axis=1)

    table = pd.DataFrame({
        "item_id": [item.item_id for item in items],
        "group": [item.group for item in items],
        "image_id": image_ids,
        "label": labels,
        "pred": pred,
        "correct": (pred == labels).astype(int),
        "margin": margin.astype(np.float32),
        "pred_shuffled": pred_shuffled,
        "correct_shuffled": (pred_shuffled == labels).astype(int),
    })
    for j in range(k):
        table[f"s{j}"] = scores[:, j].astype(np.float32)
    return table

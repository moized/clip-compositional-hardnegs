"""Fine-tune CLIP ViT-B/32 on COCO pairs with a chosen kind of negative caption.

Conditions (`--negatives`):
  none         plain CLIP fine-tuning, no extra negatives           (control, "CLIP-FT")
  random       one caption of another image as an extra negative    (control for "more negatives")
  swap         NegCLIP-style swapped captions                       (reproduced baseline)
  swap_nosym   swapped captions without meaning-preserving swaps    (ablation)
  swap_fluent  fluency-matched swapped captions                     (the improvement)

Every condition sees the same images and captions in the same number of steps, with
the same batch size and optimiser. Optimiser settings follow the NegCLIP training
code (github.com/vinid/neg_clip, an open_clip fork): AdamW with betas (0.9, 0.98),
eps 1e-6, weight decay 0.2 on weight matrices only, linear warm-up then cosine
decay, mixed precision, all layers trainable, temperature clamped at 100.
NegCLIP's additional hard negative *images* are not used here.
"""
from __future__ import annotations

import json
import math
import os
import random
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from . import CONDITIONS
from .coco import CocoImages
from .losses import batch_diagnostics, clip_loss, negclip_loss
from .models import activation_name, load_clip, save_checkpoint
from .scoring import load_image
from .utils import ensure_dir, save_json, seed_everything


class PairDataset(Dataset):
    """(image, caption, negative caption) triples for one training condition."""

    def __init__(self, table: pd.DataFrame, images: CocoImages, transform, negatives: str):
        if negatives not in CONDITIONS:
            raise ValueError(f"negatives must be one of {CONDITIONS}")
        self.paths = [images.path(int(i)) for i in table["image_id"]]
        self.captions = table["caption"].tolist()
        self.negatives = negatives
        self.transform = transform
        if negatives == "swap_fluent":
            self.choices = [[text] for text in table["neg_fluent"]]
        elif negatives in ("swap", "swap_nosym"):
            column = "neg_swap" if negatives == "swap" else "neg_swap_nosym"
            self.choices = [json.loads(texts) for texts in table[column]]
            if any(len(c) == 0 for c in self.choices):
                raise ValueError(f"Some captions have no negative in {column}.")
        else:
            self.choices = None

    def __len__(self) -> int:
        return len(self.captions)

    def negative_for(self, index: int) -> str:
        """A negative caption for item `index` (a new random choice on every call)."""
        if self.negatives == "none":
            return ""
        if self.negatives == "random":
            other = random.randrange(len(self.captions) - 1)
            return self.captions[other if other < index else other + 1]
        return random.choice(self.choices[index])

    def __getitem__(self, index: int):
        image = self.transform(load_image(self.paths[index]))
        return image, self.captions[index], self.negative_for(index)


def cosine_lr(step: int, base_lr: float, warmup: int, total_steps: int) -> float:
    """Learning rate at `step`: linear warm-up, then cosine decay to zero (as in open_clip)."""
    if step < warmup:
        return base_lr * (step + 1) / warmup
    progress = (step - warmup) / max(total_steps - warmup, 1)
    return 0.5 * (1 + math.cos(math.pi * progress)) * base_lr


def make_optimizer(model, lr: float, weight_decay: float):
    """AdamW; no weight decay on biases, norm layers and the temperature (open_clip rule)."""
    def no_decay(name: str, param) -> bool:
        return param.ndim < 2 or "bn" in name or "ln" in name or "bias" in name or "logit_scale" in name

    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    return torch.optim.AdamW(
        [{"params": [p for n, p in named if no_decay(n, p)], "weight_decay": 0.0},
         {"params": [p for n, p in named if not no_decay(n, p)], "weight_decay": weight_decay}],
        lr=lr, betas=(0.9, 0.98), eps=1e-6)


def make_grad_scaler(enabled: bool):
    """Gradient scaler for mixed precision (the constructor moved between torch versions)."""
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def encode(bundle, images: torch.Tensor, texts: list, amp: bool):
    """Normalised image and text features (with gradients)."""
    tokens = bundle.tokenizer(texts).to(bundle.device)
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
        image_features = bundle.model.encode_image(images.to(bundle.device, non_blocking=True))
        text_features = bundle.model.encode_text(tokens)
    image_features = torch.nn.functional.normalize(image_features.float(), dim=-1)
    text_features = torch.nn.functional.normalize(text_features.float(), dim=-1)
    return image_features, text_features


@torch.no_grad()
def validate(bundle, table: pd.DataFrame, images: CocoImages, batch_size: int, workers: int,
             amp: bool) -> dict:
    """Held-out check: does the model prefer the true caption over each kind of negative?

    pair_acc_fluent      true caption vs its fluency-matched negative
    pair_acc_swap        true caption vs its first NegCLIP-style negative
    pair_acc_swap_nosym  true caption vs its first non-symmetric swap
    retrieval_acc        image -> caption retrieval within a batch
    """
    bundle.model.eval()
    dataset = PairDataset(table, images, bundle.preprocess_eval, "none")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
    columns = {"fluent": table["neg_fluent"].tolist(),
               "swap": [json.loads(t)[0] for t in table["neg_swap"]],
               "swap_nosym": [json.loads(t)[0] for t in table["neg_swap_nosym"]]}
    correct = {name: 0 for name in columns}
    retrieval, seen, position = 0.0, 0, 0
    for batch_images, captions, _ in loader:
        n = len(captions)
        texts = list(captions)
        for name in columns:
            texts += columns[name][position:position + n]
        image_features, text_features = encode(bundle, batch_images, texts, amp)
        own = (image_features * text_features[:n]).sum(dim=1)
        for k, name in enumerate(columns):
            negative = (image_features * text_features[(k + 1) * n:(k + 2) * n]).sum(dim=1)
            correct[name] += int((own > negative).sum())
        retrieval += batch_diagnostics(image_features, text_features[:n])["retrieval_acc"] * n
        seen += n
        position += n
    out = {f"pair_acc_{name}": correct[name] / max(seen, 1) for name in columns}
    out["retrieval_acc"] = retrieval / max(seen, 1)
    out["n"] = seen
    return out


def train(pairs: pd.DataFrame, images: CocoImages, out_dir: str, negatives: str, seed: int = 0,
          epochs: int = 5, batch_size: int = 256, lr: float = 1e-6, weight_decay: float = 0.2,
          warmup: int = 50, workers: int = 4, device: str = "cuda", amp: bool = True,
          max_steps: int | None = None, log_every: int = 20, save_model: bool = True,
          run_info: dict | None = None) -> dict:
    """Fine-tune and return a summary (also written to <out_dir>/train_summary.json)."""
    seed_everything(seed)
    ensure_dir(out_dir)
    amp = amp and device.startswith("cuda")
    bundle = load_clip("openai", device=device, quick_gelu=True)
    model = bundle.model
    train_table = pairs[pairs["split"] == "train"].reset_index(drop=True)
    val_table = pairs[pairs["split"] == "val"].reset_index(drop=True)
    dataset = PairDataset(train_table, images, bundle.preprocess_train, negatives)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True,
                        num_workers=workers, generator=generator,
                        persistent_workers=workers > 0, pin_memory=device.startswith("cuda"))
    steps_per_epoch = len(loader)
    if steps_per_epoch == 0:
        raise ValueError(f"Batch size {batch_size} is larger than the {len(dataset)} training pairs.")
    total_steps = steps_per_epoch * epochs
    if max_steps is not None:
        total_steps = min(total_steps, max_steps)
    optimizer = make_optimizer(model, lr, weight_decay)
    scaler = make_grad_scaler(amp) if device.startswith("cuda") else None
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    history = [{"epoch": 0, **validate(bundle, val_table, images, batch_size, workers, amp)}] \
        if len(val_table) else []
    log_path = os.path.join(out_dir, "train_log.jsonl")
    step, losses = 0, []
    train_seconds, validation_seconds = 0.0, 0.0     # training time excludes validation
    with open(log_path, "w", encoding="utf-8") as log:
        for epoch in range(epochs):
            model.train()
            epoch_start = time.time()
            for batch_images, captions, negative_captions in loader:
                if step >= total_steps:
                    break
                current_lr = cosine_lr(step, lr, warmup, total_steps)
                for group in optimizer.param_groups:
                    group["lr"] = current_lr
                texts = list(captions)
                if negatives != "none":
                    texts += list(negative_captions)
                image_features, text_features = encode(bundle, batch_images, texts, amp)
                n = len(captions)
                scale = model.logit_scale.exp()
                if negatives == "none":
                    loss = clip_loss(image_features, text_features, scale)
                else:
                    loss = negclip_loss(image_features, text_features[:n], text_features[n:], scale)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Loss became {float(loss.detach())} at step {step}.")
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None and amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                with torch.no_grad():
                    model.logit_scale.clamp_(0, math.log(100))
                losses.append(loss.item())
                step += 1
                if step % log_every == 0 or step == total_steps:
                    elapsed = train_seconds + time.time() - epoch_start
                    record = {"step": step, "epoch": epoch + 1, "loss": float(np.mean(losses[-log_every:])),
                              "lr": current_lr, "seconds": round(elapsed, 1)}
                    log.write(json.dumps(record) + "\n")
                    log.flush()
                    print(f"[{negatives} seed {seed}] step {step}/{total_steps} "
                          f"loss {record['loss']:.4f} lr {current_lr:.2e} {record['seconds']:.0f}s")
            train_seconds += time.time() - epoch_start
            if len(val_table):
                validation_start = time.time()
                history.append({"epoch": epoch + 1,
                                **validate(bundle, val_table, images, batch_size, workers, amp)})
                validation_seconds += time.time() - validation_start
                print(f"[{negatives} seed {seed}] epoch {epoch + 1} validation: {history[-1]}")
            if step >= total_steps:
                break
    summary = {
        "negatives": negatives, "seed": seed, "epochs": epochs, "batch_size": batch_size, "lr": lr,
        "weight_decay": weight_decay, "warmup": warmup, "amp": amp, "steps": step,
        "steps_per_epoch": steps_per_epoch, "train_pairs": int(len(train_table)),
        "val_pairs": int(len(val_table)), "activation": activation_name(model),
        "texts_per_step": batch_size * (1 if negatives == "none" else 2),
        "train_seconds": round(train_seconds, 1),
        "validation_seconds": round(validation_seconds, 1),
        "seconds_per_step": round(train_seconds / max(step, 1), 4),
        "peak_gpu_memory_mb": round(torch.cuda.max_memory_allocated() / 2 ** 20, 1)
        if device.startswith("cuda") else None,
        "final_loss": float(np.mean(losses[-log_every:])) if losses else None,
        "validation": history,
    }
    if run_info:
        summary["run"] = run_info
    if save_model:
        checkpoint = os.path.join(out_dir, "model.pt")
        save_checkpoint(model, checkpoint, meta={"negatives": negatives, "seed": seed, "steps": step,
                                                 "lr": lr, "batch_size": batch_size})
        summary["checkpoint"] = checkpoint
    save_json(summary, os.path.join(out_dir, "train_summary.json"))
    return summary

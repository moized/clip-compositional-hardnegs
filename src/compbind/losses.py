"""Contrastive losses: CLIP, and CLIP with hard negative captions (NegCLIP).

Notation: a batch has N images and their N true captions. All feature vectors
are L2-normalised, so a dot product is a cosine similarity. `logit_scale` is
CLIP's learned temperature (about 100 for OpenAI CLIP); similarities are
multiplied by it before the softmax.

CLIP loss (Radford et al., 2021). Build the N x N similarity table. Row i should
pick caption i among the N captions (image -> text), and column i should pick
image i among the N images (text -> image). The loss is the average of the two
cross-entropies.

NegCLIP loss (Yuksekgonul et al., ICLR 2023, Section 4). Each caption also has a
hard negative caption (the same words with two elements swapped). The N negative
captions are appended to the captions, giving an N x 2N table:
* image -> text: row i must pick caption i among 2N captions, so it has to score
  the true caption above its own swapped version;
* text -> image: only the N true captions are used, because a negative caption
  has no image it belongs to.
This is the loss of the authors' open_clip fork (github.com/vinid/neg_clip),
without their additional hard negative images.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def clip_loss(image_features: torch.Tensor, text_features: torch.Tensor,
              logit_scale: torch.Tensor) -> torch.Tensor:
    """Standard symmetric CLIP loss for N matching (image, caption) pairs."""
    logits = logit_scale * image_features @ text_features.t()          # [N, N]
    labels = torch.arange(logits.shape[0], device=logits.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)) / 2


def negclip_loss(image_features: torch.Tensor, text_features: torch.Tensor,
                 negative_text_features: torch.Tensor, logit_scale: torch.Tensor) -> torch.Tensor:
    """CLIP loss with one extra hard negative caption per image (NegCLIP)."""
    n = image_features.shape[0]
    all_text = torch.cat([text_features, negative_text_features], dim=0)       # [2N, d]
    logits_per_image = logit_scale * image_features @ all_text.t()             # [N, 2N]
    logits_per_text = logit_scale * text_features @ image_features.t()         # [N, N]
    labels = torch.arange(n, device=image_features.device)
    return (F.cross_entropy(logits_per_image, labels) + F.cross_entropy(logits_per_text, labels)) / 2


@torch.no_grad()
def batch_diagnostics(image_features: torch.Tensor, text_features: torch.Tensor,
                      negative_text_features: torch.Tensor | None = None) -> dict:
    """Simple accuracies on a batch, for logging.

    retrieval_acc   how often an image's own caption is the most similar of the N captions
    pair_acc        how often the true caption beats its own hard negative (if given)
    """
    sims = image_features @ text_features.t()
    labels = torch.arange(sims.shape[0], device=sims.device)
    out = {"retrieval_acc": float((sims.argmax(dim=1) == labels).float().mean())}
    if negative_text_features is not None:
        own = sims.diag()
        negative = (image_features * negative_text_features).sum(dim=1)
        out["pair_acc"] = float((own > negative).float().mean())
    return out

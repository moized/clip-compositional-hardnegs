"""Load CLIP ViT-B/32 variants through open_clip.

Three sources are supported:
* "openai"            - the original OpenAI CLIP weights;
* a checkpoint file   - the released NegCLIP checkpoint or one of our fine-tuned models.

Activation function. OpenAI's CLIP was trained with QuickGELU, and NegCLIP was
fine-tuned from it with the same activation. Recent open_clip versions build
"ViT-B-32" with the standard GELU and only warn about the mismatch, which silently
changes the numbers. The released evaluation code of ARO and SugarCrepe loads
NegCLIP as "ViT-B-32" from a file, i.e. with GELU. This loader therefore makes
the choice explicit: `quick_gelu=True` (default, matches training) or
`quick_gelu=False` (what those scripts do). Evaluate both and report which one
reproduces the published numbers.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

ARCH_QUICKGELU = "ViT-B-32-quickgelu"
ARCH_GELU = "ViT-B-32"


@dataclass
class ClipBundle:
    """A CLIP model with its image transforms and tokenizer."""

    model: Any
    preprocess_train: Any
    preprocess_eval: Any
    tokenizer: Any
    arch: str
    source: str
    device: str


def _numpy_safe_globals() -> list:
    """NumPy constructors that old training checkpoints contain besides tensors."""
    try:
        from numpy._core.multiarray import scalar  # NumPy >= 2
    except ImportError:  # NumPy 1.x
        from numpy.core.multiarray import scalar
    allow = [np.dtype, (scalar, "numpy.core.multiarray.scalar"),
             (scalar, "numpy._core.multiarray.scalar")]
    for name in ("float64", "float32", "int64", "int32", "bool"):
        allow.append(type(np.dtype(name)))
    return allow


def read_state_dict(path: str, trust_checkpoint: bool = False) -> dict:
    """Read model weights from a checkpoint file without running arbitrary code.

    Handles open_clip training checkpoints ({"state_dict": ...}, optional "module."
    prefix). `trust_checkpoint=True` falls back to full unpickling, which can
    execute code stored in the file; use it only for files you trust.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    try:
        if hasattr(torch.serialization, "safe_globals"):
            try:
                with torch.serialization.safe_globals(_numpy_safe_globals()):
                    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            except TypeError:  # older torch: tuples not accepted in the allow-list
                checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        else:
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as err:
        if not trust_checkpoint:
            raise RuntimeError(
                f"Could not load {path} in safe mode ({type(err).__name__}). "
                "If you trust this file, pass trust_checkpoint=True / --trust-checkpoint."
            ) from err
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    return {(k[len("module."):] if k.startswith("module.") else k): v for k, v in state.items()}


def load_clip(source: str = "openai", device: str = "cpu", quick_gelu: bool = True,
              trust_checkpoint: bool = False) -> ClipBundle:
    """Build ViT-B/32 and load weights from "openai" or from a checkpoint file."""
    import open_clip

    arch = ARCH_QUICKGELU if quick_gelu else ARCH_GELU
    if source == "openai":
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # the GELU/QuickGELU warning is handled by `arch`
            model, preprocess_train, preprocess_eval = open_clip.create_model_and_transforms(
                arch, pretrained="openai")
    else:
        model, preprocess_train, preprocess_eval = open_clip.create_model_and_transforms(
            arch, pretrained=None)
        result = model.load_state_dict(read_state_dict(source, trust_checkpoint), strict=True)
        assert not result.missing_keys and not result.unexpected_keys
    model = model.float().to(device).eval()
    tokenizer = open_clip.get_tokenizer(arch)
    return ClipBundle(model, preprocess_train, preprocess_eval, tokenizer, arch, source, device)


def activation_name(model: Any) -> str:
    """'QuickGELU' or 'GELU': which activation the built model really uses."""
    names = {type(m).__name__ for m in model.modules() if "GELU" in type(m).__name__}
    return "+".join(sorted(names))


def save_checkpoint(model: Any, path: str, meta: dict | None = None) -> None:
    """Save weights (float32, on CPU) plus plain-Python metadata."""
    state = {k: v.detach().float().cpu() if v.is_floating_point() else v.detach().cpu()
             for k, v in model.state_dict().items()}
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save({"state_dict": state, "meta": meta or {}}, path)

"""The common shape of a test case, shared by all benchmarks."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Item:
    """One "pick the right caption" test case."""

    item_id: str
    image_path: str
    captions: tuple                 # candidate captions
    label: int                      # index of the correct caption
    group: str = ""                 # e.g. relation name, attribute pair, subset
    image_id: str = ""              # source image, used as the resampling unit
    box: Optional[tuple] = None     # (x, y, w, h) crop, or None for the whole image

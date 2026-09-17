"""Squeeze and roll a rod-shaped object, with no tilt: `whisk.py` with
`Config.tilt=False`.

`tilt` is exactly the maneuver this task drops, so subclassing `whisk.Config`
to flip that one field and its `label`, then handing off to `whisk.build`
unchanged, is the whole variant -- see `whisk.py` for the mechanism, bench
notes, and every other field this inherits.
"""
from dataclasses import dataclass

import torch

from ..config import HandConfig
from ..primitives import Sequence
from . import whisk


@dataclass
class Config(whisk.Config):
    label: str = "Rotate object"
    tilt: bool = False


def build(hand: HandConfig, start_mm: torch.Tensor,
          cfg: Config | None = None, **kwargs) -> Sequence:
    return whisk.build(hand, start_mm, cfg=cfg or Config(), **kwargs)

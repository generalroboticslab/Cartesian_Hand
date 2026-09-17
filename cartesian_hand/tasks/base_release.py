"""Open the base jaw, releasing whatever it is holding.

    release base jaw

The counterpart to `base_grasp.py`. Not chained after a grasp within the same
run, so there is no measured radius to open relative to (`base_grasp.py`
never names one) -- just a `Move` to a fixed open position, `release_mm`.
`loaded=True` because the jaw is expected to still be resting against
whatever it is letting go of, so a confirmed stall on the way out is the
row's own finish, not a fault.

Every goal is absolute in the hand's millimetre frame, so the hand must be
zeroed first.
"""
from dataclasses import dataclass, field

import torch

from ..config import BASE_JAW, HandConfig
from ..primitives import Move, Sequence


@dataclass
class Config:
    """Bench units. Every `tune` field becomes a slider on the studio page."""

    label: str = "Base release"
    sets_datum: bool = False

    release_mm: float = field(default=50.0, metadata={"tune": (0.0, 55.0)})
    """Absolute base jaw goal to open to, mm."""
    release_torque: float = field(default=150.0, metadata={"tune": (50.0, 500.0)})
    """Base jaw effort while opening."""
    travel_torque: float = field(default=50.0, metadata={"tune": (30.0, 200.0)})
    """Unused: this task's one row names its own effort. Carried only
    because `Sequence` takes it."""
    approach_torque: float = field(default=150.0, metadata={"tune": (50.0, 300.0)})
    """Unused: this task never probes or creeps. Carried only because
    `Sequence` takes it."""
    approach_speed: float = field(default=800.0, metadata={"tune": (25.0, 800.0)})
    """Unused, same reason as `approach_torque`."""
    travel_speed: float = field(default=1500.0, metadata={"tune": (200.0, 1500.0)})
    """Unused: every free move now runs at the servo's rated no-load top
    speed regardless of this value."""
    timeout_margin: float = field(default=1.5, metadata={"tune": (1.0, 3.0)})


def build(hand: HandConfig, start_mm: torch.Tensor,
          cfg: Config | None = None) -> Sequence:
    """Open the base jaw to `release_mm`."""
    cfg = cfg or Config()
    mm = hand.clamped_mm
    release_effort = cfg.release_torque / 1000.0

    return Sequence([
        Move(label="release", goal={BASE_JAW: mm(BASE_JAW, cfg.release_mm)},
             effort=release_effort, loaded=True),
    ], hand=hand,
       start_mm=start_mm,
       travel_torque=cfg.travel_torque,
       approach_torque=cfg.approach_torque,
       approach_speed=cfg.approach_speed,
       timeout_margin=cfg.timeout_margin,
       travel_speed=cfg.travel_speed)

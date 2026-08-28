"""Range-of-motion sweep. The first thing to run after zeroing."""

import time
from dataclasses import dataclass

DESCRIPTION = "Sweep every DOF through its full travel"


@dataclass
class Config:
    cycles: int = 1
    """Number of full min-to-max sweeps."""
    dwell: float = 0.5
    """Seconds to pause at each end of travel."""


def demo(hand, cycles: int = 1, dwell: float = 0.5):
    hand.require_zeroed()
    for i in range(cycles):
        print(f"[{hand.name}] sweep {i + 1}/{cycles}: to max")
        hand.set_pos(hand.config.upper)
        time.sleep(dwell)
        print(f"[{hand.name}] sweep {i + 1}/{cycles}: to min")
        hand.set_pos(hand.config.lower)
        time.sleep(dwell)
    print("demo complete.")


def run(hand, cfg: Config):
    try:
        demo(hand, cycles=cfg.cycles, dwell=cfg.dwell)
    finally:
        hand.release()

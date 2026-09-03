"""Unscrew a bottle cap, sizing the grip from contact rather than from a
measurement typed in by the operator.

The base jaw holds the bottle body while the aux jaw grips the cap. The aux
fingers stroke sideways to rotate the cap, releasing and re-gripping between
strokes because one stroke covers less than a full turn.

This module is a shim over `caps_motions`. The motion-program implementation
is the canonical one and lives at `caps_motions.py`; this file keeps the
historical CLI entry point alive.
"""
from dataclasses import dataclass

from .caps_motions import Config as CapsMotionsConfig, run as motions_run

DESCRIPTION = "Unscrew a bottle cap using contact-based sizing (alias of caps_motions)"


@dataclass
class Config:
    """Legacy config, kept for the existing CLI flags."""
    cap_offset: float = 20.0
    num_revs: float = 3.0
    squeeze_torque: int = 80
    approach_torque: int = 150
    approach_speed: int = 50
    release_clearance: float = 1.0
    speed: int = None
    acc: int = None
    settle_time: float = 1.0


def run(hand, cfg: Config):
    """Delegate to caps_motions.run with the same parameters."""
    mcfg = CapsMotionsConfig(
        cap_offset=cfg.cap_offset,
        num_revs=cfg.num_revs,
        squeeze_torque=cfg.squeeze_torque,
        approach_torque=cfg.approach_torque,
        travel_torque=50.0,
        release_clearance=cfg.release_clearance,
        move_timeout_s=6.0,
        probe_timeout_s=cfg.settle_time + 10.0,
    )
    motions_run(hand, mcfg)

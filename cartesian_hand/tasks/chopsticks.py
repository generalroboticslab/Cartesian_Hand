"""Grip a chopstick once, then cycle z (DOF 3) up and back down to squeezed.

    from ready: pull all four fingers inward
    close DOF 0 and DOF 4 onto the chopstick, once
    close z (DOF 3) to its squeezed position, once
    raise z all the way up, then lower it back to squeezed

Fingers and DOF 0/4 are one-time setup: closed once and never named again --
the grip on the chopstick is never released. Every button press after the
first finds fingers, jaws and z already at that setup pose and skips straight
to the z up/down cycle -- see `build`'s docstring for how that is detected.

**Every `Move` past "fingers in" is `loaded`, with its own explicit
deadline.** Bench runs on the previous version of this task (open/close on
DOF 0/4 instead of z) stalled short of goal against real resistance -- z on
the way down, then the jaws against whatever the chopstick pushed back with --
and `loaded` alone did not save it: `loaded` makes a confirmed stall count as
arrival instead of a fault, but the row still has to reach that confirmed
stall inside its deadline first, and the auto-derived margin
(`Sequence._deadline`, sized off the servo's no-load top speed) is well under
what real, resisted travel takes. `move_seconds` covers the one-time setup
moves (jaws closed, z closed); `z_cycle_seconds` covers "z up"/"z down" --
sized to the ~3 s a full up-then-down pass takes at that same real, load-slowed
speed, per the operator who timed it.

No hardware results for this task yet beyond those earlier runs: this
sequence itself is written, not bench run.
"""

from dataclasses import dataclass, field

import torch

from ..config import (AUX_JAW, AUX_LEFT, AUX_RIGHT, BASE_JAW, BASE_LEFT,
                      BASE_RIGHT, HandConfig, Z)
from ..primitives import Move, Sequence, lift_effort

FINGERS = (BASE_LEFT, BASE_RIGHT, AUX_LEFT, AUX_RIGHT)
JAWS = (BASE_JAW, AUX_JAW)

SETUP_TOLERANCE_MM = 2.0
"""How close fingers, jaws and z must already be to their setup target for a
press to count as "already set up" and skip straight to the z up/down cycle.
Loose enough to absorb wherever a `Move`'s own arrival tolerance and any
settling drift left them, tight enough that a fresh `ready` pose (fingers at
mid travel, jaws open, z at mid travel) never reads as already set up."""


@dataclass
class Config:
    """Bench units. Every `tune` field becomes a slider on the studio page."""

    label: str = "Chopsticks"
    sets_datum: bool = False

    grip_torque: float = field(default=100.0, metadata={"tune": (30.0, 500.0)})
    """Torque driving DOF 0/4 closed onto the chopstick, once, at setup."""
    z_up_fraction: float = field(default=0.9, metadata={"tune": (0.5, 0.98)})
    """How far z (DOF 3) rises for "z up", as a fraction of its own travel
    table entry -- never 1.0: the travel table is uncalibrated CAD and reads
    high, so asking for the exact limit risks the same hard stop `zero` finds
    on purpose (`demo.py`'s `open_fraction` carries the same reasoning)."""
    z_cycle_seconds: float = field(default=3.0, metadata={"tune": (1.0, 10.0)})
    """Deadline for each of "z up" and "z down" -- see module docstring."""
    move_seconds: float = field(default=10.0, metadata={"tune": (2.0, 20.0)})
    """Deadline for the one-time "jaws closed" and "z closed" setup moves --
    see the module docstring for why these need longer than a plain `Move`'s
    auto-derived margin."""
    travel_torque: float = field(default=50.0, metadata={"tune": (30.0, 200.0)})
    """Torque for the one-time fingers-in setup move."""
    approach_torque: float = field(default=100.0, metadata={"tune": (50.0, 300.0)})
    """Unused: this task never probes or creeps. Carried only because
    `Sequence` takes it."""
    approach_speed: float = field(default=300.0, metadata={"tune": (25.0, 800.0)})
    """Unused, same reason as `approach_torque`."""
    timeout_margin: float = field(default=1.5, metadata={"tune": (1.0, 3.0)})


def build(hand: HandConfig, start_mm: torch.Tensor,
          cfg: Config | None = None) -> Sequence:
    """Set up once from ready, then run one z up/down cycle.

    `start_mm` is wherever the hand actually is when the button is pressed,
    not necessarily `ready`: the setup rows (fingers in, jaws closed, z
    closed) are only included when fingers, jaws and z are not already within
    `SETUP_TOLERANCE_MM` of that target, so the first press does the full
    sequence and every press after that -- fingers, jaws and z left exactly
    where the first press put them -- runs only the z up/down cycle.
    """
    cfg = cfg or Config()
    jaw_floor = max(float(hand.gain_vector("torque_min_to_move")[dof])
                    for dof in JAWS)
    grip_effort = max(cfg.grip_torque, jaw_floor) / 1000.0
    up_z = hand.travel_mm[Z] * cfg.z_up_fraction

    setup_dofs = list(FINGERS + JAWS + (Z,))
    already_set_up = bool(
        (start_mm[:, setup_dofs] <= SETUP_TOLERANCE_MM).all())

    rows = []
    if not already_set_up:
        rows += [
            Move(label="fingers in", goal={FINGERS: 0.0}),
            # `loaded` + `move_seconds`: never fail here -- see module
            # docstring. Whatever the jaws actually close onto the chopstick
            # at is the grip; nothing downstream re-measures it.
            Move(label="jaws closed", goal={JAWS: 0.0}, effort=grip_effort,
                 loaded=True, seconds=cfg.move_seconds),
            # Same reasoning, on the way down to z's own squeezed position.
            Move(label="z closed", goal={Z: 0.0}, loaded=True,
                 seconds=cfg.move_seconds),
        ]
    rows += [
        # `loaded` + `z_cycle_seconds`: never fail here either. Rising this
        # far legitimately risks z's own hard stop (`z_up_fraction` keeps it
        # short of the limit but not clear of it under load), which a
        # confirmed stall now reports as arrival instead of a fault.
        Move(label="z up", goal={Z: up_z}, effort=lift_effort(hand),
             loaded=True, seconds=cfg.z_cycle_seconds),
        Move(label="z down", goal={Z: 0.0}, loaded=True,
             seconds=cfg.z_cycle_seconds),
    ]

    return Sequence(rows, hand=hand,
                     start_mm=start_mm,
                     travel_torque=cfg.travel_torque,
                     approach_torque=cfg.approach_torque,
                     approach_speed=cfg.approach_speed,
                     timeout_margin=cfg.timeout_margin)


# =============================================================================
# ARCHIVED, 2026-09-09 -- previous design: fingers in, z closed, jaws closed,
# then cycle DOF 0/4 open and closed (the pinch itself did the repeated
# open/close; z was one-time setup). Superseded by a design where DOF 0/4
# grip once and never release, and z does the repeated up/down cycle instead.
# Kept for reference; not wired to `build` above. To restore: rename this
# `Config`/`build` pair back to the live names (and the ones above to
# something else or delete them).
# =============================================================================
#
# @dataclass
# class _ArchivedConfig:
#     """Bench units. Every `tune` field becomes a slider on the studio page."""
#
#     label: str = "Chopsticks"
#     sets_datum: bool = False
#
#     grip_torque: float = field(default=100.0, metadata={"tune": (30.0, 500.0)})
#     """Torque driving DOF 0/4 both closed and open each cycle."""
#     open_fraction: float = field(default=0.9, metadata={"tune": (0.5, 0.98)})
#     """How far DOF 0/4 open, as a fraction of their own travel table entry --
#     never 1.0: the travel table is uncalibrated CAD and reads high, so asking
#     for the exact limit risks the same hard stop `zero` finds on purpose
#     (`demo.py`'s `open_fraction` carries the same reasoning)."""
#     wait_s: float = field(default=1.5, metadata={"tune": (0.2, 5.0)})
#     """Seconds DOF 0/4 hold closed before opening again, each cycle."""
#     move_seconds: float = field(default=10.0, metadata={"tune": (2.0, 20.0)})
#     """Deadline for "z closed", "jaws closed" and "jaws open" -- long enough
#     that a `loaded` row always gets to either arrival or a confirmed stall
#     before timing out; see the module docstring."""
#     travel_torque: float = field(default=50.0, metadata={"tune": (30.0, 200.0)})
#     """Torque for the one-time fingers-in and z-closed setup moves."""
#     approach_torque: float = field(default=100.0, metadata={"tune": (50.0, 300.0)})
#     """Unused: this task never probes or creeps. Carried only because
#     `Sequence` takes it."""
#     approach_speed: float = field(default=300.0, metadata={"tune": (25.0, 800.0)})
#     """Unused, same reason as `approach_torque`."""
#     timeout_margin: float = field(default=1.5, metadata={"tune": (1.0, 3.0)})
#
#
# def _archived_build(hand: HandConfig, start_mm: torch.Tensor,
#           cfg: "_ArchivedConfig | None" = None) -> Sequence:
#     """Set up once from ready, then run one close/wait/open cycle on DOF 0/4.
#
#     `start_mm` is wherever the hand actually is when the button is pressed,
#     not necessarily `ready`: the setup rows (fingers in, z closed) are only
#     included when fingers and z are not already within `SETUP_TOLERANCE_MM`
#     of that target, so the first press does the full sequence and every press
#     after that -- fingers and z left exactly where the first press put them --
#     runs only the DOF 0/4 cycle.
#     """
#     cfg = cfg or _ArchivedConfig()
#     jaw_floor = max(float(hand.gain_vector("torque_min_to_move")[dof])
#                     for dof in JAWS)
#     grip_effort = max(cfg.grip_torque, jaw_floor) / 1000.0
#     open_mm = {dof: hand.travel_mm[dof] * cfg.open_fraction for dof in JAWS}
#
#     setup_dofs = list(FINGERS + (Z,))
#     already_set_up = bool(
#         (start_mm[:, setup_dofs] <= SETUP_TOLERANCE_MM).all())
#
#     rows = []
#     if not already_set_up:
#         rows += [
#             Move(label="fingers in", goal={FINGERS: 0.0}),
#             # `loaded` + `move_seconds`: never fail here -- see module
#             # docstring. Wherever z actually stops is fine; this is one-time
#             # setup, not a measurement anything downstream reads.
#             Move(label="z closed", goal={Z: 0.0}, loaded=True,
#                  seconds=cfg.move_seconds),
#         ]
#     rows += [
#         # `loaded` + `move_seconds`: never fail here either -- same reasoning,
#         # against whatever resistance the chopsticks themselves push back
#         # with, on both directions of the cycle.
#         Move(label="jaws closed", goal={JAWS: 0.0}, effort=grip_effort,
#              loaded=True, seconds=cfg.move_seconds),
#         Hold(label="wait closed", seconds=cfg.wait_s),
#         Move(label="jaws open", goal=open_mm, effort=grip_effort,
#              loaded=True, seconds=cfg.move_seconds),
#     ]
#
#     return Sequence(rows, hand=hand,
#                      start_mm=start_mm,
#                      travel_torque=cfg.travel_torque,
#                      approach_torque=cfg.approach_torque,
#                      approach_speed=cfg.approach_speed,
#                      timeout_margin=cfg.timeout_margin)

"""Fixed choreography: sweep every DOF, then run a continuous wave across the fingers.

    close all -> open all
      -> close fingers, half-open jaws and z
      -> repeat(continuous wave: fingers open and close in a rolling overlap)
      -> close all -> open all -> mid all

A visual demo, not a manipulation task: every goal is free space, so there is
nothing to probe and no grip to size from contact. `Hold(seconds=cfg.dwell)`
after each `Move` is the pause a person needs to see one phase land before the
next starts -- there is no separate `Wait` row; see `primitives.Hold`.

**Ported from `cartesian_hand_old/tasks/demo.py`, whose control surface no
longer exists.** That version drove `hand.set_pos` directly every tick from a
Python loop, so it could set its own speed (`DEMO_SPEED`), acceleration
(`DEMO_ACC`), and a deliberately short per-move wait (`DEMO_TIMEOUT = 0.3s`)
so pacing came from `dwell`/`wave_stroke`, not from waiting out arrival.
`Sequence` removed all three knobs from task code, for reasons `triggers.py`'s
docstring already covers in full: every free `Move` now runs at the servo's
rated no-load top speed regardless of what a task asks (`SERVO_NO_LOAD_RPM`),
acceleration is one hand-wide gain with no per-row override, and a `Move` is
closed-loop -- it blocks until it reaches its goal or times out, not for a
fixed duration. This file has no equivalent fields.

**The wave approximates continuous motion with many small steps, not a few
named waypoints.** The old `_wave` streamed an interpolated setpoint every
tick. `Move` rows have no per-tick setpoint channel, but nothing requires a
`Hold` between them either: each DOF's own open->close leg pair is split into
`WAVE_STEPS_PER_LEG` steps, and DOF `i` starts its leg one step after DOF
`i - 1` starts its own, so several `WAVE_DOFS` are mid-leg on the same frame.
Frames chain straight into each other with no dwell, so nothing pauses the
motion between them. Each wave `Move` also widens its own arrival tolerance to
`WAVE_TOLERANCE_MM`, looser than `Move`'s 1 mm default, so a waypoint is
flagged arrived while still coasting toward it rather than after decelerating
to a full stop -- see that constant's comment. `WAVE_STEPS_PER_LEG` is sized
to stay clear of `WAVE_TOLERANCE_MM` even at `open_fraction`'s tunable floor
(0.5) -- see its own comment -- so every step is real travel rather than an
instant no-op from a goal already inside tolerance. Still discrete steps, not
a continuous stream, but small, un-paused, and loosely-toleranced enough to
read as one continuous ripple rather than fingers handing off to each other.

**`open_fraction`, not `hand.upper()`.** `HandConfig.upper` warns not to
command it: the travel table is CAD and reads high, so asking for the exact
limit risks the same hard stop `zero` finds on purpose, under a torque this
task never claims. `ready.py`'s reset button has the same rule (`fraction`,
`jaw_fraction`); this task reuses it as a single knob across every DOF.
"""
from dataclasses import dataclass, field

import torch

from ..config import (AUX_JAW, AUX_LEFT, AUX_RIGHT, BASE_JAW, BASE_LEFT,
                      BASE_RIGHT, HandConfig, Z)
from ..primitives import Hold, Loop, Move, Sequence

ALL_DOFS = (BASE_JAW, BASE_LEFT, BASE_RIGHT, Z, AUX_JAW, AUX_LEFT, AUX_RIGHT)

# Jaws and z, opened to mid-travel while the fingers close for the wave.
HALF_OPEN_DOFS = (BASE_JAW, Z, AUX_JAW)

# Base and aux fingers, waved in this order.
WAVE_DOFS = (BASE_LEFT, BASE_RIGHT, AUX_RIGHT, AUX_LEFT)

# Steps per leg (closed->open, or open->closed) of the wave. Each DOF starts
# its own leg one step after the DOF before it -- see the module docstring's
# wave note. Bounded above by `WAVE_TOLERANCE_MM` below: at `open_fraction`'s
# tunable floor (0.5) the shortest finger travel is ~27.5 mm, and 15 steps
# keeps each one at ~1.8 mm, safely past that tolerance so a step is genuine
# travel rather than a goal already inside tolerance.
WAVE_STEPS_PER_LEG = 15

# Looser than `Move`'s 1 mm default so a waypoint reports arrival while still
# coasting toward it, not after decelerating to a full stop -- that stop/start
# at every one of `WAVE_STEPS_PER_LEG` waypoints is what reads as a stepped
# wave instead of one continuous ripple, and costs the settle time on top.
# Stays below the ~1.8 mm floor-case step size above (see that comment) so a
# step is still real travel, not a goal already inside tolerance.
WAVE_TOLERANCE_MM = 1.5


@dataclass
class Config:
    """Bench units. Every `tune` field becomes a slider on the studio page."""

    label: str = "Run demo"
    sets_datum: bool = False

    open_fraction: float = field(default=0.9, metadata={"tune": (0.5, 0.98)})
    """How far each DOF opens as a fraction of its travel table entry, never
    1.0 -- see the module docstring's `open_fraction` note."""
    dwell: float = field(default=0.5, metadata={"tune": (0.0, 3.0)})
    """Seconds to pause after each move lands, so a phase is visible before
    the next one starts."""
    cycles: float = field(default=1.0, metadata={"tune": (0.0, 10.0)})
    """Wave passes over `WAVE_DOFS`. Zero repeats until the task is
    stopped (disarm torque in Studio, or Ctrl-C headless), the same
    convention `cap` and `scissors` use for their own `cycles`."""
    travel_torque: float = field(default=50.0, metadata={"tune": (30.0, 200.0)})
    approach_torque: float = field(default=150.0, metadata={"tune": (50.0, 300.0)})
    approach_speed: float = field(default=800.0, metadata={"tune": (25.0, 800.0)})
    """Unused: this task never probes or creeps, so nothing in it commands the
    approach torque or speed. Carried only because `Sequence` takes them."""
    timeout_margin: float = field(default=1.5, metadata={"tune": (1.0, 3.0)})


def build(hand: HandConfig, start_mm: torch.Tensor,
          cfg: Config | None = None) -> Sequence:
    """Sweep every DOF through its range, then run a continuous wave over the fingers."""
    cfg = cfg or Config()
    open_mm = {dof: hand.travel_mm[dof] * cfg.open_fraction for dof in ALL_DOFS}
    mid_mm = {dof: open_mm[dof] / 2 for dof in ALL_DOFS}
    half_open_mm = {dof: open_mm[dof] / 2 for dof in HALF_OPEN_DOFS}
    cycles = float("inf") if cfg.cycles <= 0 else max(1, round(cfg.cycles))

    # Continuous wave: each DOF's own open->close leg pair is split into
    # WAVE_STEPS_PER_LEG steps, and DOF i starts its leg one step after DOF
    # i - 1 starts its own; see the module docstring's wave note. Frames chain
    # straight into the next Move with no Hold between them, so nothing
    # pauses the motion.
    wave_leg_steps = 2 * WAVE_STEPS_PER_LEG
    wave_frame_count = wave_leg_steps + len(WAVE_DOFS) - 1

    def wave_frac(step: int) -> float:
        """Fraction open (0 -> 1 -> 0) at `step` steps into one DOF's leg pair."""
        if step < WAVE_STEPS_PER_LEG:
            return (step + 1) / WAVE_STEPS_PER_LEG
        return 1 - (step - WAVE_STEPS_PER_LEG + 1) / WAVE_STEPS_PER_LEG

    wave = []
    for t in range(wave_frame_count):
        goal = {}
        for i, dof in enumerate(WAVE_DOFS):
            local = t - i
            if 0 <= local < wave_leg_steps:
                goal[dof] = wave_frac(local) * open_mm[dof]
        wave.append(Move(label=f"wave frame {t}", goal=goal,
                         tolerance_mm=WAVE_TOLERANCE_MM))

    return Sequence([
        Move(label="close all", goal={ALL_DOFS: 0.0}),
        Hold(seconds=cfg.dwell),
        Move(label="open all", goal=open_mm),
        Hold(seconds=cfg.dwell),
        Move(label="close fingers, half-open jaws and z",
             goal={**{dof: 0.0 for dof in WAVE_DOFS}, **half_open_mm}),
        Hold(seconds=cfg.dwell),
        Loop(count=cycles, rows=wave),
        Move(label="close all", goal={ALL_DOFS: 0.0}),
        Hold(seconds=cfg.dwell),
        Move(label="open all", goal=open_mm),
        Hold(seconds=cfg.dwell),
        Move(label="mid all", goal=mid_mm),
    ], hand=hand,
       start_mm=start_mm,
       travel_torque=cfg.travel_torque,
       approach_torque=cfg.approach_torque,
       approach_speed=cfg.approach_speed,
       timeout_margin=cfg.timeout_margin)

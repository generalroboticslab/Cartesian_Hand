"""Fixed choreography: sweep every DOF, then chase a wave across the fingers.

    close all -> open all
      -> close DOFs one at a time (aux fingers, base fingers, jaws, z)
      -> open jaws and z to mid-travel
      -> repeat(chase: fingers open and close in a rolling overlap)
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

**The wave is a chase, not a stagger.** The old `_wave` streamed an
interpolated setpoint every tick, so four fingers could each be mid-stroke at
once, staggered by a fraction of one leg. `Move` rows are discrete
goal-and-wait phases with no per-tick setpoint stream, so the closest
equivalent here is a handful of frames: open one finger, then on each later
frame open the next while closing the one before it. It reads as a rolling
wave across `WAVE_DOFS` but is four discrete positions per cycle, not a
continuous sweep.

**`open_fraction`, not `hand.upper()`.** `HandConfig.upper` warns not to
command it: the travel table is CAD and reads high, so asking for the exact
limit risks the same hard stop `zero` finds on purpose, under a torque this
task never claims. `ready.py`'s reset button has the same rule (`fraction`,
`jaw_fraction`); this task reuses it as a single knob across every DOF.
"""
from dataclasses import dataclass, field

import torch

from ..config import (AUX_JAW, AUX_LEFT, AUX_RIGHT, BASE_JAW, BASE_LEFT,
                      BASE_RIGHT, HandConfig, LABELS, Z)
from ..primitives import Hold, Loop, Move, Sequence

ALL_DOFS = (BASE_JAW, BASE_LEFT, BASE_RIGHT, Z, AUX_JAW, AUX_LEFT, AUX_RIGHT)

# Mechanical order for the one-at-a-time close: aux fingers, base fingers,
# jaws, z. A wrong order here closes a jaw on a finger.
CLOSE_ORDER = (AUX_RIGHT, AUX_LEFT, BASE_RIGHT, BASE_LEFT, AUX_JAW, BASE_JAW, Z)

# Jaws and z, opened to mid-travel after the ordered close.
HALF_OPEN_DOFS = (BASE_JAW, Z, AUX_JAW)

# Base and aux fingers, chased in this order after the jaws/z are parked.
WAVE_DOFS = (BASE_LEFT, BASE_RIGHT, AUX_RIGHT, AUX_LEFT)


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
    """Wave chase passes over `WAVE_DOFS`. Zero repeats until the task is
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
    """Sweep every DOF through its range, then chase a wave over the fingers."""
    cfg = cfg or Config()
    open_mm = {dof: hand.travel_mm[dof] * cfg.open_fraction for dof in ALL_DOFS}
    mid_mm = {dof: open_mm[dof] / 2 for dof in ALL_DOFS}
    half_open_mm = {dof: open_mm[dof] / 2 for dof in HALF_OPEN_DOFS}
    cycles = float("inf") if cfg.cycles <= 0 else max(1, round(cfg.cycles))

    close_one_at_a_time = []
    for dof in CLOSE_ORDER:
        close_one_at_a_time.append(
            Move(label=f"close {LABELS[dof]}", goal={dof: 0.0}))
        close_one_at_a_time.append(Hold(seconds=cfg.dwell))

    # Rolling chase: open the next finger while closing the one before it, so
    # at most two fingers move on any one frame -- the discrete stand-in for
    # the old continuous stagger; see the module docstring.
    wave = []
    for i, dof in enumerate(WAVE_DOFS):
        goal = {dof: open_mm[dof]}
        if i > 0:
            goal[WAVE_DOFS[i - 1]] = 0.0
        wave.append(Move(label=f"wave open {LABELS[dof]}", goal=goal))
        wave.append(Hold(seconds=cfg.dwell))
    wave.append(Move(label=f"wave close {LABELS[WAVE_DOFS[-1]]}",
                      goal={WAVE_DOFS[-1]: 0.0}))
    wave.append(Hold(seconds=cfg.dwell))

    return Sequence([
        Move(label="close all", goal={ALL_DOFS: 0.0}),
        Hold(seconds=cfg.dwell),
        Move(label="open all", goal=open_mm),
        Hold(seconds=cfg.dwell),
        *close_one_at_a_time,
        Move(label="half open jaws and z", goal=half_open_mm),
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

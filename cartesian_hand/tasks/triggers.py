"""Squeeze a housing steady, then repeat a trigger pull.

    fingers to zero -> height to trigger_offset -> squeeze housing
      -> repeat(open trigger -> pull trigger -> hold trigger -> release trigger)

Dof 1, 2, 5, 6 (`FINGERS`: both gripper's finger pairs) go to zero once and are
never named by a later row, so -- exactly like `cap`'s "no twist" branch
leaving a DOF alone once a row has set it -- they hold that standing command
for the rest of the task. Dof 3 (`Z`) goes to `trigger_offset` the same way:
one `Move`, then no later row touches it, so it holds height for the whole
run.

Dof 0 (`BASE_JAW`) then squeezes the housing and holds it, the same `Probe`
-then-`grip` pattern `cap` uses to size its own grip from contact rather than
a number. Dof 4 (`AUX_JAW`) is the trigger: each cycle opens it to `open_mm`
(the ready position), drives it closed, then backs it off to `open_mm` again
to reset before the next cycle.

**"Max speed" and "max torque" on the pull are two different mechanisms.**
Speed is not a `Row` field at all -- `Sequence` already pins every free move to
the servo's rated no-load top speed regardless of what a task asks for (see
`SERVO_NO_LOAD_RPM` in `primitives.py`), so the pull is already at max speed by
doing nothing. Torque **is** a `Row` field, so the pull sets `effort=1.0`
explicitly -- the top of the normalized [0, 1] range `_effort` would otherwise
cap at `travel_effort`. **"Max accel" has no third mechanism to reach for**:
the servo's acceleration ramp (`HandConfig.acc`) is one number for the whole
hand, set at `config.STANDARD_ACC`, and no `Row` exposes a per-row override --
so this task cannot raise it further without editing that shared table, which
is out of scope for one task's file.

`accept_stall=True` on the pull is why 3 seconds is a "try", not a guarantee:
a free `Move` without it reports a row that stops short of its goal as a
FAULT, which would retire the whole `Sequence` the instant the trigger meets
real resistance (see `Sequence`'s "one failed row retires the whole Sequence"
note). With it, a confirmed stop under load counts as done, the same as it
does for `cap`'s loaded moves.

`cycles` follows `cap`'s convention exactly: zero repeats until the task is
stopped (disarm torque in Studio, or Ctrl-C headless); a positive value runs
that many complete open/pull/release cycles, and the task's own state resets
only at the next run's initialization, never mid-loop.

Per-object bench settings recorded during tuning (distinct from this file's
`Config` defaults above):

- **generic spray bottle**: trigger_offset 30mm, squeeze_torque 700,
  open_mm 40, pull_seconds 3, hold_seconds 5.
- **fantick drill, torch flame**: trigger_offset 20mm.
- **alcohol spray, white board spray**: trigger_offset 0mm.
"""
from dataclasses import dataclass, field

import torch

from ..config import (AUX_JAW, AUX_LEFT, AUX_RIGHT, BASE_JAW, BASE_LEFT,
                      BASE_RIGHT, HandConfig, Z)
from ..primitives import Hold, Loop, Move, Probe, Sequence

FINGERS = (BASE_LEFT, BASE_RIGHT, AUX_LEFT, AUX_RIGHT)


@dataclass
class Config:
    """Bench units. Every `tune` field becomes a slider on the studio page."""

    label: str = "Pull trigger"
    sets_datum: bool = False

    trigger_offset: float = field(default=0.0, metadata={"tune": (0.0, 40.0)})
    """Height dof3 (`Z`) holds for the whole task, mm above z zero."""
    squeeze_torque: float = field(default=600.0, metadata={"tune": (100.0, 1000.0)})
    """Torque dof0 (`BASE_JAW`) squeezes the housing with and holds, servo
    units (0-1000). 700 is 70% of the servo's torque register."""
    pull_grip_torque: float = field(default=600.0,
                                    metadata={"tune": (100.0, 1000.0)})
    """Base jaw torque during the pull only, above `squeeze_torque`.

    The pull drives dof4 shut at `effort=1.0` against a trigger spring, and
    every newton of that is a reaction pushing the housing out of the base
    jaw. So the base grip is braced for the pull and dropped back after --
    `brace` and `unbrace` in `build`, one-tick `Hold`s that change nothing but
    the cap, the same pattern `pipette` uses around its ejector press.

    Why the number may go so much higher than `pipette.eject_grip_torque`,
    which stops at 520: that task's base jaw is commanded fully shut on a thin
    pipette body, so it is saturated and the cap IS the current -- past ~550
    the firmware clears TORQUE_ENABLE within 3s and the trip LATCHES. A
    trigger housing is fat enough that the jaw closes most of the way onto it,
    so the position error is small, the current follows the error rather than
    the cap, and the cap is a ceiling that is never reached. That is also why
    `squeeze_torque` already defaults to 700, well past pipette's wall.

    If a run goes limp mid-pull that assumption was wrong for this housing --
    the jaw IS saturating -- and this wants to come down under 520, not up."""
    open_mm: float = field(default=45.0, metadata={"tune": (0.0, 50.0)})
    """Dof4 (`AUX_JAW`) position between pulls -- the trigger's ready/reset
    point."""
    pull_seconds: float = field(default=3.0, metadata={"tune": (0.5, 10.0)})
    """How long the pull tries to close dof4 before the row ends, whether or
    not it arrives or stalls first."""
    hold_seconds: float = field(default=5.0, metadata={"tune": (0.0, 15.0)})
    """How long dof4 (`AUX_JAW`) stays clamped shut after the pull, before
    releasing."""
    travel_torque: float = field(default=50.0, metadata={"tune": (30.0, 200.0)})
    approach_torque: float = field(default=150.0, metadata={"tune": (50.0, 300.0)})
    travel_speed: float = field(default=1500.0, metadata={"tune": (200.0, 1500.0)})
    """Unused: every free move now runs at the servo's rated no-load top
    speed regardless of this value -- see the module docstring."""
    approach_speed: float = field(default=800.0, metadata={"tune": (25.0, 800.0)})
    """Servo speed register while closing the base jaw to find the housing,
    counts/s."""
    timeout_margin: float = field(default=1.5, metadata={"tune": (1.0, 3.0)})
    cycles: float = field(default=1.0, metadata={"tune": (0.0, 10.0)})
    """Complete open/pull/release cycles, rounded to an integer. Zero repeats
    until the task is stopped (disarm torque in Studio, or Ctrl-C headless)."""


def build(hand: HandConfig, start_mm: torch.Tensor,
          cfg: Config | None = None) -> Sequence:
    """Zero the fingers, hold height and housing grip, then repeat the pull."""
    cfg = cfg or Config()
    mm = hand.clamped_mm
    squeeze_effort = cfg.squeeze_torque / 1000.0
    pull_grip = cfg.pull_grip_torque / 1000.0
    trigger_height = mm(Z, cfg.trigger_offset)
    open_goal = mm(AUX_JAW, cfg.open_mm)
    cycles = float("inf") if cfg.cycles <= 0 else max(1, round(cfg.cycles))

    return Sequence([
        Move(label="fingers home", goal={FINGERS: 0.0}),
        Move(label="height", goal={Z: trigger_height}),
        Probe(label="squeeze housing", group=BASE_JAW, creep=True,
              grip=squeeze_effort),
        Loop(count=cycles, rows=[
            Move(label="open trigger", goal={AUX_JAW: open_goal}),
            # The pull's reaction force is what shoves the housing out of the
            # base jaw, so brace for it and release straight after. `unbrace`
            # before "release trigger", not after, because the boost must not
            # outlive the pull it exists for: a `Sequence` that retires keeps
            # reissuing its last action forever, so a run that ended braced
            # would park the jaw at the boosted cap for the rest of the task.
            # Both carry goal 0.0, which is where `squeeze housing` already
            # left the jaw -- these rows change the cap and nothing else.
            Hold(label="brace", group=BASE_JAW, effort=pull_grip),
            Move(label="pull trigger", goal={AUX_JAW: 0.0}, effort=1.0,
                 accept_stall=True, seconds=cfg.pull_seconds),
            Hold(label="hold trigger", group=AUX_JAW, goal=0.0, effort=1.0,
                 seconds=cfg.hold_seconds),
            Move(label="release trigger", goal={AUX_JAW: open_goal}),
            Hold(label="unbrace", group=BASE_JAW, effort=squeeze_effort),
        ]),
    ], hand=hand,
       start_mm=start_mm,
       travel_torque=cfg.travel_torque,
       approach_torque=cfg.approach_torque,
       approach_speed=cfg.approach_speed,
       timeout_margin=cfg.timeout_margin,
       travel_speed=cfg.travel_speed)

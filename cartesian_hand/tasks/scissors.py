"""Open a two-handle tool by raising the stage that holds one handle.

A two-handle tool is one articulated object: the two handles share a pivot,
and the open/close primitive is the `z` DOF separating the two stages while
each stage holds one handle. The tool's own pivot supplies the reaction a
single jaw has to take from a table.

The procedure:

    1. close both jaws on their respective handles, at holding torque
    2. lift the upper stage by `open_mm`, separating the handles
    3. close back down to a quiet hold (no closing through the tool's
       own stops -- it is at the open position)

The third step is what makes this a script and not an oscillation: the upper
jaw retracts the lift it just did, so the next call is reusable from any
resting height.
"""
from dataclasses import dataclass, field

import torch

from ..config import (AUX_FINGERS, AUX_JAW, BASE_FINGERS, BASE_JAW, HandConfig, Z)
from ..motions import Program, Result, Task


@dataclass
class Config:
    label: str = ""
    """Empty: the page already has cap and zero, and a third button would
    crowd the panel before the task has any hardware results to justify it.

    Reachable as `--task scissors` regardless. A future variant that names
    its own label inherits none, by the same rule as every other task."""
    sets_datum: bool = False

    grip_torque: float = field(default=80.0, metadata={"tune": (30.0, 200.0)})
    """Holding torque while a jaw owns its handle."""
    open_mm: float = field(default=15.0, metadata={"tune": (5.0, 40.0)})
    """How far the upper stage lifts the upper handle to open the tool."""
    open_timeout_s: float = field(default=4.0, metadata={"tune": (2.0, 12.0)})
    """Budget for every move here, not only the lift.

    Tunable, and it is the knob that matters most in simulation: a profiled
    setpoint crosses a 35 mm seat stroke in about 9.5 s at the configured speed
    (see `sim.step_limit_mm`), so 4.0 s expires mid-travel on every free-space
    row. Searching it is safe because the ranking is lexicographic -- a longer
    budget only wins if it converts a failed row into a passing one, and among
    passing candidates the shorter run wins."""
    travel_torque: float = field(default=50.0, metadata={"tune": (30.0, 200.0)})
    """Free-space moves: seating the fingers before the grip."""
    seat_stroke_mm: float = field(default=35.0, metadata={"tune": (10.0, 55.0)})
    """Finger sweep to seat the fingers around their handle before gripping.

    Built from the table's full finger travel (55 mm) minus the half that
    would close past the handle rather than around it. Sized once here so a
    task that knows nothing about the tool still closes onto *something*,
    not onto the model origin.
    """


def build(hand: HandConfig, start_mm: torch.Tensor, cfg: Config | None = None,
          **kwargs) -> Task:
    """Entry point for `--task scissors`. One program, returns a `Result`.

    No measurement step: a two-handle tool reports its open angle only after
    it has opened, by which point the task is over. `Result.value` is the
    joint pose at the end of the run (hand-internal coordinates); `ok` is
    `[N]` indicating whether the program ran to completion without any joint
    timing out, since a tool that the jaw missed gives no honest success
    signal beyond "the program did what it was told".
    """
    cfg = cfg or Config()
    n_envs, n_dof = start_mm.shape
    device = start_mm.device

    p = Program(n_envs, n_dof, hand.control_hz, device)
    travel = lambda dofs, goal: p.step().set(
        dofs, goal, cfg.travel_torque, "goal", cfg.open_timeout_s)

    # Seat the fingers so the grip has somewhere to land. Travel torque
    # because this is free space, and the fingers start wherever the hand
    # was -- a follower at the bottom of its rail would otherwise grip the
    # tool halfway up its own housing.
    travel(BASE_FINGERS, cfg.seat_stroke_mm)
    travel(AUX_FINGERS, 0.0)

    # Both jaws close on their handle. stop="stuck" because the handles are
    # the stop, and the grip torque is below the stall torque the seek used
    # so a successful close does not bind the gear train before contact.
    p.step().set([BASE_JAW, AUX_JAW], 0.0, cfg.grip_torque,
                 "stuck", cfg.open_timeout_s)

    # Lift the upper stage by `open_mm`. The reaction comes from the lower
    # jaw on the lower handle -- the tool's own pivot -- so this is the
    # whole primitive.
    #
    # `frame="here"` because `open_mm` is how far to lift, not a height to
    # lift to. As an absolute goal it was a bug that only a calibrated hand
    # showed: after `zero` parks z at mid-rail, an absolute 15 mm drives the
    # stage *down* onto the tool.
    p.step().set(Z, cfg.open_mm, cfg.travel_torque, "goal",
                 cfg.open_timeout_s, frame="here")
    yield p.build()
    return Result(start_mm,  # value: pose at submit, not interesting
                  torch.ones(n_envs, dtype=torch.bool, device=device),
                  "")
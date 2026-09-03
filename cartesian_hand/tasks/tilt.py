"""Take a contact-based grip on an object with both stages, then pitch it.

The procedure:

    1. open both jaws, seat every finger at the near end of its rail
    2. close both jaws until they touch          -> the grip, and the measurement
    3. shear the base stage's fingers by `tilt_mm` while the aux stage holds

Step 3 is `primitives.tilt`. The two stages grip the same object at two
different heights (they are separated by `z`), so translating one stage's grip
along `x` while the other holds rotates the object about `y` -- a pitch. That
is the whole reason the pairing is by STAGE and not by column: both left
fingers advancing while both right fingers hold translates the object at both
heights and tilts nothing. The primitive cannot tell the two apart -- it writes
`[span, span, 0, 0]` over whichever four ids it is handed -- which is exactly
why it takes them as `stage_a, stage_b, other_a, other_b` and why this task
passes a stage's two fingers as the first pair.

One program, not three
----------------------
`cap` splits at its probe because everything after it is parameterised by the
measured radius. Nothing here is: the shear is `tilt_mm`, a number the operator
set, so the grip's outcome only decides whether the run is reported ok. One
program, one yield, and the grip check happens after it has run.

No `z` entry move
-----------------
The stage separation IS the grip height, and it is whatever the hand was left
at. A task that drove `z` first would have to know how tall the object is, and
would fight the grip it is about to take. Position the hand, then run this.

Not reset afterwards
--------------------
The object is left pitched, which is the point -- re-seating it is a second
call with the stage pairs exchanged, the same way a twist and its reset are one
`primitives.twist` call with the ids swapped.
"""
from dataclasses import dataclass, field

import torch

from ..config import (AUX_FINGERS, AUX_JAW, AUX_LEFT, AUX_RIGHT, BASE_FINGERS,
                      BASE_JAW, BASE_LEFT, BASE_RIGHT, HandConfig)
from ..motions import Program, Result, Task
from ..primitives import tilt

GRIP_STEP = 1          # which step of the program closes both jaws on the object


@dataclass
class Config:
    """Everything about this task that is not its procedure."""
    label: str = "Tilt object"
    """Button text on the studio page. Empty string means no button."""
    sets_datum: bool = False

    jaw_opening: float = field(default=25.0, metadata={"tune": (10.0, 40.0)})
    """How wide each jaw opens before closing on the object, in mm. A clearance
    the task asks for, passed through `clamped_mm` -- never a DOF's `max_mm`,
    which is CAD and reads high."""
    approach_torque: float = field(default=150.0, metadata={"tune": (50.0, 300.0)})
    """Torque while closing a jaw onto the object. Above the grip torque so the
    close finds contact rather than stalling short of it."""
    grip_torque: float = field(default=80.0, metadata={"tune": (40.0, 200.0)})
    """Holding torque, and the torque the shear itself runs at -- it is moving a
    gripped object, so it needs at least the force that took the grip."""
    travel_torque: float = field(default=50.0, metadata={"tune": (30.0, 200.0)})
    """Torque for the free-space seat. Horizontal DOFs only; nothing here lifts
    `z`, which is the one axis this number is wrong for (see `cap`'s lift)."""
    tilt_mm: float = field(default=15.0, metadata={"tune": (5.0, 40.0)})
    """How far the base stage's fingers shear, in mm. The pitch angle is this
    over the stage separation, and the separation is not something this task
    reads, so the angle is set by eye."""
    timeout_margin: float = field(default=1.5, metadata={"tune": (1.0, 3.0)})
    """Multiplier on expected travel time, as in `cap.Config.timeout_margin`.
    Flat second counts are what expired 15 of `cap`'s 18 rows."""


def build(hand: HandConfig, start_mm: torch.Tensor, cfg: Config | None = None,
          **kwargs) -> Task:
    """Entry point for `--task tilt`. One program, returns a `Result`.

    `Result.value` is the [N, J] pose the program ended at, so the jaw columns
    are the object's half-width at each stage. `ok` is [N] -- whether BOTH jaws
    ended on contact rather than on their budget. That outcome, not the
    position, is what separates "gripped a 4 mm object" from "shut on air":
    those read identically as numbers, and shearing against nothing is what
    every downstream row would then be doing.
    """
    cfg = cfg or Config()
    n_envs, n_dof = start_mm.shape
    p = Program(n_envs, n_dof, hand.control_hz, start_mm.device)

    JAWS = [BASE_JAW, AUX_JAW]
    FINGERS = BASE_FINGERS + AUX_FINGERS
    span = hand.clamped_mm(BASE_LEFT, cfg.tilt_mm)
    # Full-rail deadline at this group's own speed -- see `HandConfig.travel_budget`.
    budget = lambda dofs: hand.travel_budget(
        [dofs] if isinstance(dofs, int) else dofs, cfg.timeout_margin)

    # Seat: jaws open around the object, every finger at the near end of its
    # rail so the shear below has a full `span` to travel into. Absolute goals,
    # so the seat is the same wherever the hand was left.
    s = p.step()
    s.set(JAWS, hand.clamped_mm(BASE_JAW, cfg.jaw_opening), cfg.travel_torque,
          "goal", budget(JAWS))
    s.set(FINGERS, 0.0, cfg.travel_torque, "goal", budget(FINGERS))

    # The grip. stop="stuck", not "goal": the object is what ends this move, and
    # position convergence can legitimately never fire.
    p.step().set(JAWS, 0.0, cfg.approach_torque, "stuck", budget(JAWS))

    # The pitch. Base stage's two fingers advance, aux stage's two hold -- by
    # stage, never by column, or this translates the object instead of tilting
    # it. Grip torque, because it is turning something held.
    tilt(p.step(), BASE_LEFT, BASE_RIGHT, AUX_LEFT, AUX_RIGHT,
         span, cfg.grip_torque, budget(BASE_FINGERS))

    program = p.build()
    measured = yield program
    ok = program.all_reached(JAWS, GRIP_STEP)
    why = "" if bool(ok.all()) else (
        f"grip did not end on contact for envs "
        f"{(~ok).nonzero().flatten().tolist()} (outcome "
        f"{program.outcome[:, JAWS, GRIP_STEP].tolist()}). Is an object between "
        f"both stages?")
    return Result(measured, ok, why)

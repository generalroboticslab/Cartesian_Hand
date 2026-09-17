"""Take a contact-based grip on an object with both stages, then pitch it.

The procedure:

    1. open both jaws, seat every finger at the near end of its rail
    2. close both jaws until they touch          -> the grip, and the measurement
    3. move the aux stage's fingers out by `tilt_mm` while the base stage holds

Step 3 is `primitives.tilt`. The two stages grip the same object at two
different heights (they are separated by `z`), so moving one stage's grip along
`x` while the other holds rotates the object about `y` -- a pitch.

**The UPPER stage is the one that moves.** Aux is upper -- `z` carries it, and
the MJCF names its joints `*_up_finger_x` against base's `*_down_finger_x`. The
lower grip is the fulcrum: hold the object at the bottom, lean the top out. The
other way round pushes the base out from under a top that is still clamped,
which levers against the aux grip rather than tilting anything.

The pairing is by STAGE and not by column, for the same reason: both left
fingers advancing while both right fingers hold moves the object at both heights
and tilts nothing. The primitive cannot tell the two apart -- it writes
`[span, span, 0, 0]` over whichever four ids it is handed -- which is exactly
why it takes them as `stage_a, stage_b, other_a, other_b` and why this task
passes one stage's two fingers as the first pair.

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
    approach_torque: float = field(default=100.0, metadata={"tune": (50.0, 300.0)})
    """Torque while closing a jaw onto the object. Above the grip torque so the
    close finds contact rather than stalling short of it."""
    grip_torque: float = field(default=80.0, metadata={"tune": (40.0, 200.0)})
    """Holding torque, and the torque the shear itself runs at -- it is moving a
    gripped object, so it needs at least the force that took the grip."""
    travel_torque: float = field(default=50.0, metadata={"tune": (30.0, 200.0)})
    """Torque for the free-space seat. Horizontal DOFs only; nothing here lifts
    `z`, which is the one axis this number is wrong for (see `cap`'s lift)."""
    tilt_mm: float = field(default=40.0, metadata={"tune": (5.0, 50.0)})
    """How far the aux (upper) stage's fingers move out, in mm. The pitch angle
    is this over the stage separation, and the separation is not something this
    task reads, so the angle is set by eye.

    50 and not the 55 mm finger rail: `STANDARD_TRAVEL` is CAD and reads high,
    so `clamped_mm` cannot save a goal that asks for the limit -- the clamp uses
    the same wrong number. The base fingers cannot extend the range either;
    `primitives.tilt` drives them to absolute 0, and the seat already put them
    there, so the whole 55 mm rail is the aux stage's to spend.

    The angle this buys depends on `z`, which this task does not set. Stage
    separation is 32 mm at z=0 (the up/down finger body offset in the MJCF)
    and grows 1:1 with z, so pitch is roughly atan(tilt_mm / (32 + z_mm)):
    50 mm of shear is ~57 degrees at z=0 and ~31 degrees at z=50.

    **z=0 is the best case for angle AND for short objects, so there is nothing
    to trade off.** Low z is minimum separation, which is both the largest angle
    per mm and the smallest object the two stages can span -- at z=0 the pad
    bands are 13 mm apart, and every mm of z widens that gap 1:1. Raise z only
    when the object is tall enough to need it; it costs angle and grips nothing
    shorter."""
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
    span = hand.clamped_mm(AUX_LEFT, cfg.tilt_mm)
    # Full-rail deadline at this group's own speed -- see `HandConfig.travel_budget`.
    budget = lambda dofs: hand.travel_budget(
        [dofs] if isinstance(dofs, int) else dofs, cfg.timeout_margin)
    # No row is commanded below the measured torque that moves its joint: under
    # `torque_min_to_move` the row runs, expires, and reports nothing wrong
    # while the joint never left, and neither backend can see it. Max over the
    # group, so a mixed-floor group is satisfied for every member rather than
    # for whichever DOF happened to be listed first.
    floor = hand.gain_vector("torque_min_to_move", start_mm.device).to(torch.float32)
    drive = lambda dofs, torque: max(float(torque), float(floor[list(dofs)].max()))

    # Seat: jaws open around the object, every finger at the near end of its
    # rail so the shear below has a full `span` to travel into. Absolute goals,
    # so the seat is the same wherever the hand was left.
    s = p.step()
    s.set(JAWS, hand.clamped_mm(BASE_JAW, cfg.jaw_opening),
          drive(JAWS, cfg.travel_torque), "goal", budget(JAWS))
    s.set(FINGERS, 0.0, drive(FINGERS, cfg.travel_torque), "goal", budget(FINGERS))

    # The grip. stop="stuck", not "goal": the object is what ends this move, and
    # position convergence can legitimately never fire.
    p.step().set(JAWS, 0.0, drive(JAWS, cfg.approach_torque), "stuck",
                 budget(JAWS))

    # The pitch. Aux (upper) stage's two fingers move out, base (lower) stage's
    # two hold and act as the fulcrum -- paired by stage, never by column, or
    # this moves the object instead of tilting it. Grip torque, because it is
    # turning something held.
    tilt(p.step(), AUX_LEFT, AUX_RIGHT, BASE_LEFT, BASE_RIGHT, span,
         drive(AUX_FINGERS + BASE_FINGERS, cfg.grip_torque), budget(AUX_FINGERS))

    program = p.build()
    measured = yield program
    ok = program.all_reached(JAWS, GRIP_STEP)
    why = "" if bool(ok.all()) else (
        f"grip did not end on contact for envs "
        f"{(~ok).nonzero().flatten().tolist()} (outcome "
        f"{program.outcome[:, JAWS, GRIP_STEP].tolist()}). Is an object between "
        f"both stages?")
    return Result(measured, ok, why)

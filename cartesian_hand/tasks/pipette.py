"""Operate a pipette: hold it, work a twist-lock knob, plunge, then eject the tip.

    open the aux jaw -> z to the knob and fingers shut
      -> probe both jaws -> grip -> turn the knob
      -> release the knob -> rise -> close the aux side into a fist
      -> press to `plunger_z`, rise, twice    the plunger
      -> rest the base fingers
      -> press to `eject_z`, rise, twice      the tip ejector
      -> final pause

The row list is flat: no `Loop`, and the stroke counts are written out. `Loop`
earns its place when the count is not knowable at build time -- `cap` derives
its from a probed radius -- and here it is two. Flat also gives each press its
own phase index, so `failed in phase(s) [N]` names which pass failed instead of
pointing at a row both passes share.

**The opening is `cap`'s, because it is `cap`'s problem.** The base gripper is
fixed and the auxiliary gripper rides z, so one `Probe` of both jaws grips the
pipette body and the knob together at the height z was sent to, and one `Hold`
squeezes both. The base jaw's grip then stands for the whole run. That replaces
an open-halfway / close-fingers / probe-the-base-alone preamble which did the
same thing in four rows and left the aux fingers somewhere the twist
immediately overrode.

Nothing clears the jaw or centres the fingers before the twist either: a stroke
opens the jaw to `radius + clearance` and sweeps the fingers to the start of the
turn as its own first two phases, so a manual clear and centring are undone on
the next tick. The bench run of 2026-09-05 spent 0.9 s driving the fingers to
20 mm and the first stroke drove them straight to 40/0.

**A stroke presses to a height, and that is a deliberate reversal.** The
validated demo pressed to a stall because hand-tuned depths did not reproduce
run to run, and this file carried that. It presses to `plunger_z` instead,
because a pipette's dispensed volume is set by how far the plunger goes and a
stall only ever finds the one stop at the bottom of its travel. The cost is real
and is accepted: a pressed plunger and a slipped grip are identical as
positions, so this row can no longer tell them apart -- which is exactly what
`Probe` was doing here.

**Four z heights, all absolute, no arithmetic.** `knob_z`, `top_z`,
`plunger_z`, `eject_z`. They were briefly a height plus three offsets, which
reads as `knob_z + rise_mm - eject_mm` at the call site and hides the one thing
that matters -- that all four share the 0 to 50 mm rail and the buttons are at
fixed heights on a pipette held in a fixed grip. Absolute makes the ordering
`eject_z < plunger_z < top_z` visible on the sliders.

A press is free, so stopping short of its height is a fault and says so with
the z it stopped at. `accept_stall` was tried and is what let a fist that
plainly never reached the button report success -- see the comment in `build`.

**The two pairs press two different buttons.** The first works the plunger; the
second works the tip ejector, which sits much lower, so they are two rows at two
depths rather than one row used twice. They were one shared row until
2026-09-05, which made the second pair press the plunger again and never release
a tip -- and nothing about the motion showed it, because a press that misses and
a press that lands are the same command.

Every goal is absolute in the hand's millimetre frame, so the hand must be
zeroed first -- `knob_z` is a height above the z hard stop.

Partially validated on hardware: on 2026-09-05 everything through `close aux`
ran. The plunger and eject loops have no hardware results yet.
"""

from dataclasses import dataclass, field

import torch

from ..config import (AUX_FINGERS, AUX_JAW, AUX_LEFT, AUX_RIGHT, BASE_FINGERS,
                      BASE_JAW, BASE_LEFT, HandConfig, Z)
from ..primitives import (Hold, Move, Probe, Sequence, Twist, lift_effort,
                          strokes_for_revolutions)

JAWS = (BASE_JAW, AUX_JAW)
BASE = tuple(BASE_FINGERS)
AUX = tuple(AUX_FINGERS)
FINGERS = tuple(BASE_FINGERS + AUX_FINGERS)

Z_TOLERANCE_MM = 5.0
"""Arrival band for a z move, against `Move`'s 1.0 mm default.

Nothing here needs z placed to a millimetre: `height` only has to put the base
jaw across the body and the aux jaw around the knob, and the rise only has to
clear the knob. What the 1 mm default does buy is a fault every time the stage
parks slightly short, which is one thing this task has already failed on. Same
constant and same reasoning as `syringe.Z_TOLERANCE_MM`."""

CLOSED_TOL_MM = 10.0
"""Arrival tolerance for the aux side shutting into a fist, mm.

A `Move` is free by default, so a confirmed stop short of goal is a fault
reported the tick it becomes detectable. A finger driven to 0 does not get
there: it parks against its own stop with the last few millimetres unavailable.
Measured 2.2 mm on hand_2's aux right finger on 2026-09-05, where the gripper
had visibly closed and the row still failed. `cap.CARRY_TOL_MM` is the same
number for the same reason on the same hand; kept separate because two tasks
sharing a bench constant is not a reason to couple them.

Wide enough to pass a finger at its stop, narrow enough to still fail the jaw
when it is closing onto something -- the run before this one stalled the aux jaw
at 9.5 mm on the knob and must keep failing.

Not `loaded=True`, which would also pass by accepting any confirmed stall: that
raises the effort cap into a hard stop and drops the bound entirely, so a finger
stopping at 15 mm would read as arrival."""


@dataclass
class Config:
    """Bench units. Every `tune` field becomes a slider on the studio page."""

    label: str = "Work pipette"
    """Button text on the studio page. Empty means no button, and a variant that
    does not name its own inherits none -- see `tasks/__init__.py`.

    Named, though this port is only part-validated: reaching it through the tune
    panel's dropdown meant every unvalidated task was one collapsed folder and a
    button called **Run tuned** away from being run, which is where an operator
    does not look for "run the pipette task"."""
    sets_datum: bool = False

    knob_z: float = field(default=35.0, metadata={"tune": (10.0, 50.0)})
    """Z height of the twist-lock knob, in mm. Also the height both jaws probe
    at, so it must put the base jaw across the pipette body."""
    top_z: float = field(default=50.0, metadata={"tune": (10.0, 50.0)})
    """Z the fist returns to between presses, in mm. Must clear the knob."""
    plunger_z: float = field(default=25.0, metadata={"tune": (0.0, 40.0)})
    """Z at the bottom of a plunger press, in mm."""
    eject_z: float = field(default=5.0, metadata={"tune": (0.0, 40.0)})
    """Z at the bottom of a tip-ejector press, in mm.

    The ejector button sits much lower than the plunger, which is the whole
    reason the second pair is not the first one repeated. So `eject_z` <
    `plunger_z` < `top_z`, and the two buttons should be as far apart here as
    they are on the pipette.

    **Watch the bottom of the rail.** z zero is the hard stop, so at the
    defaults the fist starts at 30 mm and everything both buttons need has to
    fit in that 30 mm. Nothing enforces the ordering or the floor: a goal below
    zero clamps, which drives the fist into the stop instead of the button, and
    `accept_stall` reports that as success. If a run ends with z near zero, this
    is too low -- and if the real gap between the buttons will not fit above the
    stop, the fix is to grip the pipette lower on its body so both sit higher
    above z zero, not to squeeze these two numbers together."""
    knob_revs: float = field(default=0.5, metadata={"tune": (0.5, 6.0)})
    """Full turns to work the twist knob."""
    knob_clearance: float = field(default=10.0, metadata={"tune": (1.0, 20.0)})
    """How far the aux jaw backs off the knob, in mm: between twist strokes to
    reset the fingers, and once more afterward to clear it."""
    base_grip_x: float = field(default=30.0, metadata={"tune": (0.0, 55.0)})
    """Base finger position while the pipette rests in the stand, in mm."""
    final_pause_s: float = field(default=2.0, metadata={"tune": (0.0, 5.0)})
    """Pause at the end, holding the rise height."""
    approach_torque: float = field(default=150.0, metadata={"tune": (50.0, 300.0)})
    """Torque while closing a jaw onto the pipette or its knob."""
    approach_speed: float = field(default=300.0, metadata={"tune": (25.0, 500.0)})
    """Servo speed register while finding contact, counts/s.

    **Raised 50 -> 300, for `cap`'s reason.** There is no contact sensor, so
    contact is `CONFIRM_TICKS` of measured speed under `STUCK_SPEED_MM_S` =
    0.3 mm/s; at 50 counts/s the creep itself runs at 0.61 mm/s, twice the
    threshold it is tested against, and any servo hesitation reads as the
    pipette. At 300 the margin is 12x and a 20 mm close takes 5.4 s not 33."""
    travel_speed: float = field(default=1100.0, metadata={"tune": (200.0, 1500.0)})
    """Servo speed register for every free move, counts/s.

    **Not `config.STANDARD_SPEED`.** That is one number for the whole session
    and it is set by what `zero` needs -- hand_2's fingers sit at 200 counts/s,
    which makes the knob twist's 40 mm finger sweep 16 s. Deadlines derive from
    this number, so lowering it lengthens them with it."""
    squeeze_torque: float = field(default=250.0, metadata={"tune": (200.0, 800.0)})
    """Holding torque on both jaws: the base jaw on the pipette body for the
    whole run, and the aux jaw on the knob at every close and re-grip.

    One number for the two because `Hold` carries one effort for the group it
    names, and the body and the knob have never wanted different ones. A jaw
    that needs its own gets its own row."""
    base_grip_torque: float = field(default=400.0,
                                    metadata={"tune": (50.0, 400.0)})
    """Torque driving the base fingers out to `base_grip_x`.

    Above the plain finger torque because the base jaw is still squeezed on the
    body while this runs, so the fingers may have to push past it."""
    push_torque: float = field(default=400.0, metadata={"tune": (100.0, 800.0)})
    """Torque driving z down onto the plunger, for both plunge and draw.

    A descent, so it is deliberately not raised to z's `torque_min_to_move`:
    that floor is for the lifting direction, and `Sequence` only raises the
    effort of a `loaded` row. The rise between strokes is a lift and gets
    `lift_torque` instead."""
    lift_torque: float = field(default=800.0, metadata={"tune": (100.0, 1000.0)})
    """Z torque for every ascent, floored at the hand's measured gravity floor.

    **The floor alone is not enough here.** `torque_min_to_move[Z]` is bisected
    as the least torque that moves an *unloaded* stage; this one raises a
    gripped pipette. On hand_2's current 300 the rise crept, dropped under
    `STUCK_SPEED_MM_S` two millimetres short of goal, and a free `Move` reports
    a confirmed stop short of goal as a fault -- the 2026-09-05 run failed it at
    0.9 s of a 5.5 s budget, then drifted the rest of the way under the standing
    command once the task had already retired. Same field and default as `cap`."""
    travel_torque: float = field(default=50.0, metadata={"tune": (30.0, 200.0)})
    """Torque for unloaded horizontal travel."""
    finger_stroke: float = field(default=40.0, metadata={"tune": (10.0, 55.0)})
    """Full sweep of an auxiliary finger during one knob twist, in mm."""
    timeout_margin: float = field(default=1.5, metadata={"tune": (1.0, 3.0)})
    """Margin on the deadline each row derives from its own travel, at the
    speed it commands."""


def build(hand: HandConfig, start_mm: torch.Tensor,
          cfg: Config | None = None) -> Sequence:
    """Grip body and knob at once, work the knob, then plunge and draw."""
    cfg = cfg or Config()
    mm = hand.clamped_mm
    floor = hand.gain_vector("torque_min_to_move", start_mm.device)
    jaw_floor = max(float(floor[dof]) for dof in JAWS)
    squeeze = max(cfg.squeeze_torque, jaw_floor) / 1000.0
    base_grip = max(cfg.base_grip_torque, float(floor[BASE_LEFT])) / 1000.0
    push_effort = cfg.push_torque / 1000.0
    lift = lift_effort(hand, cfg.lift_torque)

    finger_span_mm = mm(AUX_LEFT, cfg.finger_stroke)
    top_z = mm(Z, cfg.top_z)
    # Mid-rail, not a tuned number: the jaw only has to be wider than the
    # pipette, and half the CAD travel is safely that without commanding a rail
    # end. Same reasoning the dropped `open half` row carried.
    jaw_open = float(hand.upper(start_mm.device)[AUX_JAW]) / 2.0
    # The three rows the tail is built from, named once and listed flat below.
    #
    # **A press is a plain `Move`: not `creep`, not `accept_stall`.** It was
    # both, and the pair is why the fist stopped visibly short of the button and
    # still reported ok. `accept_stall` ends the row at the first confirmed
    # stall and takes `grace_ticks = 0` with it, which is the mirror risk
    # `START_GRACE_TICKS` documents and does not fix: ten quiet ticks inside a
    # joint's own stiction read as arrival. At the 3.68 mm/s creep that is
    # 0.74 mm, so the row could finish before the stage had properly started.
    #
    # Free, they arrive or they fault, and a fault prints the z they stuck at --
    # the number that separates "the button is below z zero" from "the spring is
    # stiffer than `push_torque`". Tolerance stays at the 1 mm default rather
    # than `Z_TOLERANCE_MM`: on a press, five millimetres short is short.
    plunger = Move(label="plunger", goal={Z: mm(Z, cfg.plunger_z)},
                   effort=push_effort)
    eject = Move(label="eject", goal={Z: mm(Z, cfg.eject_z)},
                 effort=push_effort)
    rise = Move(label="rise", goal={Z: top_z}, effort=lift,
                tolerance_mm=Z_TOLERANCE_MM)

    return Sequence([
        # The aux jaw opens before anything else moves, and on its own row. It
        # rides z, so descending to `knob_z` with it still shut from a previous
        # run drives it into the pipette on the way down -- and a `Probe` that
        # starts already closed is at its goal on the first tick, which
        # `close_until_contact` reports as reaching the goal without contact,
        # not as a measurement. Its own row rather than folded into `height`
        # because "open, then move" is the point; opening while z travels is
        # the collision this exists to avoid.
        Move(label="open", goal={AUX_JAW: jaw_open}),
        # All four fingers shut with the z move, not after it: they are out of
        # the jaws' way before the probe either way, and doing it in one row
        # costs nothing. They share the row's `lift` effort, which is a cap and
        # not a command -- a finger ending against its own stop carries a couple
        # of millimetres of error and develops very little of it. A row that
        # swept a jaw across the workspace at this cap would be a different
        # matter.
        Move(label="height", goal={Z: mm(Z, cfg.knob_z), FINGERS: 0.0},
             effort=lift, tolerance_mm=Z_TOLERANCE_MM),
        # One probe, both jaws: the base gripper is fixed and finds the pipette
        # body, the aux gripper rides z and finds the knob. Only the aux jaw's
        # stop is a radius the twist can use.
        Probe(label="probe", group=JAWS, creep=True,
              measure={"radius": AUX_JAW}),
        Hold(label="grip", group=JAWS, effort=squeeze),
        Twist(label="knob", jaw=AUX_JAW, left=AUX_LEFT, right=AUX_RIGHT,
              radius=lambda m: m.radius, span=finger_span_mm,
              grip=squeeze, clearance=cfg.knob_clearance,
              measure={"radius": AUX_JAW},
              count=lambda m: strokes_for_revolutions(
                  torch.full_like(m.radius, cfg.knob_revs), m.radius,
                  finger_span_mm)),
        # Pulling free of a full squeeze, so it carries that effort, not travel.
        Move(label="release", effort=squeeze, loaded=True,
             goal={AUX_JAW: lambda m: m.radius + cfg.knob_clearance}),
        Move(label="to plunger", goal={Z: top_z}, effort=lift,
             tolerance_mm=Z_TOLERANCE_MM),
        Move(label="close aux", goal={AUX_JAW: 0.0, AUX: 0.0},
             tolerance_mm=CLOSED_TOL_MM),
        plunger, rise,
        plunger, rise,
        # The base fingers move while the base jaw is still squeezed on the
        # body, so they may have to push past it: their own effort, not travel.
        Move(label="rest", goal={BASE: mm(BASE_LEFT, cfg.base_grip_x)},
             effort=base_grip),
        eject, rise,
        eject, rise,
     #    Hold(label="final", seconds=cfg.final_pause_s),
        Move(label="open", goal={AUX_JAW: jaw_open}),
        Move(label="height", goal={Z: mm(Z, cfg.knob_z), FINGERS: 0.0},
             effort=lift, tolerance_mm=Z_TOLERANCE_MM),
        
    ], hand=hand,
       start_mm=start_mm,
       travel_torque=cfg.travel_torque,
       approach_torque=cfg.approach_torque,
       approach_speed=cfg.approach_speed,
       timeout_margin=cfg.timeout_margin,
       travel_speed=cfg.travel_speed)

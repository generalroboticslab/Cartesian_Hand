"""Unscrew a bottle cap, sizing the grip from contact rather than from a number.

    move to cap height -> probe both jaws -> grip
      -> repeat cycle(
           repeat(release -> reset fingers -> re-grip -> open turn)
           -> release -> centre -> hold -> re-grip -> lift -> move cap clear
           -> wait/signal -> align cap
           -> return cap to thread height
           -> repeat(release -> reset -> re-grip -> press -> close turn)
           -> let go)
      -> present

**One deliberate settle.** After the final finger centring, the task holds the
finger goals for 0.2 s before closing the jaw, so the last fraction of profiled
travel cannot overlap the re-grip. There is no pre-probe sleep: `move_to` is
closed-loop on measured position, and `close_until_contact` needs ten
consecutive ticks under 0.3 mm/s, so a hand still moving cannot register
contact.

The stroke count is written down nowhere: it is `2*pi*r / finger_span` rounded
up, with a floor of two strokes per revolution, and `r` is what the jaws
measured. That is the whole reason this is a policy and not a `Motions` row
table, whose repetition count must be fixed before anything runs.

Each stroke then turns the fingers through `revolutions*2*pi*r / stroke_count`
mm, not the full `finger_span_mm` -- the whole point of rounding the stroke
count up is that it does not divide evenly, so spending the full span on
every stroke would over-rotate by however much the rounding added. Splitting
the requested turn evenly across the rounded-up count is also what makes a
revolutions setting respond continuously: two requests that round up to the
same stroke count (0.5 and 0.7 turns of a small enough cap, say) still turn
the object by different amounts, because they divide that count differently.

Opening and closing take separate revolution counts, `num_revs_up` and
`num_revs_down`: the cap may need less than a full turn to break free but
several turns to re-seat and thread, or the other way round depending on the
bottle.

The base jaw's grip stands from the probe to the end, holding the bottle while
the auxiliary gripper removes and replaces the cap. Set `cycles` to zero to
repeat until the task is stopped; a positive value runs that many complete
open-and-close cycles.

The height and lift goals are absolute in the hand's millimetre frame, so the
hand must be zeroed first.

default offset is 10, num_revs_up  = 1, and num_revs_down = 0.8, lift_mm = 35, squeeze 80, 

centrifuge tube needed 0.65 turns for repeated task

square bottle needs 20 offset, lift_mm35, 

bottle250 needs 25mm offset cap instead of 10 for in hand off table, 1 revs up, 0.8 revs down, lift_mm=50

glue stick: 10mm offset, 0 rev up 0 rev down, squeeze 500, lift 35, same for pen with cap

for culture tube: 10mm offset, lift 35, but squeeze and lift torque at default

lightbulb: num_revs_up is 2.5 and same for down.

petri dish and well plate, 0 offset, 0revs up, 0 revs down. 40torque squeeze 

peanut butter jar: 

dropper bottle: set `dropper_bottle=True` (the "Dropper bottle" button in
`tasks/dropper_bottle.py` does this). Same cap-opening twist, but once the cap
breaks free the aux jaw squeezes and releases the dropper's bulb -- twice, once
with the tip still in the bottle to draw liquid up, once after the pipette is
lifted clear to dispense it -- before the cap is threaded back down. See the
`dropper_bottle` branch in `build` for the row-by-row mechanism.

child-safe cap: set `child_safe=True` (the "Child-safe cap" button in
`tasks/child_safe_cap.py` does this). A child-resistant closure's ratchet
only disengages while the cap is pushed down, so the plain opening twist --
which only turns -- rides the ratchet teeth instead of releasing them. This
runs the closing twist's own press-hold-turn-retract cycle (`TwistPress`) on
the *opening* twist too, so every un-locking stroke presses down, turns, and
only retracts once the jaw has let go, same as re-threading already does on
the way back down. Depth is its own field, `child_safe_press_mm`, not
`down_stroke_z`: that one is a thread-pitch press capped at a few mm by
`MAX_DOWN_TRAVEL_MM`, and a ratchet lock generally needs to be pressed
further than that to disengage.
"""
import math
from dataclasses import dataclass, field

import torch

from ..config import (AUX_FINGERS, AUX_JAW, AUX_LEFT, AUX_RIGHT, BASE_JAW,
                      BASE_LEFT, BASE_RIGHT, HandConfig, Z)
from ..primitives import (Hold, Loop, Move, Probe, Sequence, Twist, TwistPress,
                          lift_effort, strokes_for_revolutions)

JAWS = (BASE_JAW, AUX_JAW)
AUX = tuple(AUX_FINGERS)
FINGERS = (BASE_LEFT, BASE_RIGHT, AUX_LEFT, AUX_RIGHT)

CARRY_TOL_MM = 4.0
"""Arrival tolerance for a finger move that ends against its own stop, mm.

A `Move` is free by default, so a joint that stops short of its goal is a
fault reported on the tick it becomes detectable -- which is what the two
`AUX -> 0.0` rows here kept hitting. They are not faults: the fingers close
onto the cap's own stop and the last few millimetres are not available. 4 mm
covers the 2.7 mm measured on hand_2 with margin, and still fails a finger
that is genuinely obstructed halfway down its rail.

Not `loaded=True`, which would also work by accepting any confirmed stall:
that raises the effort cap into a hard stop and drops the bound entirely, so
a finger stopping at 15 mm would read as arrival."""

PUT_BACK_TOL_MM = 5.0
"""Arrival tolerance for the "put back" descent, mm.

The aux fingers are carrying the cap down onto the bottle at `reinsert_effort`
-- a deliberately gentle force cap, not a hard stop -- so this row's real
achieved speed is far below whatever nominal speed it is timed against, and
it was still 1-5 mm short of `cap_height`, and still moving, when the row's
deadline expired: neither the default 1 mm tolerance nor `accept_stall`'s
confirmed-stall check ever fired. `creep=True` (below) fixes the deadline
side by timing the row off the hand's slow approach speed instead of full
travel speed; this tolerance is the backstop for whatever gap `creep` doesn't
close, sized the same way as `CARRY_TOL_MM` but for the z axis. The following
`close` twist re-centres the cap regardless of exactly where in this margin
z lands."""

MAX_DOWN_TRAVEL_MM = 5.0
"""Hard ceiling on the closing twist's per-stroke downward z travel, mm.

`down_stroke_z` is an absolute goal clamped only against `cap_offset` (so
tightening cannot pull upward) -- it has no floor. Raise `cap_offset` on a
tall cap with `down_stroke_z` left at its default and the press would drive
z down tens of millimetres per stroke, well past any real thread pitch. This
floors the press goal at `cap_offset - MAX_DOWN_TRAVEL_MM` so the screw-down
travel (dof3, `Z`) is bounded independent of how the other two are tuned."""


@dataclass
class Config:
    """Bench units. Every `tune` field becomes a slider on the studio page."""

    label: str = "Cycle cap"
    sets_datum: bool = False

    dropper_bottle: bool = False
    """Route the opened cap through the bulb-squeeze insert instead of
    straight to the lift -- see the module docstring's "dropper bottle" entry.
    Not a slider: `tasks/dropper_bottle.py` is the button that sets it, so
    `cap` itself keeps its plain cap-cycling default."""

    child_safe: bool = False
    """Press z down through the opening twist as well as the closing one --
    see the module docstring's "child-safe cap" entry. Depth is
    `child_safe_press_mm`, not `down_stroke_z`: that field is a thread-pitch
    press bounded by `MAX_DOWN_TRAVEL_MM` (a few mm), while releasing a
    ratchet lock needs its own, usually deeper, press. Not a slider:
    `tasks/child_safe_cap.py` is the button that sets it, so `cap` itself
    keeps its plain default of pressing only while closing."""
    child_safe_press_mm: float = field(default=15.0, metadata={"tune": (0.0, 40.0)})
    """How far below `cap_offset` to press z before and during the opening
    twist, mm. `child_safe` only. Not run through `MAX_DOWN_TRAVEL_MM` --
    that ceiling exists to stop a thread press over-travelling past real
    thread pitch, which does not apply here; `hand.clamped_mm` still floors
    the goal at the rail's own 0 mm limit."""

    cap_offset: float = field(default=15, metadata={"tune": (0.0, 40.0)})
    """Height of the cap's top face above z zero, mm."""
    num_revs_up: float = field(default=1, metadata={"tune": (0.0, 6.0)})
    """Revolutions during the opening (unscrewing) twist."""
    num_revs_down: float = field(default=0.8, metadata={"tune": (0.0, 6.0)})
    """Revolutions during the closing (screwing down) twist."""
    squeeze_torque: float = field(default=80.0, metadata={"tune": (40.0, 500.0)})
    """Aux jaw (DOF4) grip effort for the initial bottle grip and the opening
    twist. The closing twist's own jaw grip is `close_squeeze_torque`, not
    this -- see that field."""
    close_squeeze_torque: float = field(default=80.0, metadata={"tune": (10.0, 500.0)})
    """Aux jaw (DOF4) grip effort for the closing twist only -- the repeated
    release/re-grip that screws the lid back on. Split from `squeeze_torque`
    so a lid that needs a gentler re-grip does not also loosen the initial
    bottle grip or the opening twist. Defaults equal to `squeeze_torque`."""
    close_torque: float = field(default=150.0, metadata={"tune": (0.0, 300.0)})
    """Finger torque while closing the cap. Zero automatically uses ten servo
    units below the effective opening torque; a positive value overrides it.
    Jaw release continues to use `squeeze_torque`; jaw grip during closing
    uses `close_squeeze_torque`."""
    approach_torque: float = field(default=150.0, metadata={"tune": (50.0, 300.0)})
    travel_speed: float = field(default=1500.0, metadata={"tune": (200.0, 1500.0)})
    """Servo speed register for every free move, counts/s.

    The longest travel in this task is the twist's 45 mm finger sweep, which
    takes about 3.3 s at 1100 counts/s. Deadlines derive from this number, so
    lowering it lengthens them with it.

    **Not `config.STANDARD_SPEED`.** That is one number for the whole session
    and it is set by what `zero` needs -- hand_2's fingers sit at 200 counts/s,
    which makes that same sweep 16 s. The seek wants slow (a fast creep
    overshoots a hard stop and climbs a gear tooth); manipulation wants fast."""
    approach_speed: float = field(default=800, metadata={"tune": (25.0, 800.0)})
    """Servo speed register while closing a jaw to find the object, counts/s.

    **Raised 50 -> 300 on 2026-09-04.** At 50 (0.61 mm/s), closing 10 mm takes
    16 s and 20 mm takes 33 s. That is the "waits 20 seconds after the first
    action" this task was reported for. At 300 those moves take 2.7 and 5.4 s.

    Faster is also *safer* here, not riskier. There is no contact sensor: with
    no `external_signal`, `Observation.contact` is all false and contact is
    detected purely as `CONFIRM_TICKS` of not moving, below
    `STUCK_SPEED_MM_S` = 0.3 mm/s. At 0.61 mm/s the creep ran at twice the
    threshold it was tested against, so any servo hesitation read as contact.
    At 300 (3.68 mm/s) the margin is 12x. The cost is 0.74 mm of overshoot into
    a jaw already capped at `approach_torque` -- inside the 1 mm tolerance the
    rest of this file works to."""
    travel_torque: float = field(default=50.0, metadata={"tune": (30.0, 200.0)})
    release_clearance: float = field(default=5.0, metadata={"tune": (1.0, 10.0)})
    """How far past the measured radius the aux jaw opens between strokes, mm."""
    timeout_margin: float = field(default=1.5, metadata={"tune": (1.0, 3.0)})
    finger_stroke: float = field(default=45.0, metadata={"tune": (10.0, 50.0)})
    """Full sweep of one auxiliary finger during a twist, mm."""
    lift_mm: float = field(default=50.0, metadata={"tune": (5.0, 50.0)})
    """Absolute z position that holds the removed cap clear of the bottle."""
    lift_torque: float = field(default=500.0,
                                metadata={"tune": (100.0, 1000.0)})
    """Z torque while lifting the gripper with the cap. This is floored at the
    hand's unloaded Z movement torque, but may need to be higher when loaded."""
    clear_wait_s: float = field(default=1.0, metadata={"tune": (0.0, 10.0)})
    """Maximum wait with the cap clear. An external signal ends it early."""
    cycles: float = field(default=1.0, metadata={"tune": (0.0, 10.0)})
    """Complete open-and-close cycles, rounded to an integer. Zero repeats
    until the task is stopped (disarm torque in Studio, or Ctrl-C headless)."""
    reinsert_z_torque: float = field(default=50.0,
                                     metadata={"tune": (50.0, 400.0)})
    """Downward z torque used to seat and thread the cap, servo units."""
    down_stroke_z: float = field(default=5.0, metadata={"tune": (0.0, 40.0)})
    """Absolute z goal while tightening, mm. Values above `cap_offset` are
    clamped to `cap_offset`, because tightening must not pull upward. Values
    that would travel down more than `MAX_DOWN_TRAVEL_MM` from `cap_offset`
    are floored the same way."""

    bulb_offset: float = field(default=30.0, metadata={"tune": (5.0, 45.0)})
    """Absolute z height of the dropper's bulb, mm -- the cap's own height
    doesn't put the bulb where the jaw already sits. `dropper_bottle` only."""
    bulb_lift_z: float = field(default=50.0, metadata={"tune": (5.0, 50.0)})
    """Absolute z the pipette rises to, clear of the bottle, before the
    dispensing squeeze. `dropper_bottle` only."""
    bulb_release_mm: float = field(default=5.0, metadata={"tune": (1.0, 20.0)})
    """How far past the grip the aux jaw opens to let go of the bulb, mm --
    separate from `release_clearance`, which is the cap-twist's own release
    and has no reason to move together with this one. `dropper_bottle` only."""
    bulb_open_mm: float = field(default=4.0, metadata={"tune": (0.0, 40.0)})
    """Absolute aux-jaw gap while carrying the bulb clear of the bottle
    without squeezing it, mm. `dropper_bottle` only."""
    bulb_squeeze_torque: float = field(default=80.0,
                                       metadata={"tune": (10.0, 300.0)})
    """Aux-jaw torque while squeezing the dropper bulb. Separate from
    `squeeze_torque`: that one is bisected to hold a rigid cap without
    slipping, and a soft bulb wants far less force to compress it than a
    hard shell wants to be gripped by. `dropper_bottle` only."""


def build(hand: HandConfig, start_mm: torch.Tensor,
          cfg: Config | None = None) -> Sequence:
    """Probe once, then repeatedly remove and replace the same cap."""
    cfg = cfg or Config()
    mm = hand.clamped_mm
    finger_span_mm = mm(AUX_LEFT, cfg.finger_stroke)
    squeeze_effort = cfg.squeeze_torque / 1000.0
    close_squeeze_effort = cfg.close_squeeze_torque / 1000.0
    finger_floor = max(float(hand.gain_vector("torque_min_to_move")[dof])
                       for dof in AUX)
    opening_torque = max(cfg.squeeze_torque, cfg.travel_torque, finger_floor)
    close_torque = (cfg.close_torque if cfg.close_torque > 0 else
                    max(0.0, opening_torque - 10.0))
    close_effort = close_torque / 1000.0
    reinsert_effort = cfg.reinsert_z_torque / 1000.0
    release_clearance_mm = cfg.release_clearance
    num_revs_up = torch.full_like(start_mm[:, 0], cfg.num_revs_up)
    num_revs_down = torch.full_like(start_mm[:, 0], cfg.num_revs_down)
    reverse = torch.ones_like(num_revs_down, dtype=torch.bool)
    cap_height = mm(Z, cfg.cap_offset)
    lift = lift_effort(hand, cfg.lift_torque)
    cycles = float("inf") if cfg.cycles <= 0 else max(1, round(cfg.cycles))

    if cfg.num_revs_up == 0 and cfg.num_revs_down == 0:
        # No twist to perform: `strokes_for_revolutions` still floors to one
        # stroke of zero span, which would run the whole release/reset/
        # re-grip/turn machinery to turn nothing. For an object that only
        # needs pulling straight out and pressing straight back in (a glue
        # stick, not a threaded cap), skip `Twist` entirely and drive the
        # jaws and z axis directly.
        return Sequence([
            Move(label="height", goal={Z: cap_height}),
            Loop(count=cycles, rows=[
                Move(label="fingers out", goal={FINGERS: finger_span_mm}),
                Probe(label="grip base", group=BASE_JAW, creep=True,
                      grip=squeeze_effort),
                Probe(label="grip aux", group=AUX_JAW, creep=True,
                      grip=squeeze_effort),
                Move(label="lift", goal={Z: mm(Z, cfg.lift_mm)}, effort=lift,
                     tolerance_mm=2.0),
                Move(label="pull back", goal={AUX: 0.0}),
                Move(label="push out", goal={AUX: finger_span_mm}),
                Move(label="close", goal={Z: cap_height}, effort=reinsert_effort,
                     accept_stall=True, tolerance_mm=PUT_BACK_TOL_MM,
                     creep=True),
            ]),
        ], hand=hand,
           start_mm=start_mm,
           travel_torque=cfg.travel_torque,
           approach_torque=cfg.approach_torque,
           approach_speed=cfg.approach_speed,
           timeout_margin=cfg.timeout_margin,
           travel_speed=cfg.travel_speed)

    down_stroke_z = max(min(cfg.down_stroke_z, cfg.cap_offset),
                         cfg.cap_offset - MAX_DOWN_TRAVEL_MM)
    press = TwistPress(
        dof=Z,
        active=reverse,
        goal_mm=torch.full_like(num_revs_down, mm(Z, down_stroke_z)),
        return_mm=torch.full_like(num_revs_down, cap_height),
        effort=torch.full_like(num_revs_down, reinsert_effort),
        return_effort=torch.full_like(num_revs_down, lift),
    )
    # Same press, mirrored onto the opening twist: a child-safe cap's ratchet
    # needs the identical press-hold-turn-retract cycle to disengage, not
    # just to re-seat threads on the way back down. `active` is its own
    # all-true tensor rather than `reverse` -- `reverse` also flips the turn
    # direction, which the opening twist must not inherit.
    press_open = TwistPress(
        dof=Z,
        active=torch.ones_like(num_revs_up, dtype=torch.bool),
        goal_mm=torch.full_like(
            num_revs_up, mm(Z, cfg.cap_offset - cfg.child_safe_press_mm)),
        return_mm=torch.full_like(num_revs_up, cap_height),
        effort=torch.full_like(num_revs_up, reinsert_effort),
        return_effort=torch.full_like(num_revs_up, lift),
    ) if cfg.child_safe else None

    def stroke_plan(revs: torch.Tensor):
        """Stroke count and per-stroke span for `revs` -- see module docstring."""
        count = lambda m: strokes_for_revolutions(revs, m.radius, finger_span_mm)
        # The full finger sweep, spent evenly over the rounded-up stroke count
        # instead of in full on every stroke. Capped at `finger_span_mm` only
        # against float rounding at the boundary; by construction `count(m)`
        # already never asks for more than that.
        span = lambda m: (revs * 2 * math.pi * m.radius
                          / count(m).to(m.radius.dtype)
                          ).clamp(max=finger_span_mm)
        return count, span

    strokes_up, stroke_span_up = stroke_plan(num_revs_up)
    strokes_down, stroke_span_down = stroke_plan(num_revs_down)
    release = Move(
        label="release",
        goal={AUX_JAW: lambda m: m.radius + release_clearance_mm},
        effort=squeeze_effort, loaded=True)
    # Raising loaded z must clear this hand's measured gravity floor.
    lift_move = Move(label="lift", goal={Z: mm(Z, cfg.lift_mm)}, effort=lift,
                     tolerance_mm=2.0)

    if cfg.dropper_bottle:
        bulb_squeeze_effort = max(
            cfg.bulb_squeeze_torque,
            float(hand.gain_vector("torque_min_to_move")[AUX_JAW])) / 1000.0
        bulb_offset_z = mm(Z, cfg.bulb_offset)
        bulb_lift_z = mm(Z, cfg.bulb_lift_z)
        bulb_open_target = mm(AUX_JAW, cfg.bulb_open_mm)
        # Reused for both the priming release (grip-relative, right after the
        # twist) and the final let-go (diameter-relative, clearing the whole
        # cap before z carries the jaw back past it) -- see the two Move rows
        # below that build on it.
        bulb_release = Move(
            label="release bulb grip",
            goal={AUX_JAW: lambda m: m.radius + cfg.bulb_release_mm})
        squeeze_bulb = Probe(label="squeeze bulb", group=AUX_JAW, creep=True,
                             grip=bulb_squeeze_effort)
        bulb_open = Move(label="bulb open", goal={AUX_JAW: bulb_open_target})
        # The twist's last stroke leaves the fingers wherever its last turn
        # stroke put them (one out, one in) -- centred, a squeeze lands even
        # on both sides of the bulb instead of mostly on one finger.
        centre_fingers = Move(label="centre fingers",
                              goal={AUX: finger_span_mm / 2})
        # The twist's last stroke ends gripping the cap (mid-turn), not
        # centred and released like the plain flow. No `regrip`: this branch
        # never lifts by the fingers' grip on the cap, so it has no reason to
        # re-close the jaw on it first.
        handling = [
            bulb_release,
            # The jaw's cap-turning height is not the bulb's -- rise to meet
            # it before closing on it.
            Move(label="reach bulb", goal={Z: bulb_offset_z}, effort=lift,
                 tolerance_mm=2.0),
            centre_fingers,
            squeeze_bulb,
            # Reopening with the tip still in the bottle draws liquid up the
            # pipette; this is priming, not the dispense -- that is the
            # second `squeeze_bulb` below, once the pipette is clear of the
            # bottle.
            bulb_release,
            bulb_open,
            Move(label="lift bulb", goal={Z: bulb_lift_z}, effort=lift,
                 tolerance_mm=2.0),
            # Squeezes again from wherever "bulb open" left the jaw -- no
            # reopening first. This stroke is the dispense, not a fresh grip
            # search. Fingers are still centred from above -- nothing moved
            # them since.
            squeeze_bulb,
            bulb_open,
            Move(label="lower bulb", goal={Z: bulb_offset_z},
                 effort=reinsert_effort, accept_stall=True,
                 tolerance_mm=PUT_BACK_TOL_MM, creep=True),
            # `m.radius` is a jaw-centre-to-object distance, so clearing the
            # cap's full width needs the diameter, not the radius, past the
            # jaw's own opening.
            Move(label="bulb let go",
                 goal={AUX_JAW: lambda m: 2 * m.radius + cfg.bulb_release_mm}),
        ]
    else:
        handling = [
            release,
            Move(label="centre", goal={AUX: finger_span_mm / 2}),
            Hold(label="centre hold", group=AUX, goal=finger_span_mm / 2,
                 seconds=0.2),
            Probe(label="regrip", group=AUX_JAW, creep=True,
                  grip=squeeze_effort),
            lift_move,
            # The aux jaw keeps holding the cap while its two fingers carry it
            # back. Closing does not start until this move has arrived.
            # Wider tolerance than the 1 mm default: a finger carrying the cap
            # parks a couple of mm short of its stop, and a free `Move` rejects
            # a confirmed stall the tick it sees one. Measured 2.7 mm.
            Move(label="cap clear", goal={AUX: 0.0}, tolerance_mm=CARRY_TOL_MM),
            Hold(label="cap clear wait", seconds=cfg.clear_wait_s,
                 interruptible=True),
            # Return to the exact centred finger position used for the lift so
            # the held cap is above the bottle again before z descends.
            Move(label="cap align", goal={AUX: finger_span_mm / 2}),
        ]

    return Sequence([
        Move(label="height", goal={Z: cap_height}),
        Probe(label="probe", group=JAWS, creep=True,
              measure={"radius": AUX_JAW}),
        Hold(label="grip", group=JAWS, effort=squeeze_effort),
        Loop(count=cycles, rows=[
            Twist(label="open", jaw=AUX_JAW, left=AUX_LEFT, right=AUX_RIGHT,
                  radius=lambda m: m.radius, span=stroke_span_up,
                  grip=squeeze_effort, clearance=release_clearance_mm,
                  press=press_open,
                  measure={"radius": AUX_JAW}, count=strokes_up),
            *handling,
            # Return to the height where the cap was first gripped. Thread
            # engagement happens in the closing Twist's downward press.
            Move(label="put back", goal={Z: cap_height},
                 effort=reinsert_effort, accept_stall=True,
                 tolerance_mm=PUT_BACK_TOL_MM, creep=True),
            Twist(label="close", jaw=AUX_JAW, left=AUX_LEFT,
                  right=AUX_RIGHT, radius=lambda m: m.radius,
                  span=stroke_span_down, grip=close_squeeze_effort,
                  turn=close_effort,
                  clearance=release_clearance_mm, reverse=reverse, press=press,
                  measure={"radius": AUX_JAW}, count=strokes_down,
                  stop_on_stall=True),
            Move(label="let go",
                 goal={AUX_JAW: lambda m: m.radius + release_clearance_mm},
                 effort=squeeze_effort, loaded=True),
        ]),
        Move(label="present", goal={AUX: 0.0}, tolerance_mm=CARRY_TOL_MM),
    ], hand=hand,
       start_mm=start_mm,
       travel_torque=cfg.travel_torque,
       approach_torque=cfg.approach_torque,
       approach_speed=cfg.approach_speed,
       timeout_margin=cfg.timeout_margin,
       travel_speed=cfg.travel_speed)

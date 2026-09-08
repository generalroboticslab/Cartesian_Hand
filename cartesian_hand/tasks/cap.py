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

    cap_offset: float = field(default=10.0, metadata={"tune": (5.0, 40.0)})
    """Height of the cap's top face above z zero, mm."""
    num_revs_up: float = field(default=1, metadata={"tune": (0.0, 6.0)})
    """Revolutions during the opening (unscrewing) twist."""
    num_revs_down: float = field(default=0.8, metadata={"tune": (0.0, 6.0)})
    """Revolutions during the closing (screwing down) twist."""
    squeeze_torque: float = field(default=80.0, metadata={"tune": (40.0, 500.0)})
    close_torque: float = field(default=150.0, metadata={"tune": (0.0, 300.0)})
    """Finger torque while closing the cap. Zero automatically uses ten servo
    units below the effective opening torque; a positive value overrides it.
    Jaw release and grip continue to use `squeeze_torque`."""
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
    lift_mm: float = field(default=35.0, metadata={"tune": (5.0, 50.0)})
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


def build(hand: HandConfig, start_mm: torch.Tensor,
          cfg: Config | None = None) -> Sequence:
    """Probe once, then repeatedly remove and replace the same cap."""
    cfg = cfg or Config()
    mm = hand.clamped_mm
    finger_span_mm = mm(AUX_LEFT, cfg.finger_stroke)
    squeeze_effort = cfg.squeeze_torque / 1000.0
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

    return Sequence([
        Move(label="height", goal={Z: cap_height}),
        Probe(label="probe", group=JAWS, creep=True,
              measure={"radius": AUX_JAW}),
        Hold(label="grip", group=JAWS, effort=squeeze_effort),
        Loop(count=cycles, rows=[
            Twist(label="open", jaw=AUX_JAW, left=AUX_LEFT, right=AUX_RIGHT,
                  radius=lambda m: m.radius, span=stroke_span_up,
                  grip=squeeze_effort, clearance=release_clearance_mm,
                  measure={"radius": AUX_JAW}, count=strokes_up),
            release,
            Move(label="centre", goal={AUX: finger_span_mm / 2}),
            Hold(label="centre hold", group=AUX, goal=finger_span_mm / 2,
                 seconds=0.2),
            Probe(label="regrip", group=AUX_JAW, creep=True,
                  grip=squeeze_effort),
            # Raising loaded z must clear this hand's measured gravity floor.
            Move(label="lift", goal={Z: mm(Z, cfg.lift_mm)}, effort=lift,
                 tolerance_mm=2.0),
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
            # Return to the height where the cap was first gripped. Thread
            # engagement happens in the closing Twist's downward press.
            Move(label="put back", goal={Z: cap_height},
                 effort=reinsert_effort, accept_stall=True,
                 tolerance_mm=PUT_BACK_TOL_MM, creep=True),
            Twist(label="close", jaw=AUX_JAW, left=AUX_LEFT,
                  right=AUX_RIGHT, radius=lambda m: m.radius,
                  span=stroke_span_down, grip=squeeze_effort,
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

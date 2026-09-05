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

The base jaw's grip stands from the probe to the end, holding the bottle while
the auxiliary gripper removes and replaces the cap. Set `cycles` to zero to
repeat until the task is stopped; a positive value runs that many complete
open-and-close cycles.

The height and lift goals are absolute in the hand's millimetre frame, so the
hand must be zeroed first.
"""
from dataclasses import dataclass, field

import torch

from ..config import (AUX_FINGERS, AUX_JAW, AUX_LEFT, AUX_RIGHT, BASE_JAW,
                      HandConfig, Z)
from ..primitives import (Hold, Loop, Move, Probe, Sequence, Twist, TwistPress,
                          lift_effort, strokes_for_revolutions)

JAWS = (BASE_JAW, AUX_JAW)
AUX = tuple(AUX_FINGERS)

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


@dataclass
class Config:
    """Bench units. Every `tune` field becomes a slider on the studio page."""

    label: str = "Cycle cap"
    sets_datum: bool = False

    cap_offset: float = field(default=10.0, metadata={"tune": (5.0, 40.0)})
    """Height of the cap's top face above z zero, mm."""
    num_revs: float = field(default=1.0, metadata={"tune": (0.5, 6.0)})
    squeeze_torque: float = field(default=80.0, metadata={"tune": (40.0, 200.0)})
    close_torque: float = field(default=0.0, metadata={"tune": (0.0, 300.0)})
    """Finger torque while closing the cap. Zero automatically uses ten servo
    units below the effective opening torque; a positive value overrides it.
    Jaw release and grip continue to use `squeeze_torque`."""
    approach_torque: float = field(default=150.0, metadata={"tune": (50.0, 300.0)})
    travel_speed: float = field(default=1100.0, metadata={"tune": (200.0, 1500.0)})
    """Servo speed register for every free move, counts/s.

    The longest travel in this task is the twist's 45 mm finger sweep, which
    takes about 3.3 s at 1100 counts/s. Deadlines derive from this number, so
    lowering it lengthens them with it.

    **Not `config.STANDARD_SPEED`.** That is one number for the whole session
    and it is set by what `zero` needs -- hand_2's fingers sit at 200 counts/s,
    which makes that same sweep 16 s. The seek wants slow (a fast creep
    overshoots a hard stop and climbs a gear tooth); manipulation wants fast."""
    approach_speed: float = field(default=300.0, metadata={"tune": (25.0, 500.0)})
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
    lift_torque: float = field(default=1000.0,
                                metadata={"tune": (100.0, 1000.0)})
    """Z torque while lifting the gripper with the cap. This is floored at the
    hand's unloaded Z movement torque, but may need to be higher when loaded."""
    clear_wait_s: float = field(default=1.0, metadata={"tune": (0.0, 10.0)})
    """Maximum wait with the cap clear. An external signal ends it early."""
    cycles: float = field(default=1.0, metadata={"tune": (0.0, 10.0)})
    """Complete open-and-close cycles, rounded to an integer. Zero repeats
    until the task is stopped (disarm torque in Studio, or Ctrl-C headless)."""
    reinsert_z_torque: float = field(default=150.0,
                                     metadata={"tune": (50.0, 400.0)})
    """Downward z torque used to seat and thread the cap, servo units."""
    down_stroke_z: float = field(default=5.0, metadata={"tune": (0.0, 40.0)})
    """Absolute z goal while tightening, mm. Values above `cap_offset` are
    clamped to `cap_offset`, because tightening must not pull upward."""


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
    num_revs = torch.full_like(start_mm[:, 0], cfg.num_revs)
    reverse = torch.ones_like(num_revs, dtype=torch.bool)
    cap_height = mm(Z, cfg.cap_offset)
    lift = lift_effort(hand, cfg.lift_torque)
    cycles = float("inf") if cfg.cycles <= 0 else max(1, round(cfg.cycles))
    press = TwistPress(
        dof=Z,
        active=reverse,
        goal_mm=torch.full_like(num_revs, mm(
            Z, min(cfg.down_stroke_z, cfg.cap_offset))),
        return_mm=torch.full_like(num_revs, cap_height),
        effort=torch.full_like(num_revs, reinsert_effort),
        return_effort=torch.full_like(num_revs, lift),
    )

    strokes = lambda m: strokes_for_revolutions(
        num_revs, m.radius, finger_span_mm)
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
                  radius=lambda m: m.radius, span=finger_span_mm,
                  grip=squeeze_effort, clearance=release_clearance_mm,
                  measure={"radius": AUX_JAW}, count=strokes),
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
                 effort=reinsert_effort, accept_stall=True),
            Twist(label="close", jaw=AUX_JAW, left=AUX_LEFT,
                  right=AUX_RIGHT, radius=lambda m: m.radius,
                  span=finger_span_mm, grip=squeeze_effort,
                  turn=close_effort,
                  clearance=release_clearance_mm, reverse=reverse, press=press,
                  measure={"radius": AUX_JAW}, count=strokes,
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

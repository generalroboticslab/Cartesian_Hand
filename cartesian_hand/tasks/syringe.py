"""Draw and dispense a syringe plunger using contact-based gripping.

    entry -> settle -> clamp the body
      -> repeat(release -> lower -> pinch the plunger -> pull)
      -> show -> pause -> pinch -> push -> release
      -> rise -> pinch -> seat by contact -> open -> release z

The base jaw clamps the syringe body on first contact and holds it for the
entire task; only the aux jaw lets go, so it can slide to a fresh grip point
between strokes.

**The pull loop releases and lowers at its start, not its end.** The last pull
has to leave the plunger drawn up at `pull_z`, because that is where the push
half re-pinches it; releasing on the way out would put z back at `clearance_z`
and the next pinch would close on barrel. Leading, both rows are no-ops on the
first pass, so the loop needs no first-or-last special case.

Every goal is absolute in the hand's millimetre frame, so the hand must be
zeroed first -- `clearance_z` is a height above the z hard stop, not above
wherever the stage happened to be parked.
"""

from dataclasses import dataclass, field

import torch

from ..config import (AUX_FINGERS, AUX_JAW, BASE_FINGERS, BASE_JAW, HandConfig,
                      Z)
from ..primitives import Hold, Loop, Move, Probe, Sequence, lift_effort

JAWS = (BASE_JAW, AUX_JAW)
FINGERS = tuple(BASE_FINGERS + AUX_FINGERS)

Z_TOLERANCE_MM = 5.0
"""Arrival band for a z ascent, against `Move`'s 1.0 mm default.

Neither `pull_z` nor `regrip_z` is a coordinate the task depends on, and
`Sequence` keeps commanding the goal through the rows that follow, so a short
park is a shorter stroke rather than a failed row. The stall detector's own
1.0 mm window is untouched: a stop further out than this still fails."""


@dataclass
class Config:
    """Bench units. Every `tune` field becomes a slider on the studio page."""

    label: str = "Draw syringe"
    sets_datum: bool = False

    clearance_z: float = field(default=5.0, metadata={"tune": (2.0, 20.0)})
    """Z height the aux jaw grips the plunger at for the low point of each pull,
    and where it pushes the plunger back down to, mm. A floor: below it the
    stage drives into the syringe's flanges before the plunger stalls it."""
    aux_min_mm: float = field(default=5.0, metadata={"tune": (0.0, 15.0)})
    """Floor for the aux jaw. Every pinch targets this instead of 0 -- closing
    past it drives the jaw into the plunger's flange guide before contact."""
    pull_z: float = field(default=25.0, metadata={"tune": (10.0, 50.0)})
    """Z height each pull stroke draws the plunger up to, mm."""
    regrip_z: float = field(default=30.0, metadata={"tune": (20.0, 50.0)})
    """Z height the aux jaw re-grips the plunger at before the final seat."""
    pull_strokes: int = 2
    """Pinch/pull/release/reset cycles needed to fully draw the plunger.
    Not tunable: one stroke's travel does not cover the draw, and changing the
    count changes the procedure rather than a parameter of it."""
    squeeze_torque: float = field(default=100.0, metadata={"tune": (50.0, 300.0)})
    """Holding torque for every contact grip: the base jaw on the body, the aux
    jaw on the plunger, and the final z seat."""
    approach_torque: float = field(default=100.0, metadata={"tune": (50.0, 300.0)})
    """Torque while closing onto an object or driving into a hard stop."""
    approach_speed: float = field(default=300.0, metadata={"tune": (25.0, 500.0)})
    """Servo speed register while finding contact, counts/s.

    **Raised 50 -> 300, for `cap`'s reason.** There is no contact sensor, so
    contact is `CONFIRM_TICKS` under `STUCK_SPEED_MM_S` = 0.3 mm/s; at 50
    counts/s the creep itself runs at 0.61 mm/s, twice the threshold it is
    tested against, and any servo hesitation reads as the plunger."""
    push_torque: float = field(default=100.0, metadata={"tune": (50.0, 500.0)})
    """Torque driving z down onto the plunger. Deliberately not raised to z's
    `torque_min_to_move`: that floor is for the lifting direction, and pressing
    down lightly is what makes the seat a measurement instead of a crush."""
    travel_torque: float = field(default=50.0, metadata={"tune": (30.0, 200.0)})
    """Torque for unloaded horizontal travel and z descent."""
    lift_torque: float = field(default=800.0, metadata={"tune": (400.0, 1000.0)})
    """Z torque while raising the gripper -- `pull` and `rise`. Floored at the
    hand's unloaded z torque, which is not enough loaded: at hand_2's 400,
    `rise` stalled at 40.7 mm on one bench run and 34.5 mm on the next, both
    against the same goal. First number to raise if either row stalls again."""
    jaw_opening: float = field(default=25.0, metadata={"tune": (10.0, 40.0)})
    """How wide the aux jaw opens to release, mm. Capped by `clamped_mm`, never
    a DOF's `max_mm` -- the travel table is CAD and reads high."""
    settle_s: float = field(default=1.0, metadata={"tune": (0.1, 3.0)})
    """Pause after the entry move, before clamping the body."""
    reveal_s: float = field(default=2.0, metadata={"tune": (0.0, 5.0)})
    """Pause with the drawn plunger visible, between the pull and push halves."""
    timeout_margin: float = field(default=1.5, metadata={"tune": (1.0, 3.0)})
    """Margin on the deadline each row derives from its own travel, at the speed
    it commands. Flat second counts expired 15 of `cap`'s 18 rows."""


def build(hand: HandConfig, start_mm: torch.Tensor,
          cfg: Config | None = None) -> Sequence:
    """Clamp the body once, draw the plunger in strokes, then dispense and seat."""
    cfg = cfg or Config()
    mm = hand.clamped_mm
    jaw_floor = max(float(hand.gain_vector("torque_min_to_move")[dof])
                    for dof in JAWS)
    squeeze_effort = max(cfg.squeeze_torque, jaw_floor) / 1000.0
    push_effort = cfg.push_torque / 1000.0
    lift = lift_effort(hand, cfg.lift_torque)

    jaw_open = mm(BASE_JAW, cfg.jaw_opening)
    clearance = mm(Z, cfg.clearance_z)
    pinch_floor = mm(AUX_JAW, cfg.aux_min_mm)

    # The aux jaw is never commanded shut: the validated implementation closed
    # it to `aux_min_mm` and only got away with it because its `set_pos` timeout
    # was unchecked -- with a plunger in the way the jaw stalls and the move
    # never converges. Every close here is a `Probe`, so contact is a
    # measurement rather than a coincidence.
    release = Move(label="release", goal={AUX_JAW: jaw_open},
                   effort=squeeze_effort, loaded=True)
    pinch = Probe(label="pinch", group=AUX_JAW, goal=pinch_floor, creep=True,
                  grip=squeeze_effort)

    return Sequence([
        Move(label="entry", goal={FINGERS: 0.0}),
        # z second and on its own: the entry move may ascend, and effort is a
        # cap, so it carries the lift floor -- far too much force to also put
        # behind a jaw sweeping through free space.
        Move(label="clearance", goal={Z: clearance}, effort=lift),
        Hold(label="settle", seconds=cfg.settle_s),
        # The body clamp becomes a standing squeeze and is never named again.
        Probe(label="body", group=BASE_JAW, creep=True, grip=squeeze_effort),
        Loop(count=cfg.pull_strokes, rows=[
            release,
            Move(label="lower", goal={Z: clearance}),
            pinch,
            Move(label="pull", goal={Z: mm(Z, cfg.pull_z)}, effort=lift,
                 tolerance_mm=Z_TOLERANCE_MM),
        ]),
        Move(label="show", goal={AUX_JAW: jaw_open}, effort=squeeze_effort,
             loaded=True),
        Hold(label="reveal", seconds=cfg.reveal_s),
        pinch,
        # Parking short is the normal outcome of a push capped at `push_torque`,
        # not a fault; without `accept_stall` the row waits out its deadline and
        # retires the sequence, so the seat never runs. `accept_stall` rather
        # than `loaded`: a stop is an arrival here, but the effort must stay the
        # descent number instead of being raised to the travel floor.
        Move(label="push", goal={Z: clearance}, effort=push_effort,
             accept_stall=True),
        release,
        Move(label="rise", goal={Z: mm(Z, cfg.regrip_z)}, effort=lift,
             tolerance_mm=Z_TOLERANCE_MM),
        pinch,
        # Contact-based, not a fixed depth. A seat that reaches `clearance_z`
        # without stalling never felt the plunger -- a slipped grip and a
        # dispense are identical as positions, and `Probe` tells them apart.
        Probe(label="seat", group=Z, goal=clearance, creep=True,
              effort=push_effort, grip=squeeze_effort),
        Move(label="open", goal={AUX_JAW: jaw_open}, effort=squeeze_effort,
             loaded=True),
        # The rack is self-locking and the stage is parked low, so holding
        # position here only heats the servo -- measured 48 -> 60 C idling,
        # against a ~70 C trip that would drop the stage uncontrolled.
        Hold(label="let go", group=Z, goal=clearance, effort=0.0),
    ], hand=hand,
       start_mm=start_mm,
       travel_torque=cfg.travel_torque,
       approach_torque=cfg.approach_torque,
       approach_speed=cfg.approach_speed,
       timeout_margin=cfg.timeout_margin)

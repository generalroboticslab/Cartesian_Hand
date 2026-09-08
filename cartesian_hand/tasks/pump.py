"""Pump dispenser and syringe: two contact-based tasks sharing one jaw
geometry and the same push/rise mechanism, in one file.

`Config`/`build` below are the pump dispenser -- this module's own name is
what `tasks/__init__.py` discovers, so they are what `--task pump` and the
GUI's "pump" entry mean. `SyringeConfig`/`build_syringe` are the syringe,
unchanged from when this file was `syringe.py`; `tasks/syringe.py` is now a
two-line variant that reaches them so the syringe keeps its own GUI button
and its own `--task syringe` -- the same pattern `tasks/__init__.py`
documents for `cap_gentle`, used here to add a task rather than restrict one.

Syringe (draw and dispense a syringe plunger)
    entry -> settle -> clamp the body
      -> repeat(release -> lower -> pinch the plunger -> pull)
      -> show -> pause -> pinch -> push -> release
      -> rise -> pinch -> seat by contact -> open -> release z

Pump dispenser (soap, lotion, spray-pump bottles)
    entry -> height -> clamp the bottle body (hard) -> close on the pump head (fast)
      -> repeat(press to contact -> reset to full height)

The pump dispenser is syringe's push-seat-release tail -- pinch/close, drive
z down onto an object it stops short of, treat the stall as arrival -- run
`num_strokes` times against a bottle instead of a barrel. It holds one grip
through every stroke instead of releasing and re-pinching between them like
syringe's pull loop: neither jaw needs to clear anything but the object it
is already on, so there is nothing to regrip. Two of syringe's numbers
change rather than the shape of the task: the base jaw squeezes at the
servo's own torque ceiling instead of syringe's 100 (the bottle has to stay
put through every one of `num_strokes` pushes, which needs holding far
harder than a syringe barrel ever does), and the aux jaw's close onto the
pump head is left at `creep`'s default (`False`, the travel-speed close)
instead of syringe's slow contact creep -- the head is a small, consistent
target across bottles, so there is nothing delicate to find the way there is
in a plunger's flange guide. It is also a `Move`, not a `Probe`, on the way
in: closing all the way to 0 (the head is thin enough to bottom the jaw out
while still gripping it) is a normal outcome here, where `Probe` would read
it as an empty-jaw failure; `accept_stall=True` still takes an earlier stop
against real resistance as arrival too, so either way of closing counts.

The base jaw clamps its object on first contact and holds it for the whole
task in both. Only syringe's aux jaw ever lets go mid-task, to slide to a
fresh grip point between pull strokes; the pump dispenser's aux jaw closes
once, on the pump head, and stays closed for every stroke.

Every goal is absolute in the hand's millimetre frame, so the hand must be
zeroed first -- every height here, `clearance_z` included, is above the z
hard stop, not above wherever the stage happened to be parked.
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
    """Pump dispenser. Bench units. Every `tune` field becomes a slider on
    the studio page."""

    label: str = "Pump dispenser"
    sets_datum: bool = False

    num_strokes: float = field(default=3.0, metadata={"tune": (1.0, 10.0)})
    """Pump strokes to run, rounded to the nearest integer. Each is a press
    to contact followed by a reset to z's own full travel, including the
    last -- the task ends with z back up, not sitting at `press_mm`."""
    squeeze_torque: float = field(default=1000.0, metadata={"tune": (500.0, 1000.0)})
    """Holding torque for the base jaw's grip on the bottle body -- maxed at
    the servo's own ceiling, not syringe's 100: the bottle has to stay put
    through every one of `num_strokes` downward pushes, which needs holding
    far harder than a syringe barrel ever does."""
    aux_close_torque: float = field(default=800.0, metadata={"tune": (300.0, 1000.0)})
    """Torque the aux jaw closes onto the pump head with, and holds for
    every stroke afterward -- has to press through the head's own
    resistance to close at all, unlike syringe's plunger pinch."""
    press_mm: float = field(default=15.0, metadata={"tune": (5.0, 30.0)})
    """Z target for each down stroke. Not z's own hard stop (0): the pump
    housing blocks the carriage well short of it, so a push reliably bottoms
    out around 20-23mm on the bench before ever reaching 0. `press_mm` asks
    for a bit more travel than that bottoming point, and `accept_stall` (see
    `Move`) still lets a push that meets resistance early stop there instead
    of forcing through it."""
    push_torque: float = field(default=1000.0, metadata={"tune": (500.0, 1000.0)})
    """Torque driving z through the pump's own resistance, both down and
    back up. Maxed out, unlike syringe's deliberately gentle `push_torque`:
    a slow, soft stroke on a pump risks resetting the head through its own
    internal detent instead of completing the dispense."""
    approach_torque: float = field(default=100.0, metadata={"tune": (50.0, 300.0)})
    """Torque while the base jaw is still seeking contact on the bottle,
    before `squeeze_torque` takes over."""
    approach_speed: float = field(default=300.0, metadata={"tune": (25.0, 500.0)})
    """Servo speed register while the base jaw creeps toward contact,
    counts/s -- same number and reason as syringe's own `approach_speed`.
    The aux jaw's close never creeps, so this does not govern it."""
    travel_torque: float = field(default=50.0, metadata={"tune": (30.0, 200.0)})
    """Torque for unloaded travel."""
    jaw_opening: float = field(default=25.0, metadata={"tune": (10.0, 40.0)})
    """How wide both jaws open at entry, before the base jaw closes on the
    bottle and the aux jaw closes on the pump head, mm."""
    jaw_settle_s: float = field(default=0.3, metadata={"tune": (0.0, 1.0)})
    """Pause after the aux jaw closes on the pump head, before the first
    stroke -- same reason as syringe's `jaw_settle_s`: a dof4 (aux jaw) row
    immediately followed by a dof3 (z) row needs this between them, or z can
    start pressing while the jaw is still mechanically settling shut."""
    timeout_margin: float = field(default=1.5, metadata={"tune": (1.0, 3.0)})
    """Margin on the deadline each row derives from its own travel, at the
    speed it commands -- same reason as syringe's own `timeout_margin`."""


def build(hand: HandConfig, start_mm: torch.Tensor,
         cfg: Config | None = None) -> Sequence:
    """Clamp the bottle hard, close on the pump head fast, then repeat a
    contact-based press/reset stroke `num_strokes` times."""
    cfg = cfg or Config()
    mm = hand.clamped_mm
    jaw_floor = max(float(hand.gain_vector("torque_min_to_move")[dof])
                    for dof in JAWS)
    squeeze_effort = max(cfg.squeeze_torque, jaw_floor) / 1000.0
    aux_close_effort = max(cfg.aux_close_torque, jaw_floor) / 1000.0
    push_effort = cfg.push_torque / 1000.0
    strokes = max(1, round(cfg.num_strokes))

    jaw_open = mm(BASE_JAW, cfg.jaw_opening)
    z_max = mm(Z, hand.travel_mm[Z])

    return Sequence([
        Move(label="entry", goal={JAWS: jaw_open, FINGERS: 0.0}),
        # z on its own, after: the entry move may ascend, and effort is a
        # cap, so it carries the (unloaded) lift floor -- far too much force
        # to also put behind a jaw sweeping through free space.
        Move(label="height", goal={Z: z_max}, effort=lift_effort(hand)),
        # The body clamp becomes a standing squeeze and is never named again
        # -- same contact-based close as syringe's, at the servo's own
        # torque ceiling instead of syringe's 100 (see `Config.squeeze_torque`).
        Probe(label="body", group=BASE_JAW, creep=True, grip=squeeze_effort),
        # A `Move`, not a `Probe`: closing all the way to 0 is a legitimate
        # outcome here (the head is thin enough to let the jaw bottom out
        # while still gripping it), and `Probe` treats reaching its goal as
        # an empty-jaw failure regardless. `accept_stall=True` still takes an
        # earlier stop against real resistance as arrival, at `Move`'s own
        # 1 mm tolerance -- so either a stall short of 0 or a full close to 0
        # counts. `creep` left at its default (False), unlike syringe's
        # plunger pinch, so this closes at full travel speed -- the head is a
        # small, consistent target across bottles, unlike a plunger's flange
        # guide, so there is nothing delicate to find by creeping.
        Move(label="pump head", goal={AUX_JAW: 0.0}, effort=aux_close_effort,
             accept_stall=True),
        Hold(label="jaw settle", seconds=cfg.jaw_settle_s),
        Loop(count=strokes, rows=[
            # Parking short of `press_mm` is the normal outcome of a stroke
            # against the pump's own resistance, not a fault -- see
            # `Config.press_mm` and syringe's identical "push" row.
            Move(label="press", goal={Z: mm(Z, cfg.press_mm)}, effort=push_effort,
                 accept_stall=True),
            # Every stroke resets to z's own full travel, including the
            # last: the task ends with z back up, not sitting at `press_mm`.
            # Same maxed-out `push_torque` as the press -- see
            # `Config.push_torque`. `Z_TOLERANCE_MM`, not the 1.0 mm default,
            # for the same reason as syringe's "pull"/"rise": z_max is not a
            # coordinate the rest of the task depends on, so a short park is
            # a shorter climb, not a failed row.
            Move(label="reset", goal={Z: z_max}, effort=push_effort,
                 tolerance_mm=Z_TOLERANCE_MM),
        ]),
        # Self-locking rack, stage parked at the top -- holding position here
        # only heats the servo, same reasoning as syringe's own "let go".
        Hold(label="let go", group=Z, goal=z_max, effort=0.0),
    ], hand=hand,
       start_mm=start_mm,
       travel_torque=cfg.travel_torque,
       approach_torque=cfg.approach_torque,
       approach_speed=cfg.approach_speed,
       timeout_margin=cfg.timeout_margin)


@dataclass
class SyringeConfig:
    """Syringe. Bench units. Every `tune` field becomes a slider on the
    studio page."""

    label: str = "Draw syringe"
    sets_datum: bool = False

    clearance_z: float = field(default=10.0, metadata={"tune": (2.0, 20.0)})
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
    jaw_settle_s: float = field(default=0.3, metadata={"tune": (0.0, 1.0)})
    """Pause after the aux jaw opens or closes, before the next z move.
    `Move`'s tolerance band and `Probe`'s contact can both be satisfied while
    the jaw is still mechanically settling into that open or closed position;
    without this, z starts a draw or a push while dof4 is still moving, which
    reads as slip rather than a clean stroke."""


def build_syringe(hand: HandConfig, start_mm: torch.Tensor,
                  cfg: SyringeConfig | None = None) -> Sequence:
    """Clamp the body once, draw the plunger in strokes, then dispense and seat."""
    cfg = cfg or SyringeConfig()
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
    # Every dof4 (aux jaw) row that a dof3 (z) row immediately follows needs
    # this between them -- see `SyringeConfig.jaw_settle_s`. "show" -> "reveal"
    # is the one exception: `reveal_s` is already a multi-second hold, far past
    # the jaw's own settling time.
    jaw_settle = Hold(label="jaw settle", seconds=cfg.jaw_settle_s)

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
            jaw_settle,
            Move(label="lower", goal={Z: clearance}),
            pinch,
            jaw_settle,
            Move(label="pull", goal={Z: mm(Z, cfg.pull_z)}, effort=lift,
                 tolerance_mm=Z_TOLERANCE_MM),
        ]),
        Move(label="show", goal={AUX_JAW: jaw_open}, effort=squeeze_effort,
             loaded=True),
        Hold(label="reveal", seconds=cfg.reveal_s),
        pinch,
        jaw_settle,
        # Parking short is the normal outcome of a push capped at `push_torque`,
        # not a fault; without `accept_stall` the row waits out its deadline and
        # retires the sequence, so the seat never runs. `accept_stall` rather
        # than `loaded`: a stop is an arrival here, but the effort must stay the
        # descent number instead of being raised to the travel floor.
        Move(label="push", goal={Z: clearance}, effort=push_effort,
             accept_stall=True),
        release,
        jaw_settle,
        Move(label="rise", goal={Z: mm(Z, cfg.regrip_z)}, effort=lift,
             tolerance_mm=Z_TOLERANCE_MM),
        pinch,
        jaw_settle,
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

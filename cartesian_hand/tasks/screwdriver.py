"""Turn a manual screwdriver, sizing the handle grip from contact.

    entry -> settle -> close the base jaw on the shaft (taut, never squeezed)
      -> probe the handle and squeeze it
      -> repeat(release -> reset fingers -> re-grip -> [press z] -> turn
                -> [back z off])

The bracketed z rows run only when `cw` is set: clockwise drives the screw down
and it must be pressed in as it turns, while counter-clockwise backs the screw
out and the screw lifts the tool itself. That is one `TwistPress` whose
`active` flag is per environment, not a second row table.

The stroke count is written down nowhere: it is `2*pi*r / finger_span` rounded
up, with a floor of two strokes per revolution, and `r` is what the aux jaw
measured -- re-measured on every re-grip, so it tracks the handle as the tool
rides up out of the screw.

Every goal is absolute in the hand's millimetre frame, so the hand must be
zeroed first -- `tool_offset` is a height above the z hard stop.

No hardware results for *this* transcription yet: the sequence is validated, the
port of it is not.
"""

from dataclasses import dataclass, field

import torch

from ..config import (AUX_FINGERS, AUX_JAW, AUX_LEFT, AUX_RIGHT, BASE_FINGERS,
                      BASE_JAW, HandConfig, Z)
from ..primitives import (Hold, Move, Probe, Sequence, Twist, TwistPress,
                          lift_effort, strokes_for_revolutions)

JAWS = (BASE_JAW, AUX_JAW)
BASE = tuple(BASE_FINGERS)
AUX = tuple(AUX_FINGERS)


@dataclass
class Config:
    """Bench units. Every `tune` field becomes a slider on the studio page."""

    label: str = "Turn screwdriver"
    """Button text on the studio page. Empty means no button, and a variant that
    does not name its own inherits none -- see `tasks/__init__.py`.

    Named, though this port has no hardware results yet: reaching it through
    the tune panel's dropdown meant every unvalidated task was one collapsed
    folder and a button called **Run tuned** away from being run, which is
    where an operator does not look for "run the screwdriver task"."""
    sets_datum: bool = False

    tool_offset: float = field(default=25.0, metadata={"tune": (5.0, 40.0)})
    """Height of the screwdriver handle above z zero, in mm."""
    num_revs: float = field(default=1.0, metadata={"tune": (0.5, 6.0)})
    """Full turns to drive per run."""
    cw: bool = True
    """Turn clockwise (screws down) if True, counter-clockwise if False.

    Only `cw` drives z down during the twist. **Not tunable**: it is not a
    number, and it selects which way the tool turns rather than how far."""
    approach_torque: float = field(default=100.0, metadata={"tune": (50.0, 300.0)})
    """Torque while closing a jaw onto the tool.

    Also the base jaw's final hold: the tip only needs to sit taut, so it is
    never squeezed harder than this."""
    handle_squeeze_torque: float = field(default=300.0,
                                         metadata={"tune": (100.0, 500.0)})
    """Holding torque on the aux jaw and fingers gripping the handle. This is
    the jaw that does the twisting, so it squeezes tight."""
    approach_speed: float = field(default=300.0, metadata={"tune": (25.0, 500.0)})
    """Servo speed register while finding contact, counts/s.

    **Raised 50 -> 300, for `cap`'s reason.** There is no contact sensor, so
    contact is `CONFIRM_TICKS` of measured speed under `STUCK_SPEED_MM_S` =
    0.3 mm/s; at 50 counts/s the creep itself runs at 0.61 mm/s, twice the
    threshold it is tested against, and any servo hesitation reads as the
    handle. At 300 the margin is 12x and a 20 mm close takes 5.4 s not 33."""
    travel_speed: float = field(default=1100.0, metadata={"tune": (200.0, 1500.0)})
    """Servo speed register for every free move, counts/s.

    **Not `config.STANDARD_SPEED`.** That is one number for the whole session
    and it is set by what `zero` needs -- hand_2's fingers sit at 200 counts/s,
    which makes the twist's 40 mm finger sweep 16 s. The seek wants slow (a
    fast creep overshoots a hard stop and climbs a gear tooth); manipulation
    wants fast. Deadlines derive from this number, so lowering it lengthens
    them with it."""
    travel_torque: float = field(default=50.0, metadata={"tune": (30.0, 200.0)})
    """Torque for unloaded horizontal travel."""
    release_clearance: float = field(default=3.0, metadata={"tune": (1.0, 10.0)})
    """How far beyond the measured handle radius the aux jaw opens between
    strokes, in mm."""
    jaw_opening: float = field(default=25.0, metadata={"tune": (10.0, 40.0)})
    """How wide each jaw opens before closing on the tool, in mm.

    Capped by `clamped_mm`, never a DOF's `max_mm` -- the travel table is CAD
    and reads high. The validated implementation used `max_mm`; this does not."""
    finger_stroke: float = field(default=40.0, metadata={"tune": (10.0, 55.0)})
    """Full sweep of an auxiliary finger during one twist, in mm."""
    down_stroke_torque: float = field(default=150.0,
                                      metadata={"tune": (50.0, 500.0)})
    """Torque on z for the cw down-stroke. Unused when `cw` is False.

    A descent, so it is deliberately not raised to z's `torque_min_to_move`:
    that floor is for the lifting direction. The *return* from this press is a
    lift and does get the floor -- see `build`."""
    down_stroke_z: float = field(default=20.0, metadata={"tune": (0.0, 45.0)})
    """Z height each cw stroke presses to and holds through the turn, in mm.
    Reset to `tool_offset` after every stroke. Unused when `cw` is False."""
    settle_s: float = field(default=1.0, metadata={"tune": (0.1, 3.0)})
    """Pause after the entry move, before probing."""
    timeout_margin: float = field(default=1.5, metadata={"tune": (1.0, 3.0)})
    """Margin on the deadline each row derives from its own travel, at the
    speed it commands."""


def build(hand: HandConfig, start_mm: torch.Tensor,
          cfg: Config | None = None) -> Sequence:
    """Take a taut tip grip and a tight handle grip, then turn `num_revs`."""
    cfg = cfg or Config()
    mm = hand.clamped_mm
    jaw_floor = max(float(hand.gain_vector("torque_min_to_move")[dof])
                    for dof in JAWS)
    tip_effort = max(cfg.approach_torque, jaw_floor) / 1000.0
    handle_effort = max(cfg.handle_squeeze_torque, jaw_floor) / 1000.0
    finger_span_mm = mm(AUX_LEFT, cfg.finger_stroke)
    tool_height = mm(Z, cfg.tool_offset)

    num_revs = torch.full_like(start_mm[:, 0], cfg.num_revs)
    # Clockwise drives the screw in, so it presses z to depth and holds it
    # through the turn; counter-clockwise leaves z alone, because the screw
    # pushes the tool out on its own as it backs free. The same flag reverses
    # which finger leads the stroke.
    clockwise = torch.full_like(num_revs, float(bool(cfg.cw))).bool()
    press = TwistPress(
        dof=Z,
        active=clockwise,
        goal_mm=torch.full_like(num_revs, mm(Z, cfg.down_stroke_z)),
        return_mm=torch.full_like(num_revs, tool_height),
        effort=torch.full_like(num_revs, cfg.down_stroke_torque / 1000.0),
        # Backing z off raises the stage, so it gets the measured floor rather
        # than the press effort -- z torque is directional.
        return_effort=torch.full_like(num_revs, lift_effort(hand)),
    )

    return Sequence([
        # Base fingers pinched at 0, not centred: the shaft wants a narrow point
        # contact, not a cradle. Aux fingers centred so the first stroke has
        # span to travel into either way.
        Move(label="entry", goal={JAWS: mm(BASE_JAW, cfg.jaw_opening),
                                  BASE: 0.0, AUX: finger_span_mm / 2}),
        # z second and on its own: the entry move may ascend and so carries the
        # lift floor, which is far too much force to also put behind a jaw
        # sweeping through free space.
        Move(label="height", goal={Z: tool_height}, effort=lift_effort(hand)),
        Hold(label="settle", seconds=cfg.settle_s),
        # The tip closes onto the shaft and stops. Never squeezed harder, and it
        # holds this taut grip for the whole task.
        Probe(label="tip", group=BASE_JAW, creep=True, grip=tip_effort),
        # The handle: probed, then squeezed tight -- this is the jaw that turns.
        Probe(label="handle", group=AUX_JAW, creep=True, grip=handle_effort,
              measure={"radius": AUX_JAW}),
        Twist(label="turn", jaw=AUX_JAW, left=AUX_LEFT, right=AUX_RIGHT,
              radius=lambda m: m.radius, span=finger_span_mm,
              grip=handle_effort, clearance=cfg.release_clearance,
              reverse=clockwise, press=press,
              measure={"radius": AUX_JAW},
              count=lambda m: strokes_for_revolutions(num_revs, m.radius,
                                                      finger_span_mm)),
    ], hand=hand,
       start_mm=start_mm,
       travel_torque=cfg.travel_torque,
       approach_torque=cfg.approach_torque,
       approach_speed=cfg.approach_speed,
       timeout_margin=cfg.timeout_margin,
       travel_speed=cfg.travel_speed)

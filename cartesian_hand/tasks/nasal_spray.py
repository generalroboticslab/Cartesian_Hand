"""Actuate a nasal spray pump: no twist, just a repeated vertical press.

    move to spray height -> probe both jaws (grip bottle & nozzle together)
      -> hold grip -> repeat cycles( press z down -> return z up )

Borrows `cap.py`'s two-jaw contact probe -- one `Probe` across both `JAWS` at
once, each latching its own contact independently -- so the bottle body
(base jaw) and the spray nozzle (aux jaw) are found and gripped in a single
row, the same as `cap.py`'s own "probe" row. Everything after that is
different: a nasal spray fires with a straight push on its own nozzle, not a
turn, so this task has no `Twist`/`TwistPress` at all -- just z pressed down
by `press_depth_mm` and back up, `cycles` times.

`Z` is the aux (bridge) side's height above the base side, not a room-frame
height -- lower moves the aux jaw's grip toward the base-held bottle, which
is the pressing motion that fires the pump; higher backs it off, which is
what a spring-return nozzle needs before the next press can register as a
fresh stroke. The press is a descent (gravity assists, so `press_torque` is
not floored -- the same reasoning `cap.py`'s `reinsert_z_torque` uses for its
own descents); the return is an ascent against gravity, so it goes through
`lift_effort`, which is floored at z's own `torque_min_to_move`, the same as
every other z ascent in this codebase.

Every goal is absolute in the hand's millimetre frame, so the hand must be
zeroed first.

No hardware results yet.
"""
from dataclasses import dataclass, field

import torch

from ..config import AUX_JAW, BASE_JAW, HandConfig, Z
from ..primitives import Hold, Loop, Move, Probe, Sequence, lift_effort

JAWS = (BASE_JAW, AUX_JAW)


@dataclass
class Config:
    """Bench units. Every `tune` field becomes a slider on the studio page."""

    label: str = "Nasal spray"
    sets_datum: bool = False

    spray_offset: float = field(default=15.0, metadata={"tune": (0.0, 40.0)})
    """Height (z) the two jaws grip the bottle and nozzle at, and the height
    each press returns to, mm above z zero."""
    press_depth_mm: float = field(default=15.0, metadata={"tune": (2.0, 30.0)})
    """How far z presses down from `spray_offset` to fire the nozzle each
    cycle, mm."""
    squeeze_torque: float = field(default=80.0, metadata={"tune": (40.0, 500.0)})
    """Jaw grip effort, latched on contact and held for the whole task."""
    approach_torque: float = field(default=150.0, metadata={"tune": (50.0, 300.0)})
    """Torque while the jaws are still searching for contact, before
    `squeeze_torque` latches in."""
    approach_speed: float = field(default=800.0, metadata={"tune": (25.0, 800.0)})
    """Servo speed register while closing the jaws to find contact, counts/s.

    Fast for `cap`'s reason: there is no contact sensor, so contact is read
    as a confirmed stop under 0.3 mm/s, and a slow creep runs close enough to
    that threshold that ordinary servo hesitation reads as contact."""
    press_torque: float = field(default=150.0, metadata={"tune": (50.0, 500.0)})
    """Z torque for the down-stroke that fires the nozzle. A descent, so
    (like `cap.py`'s `reinsert_z_torque`) this is deliberately not floored at
    z's `torque_min_to_move`: that floor is for the lifting direction."""
    release_torque: float = field(default=500.0, metadata={"tune": (100.0, 1000.0)})
    """Z torque while returning the aux jaw's grip up, off the nozzle. Floored
    at the hand's unloaded z torque via `lift_effort`; raise it if the return
    stalls part way."""
    travel_torque: float = field(default=50.0, metadata={"tune": (30.0, 200.0)})
    travel_speed: float = field(default=1500.0, metadata={"tune": (200.0, 1500.0)})
    """Unused: every free move now runs at the servo's rated no-load top
    speed regardless of this value."""
    timeout_margin: float = field(default=1.5, metadata={"tune": (1.0, 3.0)})
    cycles: float = field(default=1.0, metadata={"tune": (0.0, 20.0)})
    """Press/release cycles, rounded to an integer. Zero repeats until the
    task is stopped (disarm torque in Studio, or Ctrl-C headless)."""


def build(hand: HandConfig, start_mm: torch.Tensor,
          cfg: Config | None = None) -> Sequence:
    """Grip the bottle and nozzle once, then press and release z, `cycles` times."""
    cfg = cfg or Config()
    mm = hand.clamped_mm
    jaw_floor = max(float(hand.gain_vector("torque_min_to_move")[dof])
                    for dof in JAWS)
    squeeze_effort = max(cfg.squeeze_torque, jaw_floor) / 1000.0
    press_effort = cfg.press_torque / 1000.0
    release = lift_effort(hand, cfg.release_torque)
    spray_height = mm(Z, cfg.spray_offset)
    press_depth = mm(Z, cfg.spray_offset - cfg.press_depth_mm)
    cycles = float("inf") if cfg.cycles <= 0 else max(1, round(cfg.cycles))

    return Sequence([
        Move(label="height", goal={Z: spray_height}),
        Probe(label="probe", group=JAWS, creep=True, grip=squeeze_effort),
        Hold(label="grip", group=JAWS, effort=squeeze_effort),
        Loop(count=cycles, rows=[
            Move(label="press", goal={Z: press_depth}, effort=press_effort,
                 accept_stall=True),
            Move(label="release", goal={Z: spray_height}, effort=release,
                 tolerance_mm=2.0),
        ]),
    ], hand=hand,
       start_mm=start_mm,
       travel_torque=cfg.travel_torque,
       approach_torque=cfg.approach_torque,
       approach_speed=cfg.approach_speed,
       timeout_margin=cfg.timeout_margin,
       travel_speed=cfg.travel_speed)

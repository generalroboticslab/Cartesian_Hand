"""Open and close a bottle cap from measured dimensions.

The operator supplies the bottle and cap diameters instead of the hand probing
for them. Faster and more repeatable than `caps_contact_based` when the
container is known, and blind to a misplaced bottle.
"""

import time
from dataclasses import dataclass

import numpy as np

from .roles import AUX_JAW, AUX_LEFT, AUX_RIGHT, BASE_JAW, BASE_LEFT, BASE_RIGHT, Z

DESCRIPTION = "Open/close a bottle cap from measured dimensions"


@dataclass
class Config:
    open: bool = False
    """Run the opening sequence."""
    close: bool = False
    """Run the closing sequence."""
    bottle_diameter: float = 60.0
    """Measured bottle body diameter, in mm."""
    cap_diameter: float = 30.0
    """Measured cap diameter, in mm."""
    num_revs: float = 3.0
    """Full turns needed to free or seat the cap."""
    cap_offset: float = 20.0
    """Height of the cap's top face above z zero, in mm."""

GRIP_CLEARANCE = 10.0     # jaw opening beyond the widest part, mm
BOTTLE_INTERFERENCE = 2.0  # jaw closes this far inside the bottle radius
CAP_INTERFERENCE = 1.0     # jaw closes this far inside the cap radius
CAP_RELEASE = 5.0          # jaw backs off this far to let the cap turn free
EXTRACT_LIFT = 30.0        # z travel to pull the cap off the threads


def caps(hand, bottle_diameter: float = 60.0, cap_diameter: float = 30.0,
         num_revs: float = 3.0, cap_offset: float = 20.0,
         do_open: bool = True, do_close: bool = True,
         speed: int = None, acc: int = None, torque: int = None):
    hand.require_zeroed()
    cfg = hand.config
    m = cfg.motion
    speed = m.speed if speed is None else speed
    acc = m.acc if acc is None else acc
    torque = m.torque if torque is None else torque
    gains = dict(speed=speed, acc=acc, torque=torque)

    x_max = cfg[AUX_LEFT].max_mm
    y_max = cfg[BASE_JAW].max_mm
    z_max = cfg[Z].max_mm

    # Both aux fingers stroke in opposite directions, so one stroke turns the
    # cap by twice the finger travel.
    stroke_length = 2 * x_max
    strokes = int(np.ceil(num_revs * (np.pi * cap_diameter) / stroke_length))

    approach_mm = min(max(bottle_diameter, cap_diameter) / 2 + GRIP_CLEARANCE, y_max)
    bottle_grip = float(np.clip(bottle_diameter / 2 - BOTTLE_INTERFERENCE, 0, y_max))
    cap_grip = float(np.clip(cap_diameter / 2 - CAP_INTERFERENCE, 0, y_max))
    cap_z = min(cap_offset, z_max)
    extract_z = min(cap_offset + EXTRACT_LIFT, z_max)

    print(f"[{hand.name}] cap circumference {np.pi * cap_diameter:.1f}mm, "
          f"stroke {stroke_length:.1f}mm, {strokes} strokes")

    if do_open:
        print("caps: opening")
        hand.set_dofs({BASE_JAW: approach_mm, AUX_JAW: approach_mm}, **gains)
        hand.set_dofs({Z: cap_z}, **gains)
        time.sleep(3.0)

        hand.set_dofs({BASE_JAW: bottle_grip,
                       BASE_LEFT: x_max / 2, BASE_RIGHT: x_max / 2}, **gains)

        for i in range(strokes):
            print(f"  stroke {i + 1}/{strokes}")
            hand.set_dofs({AUX_LEFT: 0.0, AUX_RIGHT: x_max}, **gains)
            hand.set_dofs({AUX_JAW: cap_grip}, **gains)
            hand.set_dofs({AUX_LEFT: x_max, AUX_RIGHT: 0.0}, **gains)
            hand.set_dofs({AUX_JAW: cap_grip + CAP_RELEASE}, **gains)

        print("  extracting")
        hand.set_dofs({AUX_LEFT: x_max / 2, AUX_RIGHT: x_max / 2}, **gains)
        hand.set_dofs({AUX_JAW: cap_grip}, **gains)
        hand.set_dofs({Z: extract_z}, **gains)
        hand.set_dofs({BASE_LEFT: x_max, BASE_RIGHT: x_max,
                       AUX_LEFT: 0.0, AUX_RIGHT: 0.0}, **gains)
        print("cap removed, bottle presented.")

    if do_close:
        print("caps: closing")
        hand.set_dofs({BASE_LEFT: x_max / 2, BASE_RIGHT: x_max / 2,
                       AUX_LEFT: x_max / 2, AUX_RIGHT: x_max / 2}, **gains)
        hand.set_dofs({Z: cap_z}, **gains)

        for i in range(strokes):
            print(f"  stroke {i + 1}/{strokes}")
            # Press the cap down onto the threads at low torque while the
            # fingers turn it. Only the fingers are waited on: the z axis is
            # meant to stall against the cap, so it never reaches its target.
            hand.set_gains([Z], torque=50)
            hand.set_dofs({Z: 0.0, AUX_LEFT: x_max, AUX_RIGHT: 0.0},
                          wait_dofs=[AUX_LEFT, AUX_RIGHT], speed=speed, acc=acc)
            hand.set_dofs({AUX_JAW: cap_grip + CAP_RELEASE}, **gains)
            hand.set_dofs({Z: cap_z}, **gains)
            hand.set_dofs({AUX_LEFT: 0.0, AUX_RIGHT: x_max}, **gains)
            hand.set_dofs({AUX_JAW: cap_grip}, **gains)

        hand.set_dofs({Z: cap_z}, **gains)
        hand.set_dofs({BASE_JAW: approach_mm, AUX_JAW: approach_mm}, **gains)
        print("cap closed, gripper cleared.")


def run(hand, cfg: Config):
    if not cfg.open and not cfg.close:
        raise SystemExit("specify --open, --close, or both")
    try:
        caps(hand, do_open=cfg.open, do_close=cfg.close,
             bottle_diameter=cfg.bottle_diameter, cap_diameter=cfg.cap_diameter,
             num_revs=cfg.num_revs, cap_offset=cfg.cap_offset)
    finally:
        hand.release()

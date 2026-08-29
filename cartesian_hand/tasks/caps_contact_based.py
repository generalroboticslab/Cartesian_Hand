"""Unscrew a bottle cap, sizing the grip from contact rather than from a
measurement typed in by the operator.

The base jaw holds the bottle body while the aux jaw grips the cap. The aux
fingers stroke sideways to rotate the cap, releasing and re-gripping between
strokes because one stroke covers less than a full turn.
"""

import time
from dataclasses import dataclass

import numpy as np

from .primitives import approach, squeeze
from .roles import AUX_JAW, AUX_LEFT, AUX_RIGHT, BASE_JAW, BASE_LEFT, BASE_RIGHT, Z

DESCRIPTION = "Unscrew a bottle cap using contact-based sizing"


@dataclass
class Config:
    cap_offset: float = 20.0
    """Height of the cap's top face above z zero, in mm."""
    num_revs: float = 3.0
    """Full turns needed to free the cap."""
    squeeze_torque: int = 80
    """Holding torque while twisting."""
    approach_torque: int = 150
    """Torque used while closing a jaw onto the object."""
    approach_speed: int = 50
    """Speed used while closing a jaw onto the object."""


def caps_open(hand, cap_offset: float = 20.0, num_revs: float = 3.0,
              squeeze_torque: int = 80, approach_torque: int = 150,
              approach_speed: int = 50, speed: int = None, acc: int = None,
              torque: int = None, min_cap_radius: float = 1.0):
    """
    cap_offset:     height of the cap's top face above z zero, mm
    num_revs:       full turns needed to free the cap
    squeeze_torque: holding torque while twisting
    """
    hand.require_zeroed()
    cfg = hand.config
    m = cfg.motion
    speed = m.speed if speed is None else speed
    acc = m.acc if acc is None else acc
    torque = m.torque if torque is None else torque

    x_max = cfg[AUX_LEFT].max_mm
    y_max = cfg[BASE_JAW].max_mm
    cap_z = min(cap_offset, cfg[Z].max_mm)
    gains = dict(speed=speed, acc=acc, torque=torque)

    print(f"[{hand.name}] caps: open")

    # Entry: jaws wide, fingers centred, z at the cap.
    hand.set_pos([y_max, x_max / 2, x_max / 2, cap_z, y_max, x_max / 2, x_max / 2],
                 **gains)
    time.sleep(1.0)

    # Size the bottle and the cap by closing both jaws until they stall.
    print("probing...")
    contact = approach(hand, [BASE_JAW, AUX_JAW],
                       torque=approach_torque, speed=approach_speed)
    bottle_radius = contact[BASE_JAW]
    cap_radius = contact[AUX_JAW]
    if cap_radius < min_cap_radius:
        raise RuntimeError(
            f"cap probe failed: aux jaw closed to {cap_radius:.1f}mm, expected at "
            f"least {min_cap_radius}mm. Is a cap present at z={cap_z:.1f}mm?")
    print(f"  bottle radius {bottle_radius:.1f}mm, cap radius {cap_radius:.1f}mm")

    squeeze(hand, [BASE_JAW, AUX_JAW], squeeze_torque)

    # Each stroke rotates the cap by one finger travel along its circumference.
    circumference = 2 * np.pi * cap_radius
    strokes = max(1, int(np.ceil(num_revs * circumference / x_max)))
    print(f"  {strokes} strokes for {num_revs} revolutions")

    for i in range(strokes):
        print(f"  stroke {i + 1}/{strokes}")
        # Release the cap, reset the fingers, re-grip, then twist. The base jaw
        # keeps its squeeze throughout: gains are per DOF, so moving the aux
        # side does not disturb it.
        hand.set_pos({AUX_JAW: cap_radius + 1.0}, **gains)
        hand.set_pos({AUX_LEFT: x_max, AUX_RIGHT: 0.0}, speed=speed, acc=acc)
        hand.set_pos({AUX_JAW: cap_radius}, speed=speed, acc=acc)
        squeeze(hand, [AUX_JAW], squeeze_torque)
        hand.set_pos({AUX_LEFT: 0.0, AUX_RIGHT: x_max}, speed=speed, acc=acc)

    # Lift the freed cap clear, then open the base jaw to present the bottle.
    print("extracting...")
    hand.set_pos({AUX_JAW: cap_radius + 1.0}, **gains)
    hand.set_pos({AUX_LEFT: x_max / 2, AUX_RIGHT: x_max / 2}, speed=speed, acc=acc)
    approach(hand, [AUX_JAW], torque=approach_torque, speed=approach_speed)
    squeeze(hand, [AUX_JAW], squeeze_torque)
    hand.set_pos({Z: cfg[Z].max_mm}, speed=speed, acc=acc)
    hand.set_pos({BASE_LEFT: x_max, BASE_RIGHT: x_max,
                   AUX_LEFT: 0.0, AUX_RIGHT: 0.0}, **gains)
    print("cap removed.")


def run(hand, cfg: Config):
    try:
        caps_open(hand, cap_offset=cfg.cap_offset, num_revs=cfg.num_revs,
                  squeeze_torque=cfg.squeeze_torque,
                  approach_torque=cfg.approach_torque,
                  approach_speed=cfg.approach_speed)
    finally:
        hand.release()

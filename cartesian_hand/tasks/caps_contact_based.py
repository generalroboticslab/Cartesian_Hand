"""Unscrew a bottle cap, sizing the grip from contact rather than from a
measurement typed in by the operator.

The base jaw holds the bottle body while the aux jaw grips the cap. The aux
fingers stroke sideways to rotate the cap, releasing and re-gripping between
strokes because one stroke covers less than a full turn.
"""

import time
from dataclasses import dataclass
from typing import Optional

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
    release_clearance: float = 3.0
    """How far past the cap radius the aux jaw opens to release and reset
    between strokes, in mm."""
    speed: Optional[int] = None
    """Travel speed for entry, resets and strokes. Defaults to the hand's
    configured motion speed."""
    acc: Optional[int] = None
    """Travel acceleration for entry, resets and strokes. Defaults to the
    hand's configured motion acceleration."""
    settle_time: float = 1.0
    """Pause after the entry move, before probing, in seconds."""


def caps_open(hand, cap_offset: float = 20.0, num_revs: float = 3.0,
              squeeze_torque: int = 80, approach_torque: int = 150,
              approach_speed: int = 50, release_clearance: float = 1.0,
              speed: int = None, acc: int = None, settle_time: float = 1.0,
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

    # Diagnostic timing: prints elapsed time since the previous mark, so a run
    # log shows which phase actually costs the time instead of guessing from
    # gain values alone.
    t_prev = time.time()
    def mark(label):
        nonlocal t_prev
        now = time.time()
        print(f"  [{now - t_prev:5.2f}s] {label}")
        t_prev = now

    print(f"[{hand.name}] caps: open")

    # Entry: jaws wide, fingers centred, z at the cap.
    hand.set_pos([y_max, x_max / 2, x_max / 2, cap_z, y_max, x_max / 2, x_max / 2],
                 **gains)
    mark("entry move")
    time.sleep(settle_time)
    mark("settle")

    # Size the bottle and the cap by closing both jaws until they stall.
    print("probing...")
    contact = approach(hand, [BASE_JAW, AUX_JAW],
                       torque=approach_torque, speed=approach_speed)
    mark("probe")
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
        #
        # torque=squeeze_torque, not **gains's travel torque: this jaw was just
        # squeezed onto the cap at squeeze_torque (initial squeeze above, or
        # the previous stroke's re-grip below), and retracting against that
        # grip needs at least as much torque as made the grip. **gains would
        # drop it to the plain travel torque (m.torque, 50 by default), which
        # can be too weak to pull free -- set_pos then silently times out
        # (return value unchecked) and the fingers reset/twist while the jaw
        # never actually opened.
        hand.set_pos({AUX_JAW: cap_radius + release_clearance},
                     speed=speed, acc=acc, torque=squeeze_torque)
        mark("release")
        hand.set_pos({AUX_LEFT: x_max, AUX_RIGHT: 0.0}, speed=speed, acc=acc)
        mark("reset fingers")
        # wait=False: this drives back onto the cap surface, a physical
        # obstruction rather than free space, so at_target()'s position
        # convergence check can legitimately never fire (compliance/backlash
        # mean it can stall a fraction of a mm short or long of the recorded
        # cap_radius) and set_pos would otherwise block out its full timeout
        # every stroke. squeeze() immediately re-targets AUX_JAW anyway, so
        # this move's own convergence was never load-bearing.
        hand.set_pos({AUX_JAW: cap_radius}, speed=speed, acc=acc, wait=False)
        squeeze(hand, [AUX_JAW], squeeze_torque)
        mark("re-grip")
        # torque=squeeze_torque: the fingers otherwise still carry whatever
        # torque the entry move set (m.torque, STANDARD_TORQUE's finger value
        # -- 50 by default), never touched again after that. Turning a cap
        # against real friction at torque=50 stalls short of x_max, and with
        # wait=True's default 5s timeout that reads as a slow twist when it is
        # actually a torque-limited stall every stroke.
        hand.set_pos({AUX_LEFT: 0.0, AUX_RIGHT: x_max}, speed=speed, acc=acc,
                     torque=squeeze_torque)
        mark("twist")

    # Lift the freed cap clear, then open the base jaw to present the bottle.
    print("extracting...")
    # Same torque fix as the in-loop release: retracting from the last
    # squeeze needs at least squeeze_torque, not the weaker travel torque.
    hand.set_pos({AUX_JAW: cap_radius + release_clearance},
                 speed=speed, acc=acc, torque=squeeze_torque)
    hand.set_pos({AUX_LEFT: x_max / 2, AUX_RIGHT: x_max / 2}, speed=speed, acc=acc)
    approach(hand, [AUX_JAW], torque=approach_torque, speed=approach_speed)
    squeeze(hand, [AUX_JAW], squeeze_torque)
    hand.set_pos({Z: cfg[Z].max_mm}, speed=speed, acc=acc)
    hand.set_pos({BASE_LEFT: x_max, BASE_RIGHT: x_max,
                   AUX_LEFT: 0.0, AUX_RIGHT: 0.0}, **gains)
    mark("extraction")
    print("cap removed.")


def run(hand, cfg: Config):
    try:
        caps_open(hand, cap_offset=cfg.cap_offset, num_revs=cfg.num_revs,
                  squeeze_torque=cfg.squeeze_torque,
                  approach_torque=cfg.approach_torque,
                  approach_speed=cfg.approach_speed,
                  release_clearance=cfg.release_clearance,
                  speed=cfg.speed, acc=cfg.acc, settle_time=cfg.settle_time)
    finally:
        hand.release()

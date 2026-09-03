"""Unscrewing a cap, as a motion program instead of as Python.

`caps_contact_based.py` is the same task written as straight-line code. It
works, and it is the reference this file is checked against — but its
behaviour lives in a `for` loop whose bound is computed from a measurement, so
it cannot be batched and cannot be searched. Here the loop runs once, at build
time, and what comes out is tensors.

Two protocols
-------------
The human protocol is five steps and mentions no mechanism:

    1. put the gripper over the bottle, at cap height
    2. close both jaws until they touch          -> tells you both radii
    3. hold the bottle, hold the cap
    4. turn the cap three full revolutions
    5. lift the cap clear and present the bottle

The robot protocol is the motion program below. Steps 1-2 are `probe_program`,
3-4 are `stroke_program`, 5 is `extract_program`.

They are three programs rather than one because step 2 is a measurement and
steps 3-5 are parameterised by it. The measured radius sets *values* in the
later programs — goals, and how many strokes to emit. It never sets structure
at run time: by the time a program starts, its shape is fixed. That is the
whole reason a Python `for` between the programs is fine and a Python `for`
inside one would not be.

Why the base jaw only appears once
----------------------------------
It takes its grip in the first step of `stroke_program` and is never mentioned
again. A retired joint keeps its standing order (see motions.py), so the base
jaw goes on pressing the bottle at squeeze torque for the rest of the run
without occupying a row. This mirrors `caps_contact_based`, where the base
jaw's `set_pos` is issued once and left standing.
"""
import math
from dataclasses import dataclass

import numpy as np
import torch

from ..motions import Move, Program
from .primitives import twist
from .roles import AUX_JAW, AUX_LEFT, AUX_RIGHT, BASE_JAW, BASE_LEFT, BASE_RIGHT, Z

DESCRIPTION = "Unscrew a bottle cap, as a searchable motion program"

N_DOF = 7
PROBE_STEP = 1          # which step of probe_program closes the jaws on the object


@dataclass
class Config:
    """Everything the search is allowed to vary — the θ of the plan.

    Only `num_revs` changes the program's shape, and only by changing how many
    stroke steps get emitted, which is a build-time decision. By the time a
    program starts ticking its tensors are fixed.

    Defaults match caps_contact_based.Config so the two can be compared.
    """
    cap_offset: float = 20.0
    """Height of the cap's top face above z zero, in mm."""
    num_revs: float = 3.0
    """Full turns needed to free the cap."""
    squeeze_torque: float = 80.0
    """Holding torque while gripping and twisting."""
    approach_torque: float = 150.0
    """Torque used while closing a jaw onto the object."""
    travel_torque: float = 50.0
    """Torque for moves through free space."""
    release_clearance: float = 3.0
    """How far past the cap radius the aux jaw opens between strokes, in mm."""
    move_timeout_s: float = 6.0
    """Budget for a travel move before it is called a failure."""
    probe_timeout_s: float = 10.0
    """Budget for a move that ends on contact."""
    jaw_opening: float = 25.0
    """How wide to open each jaw before closing on the object, in mm.

    A jaw only has to clear the bottle. Every geometric field here is a
    clearance the task asks for, passed through `hand_cfg.clamped_mm` before it
    reaches a Move -- never a DOF's `max_mm`, which is CAD and reads high.
    """
    finger_stroke: float = 40.0
    """Full sweep of a finger during one twist, in mm."""
    lift_mm: float = 35.0
    """How high the z stage lifts the freed cap, in mm."""


def probe_program(hand_cfg, cfg, n_envs=1, hz=50.0, device="cpu"):
    """Human steps 1-2. Ends with both jaws stalled on the object.

    Where each jaw stopped IS the radius of what it is holding, so the caller
    reads `hand.positions` (or the sim's qpos) after this program and feeds the
    two numbers to the programs below.
    """
    jaw = hand_cfg.clamped_mm(BASE_JAW, cfg.jaw_opening)
    mid = hand_cfg.clamped_mm(AUX_LEFT, cfg.finger_stroke) / 2
    cap_z = hand_cfg.clamped_mm(Z, cfg.cap_offset)
    p = Program(n_envs, N_DOF, hz, device)

    move = lambda goal: Move(goal, cfg.travel_torque, "goal", cfg.move_timeout_s)
    p.step({BASE_JAW:  move(jaw),
            BASE_LEFT: move(mid), BASE_RIGHT: move(mid),
            Z:         move(cap_z),
            AUX_JAW:   move(jaw),
            AUX_LEFT:  move(mid), AUX_RIGHT: move(mid)})

    # The jaws close past 0 with reduced torque and retire on contact:
    # stop="stuck", so the object is what ends the move.
    prb = Move(0.0, cfg.approach_torque, "stuck", cfg.probe_timeout_s)
    p.step({BASE_JAW: prb, AUX_JAW: prb})
    return p.build()


def stroke_program(hand_cfg, cfg, cap_radius, hz=50.0, device="cpu"):
    """Human steps 3-4: hold both, then turn the cap `num_revs` times.

    `cap_radius` is [N] in mm, one measured value per env. A wider cap has a
    longer circumference, so it needs MORE strokes to cover the same number of
    turns, and the stroke count therefore differs per env. The Python loop runs
    to the max over envs and the ones that finish early carry `when=False` on
    the rows they do not need, idling those steps instead of branching.
    """
    span = hand_cfg.clamped_mm(AUX_LEFT, cfg.finger_stroke)
    cap_radius = torch.as_tensor(cap_radius, dtype=torch.float32, device=device)
    N = len(cap_radius)
    sq, travel_tq = cfg.squeeze_torque, cfg.travel_torque

    # One stroke rotates the cap by one finger sweep along its circumference.
    # Denominated in `span`, the distance the fingers actually travel below --
    # using the table's max here instead would under-count strokes whenever the
    # sweep is bounded, and the cap would come out short of num_revs.
    strokes = torch.ceil(
        cfg.num_revs * 2 * math.pi * cap_radius / span).clamp(min=1)

    p = Program(N, N_DOF, hz, device)

    # Both jaws take their grip on the cap: same stop rule as the probe, but at
    # holding torque rather than contact-finding torque.
    take_grip = Move(0.0, sq, "stuck", cfg.probe_timeout_s)
    p.step({BASE_JAW: take_grip, AUX_JAW: take_grip})

    release = cap_radius + cfg.release_clearance
    for i in range(int(strokes.max())):
        on = strokes > i                      # [N] bool: envs still stroking

        # Retract against our own grip at squeeze torque: pulling free of a
        # grip needs at least the torque that made it. Travel torque here is
        # the documented bug in caps_contact_based -- the jaw silently fails
        # to open and the fingers reset and twist against a held cap.
        p.step({AUX_JAW: Move(release, sq, "goal", cfg.move_timeout_s, on)})

        # Fingers reset to the far end of their sweep, ready to turn.
        p.step({AUX_LEFT:  Move(span, travel_tq, "goal", cfg.move_timeout_s, on),
                AUX_RIGHT: Move(0.0,  travel_tq, "goal", cfg.move_timeout_s, on)})

        # Back onto the cap. stop="stuck", not "goal": this drives into a
        # physical obstruction, so position convergence can legitimately
        # never fire.
        p.step({AUX_JAW: Move(0.0, sq, "stuck", cfg.probe_timeout_s, on)})

        # The turn. `twist` binds both fingers, so the step is one call.
        p.step(twist(AUX_LEFT, AUX_RIGHT, span, sq, cfg.move_timeout_s, on))
    return p.build()


def extract_program(hand_cfg, cfg, cap_radius, hz=50.0, device="cpu"):
    """Human step 5: lift the freed cap clear and present the bottle."""
    jaw = hand_cfg.clamped_mm(BASE_JAW, cfg.jaw_opening)
    span = hand_cfg.clamped_mm(AUX_LEFT, cfg.finger_stroke)
    mid = span / 2
    cap_radius = torch.as_tensor(cap_radius, dtype=torch.float32, device=device)
    N = len(cap_radius)
    sq, travel_tq = cfg.squeeze_torque, cfg.travel_torque
    p = Program(N, N_DOF, hz, device)

    move = lambda goal, tq=travel_tq: Move(goal, tq, "goal", cfg.move_timeout_s)
    # Retract at squeeze torque: same fix as the in-loop release.
    p.step({AUX_JAW: move(cap_radius + cfg.release_clearance, sq)})
    p.step({AUX_LEFT: move(mid), AUX_RIGHT: move(mid)})
    p.step({AUX_JAW: Move(0.0, sq, "stuck", cfg.probe_timeout_s)})
    p.step({Z: move(hand_cfg.clamped_mm(Z, cfg.lift_mm))})
    p.step({BASE_LEFT: move(span), BASE_RIGHT: move(span),
            AUX_LEFT:  move(0.0),   AUX_RIGHT:  move(0.0),
            BASE_JAW:  move(jaw)})
    return p.build()


def run(hand, cfg: Config, cap_radius: float = None):
    """The whole task on one hand: probe, strokes, extract.

    Three programs with plain Python between them, because the measurement in
    the middle changes what the later programs contain. The Python runs once,
    between programs -- never inside a tick -- so every program is still a fixed
    block of tensors by the time it starts.

    `cap_radius` overrides the probe: pass a value to skip the measurement and
    drive strokes against air (or any synthetic radius). Without it, the probe
    runs and its outcome is checked.
    """
    hand.require_zeroed()
    hz = hand.control_hz
    build = lambda fn, *a: fn(hand.config, cfg, *a, hz=hz)
    try:
        if cap_radius is None:
            print(f"[{hand.name}] caps: probing")
            probe = hand.run_program(probe_program(hand.config, cfg, n_envs=1, hz=hz))

            # The probe must have ended ON CONTACT, not on its timeout. Checking the
            # outcome rather than the resulting number is what tells "the cap is 2mm"
            # apart from "there was no cap and the jaw shut on air": those two are
            # indistinguishable by position alone, and taking the second for a
            # measurement is how the whole rest of the run gets built on nothing.
            if not bool(probe.succeeded()[0, AUX_JAW, PROBE_STEP]):
                raise RuntimeError(
                    f"cap probe did not end on contact (outcome "
                    f"{int(probe.outcome[0, AUX_JAW, PROBE_STEP])}). Is a cap present "
                    f"at z={hand.config.clamped_mm(Z, cfg.cap_offset):.1f}mm?")

            # Where the jaws stopped is what they are holding. One env, because a
            # real hand is one env of a batch the sim would run many of.
            pos = hand.positions
            radius = torch.from_numpy(np.asarray(pos[AUX_JAW:AUX_JAW + 1], dtype=np.float32))
            print(f"  bottle radius {pos[BASE_JAW]:.1f}mm, cap radius {float(radius):.1f}mm")
        else:
            radius = torch.tensor([cap_radius], dtype=torch.float32)
            print(f"[{hand.name}] caps: using override radius {cap_radius}mm (no probe)")

        hand.run_program(build(stroke_program, radius))
        print("  extracting")
        hand.run_program(build(extract_program, radius))
        print("cap removed.")
    finally:
        hand.release()

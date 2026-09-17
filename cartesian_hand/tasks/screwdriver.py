"""Manual screwdriver task: hand-over-hand grip transfer, 2026-09-08.

Replaces the whisk-style grip-and-roll attempt below, kept commented out
rather than deleted: if the hand-over-hand gait does not hold up on the
bench, the fallback is to delete the new `build`/`Config` and uncomment the
old ones, not to re-derive either from scratch.

Transcribed from an operator description of the gait, not yet run on
hardware -- see `build`'s docstring for the mechanism and MEMORY for
whichever of the two turns out to work.
"""

# """Rotate a screwdriver shaft by rolling it between both jaws' finger pairs --
# `whisk.py`'s grip-and-roll mechanism, minus its tilt phase.
#
#     init dof1256 to 0 -> squeeze jaws (dof0 & dof4 together)
#       -> grip (gentle search to full span) -> squeeze (latch)
#       -> repeat cycles( roll out: dof1/5 out, dof2/6 in -> roll in: dof1/5 in, dof2/6 out )
#
# A screwdriver's shaft is a plain round rod, exactly the object `whisk.py`'s
# roll maneuver is built for: squeeze it at two points (dof0 and dof4, the two
# jaws' own coarse parallel clamps) and roll the four fingers in opposite
# diagonal pairs (dof1/5 vs dof2/6) to spin it about its own long axis. That is
# the entire task -- see `whisk.py`'s module docstring for the geometry, the
# squeeze/grip/roll rationale, and its 2026-09-07 bench note on why `grip` is a
# plain `Move` (not a `Probe`) and why `contact_tolerance_mm` exists; none of
# that reasoning changes here, it is the same rod-in-fingers problem.
#
# **No tilt.** `whisk.py`'s tilt phase rocks the object at the aux jaw's fixed
# hold, which is a stirring motion a whisk wants and a screwdriver has no
# reason to reproduce -- turning a shaft is exactly what roll already does, so
# this task stops once the roll loop finishes rather than also closing the top
# fingers and tilting. `cycles` defaults to 3, not `whisk.py`'s 2, because
# "turn it" for a screwdriver means a handful of full rotations, not a couple
# of demonstration rolls.
#
# Every goal is absolute in the hand's millimetre frame, so the hand must be
# zeroed first.
#
# No hardware results yet: `whisk.py`'s own roll maneuver is authored, not
# bench-validated, at time of writing.
# """
# from dataclasses import dataclass, field
#
# import torch
#
# from ..config import (AUX_JAW, AUX_LEFT, AUX_RIGHT, BASE_JAW, BASE_LEFT,
#                       BASE_RIGHT, HandConfig)
# from ..primitives import Loop, Move, Probe, Sequence
#
# BASE = (BASE_LEFT, BASE_RIGHT)      # dof1, dof2 -- the base jaw's own pair
# AUX = (AUX_LEFT, AUX_RIGHT)         # dof5, dof6 -- the aux jaw's own pair
# LEFT = (BASE_LEFT, AUX_LEFT)        # dof1, dof5 -- both jaws' left corner
# RIGHT = (BASE_RIGHT, AUX_RIGHT)     # dof2, dof6 -- both jaws' right corner
# FINGERS = BASE + AUX                # dof1, dof2, dof5, dof6
# JAWS = (BASE_JAW, AUX_JAW)          # dof0, dof4
#
#
# @dataclass
# class Config:
#     """Bench units. Every `tune` field becomes a slider on the studio page."""
#
#     label: str = "Turn screwdriver"
#     sets_datum: bool = False
#
#     finger_stroke: float = field(default=40.0, metadata={"tune": (10.0, 55.0)})
#     """Full sweep of a finger pair, in mm. Roll drives between 0 and this
#     span; the grip searches out to the finger's full travel instead (see
#     `whisk.py`'s bench note on why)."""
#     squeeze_torque: float = field(default=100.0, metadata={"tune": (50.0, 300.0)})
#     """Torque for every row from the jaw squeeze on -- fingers and roll.
#     Low by default, matching `whisk.py`: raise it if the shaft slips in the
#     fingers during a roll."""
#     approach_torque: float = field(default=100.0, metadata={"tune": (50.0, 300.0)})
#     """Torque while the initial grip is still searching for contact, before
#     `squeeze_torque` latches in."""
#     approach_speed: float = field(default=800.0, metadata={"tune": (25.0, 1500.0)})
#     """Servo speed register for the initial contact search, counts/s.
#
#     Fast for `cap`'s reason: there is no contact sensor, so contact is read as
#     a confirmed stop under 0.3 mm/s, and a slow creep runs close enough to
#     that threshold that ordinary servo hesitation reads as contact."""
#     travel_torque: float = field(default=50.0, metadata={"tune": (30.0, 200.0)})
#     """Torque for the one unloaded row: driving dof1256 to 0 at task start."""
#     travel_speed: float = field(default=1500.0, metadata={"tune": (200.0, 1500.0)})
#     """Unused: every free move now runs at the servo's rated no-load top
#     speed regardless of this value."""
#     cycles: float = field(default=3.0, metadata={"tune": (1.0, 10.0)})
#     """Full roll out/in cycles -- that many shaft rotations -- rounded to an
#     integer."""
#     contact_tolerance_mm: float = field(default=2.0, metadata={"tune": (1.0, 5.0)})
#     """Arrival tolerance for the grip row's approach, mm.
#
#     Widened past `Move`'s 1 mm default because a rod this size stalls within a
#     millimetre of the probed ceiling on every hand tried so far -- see
#     `whisk.py`'s bench note. `accept_stall` alone does not cover this: it
#     needs `_stalled` to confirm a stop *more than* 1 mm from goal, which a
#     stall this close can never do."""
#     timeout_margin: float = field(default=1.5, metadata={"tune": (1.0, 3.0)})
#
#
# def build(hand: HandConfig, start_mm: torch.Tensor,
#           cfg: Config | None = None) -> Sequence:
#     """Squeeze a screwdriver shaft in both jaws, then roll it with the fingers."""
#     cfg = cfg or Config()
#     mm = hand.clamped_mm
#     finger_floor = max(float(hand.gain_vector("torque_min_to_move")[dof])
#                        for dof in FINGERS)
#     squeeze_effort = max(cfg.squeeze_torque, finger_floor) / 1000.0
#     approach_effort = max(cfg.approach_torque, finger_floor) / 1000.0
#     finger_span_mm = mm(BASE_LEFT, cfg.finger_stroke)
#     # The finger's true physical extreme, not a value derived from
#     # `finger_stroke` -- see `whisk.py`'s bench note for why a goal merely
#     # larger than the expected contact point is not enough, and why this is
#     # a `Move` rather than a `Probe`.
#     probe_reach_mm = mm(BASE_LEFT, hand.travel_mm[BASE_LEFT])
#     cycles = max(1, round(cfg.cycles))
#
#     return Sequence([
#         Move(label="init", goal={FINGERS: 0.0}),
#         # The two coarse parallel-jaw clamps, closed on the shaft together --
#         # one `Probe` over both dof0 and dof4, each searching its own hard
#         # stop (0) and latching independently, rather than one jaw closing
#         # fully before the other starts.
#         Probe(label="squeeze jaws", group=JAWS, creep=True,
#               grip=squeeze_effort),
#         # Gentle search creep toward the ceiling; `squeeze` below latches the
#         # real grip torque once this arrives. `tolerance_mm` (not `Probe`'s
#         # hair-trigger 1 mm) covers landing right at the ceiling; `accept_stall`
#         # covers stopping well short of it, for a thicker shaft.
#         Move(label="grip", goal={FINGERS: probe_reach_mm}, creep=True,
#              effort=approach_effort, tolerance_mm=cfg.contact_tolerance_mm,
#              accept_stall=True),
#         Move(label="squeeze", goal={FINGERS: probe_reach_mm},
#              effort=squeeze_effort, accept_stall=True),
#         Loop(count=cycles, rows=[
#             Move(label="roll out", goal={LEFT: 0.0, RIGHT: finger_span_mm},
#                  effort=squeeze_effort, accept_stall=True),
#             Move(label="roll in", goal={LEFT: finger_span_mm, RIGHT: 0.0},
#                  effort=squeeze_effort, accept_stall=True),
#         ]),
#     ], hand=hand,
#        start_mm=start_mm,
#        travel_torque=cfg.travel_torque,
#        approach_torque=cfg.approach_torque,
#        approach_speed=cfg.approach_speed,
#        timeout_margin=cfg.timeout_margin,
#        travel_speed=cfg.travel_speed)

from dataclasses import dataclass, field

import torch

from ..config import (AUX_JAW, AUX_LEFT, AUX_RIGHT, BASE_JAW, BASE_LEFT,
                      BASE_RIGHT, HandConfig, Z)
from ..primitives import Loop, Move, Probe, Sequence

BASE = (BASE_LEFT, BASE_RIGHT)      # dof1, dof2 -- base jaw's own finger pair
AUX = (AUX_LEFT, AUX_RIGHT)         # dof5, dof6 -- aux jaw's own finger pair
FINGERS = BASE + AUX
JAWS = (BASE_JAW, AUX_JAW)          # dof0, dof4


@dataclass
class Config:
    """Bench units. Every `tune` field becomes a slider on the studio page."""

    label: str = "Turn screwdriver"
    sets_datum: bool = False

    setup_fraction: float = field(default=0.5, metadata={"tune": (0.1, 0.9)})
    """Fraction of full travel every finger and jaw opens to at task start,
    before either jaw has found the shaft -- rough clearance, not a grip."""
    z_offset: float = field(default=10.0, metadata={"tune": (0.0, 40.0)})
    """Absolute z height, mm. Only the aux assembly (jaw and fingers) rides
    on z -- the base assembly is fixed -- so this is the vertical gap
    between the two grip points on the shaft, set once at task start and
    never revisited."""
    finger_stroke: float = field(default=40.0, metadata={"tune": (10.0, 55.0)})
    """Full sweep of one finger pair during a rotate/prepare stroke, mm."""
    squeeze_torque: float = field(default=100.0, metadata={"tune": (50.0, 300.0)})
    """Torque for every row from a jaw's grip on -- jaws, rotate, prepare,
    and let-go."""
    approach_torque: float = field(default=100.0, metadata={"tune": (50.0, 300.0)})
    """Torque while a jaw's grip search creeps toward contact, before
    `squeeze_torque` latches in on contact."""
    approach_speed: float = field(default=800.0, metadata={"tune": (25.0, 1500.0)})
    """Servo speed register for a grip's contact search, counts/s -- fast
    for `cap`'s reason: a slow creep is indistinguishable from a stall."""
    travel_torque: float = field(default=50.0, metadata={"tune": (30.0, 200.0)})
    """Torque for the one row nothing is gripping yet: the initial setup move."""
    travel_speed: float = field(default=1500.0, metadata={"tune": (200.0, 1500.0)})
    """Unused: every free move now runs at the servo's rated no-load top
    speed regardless of this value."""
    release_clearance: float = field(default=5.0, metadata={"tune": (1.0, 10.0)})
    """How far past a jaw's measured contact it opens when it lets go, mm."""
    cycles: float = field(default=3.0, metadata={"tune": (1.0, 20.0)})
    """Complete aux-turn-then-base-turn pairs, rounded to an integer."""
    timeout_margin: float = field(default=1.5, metadata={"tune": (1.0, 3.0)})


def build(hand: HandConfig, start_mm: torch.Tensor,
          cfg: Config | None = None) -> Sequence:
    """Walk a hand-over-hand grip up a screwdriver shaft; one jaw always holds.

        setup: fingers and jaws to setup_fraction of travel, z to z_offset
          -> aux fingers prepare for a cw stroke
          -> repeat cycles(
               stroke(aux drives, base resets)
               stroke(base drives, aux resets)
             )

    `stroke(name drives, other resets)` is:

        grip name -> [rotate name's fingers cw, simultaneously reset other's
        fingers to their own next stroke] -> grip other -> let name go

    Base and aux grip the shaft at two fixed heights (aux is the one that
    rides `Z`; base does not move on it) rather than at one shared point, so
    "rotating" one pair's fingers and "resetting" the other's can run in the
    same row without either fouling the other -- a `Move` already drives
    every DOF its goal dict names on the same tick, done when all of them
    arrive, which is what lets the reset finish for free inside whichever
    stroke is longer.

    **The shaft is never held by fewer than one jaw.** `grip other` always
    finishes -- Probe succeeds -- before `let name go` runs, so there is no
    tick where both jaws are open together and the shaft could lose its
    angular registration or drop. This is the whole difference from the
    archived roll attempt above, which drives both jaws' diagonal finger
    pairs at once and depends on friction alone to keep the shaft in place
    through every reversal.

    Every goal is absolute in the hand's millimetre frame, so the hand must
    be zeroed first.

    Transcribed from an operator description of the gait; no hardware
    results yet.
    """
    cfg = cfg or Config()
    mm = hand.clamped_mm
    finger_floor = max(float(hand.gain_vector("torque_min_to_move")[dof])
                       for dof in FINGERS)
    squeeze_effort = max(cfg.squeeze_torque, finger_floor) / 1000.0
    finger_span_mm = mm(BASE_LEFT, cfg.finger_stroke)
    setup_finger_mm = mm(BASE_LEFT, hand.travel_mm[BASE_LEFT] * cfg.setup_fraction)
    setup_jaw_mm = mm(BASE_JAW, hand.travel_mm[BASE_JAW] * cfg.setup_fraction)
    z_height = mm(Z, cfg.z_offset)
    release_clearance_mm = cfg.release_clearance
    cycles = max(1, round(cfg.cycles))

    def stroke(name: str, other: str, jaw: int, left: int, right: int,
              other_jaw: int, other_left: int, other_right: int) -> list:
        """One handoff: `name` grips and rotates while `other` resets, then
        `other` grips before `name` lets go -- see `build`'s docstring for
        why that order is what keeps the shaft always held."""
        radius = f"radius_{name}"
        return [
            Probe(label=f"grip {name}", group=jaw, creep=True,
                  grip=squeeze_effort, measure={radius: jaw}),
            Move(label=f"{name} rotate, {other} prepare",
                 goal={left: 0.0, right: finger_span_mm,
                       other_left: finger_span_mm, other_right: 0.0},
                 effort=squeeze_effort, accept_stall=True),
            Probe(label=f"grip {other}", group=other_jaw, creep=True,
                  grip=squeeze_effort, measure={f"radius_{other}": other_jaw}),
            Move(label=f"let go {name}",
                 goal={jaw: lambda m: getattr(m, radius) + release_clearance_mm},
                 effort=squeeze_effort, loaded=True),
        ]

    return Sequence([
        Move(label="setup", goal={FINGERS: setup_finger_mm, JAWS: setup_jaw_mm,
                                  Z: z_height}),
        Move(label="prepare aux", goal={AUX_LEFT: finger_span_mm, AUX_RIGHT: 0.0}),
        Loop(count=cycles, rows=[
            *stroke("aux", "base", AUX_JAW, AUX_LEFT, AUX_RIGHT,
                    BASE_JAW, BASE_LEFT, BASE_RIGHT),
            *stroke("base", "aux", BASE_JAW, BASE_LEFT, BASE_RIGHT,
                    AUX_JAW, AUX_LEFT, AUX_RIGHT),
        ]),
    ], hand=hand,
       start_mm=start_mm,
       travel_torque=cfg.travel_torque,
       approach_torque=cfg.approach_torque,
       approach_speed=cfg.approach_speed,
       timeout_margin=cfg.timeout_margin,
       travel_speed=cfg.travel_speed)

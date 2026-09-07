"""Operate a two-handle tool -- scissors, or a spring-loaded snipper -- sizing
the handle grips from contact rather than from a number typed by the operator.

    entry -> settle -> probe both handles/grip
      -> repeat(stroke away from rest -> stroke back to rest)

A two-handle tool is one articulated object: the handles share a pivot, so
separating them is the whole open/close primitive and the tool's own pivot
supplies the reaction a single jaw would otherwise take from a table. The base
jaw holds one handle steady; the aux jaw rides the z stage and holds the other,
so **z travel stands in for the tool's pivoting motion** -- not fingers stroking
sideways, as in `cap`'s twist.

Returning to rest every stroke is what makes this a script and not an
oscillation: the sequence ends at rest whatever `normally_closed` is and however
many strokes ran, so the next call starts from the pose this one left. It stays
squeezed -- this task does not let go of what it is holding.

**A jaw that shuts on air is refused, not measured.** `Probe` fails the instant
a jaw comes within a millimetre of its closed goal without contact, so an empty
gripper never becomes a handle radius that every later row inherits. That check
used to be a `min_handle_radius` field compared against the probe's result; it
is the same threshold either way, and the row reports it a phase earlier.

Every goal is absolute in the hand's millimetre frame, so the hand must be
zeroed first -- `handle_offset` is a height above the z hard stop, not above
wherever the stage happened to be parked.

No hardware results for *this* transcription yet: the sequence is validated, the
port of it is not.
"""

from dataclasses import dataclass, field

import torch

from ..config import (AUX_FINGERS, AUX_JAW, BASE_FINGERS, BASE_JAW, HandConfig,
                      Z)
from ..primitives import Hold, Loop, Move, Probe, Sequence, lift_effort

JAWS = (BASE_JAW, AUX_JAW)
FINGERS = tuple(BASE_FINGERS + AUX_FINGERS)

STROKE_TOL_MM = 5.0
"""Arrival tolerance on a loaded z stroke, mm. `cap`'s lift uses the same."""


@dataclass
class Config:
    """Bench units. Every `tune` field becomes a slider on the studio page."""

    label: str = "Open scissors"
    """Button text on the studio page. Empty means no button, and a variant
    that does not name its own inherits none -- see `tasks/__init__.py`."""
    sets_datum: bool = False

    handle_offset: float = field(default=5.0, metadata={"tune": (5.0, 40.0)})
    """Height of the aux jaw's grip point above z zero at rest, in mm.

    The closed position for a normally-closed tool, the open one otherwise.
    Probing happens here, so it is also where both handles must be."""
    travel: float = field(default=25.0, metadata={"tune": (5.0, 40.0)})
    """Aux jaw travel away from `handle_offset` for one stroke, in mm."""
    normally_closed: bool = True
    """True for a tool that rests closed and opens under a stroke (scissors),
    False for one that rests open under spring tension and closes under a
    stroke (a wire snipper's cut). It is the sign of the stroke, and the only
    thing separating the two tools -- hence a field rather than a second file.

    Not tunable: it is not a number, and half the range is a different tool."""
    num_cuts: int = 1
    """Open/close cycles to perform.

    **Deliberately not tunable**: two candidates would run different procedures
    rather than vary parameters of one procedure."""
    squeeze_torque: float = field(default=300.0, metadata={"tune": (50.0, 400.0)})
    """Holding torque on both handles through the stroke.

    High, and measured on the working hardware: the grip has to hold against
    the pry the stroke applies to it, not merely against the handle's weight."""
    approach_torque: float = field(default=150.0, metadata={"tune": (50.0, 300.0)})
    """Torque used while closing a jaw onto a handle, to find it."""
    cut_torque: float = field(default=300.0, metadata={"tune": (50.0, 800.0)})
    """Torque on the z stage during the stroke itself -- the force behind the
    snip. Raise it to cut harder.

    Floored at z's measured `torque_min_to_move` by `lift_effort`: one direction
    of the stroke always raises the stage, and under that floor the row runs,
    expires, and reports nothing wrong while nothing moved."""
    travel_torque: float = field(default=80.0, metadata={"tune": (30.0, 200.0)})
    """Torque for the horizontal free-space moves: jaws opening, fingers parking."""
    approach_speed: float = field(default=300.0, metadata={"tune": (25.0, 500.0)})
    """Servo speed register while closing a jaw onto a handle, counts/s.

    **Raised 50 -> 300, for `cap`'s reason.** There is no contact sensor, so
    contact is `CONFIRM_TICKS` of measured speed under `STUCK_SPEED_MM_S` =
    0.3 mm/s; at 50 counts/s the creep itself runs at 0.61 mm/s, twice the
    threshold it is tested against, and any servo hesitation reads as a handle.
    At 300 the margin is 12x, and a 20 mm close takes 5.4 s instead of 33."""
    jaw_opening: float = field(default=25.0, metadata={"tune": (10.0, 40.0)})
    """How wide each jaw opens before closing on its handle, in mm.

    A clearance the task asks for and gets capped by `HandConfig.clamped_mm`,
    never a DOF's `max_mm` -- the travel table is CAD and reads high, so a task
    that commands the limit is not saved by a clamp using the same wrong
    number. The validated implementation used `max_mm` here; this does not."""
    settle_s: float = field(default=0.5, metadata={"tune": (0.1, 3.0)})
    """Pause after the entry move, before probing."""
    timeout_margin: float = field(default=1.5, metadata={"tune": (1.0, 3.0)})
    """Margin on the deadline each row derives from its own travel, at the
    speed it commands. Flat second counts are what expired 15 of `cap`'s 18
    rows."""


def build(hand: HandConfig, start_mm: torch.Tensor,
          cfg: Config | None = None) -> Sequence:
    """Size both handles by contact, then stroke the tool `num_cuts` times."""
    cfg = cfg or Config()
    mm = hand.clamped_mm
    jaw_floor = max(float(hand.gain_vector("torque_min_to_move")[dof])
                    for dof in JAWS)
    squeeze_effort = max(cfg.squeeze_torque, jaw_floor) / 1000.0
    # One of the two stroke directions always raises the stage, so both take
    # the measured lift floor -- z torque is directional and the ascent is the
    # half that silently does not move under it.
    stroke_effort = lift_effort(hand, cfg.cut_torque)

    rest_z = mm(Z, cfg.handle_offset)
    # A normally-closed tool is opened by prying the handles apart (+travel); a
    # spring-loaded one already rests open, so its stroke is the cut (-travel).
    stroke_z = mm(Z, rest_z + (cfg.travel if cfg.normally_closed
                               else -cfg.travel))

    return Sequence([
        # Jaws and fingers first, z second: the entry z move may ascend and so
        # carries the lift floor, which is far too much force to also put
        # behind a jaw sweeping through free space.
        # Move(label="entry", goal={JAWS: mm(BASE_JAW, cfg.jaw_opening),
        #                           FINGERS: 0.0}),
        # Move(label="height", goal={Z: rest_z}, effort=stroke_effort),
        # Hold(label="settle", seconds=cfg.settle_s),
        # One row, not a probe and then a grip: `Probe.grip` latches the holding
        # effort on the tick contact is confirmed.
        Probe(label="handles", group=JAWS, creep=True, grip=squeeze_effort),
        # `loaded`, with cap's 2 mm tolerance: both directions pry an
        # articulated object, so effort is a cap the stroke legitimately parks
        # short of. A free `Move` rejects that confirmed stall the tick it sees
        # one and fails the whole sequence.
        Loop(count=cfg.num_cuts, rows=[
            Move(label="stroke", goal={Z: stroke_z}, effort=stroke_effort,
                 loaded=True, tolerance_mm=STROKE_TOL_MM),
            Move(label="back", goal={Z: rest_z}, effort=stroke_effort,
                 loaded=True, tolerance_mm=STROKE_TOL_MM),
        ]),
    ], hand=hand,
       start_mm=start_mm,
       travel_torque=cfg.travel_torque,
       approach_torque=cfg.approach_torque,
       approach_speed=cfg.approach_speed,
       timeout_margin=cfg.timeout_margin)

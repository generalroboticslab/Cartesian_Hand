"""Drive each DOF to its limit one at a time and leave it there, then close them all.

The hardware counterpart of the paper's `joint_monkey.mp4`. `primitive_tour`
shows what the hand can DO; this shows what it is MADE OF:

    DOF 0 opens -> hold -> DOF 1 opens -> hold -> ... -> DOF 6 opens -> hold
      -> every DOF closes together -> hold

Each DOF is LEFT open when the next starts, so the hand accumulates into its
fully open pose one degree of freedom at a time. That is what makes the closing
move possible without a reset -- by the last hold every DOF is already open --
and it is the only row in the task where more than one thing moves, which reads
as the summary it is.

A visual demo, not a manipulation task: every goal is free space, nothing is
gripped, and no row probes for contact. Run it with the jaws empty.

Seven rows, not nine
--------------------
The simulated clip spends most of its file arguing this out, because the MJCF
has NINE joints and two `<equality>` constraints welding each jaw pair, so
stepping the joints one at a time renders two poses the hand cannot reach -- one
jaw of a pair open, the other shut. Here the question does not arise: a rack
pair is ONE servo (`mjcf.py:41-45`), `LAYOUT` has seven entries, and there is no
ninth joint to get wrong.

What this task is EVIDENCE for
------------------------------
`config.LAYOUT` and the MJCF disagree about which finger is left and which is
right, and the disagreement is unresolved (`docs/hardware.md:391-398`,
`mjcf.py:36-39`). This is the one task that can settle it: exactly one finger
moves per row, and the row is labelled, so whoever films it reads off which
finger actually moved on `DoF 1: base left finger` and either confirms the table
or reaches for `--swap`. Nothing here assumes an answer.

Paced to the clip
-----------------
Every `Move` runs at `speed_mm_s`, the same 65 mm/s the simulated clips use, via
`speed_scale_for` -- see that function for why the timeout margin is divided by
the same scale. Without it the real hand runs at the servo's rated top speed and
the two videos cannot be played side by side.

No row names an `effort`
------------------------
Deliberate, and it is what makes DOF 3 work: `Sequence.travel_effort` floors
every DOF at its `torque_min_to_move`, which is 800 for z against ~250
horizontal, and a row that names its own effort opts out of that floor. See
`primitive_tour`'s docstring, which carries the full reasoning and the bug.
"""
from dataclasses import dataclass, field

import torch

from ..config import LABELS, N_DOF, HandConfig
from ..primitives import Hold, Move, Sequence, speed_scale_for

ALL_DOFS = tuple(range(N_DOF))


@dataclass
class Config:
    """Bench units. Every `tune` field becomes a slider on the studio page."""

    label: str = "Joint monkey"
    sets_datum: bool = False

    open_fraction: float = field(default=0.9, metadata={"tune": (0.5, 0.98)})
    """Far end of each DOF's reachable range, as a fraction of its travel table
    entry. Never 1.0 -- the table is CAD and reads high, so asking for the limit
    risks the hard stop `zero` finds on purpose, under a torque this task never
    claims (`HandConfig.upper` says so)."""

    margin_mm: float = field(default=1.0, metadata={"tune": (0.0, 5.0)})
    """Near end of that range, in mm off the hard stop at 0. A free `Move` is
    closed-loop, and a joint parked against the stop a millimetre short of a 0.0
    goal reads as a stall rather than an arrival."""

    dwell: float = field(default=0.5, metadata={"tune": (0.0, 3.0)})
    """Seconds to rest after each DOF arrives, so it is legible as its own step
    rather than as part of one continuous unfolding."""

    speed_mm_s: float = field(default=65.0, metadata={"tune": (10.0, 83.0)})
    """Commanded speed of every move. 65 is `animate_primitive.SPEED`, which the
    simulated clips are timed by."""

    travel_torque: float = field(default=50.0, metadata={"tune": (30.0, 200.0)})
    approach_torque: float = field(default=150.0, metadata={"tune": (50.0, 300.0)})
    approach_speed: float = field(default=800.0, metadata={"tune": (25.0, 800.0)})
    """Unused: this task never probes or creeps, so nothing in it commands the
    approach torque or speed. Carried only because `Sequence` takes them."""

    timeout_margin: float = field(default=1.5, metadata={"tune": (1.0, 3.0)})
    """Against the move's own duration at `speed_mm_s`, not at the register's
    speed -- `build` divides by the scale, see `speed_scale_for`."""


def build(hand: HandConfig, start_mm: torch.Tensor,
          cfg: Config | None = None) -> Sequence:
    """Entry point for `--task joint_monkey`. Seven opens, then one close."""
    cfg = cfg or Config()
    scale = speed_scale_for(hand, cfg.speed_mm_s)

    def reach(dof: int, fraction: float) -> float:
        """`fraction` of DOF `dof`'s reachable range, in mm. Neither end is a
        rail end; see `open_fraction` and `margin_mm`."""
        far = hand.clamped_mm(dof, hand.travel_mm[dof] * cfg.open_fraction)
        return cfg.margin_mm + fraction * (far - cfg.margin_mm)

    shut = {dof: reach(dof, 0.0) for dof in ALL_DOFS}

    # Every row commands all seven, not only the DOF it opens. The ones already
    # open are inside tolerance on the first tick and cost nothing, and it makes
    # the goal dict a full pose -- which is what the check below reads.
    at = dict(shut)
    rows = [Move(label="shut", goal=dict(at), speed_scale=scale),
            Hold(seconds=cfg.dwell)]
    poses = [dict(at)]
    for dof in ALL_DOFS:
        at[dof] = reach(dof, 1.0)
        rows += [Move(label=f"DoF {dof}: {LABELS[dof]}", goal=dict(at),
                      speed_scale=scale),
                 Hold(seconds=cfg.dwell)]
        poses.append(dict(at))
    rows += [Move(label="all DoFs", goal=dict(shut), speed_scale=scale),
             Hold(seconds=cfg.dwell)]

    # ONE DOF per row, which is the whole claim of the task and the thing that
    # would fail silently: the hand still ends up fully open either way, and a
    # row that opened two would simply look like a faster demo.
    for before, after in zip(poses, poses[1:]):
        moved = [dof for dof in ALL_DOFS if before[dof] != after[dof]]
        assert len(moved) == 1, f"row opens {len(moved)} DOFs at once: {moved}"
    assert poses[-1] == {dof: reach(dof, 1.0) for dof in ALL_DOFS}, \
        "a DOF is not open at the end; the closing move would not close it"
    assert all(0.0 <= mm <= hand.travel_mm[dof]
               for pose in poses for dof, mm in pose.items()), \
        "a goal is off the travel table -- a carriage would leave its rail"

    return Sequence(rows, hand=hand,
                    start_mm=start_mm,
                    travel_torque=cfg.travel_torque,
                    approach_torque=cfg.approach_torque,
                    approach_speed=cfg.approach_speed,
                    # A scaled row keeps its unscaled deadline; see
                    # `speed_scale_for`.
                    timeout_margin=cfg.timeout_margin / scale)

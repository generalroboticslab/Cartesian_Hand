"""Sweep the gripping face through the whole reachable volume, one slab at a time.

The hardware counterpart of the paper's `workspace_sweep.mp4`. The still figure
draws the reachable region as a box and asks the reader to take it on trust; the
clip removes the trust by having the gripping FACE paint the region itself, so
what is on screen is a record of where the face has BEEN. This task is that
motion on the real hand.

    for each (slide, z) station, in a serpentine:
        move to the station, jaws where they are
        run the jaws right across their travel
    hold

Three groups, not seven DOFs
----------------------------
    SLIDE  the four finger slides, moving together -- the region's x
    JAW    the two rack pairs, moving together -- the region's y, and the axis
           that is SWEPT rather than stepped
    SEP    z

JAW is the fast axis because the gripping face is a PLANE in y: one pose covers
none of that travel, so it has to be swept continuously. The other two are
stepped, because the face already spans most of them and standing at a few
stations covers the rest.

How many stations, and why it is arithmetic
-------------------------------------------
A station covers the face's own extent along the axis it steps, so the number
needed is the one that makes consecutive stations OVERLAP:

    need = ceil(reachable travel / face extent),  fractions k / need

The two face extents are measured off the MJCF (`animate_workspace.demo` prints
them) and transcribed here; the travel comes from `hand.travel_mm`, so a
re-measured rail moves the station count with it rather than leaving a constant
to go stale.

**Two z stations is the trap.** Fractions 0 and 1 cover the bottom 32 mm and the
top 32 mm of a 44 mm reachable range and silently leave a band in the middle the
face never touches -- and nothing about watching the hand would reveal it. On
the current table the arithmetic gives THREE, and the assertion below is that
consecutive stations are no further apart than the face is wide.

Serpentine, not raster
----------------------
Each transition moves exactly one stepped axis, at the end of a jaw sweep, and
the sweep direction alternates. A raster would fly the hand back across the
aperture between passes, which is motion that sweeps nothing and reads as a
reset.

A visual demo, not a manipulation task: every goal is free space, nothing is
gripped, and no row probes for contact. Run it with the jaws empty.

Paced to the clip
-----------------
Every `Move` runs at `speed_mm_s`, the same 65 mm/s the simulated clips use, via
`speed_scale_for` -- see that function for why the timeout margin is divided by
the same scale.

No row names an `effort`, so `Sequence.travel_effort` floors z at its measured
`torque_min_to_move`. See `primitive_tour`'s docstring for the full reasoning.
"""
import math
from dataclasses import dataclass, field

import torch

from ..config import (AUX_JAW, AUX_LEFT, AUX_RIGHT, BASE_JAW, BASE_LEFT,
                      BASE_RIGHT, HandConfig, Z)
from ..primitives import Hold, Move, Sequence, speed_scale_for

SLIDE = (BASE_LEFT, BASE_RIGHT, AUX_LEFT, AUX_RIGHT)
JAW = (BASE_JAW, AUX_JAW)
SEP = (Z,)

# The gripping face's own extent, mm, along the two STEPPED axes. Measured on
# the MJCF and printed by `animate_workspace.demo`; the y extent is absent
# because a face is a plane in y, which is why that axis is swept instead.
FACE_X_MM = 94.93
FACE_Z_MM = 32.0


@dataclass
class Config:
    """Bench units. Every `tune` field becomes a slider on the studio page."""

    label: str = "Workspace sweep"
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
    """Seconds to rest at the end of the last sweep. Only there: a pause between
    a station move and its sweep would break the volume into separate gestures,
    and the point is that it is one continuous painting."""

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
    """Entry point for `--task workspace_sweep`. One jaw sweep per station."""
    cfg = cfg or Config()
    scale = speed_scale_for(hand, cfg.speed_mm_s)

    def reach(dof: int, fraction: float) -> float:
        """`fraction` of DOF `dof`'s reachable range, in mm. Neither end is a
        rail end; see `open_fraction` and `margin_mm`."""
        far = hand.clamped_mm(dof, hand.travel_mm[dof] * cfg.open_fraction)
        return cfg.margin_mm + fraction * (far - cfg.margin_mm)

    def steps(dof: int, extent: float) -> list[float]:
        """Fractions of DOF `dof`'s range to stand at, so the face overlaps."""
        need = math.ceil((reach(dof, 1.0) - reach(dof, 0.0)) / extent)
        return [k / need for k in range(need + 1)]

    slides, seps = steps(BASE_LEFT, FACE_X_MM), steps(Z, FACE_Z_MM)

    # Serpentine: the slide order reverses on every other separation, so each
    # transition moves exactly one stepped axis.
    stations = [(slide, sep) for i, sep in enumerate(seps)
                for slide in (slides if i % 2 == 0 else slides[::-1])]

    def spread(dofs, fraction):
        return {dof: reach(dof, fraction) for dof in dofs}

    rows, jaw = [], 0.0
    for k, (slide, sep) in enumerate(stations):
        rows.append(Move(label=f"station {k + 1}/{len(stations)}: "
                               f"slide {slide:.2f}, z {sep:.2f}",
                         goal={**spread(SLIDE, slide), **spread(SEP, sep),
                               **spread(JAW, jaw)},
                         speed_scale=scale))
        jaw = 1.0 - jaw
        rows.append(Move(label="sweep jaws " + ("open" if jaw else "shut"),
                         goal=spread(JAW, jaw), speed_scale=scale))
    rows.append(Hold(seconds=cfg.dwell))

    # Consecutive stations OVERLAP, on both stepped axes. This is the one thing
    # about this task that fails invisibly: too few stations leaves an unswept
    # band in the middle of the volume, and the hand looks exactly the same
    # doing it. On the current table two z stations would cover the bottom and
    # top 32 mm of a 44 mm range and miss 12 mm between them.
    #
    # It catches fractions that BYPASS `steps` -- someone writing `[0.0, 1.0]`
    # because two stations look obviously enough. It does NOT catch a stale
    # `FACE_*_MM`, because that constant feeds `steps` and the threshold here
    # alike and moves both together; only re-measuring against the MJCF can
    # falsify those two numbers. Worth knowing before concluding this assert is
    # dead code, which is what perturbing the constant makes it look like.
    for axis, dof, extent in ((0, BASE_LEFT, FACE_X_MM), (1, Z, FACE_Z_MM)):
        apart = [abs(b[axis] - a[axis]) * (reach(dof, 1.0) - reach(dof, 0.0))
                 for a, b in zip(stations, stations[1:])]
        assert max(apart) <= extent + 1e-9, \
            f"stations are {max(apart):.1f} mm apart on a {extent:.1f} mm face"
    # And exactly one stepped axis moves per transition, which is what makes the
    # path a serpentine rather than a diagonal across the aperture.
    for a, b in zip(stations, stations[1:]):
        assert (a[0] != b[0]) != (a[1] != b[1]), f"diagonal transition {a}->{b}"
    # `isinstance`, not `getattr(row, "goal")` -- a `Hold` carries a `goal` too,
    # and it is a bare float, so the duck-typed version quietly checks nothing.
    assert all(0.0 <= mm <= hand.travel_mm[dof] for row in rows
               if isinstance(row, Move) for dof, mm in row.goal.items()), \
        "a goal is off the travel table -- a carriage would leave its rail"

    return Sequence(rows, hand=hand,
                    start_mm=start_mm,
                    travel_torque=cfg.travel_torque,
                    approach_torque=cfg.approach_torque,
                    approach_speed=cfg.approach_speed,
                    # A scaled row keeps its unscaled deadline; see
                    # `speed_scale_for`.
                    timeout_margin=cfg.timeout_margin / scale)

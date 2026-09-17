"""Run the twelve motion primitives of `tab:primitives` once each, on real hardware.

The paper's figure 3 draws twelve panels; this is the same twelve as motion. Each
panel becomes one block:

    move to the panel's START pose -> dwell -> move to its END pose -> dwell
      -> move back to START -> dwell

A visual demo, not a manipulation task: every goal is free space, nothing is
gripped, and no row probes for contact. Run it with the jaws empty.

Why out-and-back
----------------
Most of the figure's arrows are double-headed -- open/close, push/pull,
lift/lower -- so one stroke would have to pick a direction and silently drop the
other half of the primitive's name. The return also leaves the hand on a known
pose, so the next panel's start move is short and the tour reads as one hand
doing twelve things rather than twelve unrelated clips.

It dissolves one disagreement with the figure for free: the figure's *dual grasp*
panel runs its jaws LO -> HI (opening) while carrying inward "grasp" arrows. Both
directions run here, so there is nothing to pick.

`PANELS` is transcribed from the accompanying paper's primitive figure
------------------------------------------------------------------
Same twelve entries, same driven joints, same start/end fractions, with the MJCF
joint names resolved to DOF indices. That mapping is the identity on the paper's
numbering -- `config.LAYOUT` and the paper's `q_0..q_6` agree -- so `left_up_y`
is DOF 4, `right_up_finger_x` is DOF 6, and so on.

**The ordered (start, end) pair is kept, never normalized to {low, high}.** On
*twist*, *dual rotate* and *squeeze-twist* the entire content of the primitive is
the RELATIVE SIGN between the two fingers of a stage: one runs HI -> LO while its
partner runs LO -> HI. Normalizing each DOF independently would run both the same
way and turn every one of those panels into its neighbour (twist into pivot, dual
rotate into dual translate) with nothing to see -- the failure mode a symmetric
gripper cannot report.

Fractions, not millimetres
--------------------------
`reach` maps a fraction of each DOF's own range to mm, so one table serves DOFs
with different travels and the file carries no per-DOF constant.

Neither end of that map is the rail's end. `open_fraction` keeps the far end off
the travel table -- that table is CAD and reads high, so asking for the limit
risks the hard stop `zero` finds on purpose, under a torque this task never
claims (`HandConfig.upper` says so; `demo.py` and `ready.py` use the same knob).
`margin_mm` keeps the near end off the stop at the other end, because a free
`Move` is closed-loop and a joint that parks against the stop a millimetre short
of a 0.0 goal reads as a stall, not an arrival.

No row names an `effort`
------------------------
Deliberate, and it is what makes z work. `Sequence.travel_effort` floors every
DOF at its `torque_min_to_move`, which on both hands is 800 for z against ~250
horizontal. A row that names its own effort opts out of that floor (`_effort`
only raises a `loaded` row), and an under-floored z row cannot move the stage in
either direction -- it burns its deadline, and because `Sequence.step` folds
`failed` into `done`, every row after it silently never runs. `cap.py` shipped
that bug.

Nothing here presses on anything, so nothing here has a reason to override it.
"""
from dataclasses import dataclass, field

import torch

from ..config import (AUX_JAW, AUX_LEFT, AUX_RIGHT, BASE_JAW, BASE_LEFT,
                      BASE_RIGHT, HandConfig, N_DOF, Z)
from ..primitives import Hold, Move, Sequence

# Fractions of a DOF's own reachable range, resolved by `reach`. Mirrors
# `plot_primitives.py`'s LO/HI/MID/GRIP so the two tables can be read side by side.
LO, HI, MID = 0.0, 1.0, 0.5

# Every DOF a panel does not drive sits here. Mid rather than shut, for the
# reason the figure gives: with a jaw closed its two fingers meet on the
# midplane and occlude each other in the finger-slide panels.
NEUTRAL = MID

# How far a squeeze closes, as a fraction of jaw range -- NOT to zero, the one
# place this tour does not run an actuator near its stop. A squeeze closes ON
# something, so a jaw stopped partway is the honest end pose. 0.15 is the
# figure's value.
GRIP = 0.15

# (label, {dof: (start fraction, end fraction)}). An empty dict means the panel
# poses the hand and holds it, which is what a static hold is.
PANELS = [
    ("static hold (0)", {}),
    ("point push/pull (1)", {AUX_RIGHT: (LO, HI)}),
    ("horizontal open/close (2)", {AUX_JAW: (LO, HI)}),
    ("vertical open/close (3)", {Z: (LO, HI)}),
    ("pivot/translate (4)", {AUX_LEFT: (LO, HI), AUX_RIGHT: (LO, HI)}),
    ("twist (4)", {AUX_LEFT: (HI, LO), AUX_RIGHT: (LO, HI)}),
    ("dual rotate (4+4)", {BASE_LEFT: (HI, LO), AUX_LEFT: (HI, LO),
                           BASE_RIGHT: (LO, HI), AUX_RIGHT: (LO, HI)}),
    ("dual translate (4+4)", {BASE_LEFT: (LO, HI), BASE_RIGHT: (LO, HI),
                              AUX_LEFT: (LO, HI), AUX_RIGHT: (LO, HI)}),
    ("dual grasp (2+2)", {BASE_JAW: (LO, HI), AUX_JAW: (LO, HI)}),
    ("squeeze-lift/lower (2+3)", {AUX_JAW: (HI, GRIP), Z: (LO, HI)}),
    ("squeeze-twist lift/lower (2+3+4)",
     {AUX_JAW: (HI, GRIP), Z: (LO, HI),
      AUX_LEFT: (HI, LO), AUX_RIGHT: (LO, HI)}),
    ("squeeze-push/pull (1+1+2)",
     {AUX_JAW: (HI, GRIP), AUX_LEFT: (LO, HI), AUX_RIGHT: (LO, HI)}),
]


@dataclass
class Config:
    """Bench units. Every `tune` field becomes a slider on the studio page."""

    label: str = "Primitive tour"
    sets_datum: bool = False

    open_fraction: float = field(default=0.9, metadata={"tune": (0.5, 0.98)})
    """Far end of each DOF's reachable range, as a fraction of its travel table
    entry. Never 1.0 -- see the module docstring."""

    margin_mm: float = field(default=1.0, metadata={"tune": (0.0, 5.0)})
    """Near end of that range, in mm off the hard stop at 0."""

    dwell: float = field(default=0.5, metadata={"tune": (0.0, 3.0)})
    """Seconds to pause after each move lands, so a phase is visible before the
    next one starts."""

    hold_seconds: float = field(default=2.0, metadata={"tune": (0.5, 10.0)})
    """How long the static-hold panel holds. Its own field because it is the
    panel's whole content, not a pause between two things."""

    travel_torque: float = field(default=50.0, metadata={"tune": (30.0, 200.0)})
    approach_torque: float = field(default=150.0, metadata={"tune": (50.0, 300.0)})
    approach_speed: float = field(default=800.0, metadata={"tune": (25.0, 800.0)})
    """Unused: this task never probes or creeps, so nothing in it commands the
    approach torque or speed. Carried only because `Sequence` takes them."""

    timeout_margin: float = field(default=1.5, metadata={"tune": (1.0, 3.0)})


def build(hand: HandConfig, start_mm: torch.Tensor,
          cfg: Config | None = None) -> Sequence:
    """Entry point for `--task primitive_tour`. Twelve panels, six rows each."""
    cfg = cfg or Config()

    def reach(dof: int, fraction: float) -> float:
        """`fraction` of DOF `dof`'s reachable range, in mm.

        0 is `margin_mm` off the hard stop, 1 is `open_fraction` of the travel
        table. Neither end is a rail end -- see the module docstring.
        """
        far = hand.clamped_mm(dof, hand.travel_mm[dof] * cfg.open_fraction)
        return cfg.margin_mm + fraction * (far - cfg.margin_mm)

    def pose(panel: dict[int, tuple[float, float]], side: int) -> dict[int, float]:
        """Full seven-DOF goal for one side of a panel -- `side` 0 for start, 1
        for end. Driven DOFs take that fraction, every other DOF NEUTRAL."""
        fractions = {dof: NEUTRAL for dof in range(N_DOF)}
        fractions.update({dof: pair[side] for dof, pair in panel.items()})
        return {dof: reach(dof, f) for dof, f in fractions.items()}

    rows = []
    for label, panel in PANELS:
        start, end = pose(panel, 0), pose(panel, 1)
        if not panel:
            rows += [Move(label=label, goal=start),
                     Hold(seconds=cfg.hold_seconds)]
            continue
        rows += [Move(label=f"{label}: start", goal=start),
                 Hold(seconds=cfg.dwell),
                 Move(label=label, goal=end),
                 Hold(seconds=cfg.dwell),
                 Move(label=f"{label}: return", goal=start),
                 Hold(seconds=cfg.dwell)]

    return Sequence(rows, hand=hand,
                    start_mm=start_mm,
                    travel_torque=cfg.travel_torque,
                    approach_torque=cfg.approach_torque,
                    approach_speed=cfg.approach_speed,
                    timeout_margin=cfg.timeout_margin)

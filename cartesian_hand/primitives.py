"""Joint pairings, as one-line steps a task cannot line up wrongly.

Each helper fills one `Step` and returns it, so a task reads as a flat list of
moves with nothing between the helper and the engine:

    twist(p.step(), AUX_LEFT, AUX_RIGHT, span, squeeze)

**Only pairings live here.** A helper that takes an opaque `dof_ids` and treats
every element the same adds no information over `Step.set` -- it is
`set(dofs, goal, torque, stop, timeout)` under a second name, and a vocabulary
of those (`travel_to`, `probe`, `grip`, `slide_finger`) was deleted once
already. What survives is the three below, because each binds *which* joint
gets *which* goal, and that bug is silent: a stage-symmetric gripper runs a
crossed pairing without complaint.

Goals are millimetres, positive-is-extend on every DOF -- `counts_to_mm` already
applied orientation, so a task writes `-80` for "retract 80 mm" on any DOF and
never an `orientation` term. The raw-count version needs one and gets it wrong
about half the time it is re-derived.

The paired goals below are `[1, n]` rather than a bare length-n list because
`Step.set` refuses an ambiguous 1-D value -- see `Step._spread`.
"""
import torch

from .config import AUX_LEFT, AUX_RIGHT, BASE_LEFT, BASE_RIGHT
from .motions import Step


def twist(step: Step, left_joint: int, right_joint: int, span: float,
          torque: float, timeout_s: float = 6.0,
          when: torch.Tensor | None = None) -> Step:
    """The paired fingers' coordinated half-twist: left to 0, right to `span`.

    Ids and goals are built together so a caller cannot line the wrong number up
    against the wrong finger -- which is how the aux fingers ended up reversed
    once already. Swapping the two id arguments is what runs the twist backwards,
    so a stroke and its reset are the same call with the ids exchanged.

    A full turn is two of these with the object re-gripped between, which is what
    the stroke phase of `tasks.cap.build` emits.
    """
    return step.set([left_joint, right_joint],
                    torch.tensor([[0.0, float(span)]], device=step.device),
                    torque, "goal", timeout_s, when)


def tilt(step: Step, stage_a: int, stage_b: int, other_a: int, other_b: int,
         span: float, torque: float, timeout_s: float = 6.0,
         when: torch.Tensor | None = None) -> Step:
    """Pitch one stage's grip relative to the other: `stage_*` move, `other_*` hold.

    `*_a` names the same column (both left, or both right) on each stage.
    Crossing the columns pitches the grip sideways of what the task meant, which
    a symmetric gripper cannot report, so the same-id trap is refused here rather
    than built silently. Hinged-lid opens are two of these with the lid re-seated
    between.
    """
    if stage_a == stage_b or other_a == other_b:
        raise ValueError(
            f"tilt: each column must name two distinct joints, got "
            f"({stage_a}, {stage_b}) and ({other_a}, {other_b})")
    return step.set([stage_a, stage_b, other_a, other_b],
                    torch.tensor([[float(span), float(span), 0.0, 0.0]],
                                 device=step.device),
                    torque, "goal", timeout_s, when)


def rotate_in_place(step: Step, span: float, torque: float,
                    timeout_s: float = 6.0,
                    when: torch.Tensor | None = None) -> Step:
    """Yaw the object about the vertical axis without translating it.

    Two opposing twists, one per stage, so the object's centre holds still on
    average. Built here rather than as two `twist` calls in a task for the same
    reason `twist` exists: the four-way pairing is the primitive, it is
    positional in LAYOUT order, and the runtime has no other clue which joint
    opposes which.
    """
    return step.set([BASE_LEFT, BASE_RIGHT, AUX_LEFT, AUX_RIGHT],
                    torch.tensor([[float(span), 0.0, 0.0, float(span)]],
                                 device=step.device),
                    torque, "goal", timeout_s, when)



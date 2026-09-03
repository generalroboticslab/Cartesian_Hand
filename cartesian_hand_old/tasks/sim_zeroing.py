"""Sim-side zeroing as a [N, J, K] motion program.

Hardware zeroing (`zeroing.py`) is a Python loop because pre-zero mm is
undefined there. In sim, hands start at the rest pose with `q=0` defined
as the closed hard stop by construction. Zeroing is still needed (rest pose
isn't quite the hard stop — manufacturing tolerances), but the *driver*
can be the motion engine, batched over N envs on GPU.

This module is the parallel to `zeroing.zero_all` for the sim backend.
Shares PHASES and OVERTRAVEL_MM with the hardware path; the difference is
the driver and that positions are in mm.
"""
import torch

from ..hand import ZEROING_TORQUE
from ..motions import Move, Program
from .zeroing import PHASES

DESCRIPTION = "Zero every DOF in a sim batch by driving past its hard stop"

# Sim hands have ~5-10mm of compliance past the rest pose. 20mm is enough
# to push past that and seat against the rigid stop.
OVERTRAVEL_MM = 20.0

# Stall threshold in mm/s. A stopped sim joint sits at qpos noise of ~1e-4
# per step; with hz=200 that's 0.02 mm/s, comfortably below 0.3.
STUCK_SPEED_MM_S = 0.3

# Joint tolerance: tighter than the hardware default. A sim goal of
# `start - OVERTRAVEL_MM` lands close to the hard stop when the start is close
# to the stop + overtravel; the hardware's 0.5mm tolerance would read that as
# "at goal" and retire on GOAL instead of STUCK. 0.1mm keeps the stall rule
# the trigger for every joint that hits the stop.
POSITION_TOLERANCE_MM = 0.1


def zero_program(hand_cfg, n_envs: int, start_pos_mm: torch.Tensor,
                 hz: float = 200.0, device: str = "cpu") -> Program:
    """Build a [N, J, K] zeroing program. K spans the phases.

    `start_pos_mm` is [N, J] in mm — the joint positions at the moment zeroing
    begins. Each phase drives its DOFs OVERTRAVEL_MM past their per-env starting
    position with stop="stuck". The recorded zero offset per DOF is what the
    engine retires at, read after the run from `motions.held_goal`.
    """
    p = Program(n_envs, hand_cfg.n_dof, hz, device,
                stuck_speed_mm_s=STUCK_SPEED_MM_S,
                position_tolerance_mm=POSITION_TOLERANCE_MM)
    for dof_ids, _name in PHASES:
        # Per-env goal = start_pos - overtravel. The start_pos varies per env,
        # so the goal tensor is [N, len(dof_ids)].
        start = start_pos_mm[:, dof_ids]                    # [N, len(dof_ids)]
        goals = start - OVERTRAVEL_MM                       # [N, len(dof_ids)]
        moves = {d: Move(goal=goals[:, i],
                         torque=ZEROING_TORQUE[d],
                         stop="stuck",
                         timeout_s=10.0)
                 for i, d in enumerate(dof_ids)}
        p.step(moves)
    return p

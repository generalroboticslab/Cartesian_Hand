"""Send every DOF to mid travel: the studio's Reset button.
Absolute goals, so zero the hand first -- uncalibrated, mm are measured from the
startup pose and mid travel from there can run a carriage off its rail. Torque is
the floor raised enough to break out of a stop.

Two steps, jaws first: whatever the hand is still holding must be released before
the fingers move, or the fingers drag it across the bench. The step barrier is
what enforces that -- the fingers cannot start until every jaw has arrived. Jaws
then keep their standing order through step 1 and stay open.
"""
from dataclasses import dataclass, field
import torch
from ..config import AUX_JAW, BASE_JAW, HandConfig
from ..motions import Program, Result, Task

MOVE_TORQUE = 200.0     # zero.py's park_torque: what breaks OUT of a hard stop
TIMEOUT_MARGIN = 1.5

JAWS = [BASE_JAW, AUX_JAW]


@dataclass
class Config:
    label: str = "Reset to ready"   # studio button text
    fraction: float = field(default=0.5, metadata={"tune": (0.1, 0.9)})
    # Jaws sit open, not mid travel: ready is the pose a tool is handed into,
    # and mid travel is not enough clearance to get one between the jaws.
    jaw_fraction: float = field(default=0.85, metadata={"tune": (0.1, 0.95)})


def build(hand: HandConfig, start_mm: torch.Tensor, cfg: Config | None = None,
          **kwargs) -> Task:
    cfg = cfg or Config()
    rest = [d for d in range(hand.n_dof) if d not in JAWS]
    torque = hand.gain_vector("torque_min_to_move", start_mm.device).float()
    upper = hand.upper(start_mm.device)
    move_torque = torque[None].clamp(min=MOVE_TORQUE)
    p = Program(*start_mm.shape, hand.control_hz, start_mm.device)
    p.step().set(JAWS, upper[JAWS] * cfg.jaw_fraction, move_torque[:, JAWS],
                 "goal", hand.travel_budget(JAWS, TIMEOUT_MARGIN))
    p.step().set(rest, upper[rest] * cfg.fraction, move_torque[:, rest],
                 "goal", hand.travel_budget(rest, TIMEOUT_MARGIN))
    m = p.build()
    here = yield m
    return Result(here, m.all_reached(JAWS, 0) & m.all_reached(rest, 1),
                  "some joints never reached mid travel")

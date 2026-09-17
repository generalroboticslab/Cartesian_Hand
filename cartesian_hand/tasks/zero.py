"""Find every DOF's hard stop and report it as that hand's zero offset.

Per phase in mechanical order: creep into the stop, record it, back off to mid
travel. Goals are relative, so this runs uncalibrated. Returns `Result`:
[N, J] stops, [N] ok flags.
"""
from dataclasses import dataclass, field
import torch
from ..config import AUX_FINGERS, AUX_JAW, BASE_FINGERS, BASE_JAW, HandConfig, Z
from ..motions import Program, Result, Task

# Mechanical order. A wrong order here closes a jaw on a finger.
PHASES = ((tuple(BASE_FINGERS + AUX_FINGERS), "fingers"),
          ((BASE_JAW, AUX_JAW), "jaws"), ((Z,), "z"))


@dataclass
class Config:
    label: str = "Zero hand"        # studio button text; empty means none
    sets_datum: bool = True         # result becomes the hand's zero
    overtravel_mm: float = 80.0     # outruns any rail, incl. MockServo's 73.6
    seek_speed_mm_s: float = field(default=0.75, metadata={"tune": (0.4, 3.0)})
    """How fast the setpoint creeps into the stop. THE force knob, and the one
    that makes a workable torque exist at all.

    A servo develops effort from position error, so a setpoint sprinting away
    from a blocked carriage saturates whatever ceiling is in force: too fast and
    the finger climbs a gear tooth at the stop, and the torque low enough to
    prevent that is too low to break stiction anywhere else. Slow the setpoint
    and both ends resolve -- the error, and so the contact force, stays bounded.

    0.75 mm/s is the 60 counts/s of an earlier internal implementation's
    zeroing task, whose comment names this exact failure. The park is left at
    the hand's transit speed: it moves away from the stop, where there is
    nothing to hit."""
    timeout_margin: float = field(default=1.0, metadata={"tune": (1.0, 3.0)})
    park_torque: float = field(default=200.0, metadata={"tune": (100.0, 800.0)})
    """A park breaks OUT of a stop, which needs more than `torque_min_to_move` --
    that is the least force that moves a FREE joint. Deleting this once left the
    jaws too weak to leave their stop."""


def build(hand: HandConfig, start_mm: torch.Tensor, cfg: Config | None = None,
          **kwargs) -> Task:
    cfg = cfg or Config()
    floor = hand.gain_vector("torque_min_to_move", start_mm.device).float()
    stops, why = start_mm.clone(), ""
    alive = torch.ones(len(start_mm), dtype=torch.bool, device=start_mm.device)

    def move(goal, torque, stop, speed=0.0):
        """Seek and park are one row: same joints, different sign and stop rule.

        `speed=0` means the hand's transit gain. `frame="here"` throughout, so
        every goal is a distance and the uncalibrated origin cancels. The budget
        is derived from the speed actually asked for, or a creep row would be
        timed out by the transit table it is deliberately ignoring.
        """
        p = Program(*start_mm.shape, hand.control_hz, start_mm.device)
        p.step().set(ids, goal, torque, stop,
                     hand.travel_budget(ids, cfg.timeout_margin,
                                        speed_mm_s=speed or None),
                     frame="here", speed_mm_s=speed)
        return p.build()

    for dof_ids, name in PHASES:
        ids = list(dof_ids)
        # Overtravel outruns the rail, so the stop and not the number ends the
        # move -- which is why the executor must not clamp a task's goal.
        seek = move(-cfg.overtravel_mm, floor[None, ids], "stuck",
                    cfg.seek_speed_mm_s)
        here = yield seek

        # The outcome, not the position: a stall and a timeout both end
        # stationary and are indistinguishable in millimetres.
        reached = seek.all_reached(ids)
        lost, alive = ~reached & alive, reached & alive
        if not why and bool(lost.any()):
            why = (f"phase {name!r}: envs {lost.nonzero().flatten().tolist()} "
                   f"never reached a hard stop")
        stops[:, ids] = here[:, ids]

        # Back off half of each DOF's own rail. Dead envs get 0 and park in
        # place: their "stop" is wherever the budget ran out, so half a rail
        # from there aims past the open end.
        yield move(hand.upper(here.device)[None, ids] / 2 * alive[:, None],
                   floor[None, ids].clamp(min=cfg.park_torque), "goal")

        if not alive.any():
            break
    return Result(stops, alive, why)

"""Bench-time every DOF's fastest 0 -> full-travel transit.

Procedure
---------
1. Every DOF to mid travel.
2. Each finger (base left, base right, aux left, aux right) in turn: retract
   it to 0, then push it to full travel at top torque and top speed, timing
   the push.
3. Each of the base jaw, aux jaw, and z in turn: retract every finger to 0
   (clear of the jaw), retract this DOF to 0, then push it to full travel at
   top torque and top speed, timing the push.

`Result.value` is `[N, n_dof]` mm/s -- the ACHIEVED speed per DOF, all seven
columns filled since every DOF gets exactly one timed push. This is measured
distance over measured time, not the commanded goal over a budget: the stop
position is read back off `Motions.stop_position_mm` (wherever the row
actually retired, goal or timeout alike), not assumed to be the nominal
`travel_mm` table -- which MEMORY records as disagreeing with the real rail on
every DOF. A push that stalls short still reports a real, if lower, mm/s
rather than a nonsense number built on a distance that never happened.

**Top ACCEL is not settable here.** `acc` is a session-wide register
`studio.live` hoists once at startup (`cfg.gain_vector("acc")`) and reads
through `gains()` on every tick; a `Motions` row has a `torque` channel and a
`speed_mm_s` channel but no `acc` channel, so no task can touch it. Set the
page's `acc` slider to 255 (or start the hand from an `acc=255` variant)
before running this task for a genuine top-accel measurement -- otherwise it
times whatever the hand's current `acc` gain happens to produce.

**Wall-clock timing is meaningless under `sim.run`.** `sim.py` steps with no
`time.sleep` throttle, so `time.time()` deltas there measure Python loop
overhead, not motion -- and its fixed-`kp` `<position>` actuators ignore the
torque/speed/acc registers entirely (MEMORY: "Torque is a CAP"). This task is
a `studio.live` bench tool, real hardware or `--mock`, both of which share the
same real-time-paced control loop.
"""
import time
from dataclasses import dataclass, field

import torch

from ..config import AUX_FINGERS, AUX_JAW, BASE_FINGERS, BASE_JAW, HandConfig, Z
from ..motions import Program, Result, Task
from ..primitives import SERVO_NO_LOAD_RPM

FINGERS = list(BASE_FINGERS) + list(AUX_FINGERS)   # [1, 2, 5, 6]
STAGES = (BASE_JAW, AUX_JAW, Z)                    # [0, 4, 3]

TOP_TORQUE = 1000.0
"""Servo register ceiling -- `servo.py`: "torque 0-1000, a force cap."""
RETRACT_TORQUE = 200.0
"""zero.py's park_torque: what breaks a joint OUT of the hard stop it retracts
onto, not the floor that merely moves a free joint."""


@dataclass
class Config:
    label: str = "Speed test"        # studio button text
    sets_datum: bool = False
    save_json: str | None = "speed.json"
    """Append every run's per-DOF mm/s here (see `tasks.result_path`), so a
    bench session accumulates rather than only ever showing its last line."""
    timeout_margin: float = field(default=5.0, metadata={"tune": (1.5, 10.0)})
    """Generous on purpose: this drives every joint at max torque and max
    speed, the least-tested corner of its envelope, where a stall looks
    nothing like the hand's ordinary transit budget."""


def build(hand: HandConfig, start_mm: torch.Tensor, cfg: Config | None = None,
          **kwargs) -> Task:
    cfg = cfg or Config()
    device = start_mm.device
    n, j = start_mm.shape
    floor = hand.gain_vector("torque_min_to_move", device).float()
    retract_torque = floor[None].clamp(min=RETRACT_TORQUE)
    upper = hand.upper(device)
    # The servo's rated no-load top speed -- see `primitives.SERVO_NO_LOAD_RPM`,
    # the same ceiling `Sequence` pins every free move to regardless of gains.
    top_speed_mm_s = (SERVO_NO_LOAD_RPM / 60.0 * hand.counts_per_rev
                      / hand.counts_per_mm)

    def move(ids, goal_mm, torque, speed=0.0):
        p = Program(n, j, hand.control_hz, device)
        p.step().set(ids, goal_mm, torque, "goal",
                     hand.travel_budget(ids, cfg.timeout_margin,
                                        speed_mm_s=speed or None),
                     speed_mm_s=speed)
        return p.build()

    def retract(ids):
        return move(ids, 0.0, retract_torque[:, ids])

    def time_push(dof: int, start_pos: torch.Tensor):
        """One DOF, 0 -> full travel, top torque and speed.

        `t0` is set immediately before the only `yield` here, and the
        generator does not resume until `TaskRunner` has driven this program
        to completion, so the wall-clock delta is exactly that transit's real
        duration under `studio.live`'s real-time-paced loop.

        Distance is `stop_position_mm` (where the row actually retired) minus
        `start_pos` (the measured position handed back by the retract this
        push followed) -- both read off the servo, never the commanded goal
        or the nominal travel table, so a push that stalls short still
        reports the real distance it covered before that."""
        program = move([dof], upper[[dof]], TOP_TORQUE, top_speed_mm_s)
        t0 = time.time()
        yield program
        elapsed = time.time() - t0
        distance = program.stop_position_mm[:, dof, 0] - start_pos[:, dof]
        speed = distance.abs() / max(elapsed, 1.0 / hand.control_hz)
        return speed, program.all_reached([dof])

    mm_s = torch.zeros((n, j))
    succeeded = torch.ones(n, dtype=torch.bool, device=device)
    why = ""

    def note_failure(dof: int, ok_dof: torch.Tensor) -> None:
        nonlocal why
        if not why and bool((~ok_dof).any()):
            envs = (~ok_dof).nonzero().flatten().tolist()
            why = f"dof {dof}: envs {envs} never reached full travel in time"

    yield move(list(range(j)), upper / 2, floor[None])

    for dof in FINGERS:
        start_pos = yield retract([dof])
        speed, ok_dof = yield from time_push(dof, start_pos)
        mm_s[:, dof] = speed
        succeeded &= ok_dof
        note_failure(dof, ok_dof)

    for dof in STAGES:
        yield retract(FINGERS)
        start_pos = yield retract([dof])
        speed, ok_dof = yield from time_push(dof, start_pos)
        mm_s[:, dof] = speed
        succeeded &= ok_dof
        note_failure(dof, ok_dof)

    return Result(mm_s, succeeded, why)

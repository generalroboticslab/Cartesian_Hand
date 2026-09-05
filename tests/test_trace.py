"""Golden-trace check: does a task still issue the commands it used to?

`tests/` was deleted on 2026-09-04 and nothing replaced it, so any edit to the
engine, the primitives or a task file is currently unguarded. This file is the
minimum that makes a refactor falsifiable. It is not a substitute for the suite
that was lost -- it pins *behaviour*, not the invariants those tests reasoned
about -- but a change that alters what a task commands cannot pass it silently.

Why a trace and not an outcome. Only `zero` completes in `sim.run`: the model
has no bottle, cap, screw or scissor, so every manipulation task's first probe
closes on air and the run raises. Asserting on outcomes would therefore cover
one task in eight. What every task *does* have, object or not, is the sequence
of goals it puts on the wire, and that sequence changes the moment a row, a
deadline, an effort floor or a phase order does.

The plant is deliberately the crudest thing that terminates: a joint walks
toward its goal at the commanded speed and stops dead at a limit. Same model as
`MockServo` and as `sim.profile`, in millimetres and without the bus or mujoco,
so a run is milliseconds and is bit-reproducible. It has no friction, no
following error and no notion of force, so it cannot see a below-floor torque or
a one-count dither -- the two bug classes `MEMORY.md` records as invisible on
every backend. Do not read a pass here as evidence about either.

    python tests/test_trace.py            # check against the recorded traces
    python tests/test_trace.py --update   # re-record after an intended change
"""
import hashlib
import json
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from cartesian_hand import config, tasks                        # noqa: E402
from cartesian_hand.motions import TaskRunner                   # noqa: E402
from cartesian_hand.policy import Policy, PolicyRunner          # noqa: E402

GOLDEN = pathlib.Path(__file__).with_name("traces.json")

MAX_TICKS = 4000
"""Backstop, not a schedule. Long enough for `zero`'s six programs at the
shipped creep speed; a task that needs more is hanging, and its trace says so."""

OBJECT_MM = 20.0
"""Where a jaw meets the object it is closing on, mm from the closed end.

Both jaws get one, so a `Probe` ends on contact rather than running out its
deadline. The value only has to be inside the rail and clear of the goals the
tasks command; it is not a measurement of anything.
"""


def toy_plant(cfg: config.HandConfig):
    """Per-DOF `(low, high)` stops in millimetres.

    Low is 0: the closed end is the hard stop zeroing seeks, and in the
    millimetre frame every DOF's stop is at the origin. High is the travel
    table, which is what a carriage runs off the end of.
    """
    high = cfg.upper().clone()
    low = cfg.lower().clone()
    for jaw in (config.BASE_JAW, config.AUX_JAW):
        low[jaw] = OBJECT_MM
    return low[None, :], high[None, :]


def trace(name: str, cfg: config.HandConfig) -> dict:
    """Run one task against the toy plant and fingerprint what it commanded.

    Returns the tick count, the final pose, and a digest over every goal the
    controller issued. The digest is what makes this sensitive: two runs that
    end in the same place having taken different routes do not match.
    """
    low, high = toy_plant(cfg)
    position = high.clone()                 # start open, as a real hand does
    controller = tasks.make(name, cfg, position.clone())
    hz = cfg.control_hz

    if isinstance(controller, Policy):
        runner = PolicyRunner(controller, hz)
    else:
        runner = TaskRunner(controller, hold_torque=cfg.gain_vector(
            "torque_min_to_move").to(torch.float32)[None, :])
    # `Motions` carries goal and torque but no speed, so a generator task is
    # paced by the hand's configured speed, exactly as `sim.step_limit_mm` does.
    task_step = (cfg.gain_vector("speed").to(torch.float32)
                 / cfg.counts_per_mm / hz)[None, :]

    digest = hashlib.sha256()
    ticks = 0
    for ticks in range(1, MAX_TICKS + 1):
        if isinstance(controller, Policy):
            if runner.finished():
                break
            action = runner.tick(position)
            goal, step = action.goal_mm, action.max_speed_mm_s / hz
            effort = action.effort_limit * 1000.0    # to servo units, as issued
        else:
            issued = runner.tick(position)
            if issued is None:
                break
            goal, step, effort = issued[0], task_step, issued[1]
        # Effort is in the digest, not only the goal. Removing a torque floor
        # moves no joint on any backend -- the toy plant, `MockServo` and mujoco
        # all ignore force -- so a goal-only digest passed that mutation, which
        # is the one `MEMORY.md` records as costing a bench session.
        for issued_tensor in (goal, effort):
            digest.update(torch.round(issued_tensor * 100)
                          .to(torch.int32).numpy().tobytes())
        position = (position + (goal - position).clamp(-step, step)
                    ).clamp(low, high)

    return {"ticks": ticks,
            "final_mm": [round(v, 2) for v in position[0].tolist()],
            "goals_sha256": digest.hexdigest()[:16]}


def main(update: bool) -> int:
    cfg = config.HAND_1
    names = ["zero", "ready", "tilt", "cap", "scissors", "screwdriver",
             "syringe", "pipette"]
    current = {name: trace(name, cfg) for name in names}

    if update or not GOLDEN.exists():
        GOLDEN.write_text(json.dumps(current, indent=2) + "\n")
        print(f"recorded {len(current)} traces to {GOLDEN.name}")
        return 0

    golden = json.loads(GOLDEN.read_text())
    bad = [name for name in names if current[name] != golden.get(name)]
    for name in names:
        mark = "FAIL" if name in bad else "ok  "
        print(f"{mark} {name:12s} {current[name]['ticks']:5d} ticks  "
              f"{current[name]['goals_sha256']}")
        if name in bad:
            print(f"       was {golden.get(name)}")
            print(f"       now {current[name]}")
    assert not bad, f"trace changed for {bad}; --update if intended"
    print(f"all {len(names)} traces match")
    return 0


if __name__ == "__main__":
    raise SystemExit(main("--update" in sys.argv))

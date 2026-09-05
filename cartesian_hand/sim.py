"""The other backend: run a task against mujoco instead of against servos.

    python -m cartesian_hand.sim --task zero
    python -m cartesian_hand.sim --task cap

Same task and direct-policy files as `studio.live`, not ports of them. Existing
tasks yield `Motions` programs; a direct policy instead receives the canonical
typed observation and returns goal, speed, and effort every tick. What differs
between here and hardware is the adapter around either controller -- `mj_step`
and a `qpos` read where the other has `set_positions` and `read_all`.

Deliberately not covered
------------------------
**Effort is dropped on the floor here.** `Motions` torque and direct-policy
`Action.effort_limit` both reach this adapter, which cannot apply either: the
MJCF's actuators are `<position>` with a fixed `kp`, so there is no
per-environment force cap to write. Every
`stop="stuck"` in a task therefore fires against a joint limit rather than
against a grip that yields at a tuned force. That is enough to prove a task
runs, and not enough to transfer a *force*; the sim's own README calls its
dynamics placeholder, so the missing piece is upstream of this file.

Two backends, one loop
----------------------
`run` is CPU MuJoCo at N=1 -- the reference, and the fallback on a box with no
CUDA. `run_warp` is `mujoco_warp` at any N. Neither is a port of the other:
both hand `drive` the same four verbs and it owns the loop, so there is exactly
one place where a tick is decided.

What makes the batched one worth having is that `Motions` is already torch, and
`wp.to_torch` views Warp's `qpos`/`ctrl` with no copy, so the whole control loop
stays resident on the GPU -- measured millimetres in and goals out never touch
the host. Measured on an RTX 4090 with the shipped model: 74.6k steps/s on CPU
at N=1 against 8.1M steps/s at N=4096, a 109x speedup. That figure is an upper
bound: the model has no objects in it yet, so `ncon` is 0 and collision is doing
no work. Re-measure once there is something to grasp.

**`mj_step`, not `mj_forward`.** The opposite of `studio.live`, and for the
opposite reason: there the physics *is* the hand and re-simulating would
overwrite what it reported, here the physics is all there is. Stepping is also
what applies the `<equality>` couplings, so a rack pair's follower moves on its
own rather than needing the explicit write the studio loop does.
"""
import mujoco
import numpy as np
import torch
import tyro
from collections.abc import Callable

from .config import DEFAULT_HAND, HandConfig, get_hand
from .mjcf import MM_PER_M, mjcf_path, narrow_ctrlrange, qpos_addrs
from .motions import Task, TaskRunner
from .policy import Policy, PolicyRunner, PolicyState
from . import tasks


def step_limit_mm(cfg: HandConfig,
                  device: str | torch.device = "cpu") -> torch.Tensor:
    """`[1, J]` millimetres a joint may be commanded to advance in one tick.

    The servo profiles internally: `set_positions(..., speed=300)` means the
    joint tracks a setpoint that walks toward the goal at 300 counts/s, not that
    it appears there. mujoco's `<position>` actuators model none of that -- they
    are handed a target and the solver drives to it as hard as `kp` allows -- so
    without this a sim joint crosses 55 mm in the same tick it crosses 5.

    That is not a cosmetic difference, it is the one that made time meaningless
    in sim: zeroing finished in 45 ticks here against roughly 4000 on the bench,
    every distance and every timeout cost the same, and a search over travel and
    budgets found a completely flat objective (measured 2026-09-03). Stall
    detection is affected too -- `STUCK_WINDOW_STEPS` is 10 ticks of *travel*,
    and a joint that arrives instantly never spends them.

    `tests/test_tasks.py::toy_run` has always modelled the servo this way
    (`pos + (goal - pos).clamp(-step_mm, step_mm)`); this is the same two lines,
    finally on the backend that claims to be the physical one.
    """
    return (cfg.gain_vector("speed").to(device=device, dtype=torch.float32)
            / cfg.counts_per_mm / cfg.control_hz)[None, :]


def profile(goal_mm: torch.Tensor, position_mm: torch.Tensor,
            limit_mm: torch.Tensor) -> torch.Tensor:
    """Walk the commanded goal toward `goal_mm` at no more than `limit_mm`/tick.

    Applied to the *command*, not to the joint: the setpoint is what the servo
    ramps, and the joint follows it with whatever error the load imposes. Doing
    it the other way -- clamping measured motion -- would model a joint that
    cannot be pushed off course, which is the opposite of what a stall is.
    """
    return position_mm + (goal_mm - position_mm).clamp(-limit_mm, limit_mm)


def drive(controller: Task | Policy, cfg: HandConfig,
          read_mm: Callable[[], torch.Tensor],
          write_goal: Callable[[torch.Tensor], None],
          advance: Callable[[], None],
          sense: Callable[[], torch.Tensor | None],
          report: Callable[[], None],
          limit_mm: torch.Tensor, hold_torque: torch.Tensor,
          max_seconds: float) -> torch.Tensor | PolicyState:
    """The control loop both backends run, and the only copy of it.

    A backend supplies four verbs -- read millimetres, write a goal, advance the
    physics, sample an external contact signal -- and differs in nothing else.
    Keeping the loop here is what stops a fix landing in one backend only.

    `report` runs after the loop and *before* the failure check, so a run that
    ends badly still prints where the hand got to.

    A legacy generator returns its measurement; a direct policy its typed final
    state. Raises `RuntimeError` if any env failed -- a probe that closed on
    air, a DOF that never found its stop. Tasks report that as `Result.ok`
    rather than raising (see `motions.Result`); turning it back into an
    exception is this backend's policy, not the task's, since a CLI that printed
    a radius after measuring nothing is worse than one that stops. A caller
    wanting the flags per env drives `TaskRunner` and reads `runner.result`.
    """
    hz = cfg.control_hz
    direct = (PolicyRunner(controller, hz)
              if isinstance(controller, Policy) else None)
    runner = (None if direct is not None else
              TaskRunner(controller, hold_torque=hold_torque))

    for _ in range(int(max_seconds * hz)):
        mm = read_mm()
        external = sense()
        if direct is not None:
            if direct.finished():
                break
            contact = (torch.as_tensor(external, dtype=torch.bool,
                                       device=mm.device)
                       if external is not None else None)
            action = direct.tick(mm, contact)
            goal, tick_limit = action.goal_mm, action.max_speed_mm_s / hz
        else:
            step = runner.tick(mm, external)
            if step is None:
                break
            goal, tick_limit = step[0], limit_mm
        # mujoco clamps ctrl to the range `narrow_ctrlrange` set, which is what
        # turns zeroing's deliberate 120mm overtravel into "drive to the closed
        # end and lean on it" rather than an unreachable goal.
        write_goal(profile(goal, mm, tick_limit))
        advance()
    else:
        raise RuntimeError(
            f"task did not finish within {max_seconds}s of simulated time")

    report()
    if direct is not None:
        if direct.failed():
            raise RuntimeError("direct policy failed")
        return direct.state
    value, ok, why = runner.result
    if not bool(ok.all()):
        raise RuntimeError(why)
    return value


def run(controller: Task | Policy, cfg: HandConfig, xml: str | None = None,
        swap: bool = False,
        max_seconds: float = 180.0, verbose: bool = True,
        external_signal: Callable[[mujoco.MjModel, mujoco.MjData],
                                  torch.Tensor | None] | None = None
        ) -> torch.Tensor | PolicyState:
    """CPU MuJoCo at N=1: the reference backend, and the one with no CUDA.

    `max_seconds` is simulated time and is a backstop, not a schedule: a task
    that ends on its own stop conditions finishes long before it, and one that
    does not is a bug worth stopping rather than a run worth waiting out.
    """
    model = mujoco.MjModel.from_xml_path(mjcf_path(xml))
    narrow_ctrlrange(model, cfg)
    data = mujoco.MjData(model)

    # One joint per DOF is enough to read: a rack pair's two joints are held
    # equal by the model's own constraint, so the follower carries no
    # information the leader does not.
    lead = np.array([row[0] for row in qpos_addrs(model, swap)])
    substeps = max(1, round((1.0 / cfg.control_hz) / model.opt.timestep))

    def advance() -> None:
        for _ in range(substeps):
            mujoco.mj_step(model, data)

    def write_goal(goal_mm: torch.Tensor) -> None:
        data.ctrl[:] = goal_mm[0].numpy() / MM_PER_M

    def report() -> None:
        if verbose:
            pos = " ".join(f"{v:6.1f}" for v in data.qpos[lead] * MM_PER_M)
            print(f"[sim] finished at mm [{pos} ]")

    return drive(
        controller, cfg,
        read_mm=lambda: torch.from_numpy(
            data.qpos[lead] * MM_PER_M).float()[None, :],
        write_goal=write_goal,
        advance=advance,
        sense=lambda: external_signal(model, data) if external_signal else None,
        report=report,
        limit_mm=step_limit_mm(cfg),
        hold_torque=cfg.gain_vector("torque_min_to_move")
        .to(torch.float32)[None, :],
        max_seconds=max_seconds)


def run_warp(controller: Task | Policy, cfg: HandConfig, xml: str | None = None,
             swap: bool = False, max_seconds: float = 180.0,
             verbose: bool = True, n_envs: int = 1,
             device: str | torch.device = "cuda",
             external_signal: Callable[[object, object],
                                       torch.Tensor | None] | None = None
             ) -> torch.Tensor | PolicyState:
    """`run`, batched over `n_envs` on the GPU via `mujoco_warp`.

    Same contract as `run` and the same loop: read millimetres, tick the task,
    write goals, advance the physics. The differences are all in those three
    verbs -- `wp.to_torch` views instead of a numpy read, and a captured Warp
    graph instead of a Python `mj_step` loop.

    `qpos` and `ctrl` are zero-copy torch views of Warp's own buffers, so
    assigning into `ctrl` IS the write to the simulator. The task's tensors were
    built on `start_mm.device`, which `main` puts on the GPU, so nothing in the
    tick synchronises with the host.

    The substep block is captured once as a Warp graph rather than launched as
    `substeps` separate Python calls: at 50 Hz control against a 0.5 ms timestep
    that is 40 launches per tick, and the launch overhead is the loop's cost
    once the physics itself is batched.

    Torque is dropped here exactly as in `run`, and for the same reason -- see
    the module docstring. Batching does not change what the actuators can be
    told.
    """
    import mujoco_warp as mjw
    import warp as wp

    model = mujoco.MjModel.from_xml_path(mjcf_path(xml))
    narrow_ctrlrange(model, cfg)                  # before put_model: it copies
    lead = [row[0] for row in qpos_addrs(model, swap)]

    m = mjw.put_model(model)
    d = mjw.make_data(model, nworld=n_envs)
    qpos, ctrl = wp.to_torch(d.qpos), wp.to_torch(d.ctrl)   # [N, nq], [N, nu]
    substeps = max(1, round((1.0 / cfg.control_hz) / model.opt.timestep))

    # Warm up to compile the kernels, then rewind: that first step advanced the
    # sim, and the task's first measurement must be the rest pose.
    mjw.step(m, d)
    wp.synchronize()
    mjw.reset_data(m, d)
    with wp.ScopedCapture() as capture:
        for _ in range(substeps):
            mjw.step(m, d)
    graph = capture.graph

    def write_goal(goal_mm: torch.Tensor) -> None:
        ctrl[:] = goal_mm / MM_PER_M

    def report() -> None:
        if verbose:
            pos = " ".join(f"{v:6.1f}"
                           for v in (qpos[0, lead] * MM_PER_M).tolist())
            print(f"[sim] {n_envs} env(s) finished, env 0 at mm [{pos} ]")

    return drive(
        controller, cfg,
        read_mm=lambda: qpos[:, lead] * MM_PER_M,
        write_goal=write_goal,
        advance=lambda: wp.capture_launch(graph),
        sense=lambda: external_signal(m, d) if external_signal else None,
        report=report,
        limit_mm=step_limit_mm(cfg, device),
        hold_torque=cfg.gain_vector("torque_min_to_move")
        .to(device=device, dtype=torch.float32)[None, :],
        max_seconds=max_seconds)


def main(task: str = "zero",
         hand: str = DEFAULT_HAND,
         xml: str | None = None,
         swap: bool = False,
         max_seconds: float = 180.0,
         n_envs: int = 1,
         warp: bool = False) -> None:
    """Run one task in simulation.

    Args:
        task: a file stem under `tasks/`, e.g. `zero` or `cap`. An unknown name
            raises with the registry listed.
        hand: which entry in `config.HANDS` supplies travel and control rate.
        xml: MJCF path. Default $CARTESIAN_HAND_MJCF, then the sibling checkout.
        swap: exchange finger DOFs 1<->2 and 5<->6.
        max_seconds: simulated-time backstop.
        n_envs: how many worlds to simulate. Above 1 implies --warp.
        warp: use the GPU `mujoco_warp` backend instead of CPU mujoco.
    """
    cfg = get_hand(hand)
    warp = warp or n_envs > 1
    # `start_mm` fixes the batch width AND the device for every tensor the task
    # builds, since `Program` is constructed with `start_mm.device`. Putting it
    # on the GPU is what keeps the whole tick off the host.
    device = "cuda" if warp else "cpu"
    start = torch.zeros(n_envs, cfg.n_dof, device=device)
    controller = tasks.make(task, cfg, start)
    result = (run_warp(controller, cfg, xml=xml, swap=swap,
                       max_seconds=max_seconds,
                       n_envs=n_envs, device=device) if warp else
              run(controller, cfg, xml=xml, swap=swap,
                  max_seconds=max_seconds))
    if isinstance(controller, Policy):
        print(f"[sim] {task}: done={result.done.tolist()}, "
              f"failed={result.failed.tolist()}")
    else:
        print(f"[sim] {task}: {[round(float(v), 2) for v in result[0].flatten()]}"
              if result.ndim > 1 else f"[sim] {task}: {result.tolist()}")


if __name__ == "__main__":
    tyro.cli(main, prog="python -m cartesian_hand.sim")

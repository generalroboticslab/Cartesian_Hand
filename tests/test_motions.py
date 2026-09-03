"""Self-check for the motion engine and its task runner.

    python tests/test_motions.py

`Motions` is the shared artefact between the hardware backend and the sim one,
so a bug here is a bug in both at once and in the same direction -- which is
the worst possible place for one, because agreeing backends look correct.

Everything below is a property that fails *silently*. None of these would raise
in production; each one would produce a plausible number. That is the selection
rule for what is worth a test here.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from cartesian_hand.config import HAND_2
from cartesian_hand.motions import (GOAL, STUCK, STUCK_WINDOW_STEPS, TIMEOUT,
                                    Motions, Program, TaskRunner)


def one_step(dofs, goal, torque, stop="goal", timeout_s=6.0, n_joints=1):
    """A one-step, one-env program. The fixture most tests here want."""
    p = Program(1, n_joints, 50.0)
    p.step().set(dofs, goal, torque, stop, timeout_s)
    return p.build()


def drive(runner, pos, floor=None, step_mm=0.5, max_ticks=200_000):
    """Tick a runner against a toy hand until the task finishes.

    Joints move `step_mm` per tick toward whatever goal they are given and
    refuse to pass `floor`. That is the least a stand-in can do and still
    exercise both stop rules: `goal` fires on arrival, `stuck` fires on the
    floor. No bus, no mujoco, no clock -- which is the point, since a task that
    needed any of those could not run on two backends.
    """
    for _ in range(max_ticks):
        step = runner.tick(pos)
        if step is None:
            return pos
        pos = pos + (step[0] - pos).clamp(-step_mm, step_mm)
        if floor is not None:
            pos = torch.maximum(pos, floor)
    raise AssertionError("task never finished")


def test_a_joint_the_program_never_mentions_holds_still():
    """Standing orders are seeded from the measured pose, not from zero.

    Seeding them from zero would command every joint the program does not use
    to close, hard, on the first tick.
    """
    m = Motions(n_envs=1, n_joints=2, n_steps=1, hz=50.0)
    m.acts[:, 0, 0] = True                # joint 1 is never mentioned
    m.goal_mm[:, 0, 0] = 5.0
    m.timeout_steps[:] = 10
    m.wants[:, 0, 0] = GOAL

    pos = torch.tensor([[0.0, 33.0]])
    m.start(pos, 50.0)
    goal, torque = m.step_once(pos)
    assert float(goal[0, 1]) == 33.0, "an unmentioned joint must be told to stay put"
    assert float(torque[0, 1]) == 50.0, "and must keep its torque, not go limp"


def test_a_probe_that_closes_on_air_does_not_report_contact():
    """Arriving and jamming both look like zero speed. Only one is contact.

    This is the cap probe with no cap under it: `stop="stuck"` alone, the jaw
    runs all the way to its commanded goal, and there it sits reading no motion.
    Without the `~at_goal` term in `step_once` that is indistinguishable from a
    grip, so the row retires STUCK and `tasks.cap` hands the position back as the
    object's radius -- a measurement of nothing at all. It must run out its
    budget instead, which is what `Result.ok` reports as a failed probe.

    `wants` is STUCK, not GOAL: with GOAL the priority chain in `step_once`
    would score this row correct for the wrong reason. Only the stuck rule is
    armed, so the `~at_goal` term is the single thing under test -- delete it in
    `motions.py` and this goes red.

    The joint is placed ON its goal and held there, rather than driven to it: the
    bug is about what a *stationary, arrived* joint reports, and travelling there
    first would only add ticks.
    """
    budget = 3 * STUCK_WINDOW_STEPS
    m = Motions(n_envs=1, n_joints=1, n_steps=1, hz=50.0)
    m.acts[:] = True
    m.goal_mm[:] = 10.0
    m.timeout_steps[:] = budget
    m.wants[:] = STUCK

    at_goal = torch.full((1, 1), 10.0)
    m.start(at_goal, 50.0)
    for _ in range(budget + 1):          # +1: the clock is aged before it is read
        if m.done():
            break
        m.step_once(at_goal)             # never moves: zero speed, on the goal

    assert m.done(), "the row never retired at all"
    assert int(m.outcome[0, 0, 0]) == TIMEOUT, (
        f"a jaw that closed on air reported {int(m.outcome[0, 0, 0])} "
        f"(STUCK is {STUCK}) -- that position would be read back as a radius")
    assert not m.succeeded()[0, 0, 0], "and it must not count as a success"


def test_a_one_tick_timeout_expires_on_the_first_tick():
    """Clocks are aged before they are tested, or every budget is a tick long."""
    m = Motions(n_envs=1, n_joints=1, n_steps=1, hz=50.0)
    m.acts[:] = True
    m.goal_mm[:] = 99.0
    m.timeout_steps[:] = 1
    m.wants[:] = GOAL

    zero = torch.zeros(1, 1)
    m.start(zero, 50.0)
    m.step_once(zero)
    assert m.done(), "a 1-tick budget must be spent after 1 tick"
    assert int(m.outcome[0, 0, 0]) == TIMEOUT
    assert not m.succeeded()[0, 0, 0], "a timeout is not the outcome that was wanted"


def test_an_unknown_stop_rule_is_refused_where_it_is_written():
    """A typo'd stop rule must not silently become a row that only times out.

    Raised by `set` rather than by `build`, so the traceback points at the step
    that was mistyped instead of at the end of a program with K of them.
    """
    try:
        Program(1, 1, 50.0).step().set(0, 1.0, 50.0, "gaol")
        raise AssertionError("a bad stop rule was accepted")
    except ValueError as e:
        assert "gaol" in str(e), e


def test_a_relative_goal_is_measured_from_where_its_step_armed():
    """`frame="here"` freezes the origin when the step arms, and only then.

    Re-reading the position every tick would build an identical program and is
    the bug this pins: the goal would track the joint chasing it, the gap would
    never close, and `drive` would run to `max_ticks` instead of arriving. That
    is also why the assertion is on where the joint ENDED -- `goal_mm` still
    holds the offset, so reading the program back proves nothing.

    Zeroing is two of these back to back (seek the stop, then back off half a
    rail from it), so the second step measuring from where the FIRST one ended
    is the shape the task needs, not chaining for its own sake.
    """
    def once(program):
        """`TaskRunner` drives a task, not a program. This is the shortest one."""
        yield program

    from_ten = lambda p: drive(TaskRunner(once(p.build()), 50.0),
                               torch.tensor([[10.0]]))

    p = Program(1, 1, 50.0)
    p.step().set(0, 5.0, 50.0, "goal", 6.0, frame="here")
    assert abs(float(from_ten(p)) - 15.0) <= 1.0     # 10 + 5, not 5

    p = Program(1, 1, 50.0)
    p.step().set(0,  5.0, 50.0, "goal", 6.0, frame="here")
    p.step().set(0, -2.0, 50.0, "goal", 6.0, frame="here")
    assert abs(float(from_ten(p)) - 13.0) <= 1.0     # (10 + 5) - 2, not 10 - 2

    # An unknown frame raises where it was written, same as an unknown stop.
    try:
        Program(1, 1, 50.0).step().set(0, 1.0, 50.0, frame="hear")
        raise AssertionError("a bad frame was accepted")
    except ValueError as e:
        assert "hear" in str(e), e


def test_a_one_dimensional_value_is_refused_when_the_axis_is_ambiguous():
    """N == len(dofs) makes a bare vector mean two different things.

    Torch would align it on the trailing axis and write a per-env goal across
    joints instead, which is a program that runs and commands plausible wrong
    numbers -- the failure mode this whole module is arranged to avoid.
    """
    s = Program(2, 2, 50.0).step()
    try:
        s.set([0, 1], torch.tensor([3.0, 4.0]), 50.0)
        raise AssertionError("an ambiguous 1-D goal was accepted")
    except ValueError as e:
        assert "ambiguous" in str(e), e
    # Saying which axis it varies over is accepted, both ways round.
    s.set([0, 1], torch.tensor([[3.0, 4.0]]), 50.0)          # per joint
    assert s.goal_mm.tolist() == [[3.0, 4.0], [3.0, 4.0]]
    s.set([0, 1], torch.tensor([[3.0], [4.0]]), 50.0)        # per env
    assert s.goal_mm.tolist() == [[3.0, 3.0], [4.0, 4.0]]


def test_the_runner_hands_the_measurement_back_to_the_task():
    """The measurement bus: `yield` evaluates to where the program left the hand.

    This is the entire mechanism by which zeroing learns where a stop was and
    the cap task learns a radius. If `send` passed anything else -- the goal, a
    stale pose, the position at submit time -- both tasks would still run to
    completion and both would be built on a number that was never measured.
    """
    seen = []

    def task():
        seen.append((yield one_step(0, 7.0, 50.0, "goal", 10.0)))
        return "done"

    runner = TaskRunner(task(), hold_torque=50.0)
    end = drive(runner, torch.zeros(1, 1))
    assert runner.result == "done", runner.result
    assert len(seen) == 1, seen
    assert abs(float(seen[0]) - float(end)) < 1e-6, (seen, end)
    assert abs(float(seen[0]) - 7.0) <= 1.0, f"not the position it reached: {seen[0]}"


def test_a_finished_runner_stays_finished():
    """`tick` returning None must be terminal.

    The hardware loop reads None as "the task is over, hand the joints back to
    the sliders". A runner that resumed on a later tick would fight the goal
    source that had already taken over, on the same servos.
    """
    def task():
        yield one_step(0, 1.0, 50.0, "goal", 1.0)
        return 42

    runner = TaskRunner(task(), hold_torque=50.0)
    pos = drive(runner, torch.zeros(1, 1))
    assert runner.result == 42
    for _ in range(5):
        assert runner.tick(pos) is None, "a finished runner resumed"


def test_a_task_that_yields_nothing_finishes_without_a_tick():
    """An empty task must not deadlock the loop that is executing it."""
    def task():
        return "empty"
        yield                                  # noqa: unreachable, makes it a generator

    runner = TaskRunner(task(), hold_torque=50.0)
    assert runner.tick(torch.zeros(1, 1)) is None
    assert runner.result == "empty"


def test_a_grip_outlives_the_motion_that_made_it():
    """A retired joint keeps its standing order.

    The base jaw takes hold of the bottle in the first step of the stroke
    program and is never named again. If retiring dropped its goal to wherever
    it happened to be, or dropped its torque, it would let go at that instant
    and every later step would run against a bottle free to spin.
    """
    from cartesian_hand.config import BASE_JAW
    from cartesian_hand.tasks import cap

    cfg = cap.Config(num_revs=1.0)
    # With a radius supplied the probe is skipped, so the first program the task
    # yields is the stroke program. Taken through `build` rather than from a
    # helper: what this pins is the engine, and the task is the fixture.
    m = next(cap.build(HAND_2, torch.zeros(1, HAND_2.n_dof), cfg=cfg,
                       cap_radius=torch.tensor([6.0])))
    assert m.acts[0, BASE_JAW].nonzero().flatten().tolist() == [0], "acts more than once"

    pos = torch.zeros(1, HAND_2.n_dof)
    floor = torch.zeros(1, HAND_2.n_dof)
    floor[0, BASE_JAW] = floor[0, 4] = 6.0
    m.start(pos, 50.0)
    for _ in range(200_000):
        if m.done():
            break
        goal, _t = m.step_once(pos)
        pos = torch.maximum(pos + (goal - pos).clamp(-0.5, 0.5), floor)
    assert m.done(), "stroke program never finished"
    assert float(m.held_goal[0, BASE_JAW]) == 0.0, "the jaw stopped pressing"
    assert float(m.held_torque[0, BASE_JAW]) == cfg.squeeze_torque, "the grip went limp"


def test_batch_rows_never_interact():
    """N=2 must equal two runs at N=1: nothing in the tick couples env i to j.

    This is what makes a row mean "one env in sim" and "one physical hand on
    real" at the same time, and it is the admission rule for ever running two
    different theta on two hands at once.

    Compared on the engine's own state, not on where the toy hand ended up. The
    two are not the same check and only one of them is about the engine: a
    batch runs until its *slowest* env is done, so a 6mm cap sharing a batch
    with an 11mm one gets extra ticks to converge that it would not get alone,
    and its final position differs by exactly the driver's step size. That is
    the harness, not coupling -- `held_goal` and `outcome` are identical.
    """
    from cartesian_hand.config import AUX_JAW, BASE_JAW
    from cartesian_hand.tasks import cap

    cfg = cap.Config(num_revs=1.0)

    def run(radii):
        radii = torch.tensor(radii)
        n = len(radii)
        pos = torch.zeros(n, HAND_2.n_dof)
        floor = torch.zeros(n, HAND_2.n_dof)
        floor[:, BASE_JAW] = floor[:, AUX_JAW] = radii
        m = next(cap.build(HAND_2, pos, cfg=cfg, cap_radius=radii))
        m.start(pos, 50.0)
        for _ in range(200_000):
            if m.done():
                break
            goal, _t = m.step_once(pos)
            pos = torch.maximum(pos + (goal - pos).clamp(-0.5, 0.5), floor)
        assert m.done()
        return m

    radii = [6.0, 11.0]
    together = run(radii)
    for i, r in enumerate(radii):
        alone = run([r])
        for field in ("held_goal", "held_torque"):
            a, b = getattr(alone, field)[0], getattr(together, field)[i]
            assert torch.allclose(a, b, atol=1e-5), (field, i, a, b)
        # Outcomes too, over the rows this env actually ran: a batch is padded
        # to the widest program, so the extra columns are ones this env idled.
        k = alone.K
        assert torch.equal(alone.outcome[0, :, :k], together.outcome[i, :, :k]), i


def test_external_condition_and_runtime_feedback_are_batched():
    """Contact result from one row selects later rows independently per env."""
    from cartesian_hand.motions import EXTERNAL, When

    p = Program(2, 2, 10.0)
    p.step().set(0, -10.0, 50.0, "external", 0.2)
    p.step().set(1, 5.0, 60.0, "goal", 1.0,
                 when=When("stopped_ok", 0), frame="here")
    p.step().set(1, -3.0, 40.0, "goal", 1.0,
                 when=When("stopped_failed", 0), frame="here")
    m = p.build()
    pos = torch.zeros(2, 2)
    m.start(pos, 20.0)

    signal = torch.tensor([[True, False], [False, False]])
    m.step_once(pos, signal)
    m.step_once(pos)
    assert m.outcome[:, 0, 0].tolist() == [EXTERNAL, TIMEOUT]
    assert m.stopped_ok[:, 0].tolist() == [True, False]
    assert m.executed[:, 1, 1].tolist() == [True, False]

    pos[0, 1] = 5.0
    m.step_once(pos)
    m.step_once(pos)
    assert m.executed[:, 1, 2].tolist() == [False, True]
    assert m.held_goal[:, 1].tolist() == [5.0, -3.0]


def test_timeout_does_not_overwrite_last_trustworthy_stop():
    """Timeout invalidates latest result without fabricating contact position."""
    from cartesian_hand.motions import When

    p = Program(1, 1, 10.0)
    p.step().set(0, 7.0, 50.0, "goal", 1.0)
    p.step().set(0, 20.0, 50.0, "goal", 0.1,
                 when=When("stopped_ok", 0))
    m = p.build()
    pos = torch.tensor([[7.0]])
    m.start(pos, 20.0)
    m.step_once(pos)
    assert m.stopped_ok.item() and m.stopped_at.item() == 7.0
    m.step_once(pos)
    assert not m.stopped_ok.item()
    assert m.stopped_at.item() == 7.0


def test_external_shape_is_exact_not_broadcast():
    p = Program(2, 3, 10.0)
    p.step().set(0, 0.0, 1.0, "external", 1.0)
    m = p.build()
    pos = torch.zeros(2, 3)
    m.start(pos, 1.0)
    try:
        m.step_once(pos, torch.ones(2))
        raise AssertionError("[N] external signal was broadcast across joints")
    except ValueError as e:
        assert "[N, J]" in str(e)


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  ok    {t.__name__}")
        except BaseException as e:
            failed += 1
            print(f"  FAIL  {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

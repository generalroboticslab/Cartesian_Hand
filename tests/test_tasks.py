"""Self-check for the two tasks, on both backends.

    python tests/test_tasks.py

The claim under test is the one the whole refactor exists to make: a task is a
generator of motion programs, the hand is an executor, and the same task file
runs on hardware and in mujoco with nothing swapped. So zeroing is driven three
ways here -- through `studio.live` against a mock bus, through `sim.run`
against mujoco, and against a toy hand with neither -- and it has to land in
the same place every time.

`CARTESIAN_HAND_CALIB` is redirected before `config` is imported. That is not
tidiness: `config.CALIB_PATH` is read from the environment at import time, and
a test that wrote the default would overwrite the calibration of whatever hand
is plugged into this machine -- a real one, that costs a bench cycle to
recover.
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["CARTESIAN_HAND_CALIB"] = os.path.join(
    tempfile.mkdtemp(prefix="cartesian_hand_test_"), "zero_offsets.json")

import contextlib
import io

import torch

from cartesian_hand.config import (AUX_LEFT, AUX_RIGHT, BASE_LEFT, BASE_RIGHT,
                                   CALIB_PATH, HAND_1, HAND_2, ORIENTATION,
                                   load_offsets, save_offsets)
from cartesian_hand.motions import STUCK, TaskRunner
from cartesian_hand.servo import MockServo
from cartesian_hand.tasks import zero

assert "cartesian_hand_test_" in CALIB_PATH, \
    f"tests would write the real calibration at {CALIB_PATH}"


def toy_run(task, cfg, start=None, floor=None, step_mm=0.5, max_ticks=200_000,
            programs=None):
    """Drive a task against a hand made of two lines of arithmetic.

    Joints move toward their goal and refuse to pass `floor`, which stands in
    for the hard stops. Neither a bus nor a simulator, which is the point: if a
    task needs either, it is not portable and this call is where that shows.

    `programs`, if given, collects each program the task issued. The runner
    consumes and discards them, so this is the only way to assert on what a task
    *commanded* rather than on where the joints ended up.
    """
    pos = torch.zeros(1, cfg.n_dof) if start is None else start
    runner = TaskRunner(task, hold_torque=50.0)
    for _ in range(max_ticks):
        step = runner.tick(pos)
        if step is None:
            return runner, pos
        if programs is not None and runner.program is not (
                programs[-1] if programs else None):
            programs.append(runner.program)
        pos = pos + (step[0] - pos).clamp(-step_mm, step_mm)
        if floor is not None:
            pos = torch.maximum(pos, floor)
    raise AssertionError("task never finished")


def test_zeroing_through_the_executor_lands_on_the_hard_stops():
    """The whole hardware path: submit, tick, measure, convert, save.

    Driven through `studio.live` rather than by calling the task directly,
    because everything between the task and the servo is what this is for --
    the millimetre frame, `mm_to_counts` and its orientation term, and the
    handover back to the sliders. `MockServo`'s own rails are the hard stops.

    The expected sign per DOF is the check that matters. A seek commands
    `here - overtravel` in *millimetres*, and `counts_to_mm` applies
    `orientation` on the way in, so the raw counts a DOF stalls at land on
    opposite rails depending on its orientation. Getting that backwards is the
    bug that recorded a stop at the far end of the rail from the real one, and
    it is invisible in millimetres -- which is exactly why it is asserted here
    in counts.
    """
    from cartesian_hand import studio

    if os.path.exists(CALIB_PATH):
        os.unlink(CALIB_PATH)
    with contextlib.redirect_stdout(io.StringIO()):
        studio.live(hand="hand_2", mock=True, viewer=False, studio=False,
                    task="zero", seconds=120)

    saved = load_offsets("hand_2", HAND_2.n_dof)
    assert saved is not None, "the executor did not save a calibration"
    for dof, orientation in enumerate(ORIENTATION):
        # mm decrease toward the stop on every DOF, so the count it stalls at is
        # the rail on the side `orientation` puts the negative direction.
        want = MockServo.STOP_HIGH if orientation < 0 else MockServo.STOP_LOW
        assert saved[dof] == want, (
            f"DOF {dof} (orientation {orientation:+d}) stopped at "
            f"{saved[dof]}, expected the {want} rail")

    # And the offsets mean what they claim: at the stop, the hand reads 0.0 mm.
    off = torch.tensor(saved, dtype=torch.float32)
    mm = HAND_2.counts_to_mm(off, off)
    assert torch.allclose(mm, torch.zeros_like(mm), atol=1e-3), mm.tolist()


def test_a_dof_that_never_stalls_reports_not_ok_and_saves_nothing():
    """A timed-out DOF must not be recorded where it happened to stop.

    This is the load-bearing branch of the whole procedure. A DOF that ran out
    of budget also stopped moving, so position alone cannot tell it from one
    that found its stop -- and recording it puts the origin somewhere mid
    travel, which makes every later millimetre on that axis wrong by however
    far it fell short, in the direction of the open end of the rail. Silent,
    and it walks a carriage off a slider.

    Reported as `ok=False` rather than raised, so this asserts both halves: the
    flag is down, and the hardware path *obeys* the flag. The disk assertion is
    the one that matters -- an `ok` nobody reads is not a check, and `finish` is
    where the only irreversible step in `studio.live` lives.

    The toy hand has no floor here, so nothing ever stalls.
    """
    from cartesian_hand import studio

    sentinel = {"hand_2": {"offsets": [1] * HAND_2.n_dof, "timestamp": "sentinel"}}
    os.makedirs(os.path.dirname(CALIB_PATH), exist_ok=True)
    with open(CALIB_PATH, "w") as f:
        json.dump(sentinel, f)

    runner, pos = toy_run(zero.build(HAND_2, torch.zeros(1, HAND_2.n_dof)),
                          HAND_2, step_mm=5.0, max_ticks=20_000)
    assert not bool(runner.result.ok.all()), \
        "a DOF that never stalled was accepted as zeroed"
    assert "never reached a hard stop" in runner.result.why, runner.result.why

    # It stopped at the phase that failed instead of running the other two. At
    # N=1 one dead env is every env, and a hand that goes on to drive its jaws
    # and z stage for another 40s after the fingers failed is 40s the operator
    # spends watching a run whose result is already void. Phase 1 is the
    # fingers, so the jaws and z must still be where they started.
    from cartesian_hand.config import AUX_JAW, BASE_JAW, Z
    untouched = pos[0, [BASE_JAW, AUX_JAW, Z]]
    assert torch.allclose(untouched, torch.zeros(3), atol=1e-3), untouched.tolist()

    datum = torch.zeros(HAND_2.n_dof)
    with contextlib.redirect_stdout(io.StringIO()):
        kept, ok = studio.finish("zero", runner, HAND_2, datum,
                                 torch.zeros(HAND_2.n_dof), None)
    assert not ok and kept is datum, "a failed run changed the datum"
    with open(CALIB_PATH) as f:
        assert json.load(f) == sentinel, "a failed run clobbered the calibration"
    os.unlink(CALIB_PATH)


def test_one_env_failing_does_not_take_the_batch_with_it():
    """Envs fail independently, which is the whole reason `ok` is [N].

    Two envs, one with hard stops and one without. The old code reduced the
    outcome with a bare `.all()` and raised, so a single env that ran out of
    budget discarded every good measurement in the batch -- and raising from
    inside the generator meant they could not be recovered from the exception
    either. At N=4096 under domain randomisation some envs are *supposed* to
    fail; that is data, not an error.

    The second assertion is the one that costs something to get right: the
    survivor must park at mid travel while the failure holds its place, because
    `stop + travel/2` measured from a "stop" that is really mid rail aims past
    the open end of the rail. That bound is what lets a failure be reported
    instead of raised -- see the park in `tasks.zero.build`.
    """
    stops = torch.tensor([[-4.0, -1.0, -7.0, -2.5, -3.0, -6.0, -5.0]])
    floor = torch.cat([stops, torch.full_like(stops, -1e4)])   # env 1: unreachable
    start = torch.zeros(2, HAND_2.n_dof)
    runner, pos = toy_run(zero.build(HAND_2, start), HAND_2, start=start,
                          floor=floor, step_mm=5.0, max_ticks=20_000)
    value, ok, why = runner.result
    assert ok.tolist() == [True, False], ok.tolist()
    assert torch.allclose(value[0], stops[0], atol=1.1), value[0].tolist()
    assert "envs [1]" in why, why

    # The survivor parked at mid travel, measured from its own stop.
    want = stops[0] + torch.tensor([t / 2 for t in HAND_2.travel_mm])
    assert torch.allclose(pos[0], want, atol=1.1), (pos[0].tolist(), want.tolist())

    # The failure parked in PLACE, and this is the assertion the whole change
    # rests on. Its "stop" is wherever the seek ran out of budget -- here the
    # overtravel goal, since nothing stopped it -- and `that + travel/2` aims
    # travel/2 past the open end of the rail, which on hardware is where a
    # carriage leaves its slider. Parking in place is what makes reporting a
    # failure safe instead of having to raise on one.
    expired_at = torch.full_like(pos[1], -zero.Config().overtravel_mm)
    assert torch.allclose(pos[1], expired_at, atol=1.1), pos[1].tolist()


def test_the_park_is_measured_from_each_dof_s_own_stop():
    """Mid travel means half of *this* joint's stroke from *this* joint's stop.

    Two ways to get it wrong, and both have been shipped. Hoisting the datum
    out -- parking against 0 instead of against the stop just found -- commands
    absolute millimetres near the frame's origin, which before a calibration is
    an arbitrary place on the rail and on hardware is where a carriage leaves
    its slider. Hoisting the *travel* out and using one DOF's `max_mm` for the
    whole phase leaves the jaws 2.5 mm off, which looks like slop.

    Stops are staggered per DOF so a hoisted datum cannot pass by coincidence,
    and the travel table is deliberately NOT the shipped one. Today every DOF
    inside a phase happens to share a travel -- fingers 55, jaws 50 -- so the
    hoisted-travel bug is invisible against `STANDARD_TRAVEL` and was in fact
    shipped once without a test noticing. Pinning it against a table where a
    phase's DOFs differ makes the invariant hold on its own terms rather than
    on a coincidence the next edit to the table could remove.
    """
    mixed = HAND_2.variant(
        travel_mm=[50.0, 55.0, 44.0, 50.0, 38.0, 55.0, 30.0])
    stops = torch.tensor([[-4.0, -1.0, -7.0, -2.5, -3.0, -6.0, -5.0]])
    _runner, pos = toy_run(zero.build(mixed, torch.zeros(1, mixed.n_dof)),
                           mixed, floor=stops)
    want = stops + torch.tensor([[t / 2 for t in mixed.travel_mm]])
    # 1.1 mm: a "goal" move retires inside the engine's own position tolerance.
    assert torch.allclose(pos, want, atol=1.1), (pos.tolist(), want.tolist())


def test_each_hand_seeks_at_its_own_torque_min_to_move():
    """Friction is per unit, so the seek torque has to be. A module constant
    inlined into the seek would tune both hands at once: raising it for a stiff
    gear train would push the other hand's fingers into their stops hard enough
    to deflect the rack, and the recorded stop would move with it.

    Read off the programs the task actually issued rather than off a helper's
    return value: config -> `gain_vector` -> `Move` -> program is the whole path
    the number can travel, and asserting on the far end of it tests the routing
    rather than a table. The expected values are derived from the config for the
    same reason -- these are bench numbers and get retuned, so pinning literals
    here would break this test every time a hand is measured. The second case is
    what proves the routing is real: a hand whose table says 222 seeks at 222.
    """
    def creep(cfg):
        """{dof: torque} over every DOF whose seek retires on contact."""
        issued = []
        toy_run(zero.build(cfg, torch.zeros(1, cfg.n_dof)), cfg,
                floor=-torch.ones(1, cfg.n_dof), programs=issued)
        return {d: int(p.torque[0, d, 0]) for p in issued
                for d in range(cfg.n_dof) if int(p.wants[0, d, 0]) == STUCK}

    want = HAND_2.gain_vector("torque_min_to_move").tolist()
    assert creep(HAND_2) == {d: want[d] for d in range(HAND_2.n_dof)}
    # And the other hand is untouched by that table.
    assert creep(HAND_1.variant(torque_min_to_move=[222] * 7)) == {d: 222 for d in range(7)}


def test_the_park_can_actually_move_every_dof_it_commands():
    """A park goal below the DOF's own `torque_min_to_move` moves nothing.

    Both halves of this shipped broken and neither backend could see it. The z
    stage is the only DOF under load, and its floor is 350 on hand_2 and 800 on
    hand_1 against a flat `park_torque` of 300 -- so phase 3 commanded mid travel
    at a torque that cannot lift the stage, and the run ended with it sitting on
    its bottom stop. mujoco drops torque entirely and `MockServo` ignores it, so
    the only place this is visible is the program the task issued.

    The budget half is the same bug in time: at the configured 300 counts/s the
    hand moves 3.68 mm/s, so half a finger rail is 7.5 s against a 6 s park
    budget. Every park expired mid-move, and nothing noticed because only the
    seek's outcome is ever read. Asserted against the distance and speed rather
    than a literal, since both are bench numbers that get retuned.
    """
    from cartesian_hand.config import Z

    for cfg in (HAND_1, HAND_2):
        issued = []
        start = torch.zeros(1, cfg.n_dof)
        toy_run(zero.build(cfg, start), cfg, start=start,
                floor=torch.zeros(1, cfg.n_dof), step_mm=5.0, programs=issued)
        floor = cfg.gain_vector("torque_min_to_move").tolist()
        # Per DOF, not the batch minimum: z runs at its own speed so it can
        # break its static friction, and pricing its budget at the fingers'
        # speed would assert against a move nobody commanded.
        speeds = cfg.gain_vector("speed").tolist()

        # Odd programs are the parks; each retires on arrival, not on contact.
        for phase, park in enumerate(issued[1::2]):
            for d in range(cfg.n_dof):
                if not bool(park.acts[0, d, 0]):
                    continue
                assert int(park.torque[0, d, 0]) >= floor[d], (
                    f"[{cfg.name}] phase {phase} parks DOF {d} at "
                    f"{int(park.torque[0, d, 0])}, below its {floor[d]} floor")
                budget = float(park.timeout_steps[0, d, 0]) / cfg.control_hz
                needs = (cfg.travel_mm[d] / 2) / (speeds[d] / cfg.counts_per_mm)
                assert budget > needs, (
                    f"[{cfg.name}] phase {phase} gives DOF {d} {budget:.1f}s to "
                    f"travel {cfg.travel_mm[d] / 2:.1f}mm, which takes {needs:.1f}s")
        assert issued, "the task issued no programs"
        assert bool(issued[5].acts[0, Z, 0]), "phase 3's park is not the z stage"


def test_primitives_bind_their_pairings_at_build_time():
    """`twist`, `tilt`, `rotate_in_place` carry the id/goal pairing the task
    would otherwise get wrong silently on a symmetric gripper.

    The fixtures are pairs the helpers must reject or accept by id, with the
    goal the engine actually wrote -- not the literal the caller passed in.
    """
    from cartesian_hand import primitives
    from cartesian_hand.motions import Program

    span = 11.0
    J = HAND_2.n_dof
    step = lambda: Program(1, J, 50.0).step()
    goals = lambda s, ids: s.goal_mm[0, list(ids)].tolist()

    # twist: ids in argument order; first id goes to 0, second to span.
    s = primitives.twist(step(), AUX_LEFT, AUX_RIGHT, span, 80.0)
    assert goals(s, [AUX_LEFT, AUX_RIGHT]) == [0.0, span]
    # Only the named joints act; a joint left out must not be commanded to 0.0,
    # which on this hand is a full-stroke move into a hard stop.
    assert s.acts[0].tolist().count(True) == 2

    # tilt: one stage's pair advances by `span`, the other holds still.
    # Pairs by column: left on stage 0 / left on stage 1, right likewise.
    s = primitives.tilt(step(), AUX_LEFT, BASE_LEFT, AUX_RIGHT, BASE_RIGHT,
                        span, 80.0)
    assert goals(s, [AUX_LEFT, BASE_LEFT, AUX_RIGHT, BASE_RIGHT]) == \
        [span, span, 0.0, 0.0]
    # Crossed columns would invert the grip on the runtime; the helper refuses
    # the symmetric-id trap instead of silently building it.
    try:
        primitives.tilt(step(), AUX_LEFT, AUX_LEFT, BASE_RIGHT, BASE_RIGHT,
                        span, 80.0)
        raise AssertionError("a same-id pair was accepted")
    except ValueError:
        pass

    # rotate_in_place: opposing twists on both stages, no translation on
    # average. The pairing is positional in LAYOUT order so the runtime has
    # no other clue which joint opposes which.
    s = primitives.rotate_in_place(step(), span, 80.0)
    assert goals(s, [BASE_LEFT, BASE_RIGHT, AUX_LEFT, AUX_RIGHT]) == \
        [span, 0.0, 0.0, span]

    # Only pairings live in `primitives`. A helper that took an opaque
    # `dof_ids` and treated every element alike would be `Step.set` under a
    # second name -- `move_until_stuck` was exactly that and is now the one
    # `set(ids, -overtravel, floor, "stuck", t, frame="here")` in `zero.py`.
    assert not hasattr(primitives, "move_until_stuck")

    # `torque` and `when` are ordinary arguments, so they reach the program.
    # When they were fields of a returned tuple that `Step.set` re-unpacked,
    # a caller's overrides were parsed and then silently discarded.
    s = primitives.twist(step(), AUX_LEFT, AUX_RIGHT, span, 99.0, when=[False])
    assert not s.acts[0].any(), "when=False must idle every named joint"
    assert s.torque[0, AUX_LEFT] == 99.0


def test_scissors_task_seats_fingers_then_lifts_on_z():
    """The two-handle primitive: seat, grip both handles, lift `z`.

    Drove end-to-end against a toy hand rather than asserting on the program's
    raw tensors: the task is the fixture, and the right check is that the program
    issues the moves in order at all. The toy hand stops everything on the
    floor, so the lift at `z` runs to its goal and the program finishes.
    """
    from cartesian_hand.tasks import scissors

    issued = []
    toy_run(scissors.build(HAND_2, torch.zeros(1, HAND_2.n_dof)),
            HAND_2, floor=-50.0 * torch.ones(1, HAND_2.n_dof), programs=issued)
    assert issued, "scissors task yielded no program"
    prog = issued[-1]
    K = prog.goal_mm.shape[2]
    # The lift lives at z (DOF 3) and only there, on its own step, after both
    # jaws have closed. A program that drops or reorders the lift is a tool
    # primitive that doesn't work.
    lift_steps = [k for k in range(K) if bool(prog.acts[0, 3, k])]
    assert len(lift_steps) == 1, f"expected one lift step on z, got steps {lift_steps}"
    assert prog.goal_mm[0, 3, lift_steps[0]].item() == scissors.Config.open_mm, (
        f"lift was {prog.goal_mm[0, 3, lift_steps[0]].item()}, expected "
        f"{scissors.Config.open_mm}")


def test_tilt_task_grips_on_contact_then_shears_one_stage():
    """Grasp then pitch: the grip is contact-based, the shear is by stage.

    Driven against the toy hand for the same reason `scissors` is -- the task is
    the fixture. Two things can silently go wrong here and neither shows in a
    run that merely completes: a grip commanded with `stop="goal"` would report
    success against thin air, and a shear paired by COLUMN (both left fingers)
    would translate the object instead of tilting it while writing exactly the
    same four goals. Both are asserted on the program, not on the pose.
    """
    from cartesian_hand.config import AUX_JAW, BASE_JAW
    from cartesian_hand.tasks import tilt

    issued = []
    toy_run(tilt.build(HAND_2, torch.zeros(1, HAND_2.n_dof)),
            HAND_2, floor=-50.0 * torch.ones(1, HAND_2.n_dof), programs=issued)
    assert len(issued) == 1, f"tilt should yield one program, got {len(issued)}"
    prog = issued[0]

    # The grip is the only contact-based row, and it is on both jaws.
    for jaw in (BASE_JAW, AUX_JAW):
        assert prog.wants[0, jaw, tilt.GRIP_STEP] == STUCK, (
            f"DOF {jaw} at step {tilt.GRIP_STEP} wants "
            f"{prog.wants[0, jaw, tilt.GRIP_STEP]}, expected STUCK")

    # The shear: base stage's two fingers advance together, aux stage's two
    # hold. Crossing this into columns is the failure the pairing exists to stop.
    shear = prog.goal_mm.shape[2] - 1
    goals = prog.goal_mm[0, [BASE_LEFT, BASE_RIGHT, AUX_LEFT, AUX_RIGHT], shear]
    span = HAND_2.clamped_mm(BASE_LEFT, tilt.Config.tilt_mm)
    assert goals.tolist() == [span, span, 0.0, 0.0], goals.tolist()


def test_the_same_zeroing_task_runs_in_simulation():
    """The portability claim, executed rather than asserted.

    Same generator, same `TaskRunner`, a different backend underneath. In sim
    the hard stop is the joint's lower limit, so every DOF should report a stop
    at 0 mm -- which is also what makes this a real check and not a tautology:
    a task that had smuggled in a hardware assumption would not converge here.
    """
    from cartesian_hand import sim

    start = torch.zeros(1, HAND_2.n_dof)
    with contextlib.redirect_stdout(io.StringIO()):
        stops = sim.run(zero.build(HAND_2, start), HAND_2, max_seconds=120.0)
    assert stops.shape == (1, HAND_2.n_dof), stops.shape
    # The z stage sags a little past its limit under gravity; the rest sit on it.
    assert torch.allclose(stops, torch.zeros_like(stops), atol=1.0), stops.tolist()


def test_the_cap_task_measures_the_cap_it_is_holding():
    """The probe's number is the radius, and the outcome is what proves it.

    A jaw closing on nothing also stalls -- against its own limit -- so the
    position is not evidence on its own. `cap.build` checks the probe retired on
    contact before it believes the measurement.
    """
    from cartesian_hand.config import AUX_JAW, BASE_JAW
    from cartesian_hand.tasks import cap as cap_task

    cap = 6.0
    start = torch.zeros(1, HAND_2.n_dof)
    floor = torch.zeros(1, HAND_2.n_dof)
    floor[0, BASE_JAW] = floor[0, AUX_JAW] = cap
    runner, _pos = toy_run(
        cap_task.build(HAND_2, start, cfg=cap_task.Config(num_revs=1.0)),
        HAND_2, floor=floor)
    assert abs(float(runner.result.value) - cap) < 1.0, runner.result


def test_the_cap_lift_is_commanded_hard_enough_and_long_enough_to_happen():
    """z is the only loaded axis, and the extraction lift is the only row that
    raises it. Both halves of that row are invisible in every automated check
    available here: mujoco drops torque on the floor, `MockServo` ignores it,
    and a budget that expires still ends the program tidily. So the failure --
    a run that reports success with the cap still screwed onto the bottle -- is
    only observable on the bench, which is where it was found.

    Torque: `travel_torque` is a horizontal-DOF number (hand_2's bisect puts the
    unloaded lifting cliff between 150 and 200), so the shipped 50 lifts
    nothing. Asserted against `torque_min_to_move`, the per-hand measured floor,
    rather than a literal -- a hand whose floor is raised in `config` must raise
    this row with it.

    Budget: asserted across `lift_mm`'s whole declared tuning range, because a
    flat `move_timeout_s` covers the bottom of that range and nothing else, and
    dragging the slider is how anyone would find out.
    """
    from cartesian_hand.config import Z
    from cartesian_hand.tasks import cap as cap_task

    lo, hi = tasks_tunable_range("cap", "lift_mm")
    for lift_mm in (lo, cap_task.Config().lift_mm, hi):
        gen = cap_task.build(HAND_2, torch.zeros(1, HAND_2.n_dof),
                             cfg=cap_task.Config(lift_mm=lift_mm, num_revs=1.0),
                             cap_radius=6.0)
        programs = [next(gen)]
        while True:
            try:
                programs.append(gen.send(torch.zeros(1, HAND_2.n_dof)))
            except StopIteration:
                break
        final = programs[-1]

        rows = [k for k in range(final.goal_mm.shape[2]) if final.acts[0, Z, k]]
        assert len(rows) == 1, f"expected one z row in the extraction, got {rows}"
        k = rows[0]

        want = float(HAND_2.gain_vector("torque_min_to_move")[Z])
        got = float(final.torque[0, Z, k])
        assert got >= want, (
            f"lift_mm={lift_mm}: z commanded at {got}, below its {want} floor "
            f"-- the row runs and the stage does not move")

        goal = float(final.goal_mm[0, Z, k])
        mm_per_s = HAND_2.gain_vector("speed")[Z].item() / HAND_2.counts_per_mm
        budget_s = int(final.timeout_steps[0, Z, k]) / HAND_2.control_hz
        assert budget_s >= goal / mm_per_s, (
            f"lift_mm={lift_mm}: {budget_s:.1f}s budget for a {goal:.1f}mm lift "
            f"at {mm_per_s:.1f}mm/s -- expires part-way up")


def tasks_tunable_range(task, field):
    from cartesian_hand import tasks
    _value, lo, hi = tasks.tunables(task)[field]
    return lo, hi


def test_stops_round_trip_back_to_absolute_counts():
    """The one hardware-only step: relative millimetres to storable counts.

    `zero` reports where the stops were in whatever frame it was given. Turning
    that into the datum means adding the frame back, and if the conversion lost
    the orientation term the new datum would be the reflection of the real one
    -- a hand that reads negative millimetres everywhere and clamps at the
    wrong end.
    """
    datum = torch.tensor([[100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0]])
    stops_mm = torch.tensor([[-10.0, -12.0, -8.0, -5.0, -11.0, -9.0, -7.0]])
    counts = HAND_2.mm_to_counts(stops_mm, datum)
    # Read the same counts back against the new datum: the stops are now 0.0.
    back = HAND_2.counts_to_mm(counts.float(), counts.float())
    assert torch.allclose(back, torch.zeros_like(back), atol=1e-3), back
    # And against the OLD datum they are still where they were measured.
    old = HAND_2.counts_to_mm(counts.float(), datum)
    assert torch.allclose(old, stops_mm, atol=0.02), (old, stops_mm)


def test_save_load_round_trip_and_a_second_hand_is_preserved():
    """One file, many hands. A wrong-length entry reads as missing, not as data."""
    path = tempfile.mktemp(suffix=".json")
    try:
        save_offsets("hand_1", [100] * HAND_1.n_dof, path=path)
        save_offsets("hand_2", [200, 250, 300, 350, 400, 450, 500], path=path)
        assert load_offsets("hand_1", HAND_1.n_dof, path=path) == [100] * HAND_1.n_dof
        assert load_offsets("hand_2", HAND_2.n_dof, path=path) == \
            [200, 250, 300, 350, 400, 450, 500]
        save_offsets("hand_2", [1, 2, 3], path=path)
        assert load_offsets("hand_2", HAND_2.n_dof, path=path) is None
        assert load_offsets("hand_1", HAND_1.n_dof, path=path) == [100] * HAND_1.n_dof
        os.unlink(path)
        assert load_offsets("hand_1", HAND_1.n_dof, path=path) is None
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_a_corrupt_calibration_is_renamed_not_lost():
    """A bad file must not cost a fresh calibration run its result."""
    path = tempfile.mktemp(suffix=".json")
    try:
        with open(path, "w") as f:
            f.write("{not json")
        save_offsets("hand_1", [10] * HAND_1.n_dof, path=path)
        assert os.path.exists(path + ".bad")
        assert load_offsets("hand_1", HAND_1.n_dof, path=path) == [10] * HAND_1.n_dof
    finally:
        for p in (path, path + ".bad"):
            if os.path.exists(p):
                os.unlink(p)


def test_the_task_name_is_the_file_name():
    """`--task zero` runs `tasks/zero.py`, and nothing maps one to the other.

    The name is the import, so knowing the flag is knowing which file to open.
    A dict from name to module would be a table edited on one side only.
    """
    from cartesian_hand import tasks

    assert tasks.names() == ["cap", "scissors", "tilt", "zero"], tasks.names()
    for name in tasks.names():
        assert tasks.module(name).__name__ == f"cartesian_hand.tasks.{name}"
        assert callable(tasks.module(name).build), f"{name}.py has no build()"


def test_a_variant_is_a_file_that_reuses_the_parts_it_names():
    """Dropping a file into `tasks/` makes a task, with nothing else edited.

    Written to a real file rather than defined inline, because that is the
    whole mechanism -- there is no registration step to fake, and a variant
    that cannot be written as a file is not a variant.

    `Config(...)` is the configuration: the dataclass constructor leaves every
    knob the variant does not name at its default, which is what lets the
    docstring honestly claim "one change from cap.py".

    It reaches that class as `cap.Config`, never `from .cap import Config`, and
    gets no button as a result -- `tasks.config` reads `Config` off the variant's
    own namespace, so importing the name would hand back cap's label and put a
    second "Open cap" on the panel. A variant is reachable by `--task`, and every
    variant sprouting a button would turn the panel into a wall.
    """
    from cartesian_hand import tasks
    from cartesian_hand.tasks.cap import AUX_JAW, BASE_JAW, Config

    path = os.path.join(os.path.dirname(tasks.__file__), "cap_test_variant.py")
    with open(path, "w") as f:
        f.write('"""Test-only variant: one turn instead of three."""\n'
                'from . import cap\n\n\n'
                'def build(hand, start_mm, **kwargs):\n'
                '    return cap.build(hand, start_mm,\n'
                '                     cfg=cap.Config(num_revs=1.0), **kwargs)\n')
    try:
        assert "cap_test_variant" in tasks.names()
        assert dict(tasks.buttons()) == {"zero": "Zero hand", "cap": "Open cap",
                                         "tilt": "Tilt object"}
        assert not tasks.sets_datum("cap_test_variant")
        assert tasks.sets_datum("zero"), "zeroing must own the datum"

        cap = 6.0
        floor = torch.zeros(1, HAND_2.n_dof)
        floor[0, BASE_JAW] = floor[0, AUX_JAW] = cap
        runner, _pos = toy_run(
            tasks.make("cap_test_variant", HAND_2, torch.zeros(1, HAND_2.n_dof)),
            HAND_2, floor=floor)
        assert abs(float(runner.result.value) - cap) < 1.0, runner.result

        # It changed the knob it named and nothing else.
        assert Config(num_revs=1.0).squeeze_torque == Config().squeeze_torque
    finally:
        os.unlink(path)


def test_every_tunable_bound_brackets_the_number_it_ships_with():
    """A declared range has to contain the number the task actually ships.

    Three ways this goes wrong, all silent:

    Bounds that exclude the default mean the panel opens with its slider already
    somewhere the task was never run at, and the shipped configuration -- the one
    that is known to work -- is the one setting nobody can drag back to.

    A field that changes `K` cannot be tuned at all: two configurations whose
    programs are different lengths are not comparable. `cap.max_cap_radius` is
    the only such field, and it is pinned here because nothing about
    `field(metadata=...)` stops the next edit from marking it.

    A non-numeric field would be handed to a slider that would do arithmetic on
    it. `label` and `sets_datum` are the fields that would break, so they are the
    ones asserted absent.
    """
    from cartesian_hand import tasks

    for name in tasks.names():
        knobs = tasks.tunables(name)
        for field_name, (value, lo, hi) in knobs.items():
            assert lo < hi, f"{name}.{field_name}: empty range [{lo}, {hi}]"
            assert lo <= value <= hi, (
                f"{name}.{field_name} ships {value}, outside its own "
                f"[{lo}, {hi}] -- nothing can set it back to the default")
            assert isinstance(value, (int, float)) and not isinstance(value, bool), \
                f"{name}.{field_name} is {type(value).__name__}, not a number"
        assert not {"label", "sets_datum"} & set(knobs)

    assert tasks.tunables("zero"), "zero exposes nothing to tune"
    assert "max_cap_radius" not in tasks.tunables("cap"), \
        "max_cap_radius sets cap's program length; tuning it compares programs " \
        "of different K"


def test_an_unknown_task_name_lists_the_registry():
    """A typo at the CLI must say what the options were, not raise a bare key."""
    from cartesian_hand import tasks

    try:
        tasks.module("zeroing")       # the old module name
    except KeyError as e:
        assert "zero" in str(e) and "cap" in str(e), e
    else:
        raise AssertionError("an unknown task name was accepted")


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

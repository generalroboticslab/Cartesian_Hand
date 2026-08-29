"""Self-check for the control stack, run against the mock backend.

    python test_cartesian_hand.py

Covers the logic that has no business failing on hardware: unit conversion,
the normalized action contract, per-DOF gain isolation, slew limiting, and
rollout round-tripping. It does not and cannot verify servo behaviour.
"""

import os
import time
import tempfile

import numpy as np

from cartesian_hand.hand import connect
from cartesian_hand.hands import Dof, Geometry, HandConfig, Motion, get_hand
from cartesian_hand.policy import (
    ContractMismatch,
    FunctionPolicy,
    HoldPolicy,
    ReplayPolicy,
    Rollout,
    run_policy,
)
from cartesian_hand.tasks import registry


def test_config_validation():
    ok = 0
    for bad in (dict(servo_id=0, axis="w", orientation=1),
                dict(servo_id=0, axis="x", orientation=2),
                dict(servo_id=0, axis="x", orientation=1, min_mm=10, max_mm=5)):
        try:
            Dof(**bad)
        except ValueError:
            ok += 1
    assert ok == 3, "Dof accepted an invalid definition"

    try:
        HandConfig(name="dup", port="/dev/null",
                   dofs=[Dof(1, "x", 1), Dof(1, "y", 1)])
        raise AssertionError("duplicate servo_id accepted")
    except ValueError:
        pass


def test_gain_vector():
    dofs = [Dof(i, "x", 1) for i in range(3)]

    scalar = HandConfig(name="s", port="/dev/null", dofs=dofs,
                        motion=Motion(torque=50))
    assert scalar.gain_vector("torque").tolist() == [50, 50, 50], "scalar did not broadcast"

    per_dof = HandConfig(name="v", port="/dev/null", dofs=dofs,
                         motion=Motion(torque=[50, 150, 50]))
    assert per_dof.gain_vector("torque").tolist() == [50, 150, 50], "vector not passed through"

    # The caller gets a copy: CartesianHand mutates this array via set_gains,
    # and the shared HAND_* configs are module-level singletons.
    per_dof.gain_vector("torque")[0] = 999
    assert per_dof.gain_vector("torque").tolist() == [50, 150, 50], "gain_vector aliases config"

    try:
        HandConfig(name="bad", port="/dev/null", dofs=dofs,
                   motion=Motion(torque=[50, 150]))
        raise AssertionError("wrong-length gain accepted")
    except ValueError:
        pass


def test_counts_roundtrip():
    cfg = get_hand("hand_1")
    for dof_id in range(cfg.n_dof):
        for mm in (0.0, 12.5, 60.0):
            counts = cfg.mm_to_counts(dof_id, mm, zero_offset=2048)
            back = cfg.counts_to_mm(dof_id, counts, zero_offset=2048)
            assert abs(back - mm) < 0.02, f"dof {dof_id} {mm} -> {back}"
    assert cfg.counts_to_mm(0, None, 0) is None


def test_geometry_calibration_override():
    derived = Geometry(gear_pitch_diameter_mm=16.0, counts_per_rev=4096)
    assert abs(derived.counts_per_mm - 4096 / (np.pi * 16)) < 1e-9
    # The calibration knob must win over the derived value.
    assert Geometry(counts_per_mm=80.0).counts_per_mm == 80.0


def test_normalization():
    cfg = get_hand("hand_1")
    assert np.allclose(cfg.normalize(cfg.lower), -1.0)
    assert np.allclose(cfg.normalize(cfg.upper), +1.0)
    assert np.allclose(cfg.denormalize(np.full(cfg.n_dof, -1.0)), cfg.lower)
    assert np.allclose(cfg.denormalize(np.full(cfg.n_dof, +1.0)), cfg.upper)
    mid = cfg.denormalize(np.zeros(cfg.n_dof))
    assert np.allclose(mid, (cfg.lower + cfg.upper) / 2)
    # Out-of-range actions clip to the travel limit rather than escaping it.
    assert np.allclose(cfg.denormalize(np.full(cfg.n_dof, 9.0)), cfg.upper)
    assert np.allclose(cfg.denormalize(np.full(cfg.n_dof, -9.0)), cfg.lower)


def test_fingerprint_tracks_contract_only():
    a = get_hand("hand_1")
    b = get_hand("hand_2")
    # Different ports and servo IDs, identical kinematics: a policy transfers.
    assert a.fingerprint() == b.fingerprint()
    assert a.port != b.port and a.servo_ids != b.servo_ids

    # Different travel: a policy must not transfer.
    narrow = HandConfig(name="narrow", port=a.port,
                        dofs=[Dof(d.servo_id, d.axis, d.orientation,
                                  d.min_mm, d.max_mm / 2) for d in a.dofs])
    assert narrow.fingerprint() != a.fingerprint()

    # Different control rate changes the meaning of a step.
    slow = HandConfig(name="slow", port=a.port, dofs=list(a.dofs),
                      motion=Motion(control_hz=10))
    assert slow.fingerprint() != a.fingerprint()


def test_config_roundtrip():
    a = get_hand("hand_1")
    b = HandConfig.from_dict(a.to_dict())
    assert b.fingerprint() == a.fingerprint()
    assert b.servo_ids == a.servo_ids
    assert b.port == a.port


def test_task_registry():
    tasks = registry()
    for name in ("zeroing", "demo", "caps_contact_based", "caps_measured"):
        assert name in tasks, f"{name} missing from registry"
    for name, module in tasks.items():
        assert callable(module.run), f"{name}.run is not callable"
    assert "primitives" not in tasks and "roles" not in tasks


def _mock_hand(**kwargs):
    hand = connect("hand_1", mock=True, register_signal=False, **kwargs)
    hand.servo.time_scale = 500.0     # run the kinematic model far faster than real time
    return hand


def test_zeroing_and_motion():
    with _mock_hand(load_calibration=False) as hand:
        assert not hand.is_zeroed
        try:
            hand.set_pos(hand.config.upper)
            raise AssertionError("set_pos ran without zero offsets")
        except Exception as e:
            assert "zero offsets" in str(e)

        from cartesian_hand.tasks.zeroing import zero_all
        assert zero_all(hand, save=False, stall_timeout=10.0), "zeroing failed"
        assert hand.is_zeroed

        hand.set_pos(hand.config.upper, timeout=10.0)
        assert np.allclose(hand.targets, hand.config.upper)


def test_gains_are_per_dof():
    """The old set_pos broadcast torque to all seven DOFs, so tasks had to
    reapply a squeeze after every unrelated move."""
    with _mock_hand(load_calibration=False) as hand:
        hand.is_zeroed = True
        hand.set_gains([0], torque=999)
        hand.set_pos([None, 10.0, None, None, None, None, None],
                     torque=42, wait=False)
        assert hand.gains(0)["torque"] == 999, "gain on DOF 0 was clobbered"
        assert hand.gains(1)["torque"] == 42


def test_set_pos_accepts_a_mapping():
    """The mapping form must command exactly the DOFs it names."""
    with _mock_hand(load_calibration=False) as hand:
        hand.is_zeroed = True
        hand.set_pos({3: 20.0, 5: 15.0}, wait=False)
        t = hand.targets
        assert t[3] == 20.0 and t[5] == 15.0
        assert t[0] == 0.0, "mapping form touched an uncommanded DOF"

        # Mapping and vector forms are the same call.
        hand.set_pos([None, None, None, 40.0, None, 35.0, None], wait=False)
        v = hand.targets
        hand.set_pos({3: 20.0, 5: 15.0}, wait=False)
        hand.set_pos({3: 40.0, 5: 35.0}, wait=False)
        assert np.allclose(hand.targets, v), "mapping and vector forms disagree"


def test_position_clamping():
    with _mock_hand(load_calibration=False) as hand:
        hand.is_zeroed = True
        hand.set_pos([999.0] * hand.n_dof, wait=False)
        assert np.allclose(hand.targets, hand.config.upper)
        hand.set_pos([-999.0] * hand.n_dof, wait=False)
        assert np.allclose(hand.targets, hand.config.lower)


def test_policy_contract_enforced():
    with _mock_hand(load_calibration=False) as hand:
        hand.is_zeroed = True
        policy = HoldPolicy(np.zeros(hand.n_dof), horizon=1)
        policy.fingerprint = "deadbeef1234"
        try:
            run_policy(hand, policy, steps=1, settle=0, strict=True)
            raise AssertionError("mismatched policy was allowed to run")
        except ContractMismatch:
            pass
        # Non-strict downgrades to a warning.
        run_policy(hand, policy, steps=1, settle=0, strict=False)


def test_policy_rejects_bad_actions():
    with _mock_hand(load_calibration=False) as hand:
        hand.is_zeroed = True
        for bad, why in ((lambda o: np.full(hand.n_dof, np.nan), "non-finite"),
                         (lambda o: np.zeros(3), "wrong width")):
            try:
                run_policy(hand, FunctionPolicy(bad), steps=2, settle=0)
                raise AssertionError(f"{why} action was accepted")
            except ValueError:
                pass


def test_slew_limit():
    """A policy that teleports in the twin must ramp on hardware."""
    with _mock_hand(load_calibration=False) as hand:
        hand.is_zeroed = True
        hand.set_pos(hand.config.lower, timeout=5.0)

        jump = FunctionPolicy(lambda o: np.ones(hand.n_dof))
        rollout = run_policy(hand, jump, steps=5, hz=200, max_delta=0.05, settle=0)
        steps = np.diff(rollout.actions, axis=0)
        assert np.all(np.abs(steps) <= 0.05 + 1e-9), f"slew limit exceeded: {steps.max()}"

        unlimited = run_policy(hand, jump, steps=2, hz=200, max_delta=None, settle=0)
        assert np.allclose(unlimited.actions[0], 1.0), "unlimited run was still clamped"


def test_rollout_roundtrip_and_replay():
    with _mock_hand(load_calibration=False) as hand:
        hand.is_zeroed = True
        target = np.linspace(-0.5, 0.5, hand.n_dof)
        # Recorded and replayed at the hand's configured rate. Overriding hz
        # here would trip the contract check, which is the point of that check.
        recorded = run_policy(hand, HoldPolicy(target, horizon=6),
                              max_delta=None, settle=0)
        assert len(recorded) == 6
        assert recorded.fingerprint == hand.config.fingerprint()

        path = os.path.join(tempfile.mkdtemp(), "rollout.npz")
        recorded.save(path)
        loaded = Rollout.load(path)
        assert loaded.fingerprint == recorded.fingerprint
        assert np.allclose(loaded.actions, recorded.actions)

        # Replaying a rollout recorded on this hand must pass the contract check.
        replayed = run_policy(hand, ReplayPolicy(path), max_delta=None, settle=0)
        assert np.allclose(replayed.actions, recorded.actions)


def test_rollout_saves_without_optional_fields():
    """A trajectory authored in a twin has actions but no measured positions.
    np.asarray(None) makes an object array that will not load without pickle."""
    cfg = get_hand("hand_1")
    r = Rollout(fingerprint=cfg.fingerprint(), hz=50.0,
                actions=np.zeros((4, cfg.n_dof)))
    path = os.path.join(tempfile.mkdtemp(), "twin.npz")
    loaded = Rollout.load(r.save(path))
    assert loaded.fingerprint == cfg.fingerprint()
    assert loaded.actions.shape == (4, cfg.n_dof)
    assert loaded.positions.shape == (0, cfg.n_dof)


def test_zeroing_aborts_when_a_dof_never_stalls():
    """A timed-out DOF must not be recorded as a zero: the origin would land
    mid-travel and every later mm command on that axis would be wrong."""
    from cartesian_hand.tasks import primitives, zeroing

    # The DOF must still be travelling when the timeout fires. A servo that is
    # not moving at all is indistinguishable from one against a hard stop, on
    # the mock and on real hardware alike.
    with _mock_hand(load_calibration=False) as hand:
        hand.servo.time_scale = 1.0       # slow enough not to reach the stop
        before = hand.zero_offset.copy()

        assert not zeroing.zero_all(hand, save=False, stall_timeout=0.5)
        assert not hand.is_zeroed
        assert np.array_equal(hand.zero_offset, before), "offsets were modified"

        sid = hand.config[0].servo_id
        hand.servo.enable_torque(sid, True)
        hand.servo.set_position(sid, hand.servo.read_position(sid) - 10000, 50, 20, 50)
        stops = primitives.wait_for_stall_counts(hand, [0], timeout=0.5, verbose=False)
        assert stops[0] is None, "timeout reported a moving DOF as a hard stop"


def test_unfingerprinted_replay_is_refused():
    """A recorded trajectory is only valid for the geometry it was recorded on.
    An unlabelled one must not be allowed to drive an arbitrary hand."""
    cfg = get_hand("hand_1")
    path = os.path.join(tempfile.mkdtemp(), "anon.npz")
    Rollout(fingerprint=None, hz=50.0, actions=np.zeros((3, cfg.n_dof))).save(path)

    with _mock_hand(load_calibration=False) as hand:
        hand.is_zeroed = True
        try:
            run_policy(hand, ReplayPolicy(path), steps=1, settle=0, strict=True)
            raise AssertionError("unfingerprinted rollout was allowed to run")
        except ContractMismatch:
            pass
        # A hand-written policy without a fingerprint is still fine: it is
        # geometry-agnostic by construction, unlike a fixed trajectory.
        run_policy(hand, HoldPolicy(np.zeros(hand.n_dof), horizon=1),
                   steps=1, settle=0, strict=True)


def test_rate_override_is_caught():
    """control_hz is in the fingerprint, so running at another rate invalidates
    the check that just passed."""
    with _mock_hand(load_calibration=False) as hand:
        hand.is_zeroed = True
        policy = HoldPolicy(np.zeros(hand.n_dof), horizon=1)
        policy.fingerprint = hand.config.fingerprint()
        try:
            run_policy(hand, policy, steps=1, hz=200.0, settle=0, strict=True)
            raise AssertionError("mismatched control rate was accepted")
        except ContractMismatch:
            pass
        run_policy(hand, policy, steps=1, hz=hand.config.motion.control_hz,
                   settle=0, strict=True)


def test_actions_stay_in_range_from_an_out_of_travel_pose():
    """The slew window is applied around the last action. Seeded from an
    out-of-travel measurement it would drag commands outside [-1, 1]."""
    with _mock_hand(load_calibration=False) as hand:
        hand.is_zeroed = True
        # Park the model far outside configured travel.
        for d in range(hand.n_dof):
            hand.servo._pos[hand.config[d].servo_id] = -5000.0
        hand._step()
        assert np.any(np.abs(hand.normalized_positions()) > 1.0), "setup failed"

        rollout = run_policy(hand, HoldPolicy(np.ones(hand.n_dof), horizon=5),
                             hz=200, max_delta=0.05, settle=0)
        assert np.all(np.abs(rollout.actions) <= 1.0 + 1e-9), \
            f"action escaped [-1, 1]: {rollout.actions.min()} {rollout.actions.max()}"


def test_geometry_rejects_unusable_calibration():
    for bad in (0.0, -5.0, float("nan"), float("inf")):
        try:
            Geometry(counts_per_mm=bad)
            raise AssertionError(f"counts_per_mm={bad} accepted")
        except ValueError:
            pass
    try:
        Geometry(gear_pitch_diameter_mm=0.0)
        raise AssertionError("zero pitch diameter accepted")
    except ValueError:
        pass


def test_enable_is_not_racy():
    """Two callers must not each start a control loop onto one serial bus."""
    import threading as th

    with _mock_hand(load_calibration=False) as hand:
        hand.is_zeroed = True
        threads = [th.Thread(target=hand.enable) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        live = [t for t in th.enumerate() if t.name.startswith("Thread-")
                and t.is_alive() and t.daemon]
        assert hand._thread is not None and hand._thread.is_alive()
        hand.release()
        assert not hand.running
        assert not hand._thread or not hand._thread.is_alive()


def test_zeroing_rolls_back_a_partial_failure():
    """A phase that succeeds then a later one that fails must not leave the hand
    holding a mix of new and stale offsets while still flagged as zeroed."""
    from cartesian_hand.tasks import zeroing

    with _mock_hand(load_calibration=False) as hand:
        # Pretend the hand already had a good calibration loaded.
        previous = np.arange(1, hand.n_dof + 1, dtype=int) * 100
        hand.zero_offset[:] = previous
        hand.is_zeroed = True

        # Fail the second phase only: DOF 4's servo stops answering.
        failing_sid = hand.config[zeroing.PHASES[1][1][0]].servo_id
        real_read = hand.servo.read_position
        hand.servo.read_position = lambda sid: (None if sid == failing_sid
                                                else real_read(sid))
        try:
            assert not zeroing.zero_all(hand, save=False, stall_timeout=5.0)
        finally:
            hand.servo.read_position = real_read

        assert np.array_equal(hand.zero_offset, previous), \
            f"offsets left mixed: {hand.zero_offset.tolist()} vs {previous.tolist()}"
        assert hand.is_zeroed, "previous good calibration was discarded"


def test_control_loop_drops_torque_on_driver_failure():
    """The loop is a daemon thread. An exception there must not end silently
    with the servos still energized against a load."""
    with _mock_hand(load_calibration=False) as hand:
        hand.is_zeroed = True
        hand.set_pos(np.full(hand.n_dof, 10.0), wait=False)
        time.sleep(0.1)

        def explode(sid):
            raise IOError("serial port went away")

        hand.servo.read_position = explode
        deadline = time.time() + 3.0
        while hand.running and time.time() < deadline:
            time.sleep(0.02)

        assert not hand.running, "loop kept running after a driver exception"
        assert hand._loop_error is not None
        assert not any(hand.servo._torque_on.values()), "servos left energized"


def test_cli_parses_without_hardware():
    import tyro

    from cartesian_hand.__main__ import build_cli

    cmd = tyro.cli(build_cli(),
                   args=["caps_contact_based", "--mock", "--cap-offset", "25"])
    assert cmd.opts.mock and cmd.opts.hand is None
    assert cmd.cfg.cap_offset == 25.0
    assert cmd.cfg.num_revs == 3.0, "unspecified flag lost its default"
    assert callable(cmd.execute)

    policy = tyro.cli(build_cli(), args=["policy", "run.npz", "--max-delta", "0.1"])
    assert policy.source == "run.npz" and policy.max_delta == 0.1
    assert policy.strict, "strict must default on"


def test_every_task_exposes_a_config():
    """The CLI is generated from Config, so a task without one is invisible."""
    import inspect

    from cartesian_hand.tasks import _discover
    for name, module in _discover().items():
        assert hasattr(module, "Config"), f"{name} has no Config dataclass"
        assert hasattr(module, "DESCRIPTION"), f"{name} has no DESCRIPTION"
        params = list(inspect.signature(module.run).parameters)
        assert params[:2] == ["hand", "cfg"], f"{name}.run signature is {params}"


def test_vector_conversion_matches_per_dof():
    """The loop converts whole vectors; tasks still convert one DOF at a time."""
    with _mock_hand() as hand:
        cfg = hand.config
        offsets = np.arange(cfg.n_dof, dtype=int) * 137 - 300
        counts = np.arange(cfg.n_dof, dtype=int) * 411 + 50

        vector_mm = cfg.counts_to_mm_all(counts, offsets)
        for d in range(cfg.n_dof):
            scalar_mm = cfg.counts_to_mm(d, int(counts[d]), int(offsets[d]))
            assert abs(vector_mm[d] - scalar_mm) < 1e-9, \
                f"DOF {d}: vector {vector_mm[d]} vs scalar {scalar_mm}"

        mm = np.linspace(0.0, 55.0, cfg.n_dof)
        vector_counts = cfg.mm_to_counts_all(mm, offsets)
        for d in range(cfg.n_dof):
            scalar_counts = cfg.mm_to_counts(d, float(mm[d]), int(offsets[d]))
            assert vector_counts[d] == scalar_counts, \
                f"DOF {d}: vector {vector_counts[d]} vs scalar {scalar_counts}"


def test_dropped_read_does_not_move_state():
    """A servo that fails to answer must keep its last position, not gain a new one.

    ReadPos decodes the SDK's -1 error into +32769 counts (~402mm), so before
    the driver reported failures as None this wrote a plausible-looking but
    invented position straight into the state vector.
    """
    with _mock_hand() as hand:
        hand.set_pos([25.0] * hand.n_dof, timeout=5.0, tolerance=1.0)
        # Stop the loop first: it would race the manual steps below and refresh
        # the very DOF the test is checking stayed put.
        hand.stop_loop()
        hand._step()
        settled = hand.positions.copy()

        real = hand.servo.read_positions
        dropped = 2
        hand.servo.read_positions = lambda sids: [
            None if i == dropped else v for i, v in enumerate(real(sids))]
        try:
            # Move the target so a live DOF demonstrably tracks while the
            # dropped one holds; otherwise a frozen state vector would pass too.
            hand.target[:] = 40.0
            for _ in range(5):
                hand._step()
            after = hand.positions
        finally:
            hand.servo.read_positions = real

        assert after[dropped] == settled[dropped], \
            f"dropped DOF moved {settled[dropped]} -> {after[dropped]}"
        assert after[0] != settled[0], \
            "no DOF moved at all, so holding proves nothing"


def test_enable_holds_position_instead_of_commanding_zero():
    """enable() must seed the target from the hardware.

    The loop writes the whole target vector every step and target starts as
    zeros, so without seeding the first step drives every joint to 0mm — which
    on real hardware means into the hard stops.
    """
    with _mock_hand() as hand:
        # Park somewhere clearly away from both zero and the stops.
        hand.set_pos([25.0] * hand.n_dof, timeout=5.0, tolerance=1.0)
        hand.stop_loop()
        parked = hand.positions.copy()

        hand.target[:] = 0.0          # what construction leaves behind
        hand.enable()
        time.sleep(0.2)

        assert np.allclose(hand.targets, parked, atol=1.0), \
            f"enable() did not hold: target {hand.targets} vs parked {parked}"
        assert np.allclose(hand.positions, parked, atol=1.0), \
            f"hand moved on enable: {hand.positions} vs {parked}"


def test_set_pos_accepts_per_dof_gains():
    """A per-DOF gain vector through set_pos lands on each DOF individually.

    An earlier revision only accepted scalars, so passing m.torque (a list,
    since Motion was vectorized) silently TypeError'd. Tasks rely on this.
    """
    with _mock_hand() as hand:
        per_dof = [50, 50, 50, 300, 50, 50, 50]
        hand.set_pos([20.0] * hand.n_dof, torque=per_dof, wait=False)
        got = [hand.gains(d)["torque"] for d in range(hand.n_dof)]
        assert got == per_dof, f"per-DOF torque lost: {got}"

        hand.set_pos([25.0] * hand.n_dof, torque=200, wait=False)
        got = [hand.gains(d)["torque"] for d in range(hand.n_dof)]
        assert all(t == 200 for t in got), f"scalar broadcast lost: {got}"

        # Wrong-length sequence raises — same rule Motion enforces.
        try:
            hand.set_pos([25.0] * hand.n_dof, torque=[100, 200], wait=False)
            raise AssertionError("a 2-value gain vector should not be accepted")
        except ValueError:
            pass


def test_per_dof_gains_ride_in_one_write():
    """Differing gains must not fall back to one packet per servo."""
    with _mock_hand() as hand:
        hand.set_gains([3], torque=300)          # z differs from the rest
        calls = []
        real = hand.servo.set_positions
        hand.servo.set_positions = lambda *a, **k: (calls.append((a, k)), real(*a, **k))[1]
        try:
            hand._step()
        finally:
            hand.servo.set_positions = real

        assert len(calls) == 1, f"expected one sync write, got {len(calls)}"
        (_, _, speed, acc, torque), _ = calls[0]
        assert torque[3] == 300, f"per-DOF torque lost: {torque}"
        assert torque[0] == hand.config.gain_vector("torque")[0], \
            f"other DOFs disturbed: {torque}"
        assert len(speed) == hand.n_dof and len(acc) == hand.n_dof


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  ok    {t.__name__}")
        except Exception as e:
            failed += 1
            print(f"  FAIL  {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Self-check for the mock servo model.

    python tests/test_servo.py

The mock is what every layer above it is tested against, so a lie here is a
lie everywhere. These pin the three behaviours the stack depends on: motion
toward a target, stopping at a stop, and reporting a stall honestly on the
speed and load channels.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time

from cartesian_hand.servo import MockServo


def _settle(m, sids, seconds=0.2):
    """Let the model integrate. time_scale makes this fast in wall clock."""
    time.sleep(seconds)
    return m.read_all(sids)


def test_a_servo_moves_toward_its_target():
    m = MockServo("mock", time_scale=100.0)
    m.enable_torques([0], True)
    m.set_positions([0], [1000], speed=300)
    _settle(m, [0])
    assert m.read_position(0) == 1000, "did not reach an in-range target"


def test_torque_off_means_no_motion():
    """The zeroing protocol parks a phase and drops torque; a mock that moved
    anyway would hide a task forgetting to re-enable it."""
    m = MockServo("mock", time_scale=100.0)
    m.set_positions([0], [1000], speed=300)      # torque never enabled
    _settle(m, [0])
    assert m.read_position(0) == 0, "moved with torque disabled"


def test_a_servo_stops_at_its_stop_and_reports_the_stall():
    """The event the whole contact story rests on: commanded to move, not
    moving, and both the speed and load channels say so."""
    m = MockServo("mock", time_scale=100.0)
    m.enable_torques([0], True)
    m.set_positions([0], [MockServo.STOP_HIGH + 5000], speed=3000)
    for _ in range(6):                            # drive it into the stop
        _settle(m, [0], 0.05)
    pos, speed, load = m.read_all([0])[0]
    assert pos == MockServo.STOP_HIGH, f"walked past the stop: {pos}"
    assert speed == 0, f"reports motion while blocked: {speed}"
    assert load == MockServo.BLOCKED_LOAD, f"no load while blocked: {load}"


def test_an_obstruction_stalls_partway_through_travel():
    """A cap is not a rail end. Without a movable stop the only thing a mock
    joint can hit is the end of its own travel, and 'the jaw closed on a 12mm
    cap' -- the measurement the entire cap task is built on -- is untestable
    offline."""
    m = MockServo("mock", time_scale=100.0, start_counts=3000)
    m.set_stops(0, low=1000)                      # the object
    m.enable_torques([0], True)
    m.set_positions([0], [-6000], speed=3000)     # drive well past it
    for _ in range(6):
        _settle(m, [0], 0.05)
    pos, speed, load = m.read_all([0])[0]
    assert pos == 1000, f"drove through the obstruction: {pos}"
    assert speed == 0 and load == MockServo.BLOCKED_LOAD, (speed, load)


def test_arriving_is_not_reported_as_a_stall():
    """A joint that ARRIVED also has zero speed. If the mock loaded up on
    arrival, `stuck` would be indistinguishable from `at goal` and an empty
    gripper closing on nothing would report a contact."""
    m = MockServo("mock", time_scale=100.0)
    m.enable_torques([0], True)
    m.set_positions([0], [500], speed=3000)
    for _ in range(4):
        _settle(m, [0], 0.05)
    pos, speed, load = m.read_all([0])[0]
    assert pos == 500
    assert speed == 0 and load == 0, f"arrival faked a contact: {speed}, {load}"


def test_read_all_is_one_entry_per_id_in_order():
    """The control loop zips this against its DOF table, so order and length
    are the contract -- a short list would silently shift every joint."""
    m = MockServo("mock", time_scale=100.0)
    sids = [3, 1, 7]
    m.enable_torques(sids, True)
    m.set_positions(sids, [100, 200, 300], speed=3000)
    _settle(m, sids)
    out = m.read_all(sids)
    assert len(out) == 3
    assert [r[0] for r in out] == [100, 200, 300], f"order not preserved: {out}"


def test_gains_behave_the_way_the_real_driver_does():
    """The mock's whole job is to fail the same way hardware does. All five
    cases were verified against a real FtServo on hand_2.

    speed=0 is the trap: it means *as fast as it can*, not "do not move". A
    mock that read a falsy speed as "leave it alone" would pass offline and
    then run at whatever speed the previous command happened to set.
    """
    m = MockServo("mock", time_scale=100.0)
    try:
        m.set_positions([0], [100], None)
        raise AssertionError("None gain accepted; the real driver raises TypeError")
    except TypeError:
        pass
    try:
        m.set_positions([0], [100], [300, 300], [50, 50], [50, 50])
        raise AssertionError("wrong-length gain accepted; all lists must match")
    except ValueError:
        pass
    # All three gains must ask for the same overload. This one reached hardware:
    # `set_positions(ids, pos, 300, 25, [50]*7)` matches neither the all-Sequence
    # overload nor the all-int one, and the mock used to broadcast each gain
    # independently and accept it.
    try:
        m.set_positions([0], [100], [300], 50, 50)
        raise AssertionError("mixed scalar and sequence gains accepted")
    except TypeError:
        pass
    m.enable_torques([0], True)
    m.set_positions([0], [MockServo.STOP_HIGH], speed=0)
    _settle(m, [0], 0.15)
    assert m.read_position(0) == MockServo.STOP_HIGH, "speed=0 did not run at max"

    # Per-servo list applied per-servo, not broadcast. Broadcasting would give
    # both joints speed 10; the real driver takes a scalar OR one value per servo.
    m2 = MockServo("mock", time_scale=100.0)
    m2.enable_torques([0, 1], True)
    m2.set_positions([0, 1], [6000, 6000], [10, 6000], [50, 50], [500, 500])
    _settle(m2, [0, 1], 0.1)
    slow, fast = m2.read_position(0), m2.read_position(1)
    assert fast > slow, f"speeds not applied per servo: {slow}, {fast}"


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

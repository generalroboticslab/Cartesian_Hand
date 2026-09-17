"""The servo bus: one open serial port and the servos on it.

Get a driver
------------
    from cartesian_hand.servo import open_driver

    bus = open_driver("/dev/ttyACM0")        # real hardware
    bus = open_driver("mock", mock=True)     # offline model, no hardware

Both return an object with the same five methods. `open_driver` is the only
place that knows how to find the compiled extension.

Commands
--------
    bus.enable_torques(ids, on)                  energize (True) or release
    bus.set_positions(ids, pos,                  command every servo, one packet
                      speed=0, acc=50, torque=500)
    bus.close()                                  release torque FIRST, see below

`ids` and `pos` are equal-length lists. Each gain is either one number shared by
every servo, or a list with one entry per servo -- both cost the same packet,
because `INST_SYNC_WRITE` broadcasts once and each servo reads its own slice.

Reading
-------
    bus.read_all(ids)

Returns a list as long as `ids`, in the same order. Each entry is either a
3-tuple `(position, speed, load)` for that servo, or `None` if that one servo
did not reply:

    >>> bus.read_all([7, 8, 9])
    [(1740, 0, 0), None, (2505, 0, 0)]
     ^servo 7      ^servo 8 silent

    for sid, reading in zip(ids, bus.read_all(ids)):
        if reading is None:
            continue                  # leave your buffer alone -- see below
        position, speed, load = reading

All three values come from one packet: registers 56-61 are contiguous, so the
reply carrying position carries speed and load too. Load is free on the bus.

    bus.read_position(id)        one servo. int, or None if it did not reply.
    bus.ping(id)                 that id if it answers, else -1
    bus.scan(start_id=0, end_id=253)     list of ids that answer
    bus.get_voltage(id)          tenths of a volt, or None
    bus.get_temperature(id)      Celsius, or None

Units
-----
Everything is a raw servo register value, never SI.

    position   **counts**, not mm. `config.HandConfig.counts_to_mm` converts,
               and needs a zero offset only the zeroing protocol can supply.
    speed      roughly counts/second. `speed=0` on a command means *as fast as
               it can*, NOT "do not move".
    torque     0-1000, a force cap. It only reaches the servo attached to a
               goal, so changing it alone does nothing.
    load       uncalibrated. No mapping to newtons exists.

Two things that bite
--------------------
1. **`None` means that servo did not answer, and must stay `None`.** It is
   never a sentinel dressed up as a number. Skip the slot and leave your
   buffer holding the last good value -- writing a placeholder hands the layer
   above a fabricated position, which reads as a large jump, the exact
   opposite of the stall it is watching for.
2. **Counts go legitimately negative.** They are sign-magnitude on the wire and
   the driver has already decoded them, so a negative position is data, not an
   error code.

Typical use, and the order matters
----------------------------------
    bus.enable_torques(ids, True)              # snaps to whatever goal is
                                               # already in the register, so
                                               # write the present position first
    bus.set_positions(ids, targets, speed=300, torque=50)
    for pos, speed, load in bus.read_all(ids):
        ...
    bus.enable_torques(ids, False)             # close() does NOT do this
    bus.close()

Not exposed here
----------------
The compiled driver has ~25 methods; this module wraps the 9 above. The
omissions are deliberate: the `get_*` family reads a cache that a background
poll thread fills and returns **zeros** when that thread is not running --
indistinguishable from a bus parked at origin -- and `set_mode`/`write_id`
change servo firmware state. Reach for `ft_servo_ext.FtServo` directly if you
need them, and read `help(FtServo)` first.

`MockServo` implements the same surface against a kinematic model with stops,
so everything above this layer is testable with no hardware attached. It has
one extra method, `set_stops`, which is how a test grows an obstruction to
stall against.
"""

import sys
import threading
import time
from collections.abc import Sequence
from typing import Any

# A gain on `set_positions`: one number for the whole bus, or one per servo.
# Not interchangeable at the driver -- the compiled overloads take all
# sequences or all scalars, never a mix. See `MockServo.set_positions`.
Gain = int | Sequence[int]
# What one servo answered with, or None if it did not answer at all. The `None`
# is load-bearing and must stay `None` -- see the module docstring.
Reading = tuple[int, int, int] | None


def _import_ft_servo() -> type:
    """Import the compiled driver.

    Lazy on purpose -- inside the call, not at module scope. Importing anything
    in this package therefore touches no hardware and needs no compiler, which
    is what lets the digital twin import `config` on a machine that cannot build
    the extension.

    One location, not a search. `pip install -e .` puts the extension inside
    this package (pyproject: `wheel.packages = ["cartesian_hand"]`), and that is
    the only supported build. An earlier version also probed site-packages and
    two `build/` directories, inserting each into `sys.path`; verified on this
    machine, the first import resolves and those fallbacks never fire. They cost
    two real things: mutating `sys.path` as a side effect of an import can
    shadow unrelated packages, and a genuinely missing build reported as a
    three-path search instead of as the one command that fixes it.
    """
    try:
        from .ft_servo_ext import FtServo
        return FtServo
    except ImportError as e:
        # sys.executable, not a hardcoded interpreter: the one that matters is
        # the one that just failed to import, and naming any other sends the
        # reader to build the extension into an environment they are not using.
        raise ImportError(
            "cartesian_hand.ft_servo_ext is not built. Run:\n"
            f"  uv pip install -e . --python {sys.executable}\n"
            "(plain `pip install -e .` fails with BackendUnavailable: scikit_build_core.)\n"
            "Or pass mock=True for the offline model."
        ) from e


def open_driver(port: str, mock: bool = False, **mock_kwargs) -> Any:
    """Open the bus. This is the entry point -- start here.

    Returns `Any`, not a union: `FtServo` and `MockServo` share the five-method
    surface documented above but no base class, and `FtServo` is a nanobind
    extension with no stubs, so naming it in a return type would make importing
    this module require the build that `_import_ft_servo` exists to defer.

        bus = open_driver("/dev/ttyACM0")            # hardware
        bus = open_driver("mock", mock=True)         # offline model

    `port` is a device path. Prefer `/dev/serial/by-id/...`, which is tied to
    the adapter's serial number, over `/dev/ttyACM0`: ACM numbers are handed
    out in plug order, so a fixed one silently addresses whichever hand
    enumerated first. `config.HAND_2.port` is already a by-id path.

    `mock_kwargs` go to `MockServo` and are ignored on hardware. The useful one
    is `time_scale`, which runs the model faster than real time.

    Raises ImportError with the build command if the extension is missing.
    """
    if mock:
        return MockServo(port, **mock_kwargs)
    return _import_ft_servo()(port)


class MockServo:
    """Offline stand-in for FtServo.

    Each servo ramps toward its target at the commanded speed until it meets a
    stop. Positions advance from wall-clock time on read, so there is no
    background thread and `time_scale` can make a self-check run in
    milliseconds.

    Deliberately crude: no friction, no following error, and torque does not
    affect motion -- a servo either moves at its commanded speed or is blocked.
    This models control flow, unit conversion and contact *events*, not
    dynamics. Anything that depends on how hard a joint pushes belongs in the
    mujoco backend, not here.
    """

    # Stops span several revolutions because the hands do. 60mm of travel at the
    # configured 16mm pitch diameter is 4889 counts, past the 4096 of a single
    # turn, so these servos run multi-turn. A 0-4095 model would make travel
    # unreachable in mock and hide that.
    STOP_LOW = -6000
    STOP_HIGH = 6000

    # What a blocked servo reports on the load channel. A number, not a force:
    # the real load byte is uncalibrated sign-magnitude counts with no mapping
    # to newtons, so a mock that returned newtons would be inventing precision
    # the hardware does not have.
    BLOCKED_LOAD = 500

    # `speed=0` means "as fast as it can" on the real servo, not "do not move".
    # The model needs a finite number for that, and this is it.
    MAX_SPEED = 3000

    def __init__(self, port: str, n_servos: int = 32, time_scale: float = 20.0,
                 start_counts: int = 0) -> None:
        self.port = port
        # Default faster than real time: a creep-speed zeroing sweep takes two
        # minutes of wall clock at 1x, which makes --mock useless as a smoke test.
        self.time_scale = time_scale
        # ponytail: one lock for the whole bus, matching the real driver's serial
        # port. Per-servo locks would buy nothing -- the bus is the bottleneck.
        self._lock = threading.Lock()
        self.closed = False
        now = time.time()
        ids = range(n_servos)
        self._pos = {i: float(start_counts) for i in ids}
        self._target = dict(self._pos)
        self._speed = {i: 300.0 for i in ids}
        self._torque_on = {i: False for i in ids}
        self._t = {i: now for i in ids}
        self._moved = {i: 0.0 for i in ids}      # counts/s over the last advance
        self._stops = {i: (self.STOP_LOW, self.STOP_HIGH) for i in ids}

    # ── Model ─────────────────────────────────────────────────────────────────

    def set_stops(self, sid: int, low: float | None = None,
                  high: float | None = None) -> None:
        """Move a servo's stops. This is how the mock grows an obstruction.

        A rail end and an object in the jaws are the same event to everything
        above this line -- commanded to move, not moving -- so one mechanism
        covers both. Without it the only thing a mock joint can hit is the end
        of its own travel, and a jaw closing on a 12mm cap is untestable
        offline: the whole cap task turns on stalling *partway* through the
        stroke, which is exactly what a rail stop cannot express.

        `None` leaves that side where it is.
        """
        with self._lock:
            lo, hi = self._stops[sid]
            self._stops[sid] = (lo if low is None else float(low),
                                hi if high is None else float(high))

    def _advance(self, sid: int) -> float:
        """Integrate one servo up to now. Caller holds the lock."""
        now = time.time()
        dt = (now - self._t[sid]) * self.time_scale
        self._t[sid] = now
        if dt <= 0:
            return self._pos[sid]
        before = self._pos[sid]
        if self._torque_on[sid]:
            # Feetech speed units are approximately steps/sec.
            step = self._speed[sid] * dt
            delta = self._target[sid] - before
            if abs(delta) <= step:
                self._pos[sid] = self._target[sid]
            else:
                self._pos[sid] = before + step * (1 if delta > 0 else -1)
            lo, hi = self._stops[sid]
            self._pos[sid] = min(hi, max(lo, self._pos[sid]))
        # Speed is reported from motion that actually happened, not from the
        # commanded speed. A servo pressed against a stop still has a distant
        # target, so a commanded-speed model would report it sprinting forever
        # and nothing reading the speed channel could ever see a stall.
        self._moved[sid] = (self._pos[sid] - before) / dt
        return self._pos[sid]

    def _blocked(self, sid: int) -> bool:
        """Commanded to move, not moving. Caller holds the lock."""
        return (self._torque_on[sid]
                and abs(self._target[sid] - self._pos[sid]) > 1.0
                and abs(self._moved[sid]) < 1.0)

    # ── Commands ──────────────────────────────────────────────────────────────

    def set_positions(self, ids: Sequence[int], pos: Sequence[float],
                      speed: Gain = 0, acc: Gain = 50,
                      torque: Gain = 500) -> bool:
        """Command every servo in one packet. Gains are scalars or per-servo lists.

        Signature and defaults copied from the real driver, which is overloaded:
        a list gain picks the per-servo overload, an int picks the shared one.
        They are not interchangeable defaults -- an earlier version of this mock
        took `speed=None` and treated any falsy speed as "leave it alone", which
        differed from hardware twice over. `None` raises TypeError on the real
        driver, and real `speed=0` means *as fast as it can*, not "do not move".
        Both would have passed every offline test and then behaved differently
        on the bench.

        The three gains must agree on which overload they are asking for: all
        sequences, or all scalars. A mix matches neither and raises `TypeError`
        on hardware, and this mock used to accept it -- it broadcast each gain
        independently, so `(speed=300, acc=25, torque=[...])` passed every
        offline test and then failed on the first real bus. Rejecting it here is
        the whole point of a mock whose signature is copied from the driver.

        `acc` and `torque` are accepted and ignored: this model has no
        acceleration ramp, and torque does not gate motion here. A joint either
        runs at its commanded speed or is blocked. Anything that turns on how
        hard a joint pushes belongs in the mujoco backend.
        """
        n = len(ids)
        if len(pos) != n:
            raise ValueError(f"ids and pos differ in length: {n} vs {len(pos)}")

        seq = [isinstance(g, (list, tuple)) for g in (speed, acc, torque)]
        if any(seq) and not all(seq):
            raise TypeError(
                f"set_positions gains must be all sequences or all scalars, "
                f"got speed={type(speed).__name__}, acc={type(acc).__name__}, "
                f"torque={type(torque).__name__}")

        def per_servo(gain: Gain, name: str) -> Sequence[int]:
            if isinstance(gain, (list, tuple)):
                if len(gain) != n:
                    raise ValueError(
                        f"{name} has {len(gain)} values, expected {n} or a scalar")
                return gain
            return [gain] * n

        speeds = per_servo(speed, "speed")
        # Length-checked and discarded: this model ignores both, but a wrong
        # length is a wrong packet on hardware and must not pass offline.
        per_servo(acc, "acc")
        per_servo(torque, "torque")
        with self._lock:
            for sid, p, s in zip(ids, pos, speeds):
                self._advance(sid)
                self._target[sid] = float(p)
                # float(s) first, so a None gain raises TypeError here exactly
                # as it does on the real driver. `if s else` would short-circuit
                # and silently accept it. `or` then maps a real 0 to max speed.
                self._speed[sid] = float(s) or float(self.MAX_SPEED)
        return True

    def enable_torques(self, sids: Sequence[int], on: bool) -> bool:
        with self._lock:
            for sid in sids:
                self._advance(sid)
                self._torque_on[sid] = bool(on)
        return True

    def set_torque_limit(self, sid: int, limit: int) -> bool:
        """No-op here, matching `set_positions`'s torque: this model has no
        concept of a force ceiling, real or otherwise. Present so code that
        calls it (real hardware's separate TORQUE_LIMIT register, 0-1000,
        distinct from `set_positions`'s own per-move torque -- see
        `ft_servo_driver.hpp`) runs the same against `--mock`."""
        return True

    def set_torque_limits(self, sids: Sequence[int], limits: Sequence[int]) -> None:
        if len(limits) != len(sids):
            raise ValueError(
                f"set_torque_limits: ids and limits must be the same length, "
                f"got {len(sids)} vs {len(limits)}")
        for sid, limit in zip(sids, limits):
            self.set_torque_limit(sid, limit)

    # ── Reads ─────────────────────────────────────────────────────────────────

    def read_all(self, sids: Sequence[int]) -> list[Reading]:
        """(position, speed, load) per servo, one entry each, in ID order.

        The real driver gets all three from a single sync-read: registers 56-61
        are contiguous, so the 6-byte reply that carries position carries speed
        and load too. Load is free on the bus.

        A servo that did not answer is `None` in place of the tuple. The mock
        never drops a reply, so callers that need to exercise a dropped read
        patch this method -- fabricating drops on a timer would make every test
        that uses the mock intermittently flaky.
        """
        with self._lock:
            out = []
            for sid in sids:
                pos = self._advance(sid)
                load = self.BLOCKED_LOAD if self._blocked(sid) else 0
                out.append((int(pos), int(self._moved[sid]), load))
            return out

    def read_position(self, sid: int) -> int:
        with self._lock:
            return int(self._advance(sid))

    def get_voltage(self, sid: int) -> int:
        return 120        # tenths of a volt

    def get_temperature(self, sid: int) -> int:
        return 30

    def ping(self, sid: int) -> int:
        return sid if sid in self._pos else -1

    def scan(self, start_id: int = 0, end_id: int = 253) -> list[int]:
        """Which servos are on the bus. Servo IDs settle which hand is plugged
        in: 0-6 is hand_1, 7-13 is hand_2."""
        return [s for s in self._pos if start_id <= s <= end_id]

    def close(self) -> None:
        """Does NOT release torque, matching the real driver. Callers that want
        the hand limp call enable_torques(ids, False) first."""
        self.closed = True

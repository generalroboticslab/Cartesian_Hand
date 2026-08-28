"""Servo backends.

`open_driver()` is the single place that knows how to find the compiled
extension, replacing the import shim that used to sit at the top of
cartesian_hand.py.

`MockServo` implements the same surface in pure Python against a simple
kinematic model with hard stops. It is what makes the rest of the stack
runnable with no hardware attached, which matters for developing a policy
before it ever touches a real hand.
"""

import os
import sys
import time
import threading


def _import_ft_servo():
    """Locate the compiled FtServo extension. Tries, in order: the installed
    package, site-packages, then a local build/ directory."""
    try:
        from .ft_servo_ext import FtServo
        return FtServo
    except ImportError:
        pass

    import sysconfig
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(sysconfig.get_path("platlib"), "cartesian_hand"),
        os.path.join(os.path.dirname(here), "build"),
        os.path.join(here, "build"),
    ]
    for d in candidates:
        if os.path.isdir(d) and d not in sys.path:
            sys.path.insert(0, d)
    try:
        from ft_servo_ext import FtServo
        return FtServo
    except ImportError as e:
        raise ImportError(
            "Could not import ft_servo_ext. Build it with `pip install -e .`, "
            "or run with --mock to use the simulated backend.\n"
            f"Searched: {candidates}\nOriginal error: {e}"
        ) from e


def open_driver(port: str, mock: bool = False, **mock_kwargs):
    """Return a servo driver for `port`. Set mock=True for the offline model."""
    if mock:
        return MockServo(port, **mock_kwargs)
    return _import_ft_servo()(port)


class MockServo:
    """Offline stand-in for FtServo.

    Each servo is a position source that ramps toward its target at the
    commanded speed, bounded by a hard stop. Positions advance from wall-clock
    time on read, so no background thread is needed and `time_scale` can make a
    self-check run in milliseconds.

    The model is deliberately crude: no friction, no load-dependent stall, no
    following error. It exercises control flow and unit conversion, not physics.
    """

    # ponytail: one lock for the whole bus, matching the real driver's serial
    # port. Per-servo locks would buy nothing since the bus is the bottleneck.

    # Hard stops span several revolutions because the hands do. 60mm of travel
    # at the configured 16mm pitch diameter is 4889 counts, past the 4096 of a
    # single turn, so these servos must be running multi-turn. A 0-4095 model
    # would make travel unreachable in mock and mask that.
    STOP_LOW = -6000
    STOP_HIGH = 6000

    # Default faster than real time: a creep-speed zeroing sweep takes two
    # minutes of wall clock at 1x, which makes `--mock` useless as a smoke test.
    def __init__(self, port: str, n_servos: int = 32, time_scale: float = 20.0,
                 start_counts: int = 0):
        self.port = port
        self.time_scale = time_scale
        self._lock = threading.Lock()
        self._closed = False
        now = time.time()
        self._pos = {i: float(start_counts) for i in range(n_servos)}
        self._target = dict(self._pos)
        self._speed = {i: 300.0 for i in range(n_servos)}
        self._torque_on = {i: False for i in range(n_servos)}
        self._t = {i: now for i in range(n_servos)}

    # ── Model ─────────────────────────────────────────────────────────────────

    def _advance(self, sid: int) -> float:
        """Integrate one servo up to now. Caller holds the lock."""
        now = time.time()
        dt = (now - self._t[sid]) * self.time_scale
        self._t[sid] = now
        if not self._torque_on[sid] or dt <= 0:
            return self._pos[sid]
        # Feetech speed units are approximately steps/sec.
        step = self._speed[sid] * dt
        delta = self._target[sid] - self._pos[sid]
        if abs(delta) <= step:
            self._pos[sid] = self._target[sid]
        else:
            self._pos[sid] += step * (1 if delta > 0 else -1)
        # Hard stop: this is what makes zeroing-by-stall terminate.
        self._pos[sid] = min(self.STOP_HIGH, max(self.STOP_LOW, self._pos[sid]))
        return self._pos[sid]

    # ── Commands ──────────────────────────────────────────────────────────────

    def set_position(self, sid, position, speed=None, acc=None, torque=None):
        with self._lock:
            self._advance(sid)
            self._target[sid] = float(position)
            if speed:
                self._speed[sid] = float(speed)
        return True

    def set_positions(self, sids, positions, speed=None, acc=None, torque=None):
        for sid, p in zip(sids, positions):
            self.set_position(sid, p, speed, acc, torque)
        return True

    def set_speed(self, sid, speed):
        with self._lock:
            self._speed[sid] = float(speed)
        return True

    def set_speeds(self, sids, speeds):
        for sid, s in zip(sids, speeds):
            self.set_speed(sid, s)
        return True

    def enable_torque(self, sid, on):
        with self._lock:
            self._advance(sid)
            self._torque_on[sid] = bool(on)
        return True

    def enable_torques(self, sids, on):
        for sid in sids:
            self.enable_torque(sid, on)
        return True

    def set_mode(self, sid, mode):
        return True

    # ── Reads ─────────────────────────────────────────────────────────────────

    def read_position(self, sid):
        with self._lock:
            return int(self._advance(sid))

    def get_position(self, sid):
        return self.read_position(sid)

    def get_positions(self, sids):
        return [self.read_position(s) for s in sids]

    def read_speed(self, sid):
        with self._lock:
            moving = abs(self._target[sid] - self._pos[sid]) > 1
            return int(self._speed[sid]) if moving and self._torque_on[sid] else 0

    def get_speed(self, sid):
        return self.read_speed(sid)

    def get_speeds(self, sids):
        return [self.read_speed(s) for s in sids]

    def read_load(self, sid):
        # Nonzero only while pressed against a hard stop, so stall-detection code
        # sees something plausible.
        with self._lock:
            at_stop = self._pos[sid] in (self.STOP_LOW, self.STOP_HIGH)
            return 500 if at_stop and self._torque_on[sid] else 0

    def get_load(self, sid):
        return self.read_load(sid)

    def get_loads(self, sids):
        return [self.read_load(s) for s in sids]

    def get_voltage(self, sid):
        return 120        # tenths of a volt

    def get_temperature(self, sid):
        return 30

    def ping(self, sid):
        return 0 if sid in self._pos else -1

    def scan(self, start_id: int = 0, end_id: int = 253):
        return [s for s in self._pos if start_id <= s <= end_id]

    def write_id(self, sid, new_id):
        with self._lock:
            for d in (self._pos, self._target, self._speed, self._torque_on, self._t):
                d[new_id] = d.pop(sid)
        return True

    def start_poll(self, *a, **kw):
        return True

    def stop_poll(self):
        return True

    def close(self):
        self._closed = True

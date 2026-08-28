"""CartesianHand: the control layer over a bus of FT servos.

Holds no hardware constants of its own. Everything dimensional comes from the
HandConfig it is given, so the same class drives any hand defined in hands.py,
and the same code path drives the mock backend with no hardware attached.
"""

import json
import os
import signal
import sys
import threading
import time
from datetime import datetime

import numpy as np

from .driver import MockServo, open_driver
from .hands import HandConfig


class NotZeroedError(RuntimeError):
    """Raised when a motion is commanded before zero offsets are known."""


# ── Zero-offset persistence ───────────────────────────────────────────────────
#
# Offsets are measured by the zeroing routine at runtime, not authored, so
# unlike hands.py they live in a generated data file. Entries are keyed by hand
# name. An earlier version keyed them by `dof_config is config_1` object
# identity, which mislabelled any hand built from a copied dict.

# Measured offsets outlive any one install, so they must not sit inside the
# package directory: `pip install -e .` wipes it, and a lost calibration means
# re-driving every DOF into its hard stop. Override with CARTESIAN_HAND_CALIB.
CALIB_PATH = os.environ.get(
    "CARTESIAN_HAND_CALIB",
    os.path.join(os.path.expanduser("~"), ".cartesian_hand", "zero_offsets.json"))

# Offsets written by an older version landed next to this file. Read them if the
# new location has nothing yet, so an existing calibration is not silently lost.
_LEGACY_CALIB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "zero_offsets.json")


def save_offsets(name: str, offsets, path: str = CALIB_PATH) -> str:
    """Merge one hand's offsets into the shared file, leaving other hands alone."""
    data = {}
    if os.path.exists(path):
        try:
            with open(path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            # A corrupt file must not cost us a fresh calibration run.
            backup = path + ".bad"
            os.replace(path, backup)
            print(f"Existing {path} was unreadable, moved to {backup}")
    data[name] = {
        "offsets": [int(v) for v in offsets],
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return path


def load_offsets(name: str, n_dof: int, path: str = CALIB_PATH):
    """Return the saved offset vector for a hand, or None if missing or unusable."""
    if not os.path.exists(path) and os.path.exists(_LEGACY_CALIB_PATH):
        print(f"Reading offsets from the old location {_LEGACY_CALIB_PATH}. "
              f"Re-run zeroing to move them to {path}.")
        path = _LEGACY_CALIB_PATH
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"Could not read {path}: {e}")
        return None
    entry = data.get(name)
    if entry is None:
        return None
    offsets = entry.get("offsets") if isinstance(entry, dict) else entry
    if not offsets or len(offsets) != n_dof:
        print(f"Ignoring offsets for {name!r}: expected {n_dof} values, "
              f"got {len(offsets) if offsets else 0}")
        return None
    return np.array(offsets, dtype=int)


def offset_timestamp(name: str, path: str = CALIB_PATH):
    if not os.path.exists(path) and os.path.exists(_LEGACY_CALIB_PATH):
        path = _LEGACY_CALIB_PATH
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            entry = json.load(f).get(name)
    except (json.JSONDecodeError, OSError):
        return None
    return entry.get("timestamp") if isinstance(entry, dict) else None


class CartesianHand:
    """Position controller for one hand.

    Runs a background loop at config.motion.control_hz that reads every DOF and
    writes the current target vector. Commands set targets; the loop does the
    talking, so callers never block on the serial bus.
    """

    def __init__(self, config: HandConfig, driver=None, mock: bool = False,
                 register_signal: bool = True, load_calibration: bool = True):
        self.config = config
        self.name = config.name
        self.n_dof = config.n_dof
        self.servo = driver if driver is not None else open_driver(config.port, mock=mock)
        self.control_hz = config.motion.control_hz

        # Offsets from a mock run are the mock's hard-stop constants, not a
        # measurement, so they must never load onto the real hand. They are
        # still worth persisting (task development, self-checks), so key them
        # separately rather than refusing to save.
        self.calib_key = (f"{self.name}_mock" if isinstance(self.servo, MockServo)
                          else self.name)

        self.lock = threading.Lock()
        # Separate from `lock`, which guards the state arrays. This one guards
        # start/stop so two callers cannot race enable() into two loop threads
        # issuing competing commands on one serial bus.
        self._lifecycle = threading.RLock()
        self.zero_offset = np.zeros(self.n_dof, dtype=int)
        self.is_zeroed = False

        self.target = np.zeros(self.n_dof, dtype=float)
        self.actual = np.zeros(self.n_dof, dtype=float)

        # Each gain is scalar-or-per-DOF in the config; broadcast once here so
        # the loop only ever sees arrays.
        self._speed = config.gain_vector("speed")
        self._acc = config.gain_vector("acc")
        self._torque = config.gain_vector("torque")

        self.running = False
        self._released = False
        self._thread = None
        self._loop_error = None

        if load_calibration:
            self.load_calibration()

        if register_signal:
            signal.signal(signal.SIGINT, self._on_sigint)

    # ── Calibration ───────────────────────────────────────────────────────────

    def load_calibration(self) -> bool:
        offsets = load_offsets(self.calib_key, self.n_dof)
        if offsets is None:
            return False
        self.zero_offset[:] = offsets
        self.is_zeroed = True
        print(f"[{self.name}] loaded zero offsets ({offset_timestamp(self.calib_key)})")
        return True

    def save_calibration(self) -> str:
        return save_offsets(self.calib_key, self.zero_offset)

    def require_zeroed(self):
        if not self.is_zeroed:
            raise NotZeroedError(
                f"[{self.name}] no zero offsets. Run: "
                f"python -m cartesian_hand zeroing --hand {self.name}")

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def enable(self):
        with self._lifecycle:
            if self.running:
                return
            # Seed the target from where the hand actually is. The loop writes
            # the whole target vector every step, and target starts as zeros, so
            # without this the first step commands every joint to 0mm — driving
            # the entire hand into the hard stops the instant the loop starts.
            # Commanding a subset (set_dofs on one DOF, hold() on the jaws) is
            # what exposes it: the DOFs left alone are not left where they are,
            # they are left at zero.
            self._seed_target_from_hardware()
            self.servo.enable_torques(self.config.servo_ids, True)
            self.running = True
            self._released = False
            self._thread = threading.Thread(target=self._control_loop, daemon=True)
            self._thread.start()

    def _seed_target_from_hardware(self):
        """Point the target vector at the current position, so enabling holds."""
        if not self.is_zeroed:
            # No offsets means no mm, so there is nothing meaningful to seed.
            # set_pos/move both require_zeroed before reaching enable(), so the
            # only way here is a caller driving the loop unzeroed on purpose.
            return
        raw = self.servo.read_positions(self.config.servo_ids)
        ok = np.array([c is not None for c in raw])
        if not ok.any():
            raise RuntimeError(
                f"[{self.name}] no servo answered while seeding the target; "
                f"refusing to start the control loop")
        counts = np.array([c if c is not None else 0 for c in raw], dtype=float)
        mm = self.config.counts_to_mm_all(counts, self.zero_offset)
        with self.lock:
            self.actual[ok] = mm[ok]
            self.target[ok] = mm[ok]
            if not ok.all():
                # A joint we could not read gets its target left as-is rather
                # than guessed; report it instead of silently commanding it.
                print(f"[{self.name}] could not read DOFs "
                      f"{[d for d in range(self.n_dof) if not ok[d]]} while seeding")

    def stop_loop(self):
        """Stop the control loop but leave torque on, holding the last target.

        Needed before issuing raw servo commands: otherwise the loop keeps
        writing its own target vector and overwrites them mid-move.
        """
        with self._lifecycle:
            if not self.running:
                return
            self.running = False
            self._join_loop()

    def _join_loop(self):
        t = self._thread
        if t is not None and t.is_alive() and t is not threading.current_thread():
            # Let the loop finish its in-flight bus transaction rather than
            # writing to the port from two threads at once.
            t.join(timeout=2.0 / max(self.control_hz, 1.0) + 0.5)
        self._thread = None

    def release(self, dof_ids: list = None):
        """Disable torque on the given DOFs, or all of them. Idempotent."""
        with self._lifecycle:
            if dof_ids is None and self._released:
                return
            if dof_ids is None:
                # Stop the loop before dropping torque, so it cannot re-command
                # a target onto servos that are being released.
                self.running = False
                self._join_loop()
            time.sleep(0.1)
            ids = (self.config.servo_ids if dof_ids is None
                   else [self.config[d].servo_id for d in dof_ids])
            self.servo.enable_torques(ids, False)
            if dof_ids is None:
                self._released = True
            print(f"[{self.name}] torque disabled"
                  f"{'' if dof_ids is None else f' on DOFs {dof_ids}'}")

    def close(self):
        """Stop the loop, drop torque, close the bus. Safe to call twice."""
        with self._lifecycle:
            try:
                self.release()
            except Exception as e:
                print(f"[{self.name}] release during close failed: {e}")
            try:
                self.servo.close()
            except Exception as e:
                print(f"[{self.name}] close failed: {e}")

    def _on_sigint(self, sig, frame):
        print("\nCtrl+C, releasing servos...")
        self.close()
        sys.exit(0)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    # ── State ─────────────────────────────────────────────────────────────────

    @property
    def positions(self) -> np.ndarray:
        """Measured position vector in mm."""
        with self.lock:
            return self.actual.copy()

    @property
    def targets(self) -> np.ndarray:
        with self.lock:
            return self.target.copy()

    def normalized_positions(self) -> np.ndarray:
        """Measured position in the [-1, 1] space that policies work in."""
        return self.config.normalize(self.positions)

    def at_target(self, dof_ids=None, tolerance: float = 1.0) -> bool:
        ids = range(self.n_dof) if dof_ids is None else dof_ids
        with self.lock:
            return all(abs(self.actual[d] - self.target[d]) <= tolerance for d in ids)

    # ── Gains ─────────────────────────────────────────────────────────────────

    def set_gains(self, dof_ids=None, speed=None, acc=None, torque=None):
        """Set servo gains on specific DOFs. Omitted values are left alone.

        Gains are per DOF on purpose. A task that squeezes with one jaw while
        transiting another needs those two torques to coexist.
        """
        ids = list(range(self.n_dof)) if dof_ids is None else list(dof_ids)
        with self.lock:
            for d in ids:
                if speed is not None:
                    self._speed[d] = int(speed)
                if acc is not None:
                    self._acc[d] = int(acc)
                if torque is not None:
                    self._torque[d] = int(torque)

    def gains(self, dof_id: int) -> dict:
        with self.lock:
            return {"speed": int(self._speed[dof_id]),
                    "acc": int(self._acc[dof_id]),
                    "torque": int(self._torque[dof_id])}

    # ── Motion ────────────────────────────────────────────────────────────────

    def set_pos(self, positions_mm, speed=None, acc=None, torque=None,
                wait: bool = True, wait_dofs: list = None,
                tolerance: float = 1.0, timeout: float = 5.0) -> bool:
        """Command target positions in mm. Returns True if the wait converged.

        Pass None for a DOF to leave its target unchanged. speed/acc/torque
        apply only to the DOFs actually commanded in this call, so a gain set
        for a squeezing jaw survives a later move of a different DOF.
        """
        self.require_zeroed()
        if not self.running:
            self.enable()

        moved = [d for d, v in enumerate(positions_mm) if v is not None]
        with self.lock:
            for d in moved:
                cfg = self.config[d]
                self.target[d] = min(cfg.max_mm, max(cfg.min_mm, float(positions_mm[d])))
                if speed is not None:
                    self._speed[d] = int(speed)
                if acc is not None:
                    self._acc[d] = int(acc)
                if torque is not None:
                    self._torque[d] = int(torque)

        if not wait:
            return True

        check = moved if wait_dofs is None else wait_dofs
        if not check:
            return True
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.at_target(check, tolerance):
                return True
            if self._loop_error is not None:
                # Nothing is driving the servos any more, so waiting out the
                # full timeout would only hide the real failure.
                raise RuntimeError(
                    f"[{self.name}] control loop died: {self._loop_error}")
            time.sleep(0.02)
        print(f"[{self.name}] set_pos timed out on DOFs {check}")
        return False

    def set_dofs(self, updates: dict, **kwargs) -> bool:
        """Command named DOFs by id: `hand.set_dofs({AUX_JAW: 12.0, Z: 20.0})`.

        Tasks move two or three DOFs at a time out of seven, and spelling that
        as a positional list is mostly Nones with the meaning hidden in the
        index of the one entry that is not.
        """
        positions = [None] * self.n_dof
        for dof_id, mm in updates.items():
            positions[dof_id] = mm
        return self.set_pos(positions, **kwargs)

    def move(self, action, **kwargs) -> bool:
        """Command a normalized [-1, 1] action vector. The policy-facing move."""
        return self.set_pos(self.config.denormalize(action), **kwargs)

    def hold(self, dof_ids, torque: int, position_mm: float = 0.0):
        """Drive DOFs toward a position with reduced torque so they stall on
        contact and keep pressing. This is how the hand grips."""
        self.require_zeroed()
        if not self.running:
            self.enable()
        with self.lock:
            for d in dof_ids:
                self._torque[d] = int(torque)
                self.target[d] = float(position_mm)

    # ── Monitoring ────────────────────────────────────────────────────────────

    def publish(self, hz: float = 1.0, dof_ids: list = None):
        """Print DOF state until interrupted."""
        ids = list(range(self.n_dof)) if dof_ids is None else dof_ids
        print(f"[{self.name}] publishing at {hz}Hz, Ctrl+C to stop")
        header = f"{'DOF':<5}{'label':<26}{'actual':>9}{'target':>9}{'load':>7}{'volt':>7}{'temp':>6}"
        try:
            while True:
                print(header)
                print("-" * len(header))
                for d in ids:
                    sid = self.config[d].servo_id
                    counts = self.servo.read_position(sid)
                    mm = self.config.counts_to_mm(d, counts, self.zero_offset[d])
                    volt = self.servo.get_voltage(sid)
                    with self.lock:
                        tgt = f"{self.target[d]:.1f}" if self.running else "-"
                    print(f"{d:<5}{self.config[d].label[:25]:<26}"
                          f"{('?' if mm is None else f'{mm:.1f}'):>9}{tgt:>9}"
                          f"{str(self.servo.read_load(sid)):>7}"
                          f"{('?' if volt is None else f'{volt/10:.1f}V'):>7}"
                          f"{str(self.servo.get_temperature(sid)):>6}")
                time.sleep(1.0 / hz)
        except KeyboardInterrupt:
            print("stopped.")

    # ── Control loop ──────────────────────────────────────────────────────────

    def _control_loop(self):
        period = 1.0 / self.control_hz
        while self.running:
            t0 = time.time()
            try:
                self._step()
            except Exception as e:
                # The loop is a daemon thread, so an exception here would
                # otherwise end silently and leave the servos energized, holding
                # the last target indefinitely with nobody driving them. Drop
                # torque before giving up.
                print(f"[{self.name}] control loop error: {e}")
                self.running = False
                self._loop_error = e
                try:
                    self.servo.enable_torques(self.config.servo_ids, False)
                    self._released = True
                    print(f"[{self.name}] torque dropped after control loop error")
                except Exception as release_error:
                    print(f"[{self.name}] FAILED to drop torque after loop error: "
                          f"{release_error}. Servos may still be holding. "
                          f"Cut power if the hand is loaded.")
                return
            time.sleep(max(0.0, period - (time.time() - t0)))

    def _step(self):
        """One control step: read the joint vector, write the joint vector.

        Two bus packets regardless of gains. The servos are addressed as one
        robot rather than seven devices, which is also how the twin sees them.
        """
        cfg = self.config
        sids = cfg.servo_ids

        # One sync-read TX covers every servo. A servo that does not answer
        # comes back None and keeps its previous value: a stale reading is
        # recoverable, a fabricated one silently corrupts the state vector.
        raw = self.servo.read_positions(sids)
        ok = np.array([c is not None for c in raw])
        if ok.any():
            counts = np.array([c if c is not None else 0 for c in raw], dtype=float)
            mm = cfg.counts_to_mm_all(counts, self.zero_offset)
            with self.lock:
                self.actual[ok] = mm[ok]

        with self.lock:
            target = self.target.copy()
            speed = self._speed.copy()
            acc = self._acc.copy()
            torque = self._torque.copy()

        # Per-servo gains ride in the same sync-write packet as the positions,
        # so a jaw squeezing at one torque and a stage lifting at another cost
        # exactly one packet between them.
        self.servo.set_positions(
            sids, cfg.mm_to_counts_all(target, self.zero_offset).tolist(),
            speed.tolist(), acc.tolist(), torque.tolist())


def connect(hand: str = None, mock: bool = False, port: str = None,
            **kwargs) -> CartesianHand:
    """Open the named hand. The one-liner every task and script starts from."""
    from .hands import get_hand
    config = get_hand(hand)
    if port:
        config = config.variant(port=port)
    return CartesianHand(config, mock=mock, **kwargs)

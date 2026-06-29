import time
import json
import os
import numpy as np
import threading
import signal
import sys

try:
    from .ft_servo_ext import FtServo          # installed package (normal case)
except ImportError:
    # Source tree is shadowing the installed package — find the .so in site-packages.
    import sysconfig
    _sp = os.path.join(sysconfig.get_path("platlib"), "cartesian_hand")
    _build = os.path.join(os.path.dirname(os.path.abspath(__file__)), "build")
    for _d in (_sp, _build):
        if os.path.isdir(_d):
            sys.path.insert(0, _d)
            break
    from ft_servo_ext import FtServo

# ── Ports ─────────────────────────────────────────────────────────────────────
PORT_1 = "/dev/ttyACM0"
PORT_2 = "/dev/ttyACM1"

# ── Motion defaults ───────────────────────────────────────────────────────────
CONTROL_HZ     = 50
DEFAULT_TORQUE = 50   # 0–1000
DEFAULT_SPEED  = 300
DEFAULT_ACC    = 25

# ── DOF configs ───────────────────────────────────────────────────────────────
config_1 = {
    0: {"servo_id":  0, "axis": "y", "orientation": -1, "min_mm": 0.0, "max_mm": 60.0},  # base parallel actuation, y-dir
    1: {"servo_id":  1, "axis": "x", "orientation": +1, "min_mm": 0.0, "max_mm": 60.0},  # base left finger, x-dir
    2: {"servo_id":  2, "axis": "x", "orientation": -1, "min_mm": 0.0, "max_mm": 60.0},  # base right finger, x-dir
    3: {"servo_id":  3, "axis": "z", "orientation": -1, "min_mm": 0.0, "max_mm": 60.0},  # vertical translation, z-dir
    4: {"servo_id":  4, "axis": "y", "orientation": -1, "min_mm": 0.0, "max_mm": 60.0},  # aux parallel actuation, y-dir
    5: {"servo_id":  5, "axis": "x", "orientation": +1, "min_mm": 0.0, "max_mm": 60.0},  # aux left finger, x-dir
    6: {"servo_id":  6, "axis": "x", "orientation": -1, "min_mm": 0.0, "max_mm": 60.0},  # aux right finger, x-dir
}

config_2 = {
    0: {"servo_id":  7, "axis": "y", "orientation": -1, "min_mm": 0.0, "max_mm": 60.0},
    1: {"servo_id":  8, "axis": "x", "orientation": +1, "min_mm": 0.0, "max_mm": 60.0},
    2: {"servo_id":  9, "axis": "x", "orientation": -1, "min_mm": 0.0, "max_mm": 60.0},
    3: {"servo_id": 10, "axis": "z", "orientation": -1, "min_mm": 0.0, "max_mm": 60.0},
    4: {"servo_id": 11, "axis": "y", "orientation": -1, "min_mm": 0.0, "max_mm": 60.0},
    5: {"servo_id": 12, "axis": "x", "orientation": +1, "min_mm": 0.0, "max_mm": 60.0},
    6: {"servo_id": 13, "axis": "x", "orientation": -1, "min_mm": 0.0, "max_mm": 60.0},
}

GEAR_PITCH_DIAMETER = 16          # mm
COUNTS_PER_REV      = 4096
MM_PER_REV          = np.pi * GEAR_PITCH_DIAMETER
COUNTS_PER_MM       = COUNTS_PER_REV / MM_PER_REV   # ≈ 81.49 counts/mm


class CartesianHand:
    """7-DOF FT servo gripper controller."""

    def __init__(self, port: str = PORT_2, dof_config: dict = config_2,
                 control_hz: float = CONTROL_HZ, register_signal: bool = True):
        self.servo      = FtServo(port)
        self.control_hz = control_hz
        self.dof_config = dof_config
        self._servo_ids = [cfg["servo_id"] for cfg in dof_config.values()]

        self.zero_offset = np.zeros(7, dtype=int)
        self.is_zeroed   = False
        self._try_load_offsets()

        self.target = np.zeros(7, dtype=float)
        self.actual = np.zeros(7, dtype=float)
        self.lock   = threading.Lock()

        self.running = False
        self._thread = None

        self._speed  = np.full(7, DEFAULT_SPEED,  dtype=int)
        self._acc    = np.full(7, DEFAULT_ACC,    dtype=int)
        self._torque = np.full(7, DEFAULT_TORQUE, dtype=int)

        if register_signal:
            signal.signal(signal.SIGINT, self._shutdown)

    def _config_key(self) -> str:
        return "config_1" if self.dof_config is config_1 else "config_2"

    def _try_load_offsets(self):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "commons", "zero_offsets.json")
        key  = self._config_key()
        try:
            with open(path) as f:
                data = json.load(f)
            offsets = data.get(key)
            if offsets and len(offsets) == 7:
                self.zero_offset[:] = offsets
                self.is_zeroed = True
                print(f"[{key}] Loaded zero offsets from {os.path.basename(path)}")
        except FileNotFoundError:
            pass
        except Exception as e:
            print(f"[{key}] Could not load zero offsets: {e}")

    def _shutdown(self, sig, frame):
        print("\nCtrl+C caught — releasing servos...")
        try:
            self.release()
            self.servo.close()
        except Exception as e:
            print(f"Shutdown error: {e}")
        sys.exit(0)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        print("\nExiting — releasing servos...")
        try:
            self.release()
            self.servo.close()
        except Exception as e:
            print(f"Shutdown error: {e}")
        return False

    # ── Conversions ───────────────────────────────────────────────────────────

    def _counts_to_mm(self, dof_id: int, counts: int) -> float:
        if counts is None:
            return None
        orientation = self.dof_config[dof_id]["orientation"]
        return (counts - self.zero_offset[dof_id]) * orientation / COUNTS_PER_MM

    def _mm_to_counts(self, dof_id: int, val_mm: float) -> int:
        orientation = self.dof_config[dof_id]["orientation"]
        return int(self.zero_offset[dof_id] + val_mm * COUNTS_PER_MM * orientation)

    # ── set_pos ───────────────────────────────────────────────────────────────

    def set_pos(self, positions_mm, speed: int = None, acc: int = None,
                torque: int = None, wait: bool = True,
                wait_dofs: list = None, tolerance: float = 1.0, timeout: float = 5.0):
        """Command all 7 DOFs by target position vector (mm).

        Pass None for any DOF to leave its target unchanged.
        speed / acc / torque: scalar, broadcasts to all 7 DOFs.
        wait_dofs: subset of DOF ids to check; defaults to all moved DOFs.
        """
        if not self.is_zeroed:
            print("Must zero the controller first.")
            return
        if not self.running:
            self.enable()

        moved_dofs = []
        with self.lock:
            if speed  is not None: self._speed[:]  = speed
            if acc    is not None: self._acc[:]    = acc
            if torque is not None: self._torque[:] = torque
            for dof_id, val in enumerate(positions_mm):
                if val is not None:
                    clamped = max(self.dof_config[dof_id]["min_mm"],
                                  min(self.dof_config[dof_id]["max_mm"], float(val)))
                    self.target[dof_id] = clamped
                    moved_dofs.append(dof_id)

        if wait:
            check_dofs = wait_dofs if wait_dofs is not None else moved_dofs
            start = time.time()
            while time.time() - start < timeout:
                with self.lock:
                    if all(abs(self.actual[d] - self.target[d]) <= tolerance
                           for d in check_dofs):
                        return
                time.sleep(0.02)
            print(f"Warning: set_pos timed out for DOFs {check_dofs}")

    # ── Enable / Release ──────────────────────────────────────────────────────

    def enable(self):
        if self.running:
            return
        self.servo.enable_torques(self._servo_ids, True)
        self.running = True
        self._thread = threading.Thread(target=self._control_loop, daemon=True)
        self._thread.start()

    def release(self, dof_ids: list = None):
        """Disable torque. Pass DOF ids or None for all."""
        time.sleep(0.1)
        if dof_ids is None:
            self.servo.enable_torques(self._servo_ids, False)
        else:
            sids = [self.dof_config[d]["servo_id"] for d in dof_ids]
            self.servo.enable_torques(sids, False)
        print("Torque disabled.")

    # ── Monitoring ────────────────────────────────────────────────────────────

    def publish(self, hz: float = 1, dof_ids: list = None):
        """Continuously print DOF state. Ctrl+C to stop."""
        ids = dof_ids if dof_ids is not None else list(self.dof_config.keys())
        print(f"Publishing at {hz}Hz — Ctrl+C to stop")
        print(f"{'DOF':<7} {'actual(mm)':>10} {'target(mm)':>10} {'load':>6} {'volt':>6} {'temp':>6}")
        print("-" * 58)
        try:
            while True:
                for dof_id in ids:
                    sid    = self.dof_config[dof_id]["servo_id"]
                    counts = self.servo.read_position(sid)
                    load   = self.servo.read_load(sid)
                    volt   = self.servo.get_voltage(sid)
                    temp   = self.servo.get_temperature(sid)
                    actual_mm  = self._counts_to_mm(dof_id, counts)
                    actual_str = f"{actual_mm:.1f}" if actual_mm is not None else "?"
                    with self.lock:
                        target_str = f"{self.target[dof_id]:.1f}" if self.running else "N/A"
                    volt_v = f"{volt/10.0:.1f}V" if volt is not None else "?"
                    print(f"{dof_id:<7} {actual_str:>10} {target_str:>10} {str(load):>6} {volt_v:>6} {str(temp):>6}")
                print("-" * 58)
                time.sleep(1 / hz)
        except KeyboardInterrupt:
            print("Stopped.")

    # ── Control Loop ──────────────────────────────────────────────────────────

    def _control_loop(self):
        while self.running:
            t0 = time.time()

            for dof_id in range(7):
                sid    = self.dof_config[dof_id]["servo_id"]
                counts = self.servo.read_position(sid)
                if counts is not None:
                    with self.lock:
                        self.actual[dof_id] = self._counts_to_mm(dof_id, counts)

            with self.lock:
                target_mm = self.target.copy()
                speeds    = self._speed.copy()
                accs      = self._acc.copy()
                torques   = self._torque.copy()

            sids   = [self.dof_config[d]["servo_id"] for d in range(7)]
            counts = [self._mm_to_counts(d, float(target_mm[d])) for d in range(7)]

            if np.all(speeds == speeds[0]) and np.all(accs == accs[0]) and np.all(torques == torques[0]):
                self.servo.set_positions(sids, counts, int(speeds[0]), int(accs[0]), int(torques[0]))
            else:
                for dof_id in range(7):
                    self.servo.set_position(sids[dof_id], counts[dof_id],
                                            int(speeds[dof_id]), int(accs[dof_id]), int(torques[dof_id]))

            elapsed = time.time() - t0
            time.sleep(max(0.0, 1.0 / self.control_hz - elapsed))


if __name__ == "__main__":
    from tasks.zeroing import zero_all
    from tasks.demo import demo
    from tasks.caps_contact_based import caps_open

    with CartesianHand(PORT_1, config_2) as hand:
        zero_all(hand, zero_torque=50)
        # caps_open(hand, cap_offset=25, num_revs=2, squeeze_torque=80)
        # demo(hand)

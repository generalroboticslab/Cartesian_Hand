"""What a hand is, which hands exist, and how to drive one.

Read it top to bottom. The tunables come first, because they are the only part
most people ever edit: travel limits, motion gains, the creep torque zeroing
presses with. Then the types those numbers fill in, then the hand definitions
themselves, then CartesianHand, which turns any of it into servo traffic.

This was two files, `hands.py` and `hand.py`, split description from behaviour.
The names differed by one character and nothing else told you which was which.
The split was meant to keep the digital twin from importing a serial driver it
cannot build, but that was never the reason it worked: `servo.open_driver`
imports the compiled extension lazily, inside the call, so importing this module
touches no hardware. The twin gets HandConfig.contract() and fingerprint()
without a serial port either way.

CartesianHand holds no hardware constants of its own. Everything dimensional is
in the tables at the top, so the same class drives any hand defined below, and
the same code path drives the mock backend with no hardware attached.
"""

import hashlib
import json
import os
import signal
import sys
import threading
import time
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime

import numpy as np

from .servo import MockServo, open_driver


# ══ Tunables ══════════════════════════════════════════════════════════════════
#
# Everything below is meant to be edited. Nothing downstream carries a hardware
# constant of its own, so retuning a gear ratio or a travel limit never means
# opening control code.

AXES = ("x", "y", "z")

DEFAULT_HAND = "hand_2"

# Physical layout shared by every hand built so far: a base gripper (DOF 0-2),
# a z stage (DOF 3), and an auxiliary gripper (DOF 4-6). DOF index is the
# controller's addressing scheme; servo_id is what sits on the serial bus.
#
# This ordering is the one the simulation model must match, not the other way
# round. See asset/duke_v2/cartesian_hand_v2/ in the legged_env_v2 repo.
LAYOUT = [
    ("y", -1, "base parallel actuation"),
    ("x", +1, "base left finger"),
    ("x", -1, "base right finger"),
    ("z", -1, "vertical translation"),
    ("y", -1, "aux parallel actuation"),
    ("x", +1, "aux left finger"),
    ("x", -1, "aux right finger"),
]

# Travel per DOF, from the v2 CAD. A joint limit belongs to the model, not to
# the build: hand_1 and hand_2 are two physical realizations of the same design,
# so they share this table the way they share LAYOUT. Give a hand its own only
# if a measurement forces it apart.
#
# These are not all the same number. The jaws and the z stage run on 50mm rails
# and the fingers on 55mm, so the single 60.0 that used to be stamped across all
# seven was wrong on every DOF -- over-travel on all of them, by 10mm on the
# jaws and z. The far end of each rail is open by design: drive past it and the
# carriage leaves the slider and the servo spins free. This table is the only
# thing that stops that.
#
# Still CAD, not calipers. The one measurement that exists disagrees: DOF 0 on
# hand_2 stalled 29.8mm from its zero, against the 50 below. It came from a
# `travel` task that took six of seven carriages off their rails before being
# deleted, so it is one datum from a run that was itself going wrong -- enough
# to distrust the 50, not enough to replace it. Measure and type the numbers in.
#
# One motor turn is 4096 counts, and at the derived 81.487 counts/mm that is
# 50.27mm. Any DOF above it makes the encoder angle ambiguous, and
# _reconcile_offsets tests `(span < turn).all()` -- so a single finger at 55
# keeps all seven in the branch that cannot recover saved offsets after a power
# cycle. Bringing the fingers under 50.27 would buy that back.
STANDARD_TRAVEL = [50.0, 55.0, 55.0, 50.0, 50.0, 55.0, 55.0]

# The z stage lifts the aux gripper against gravity, so it cannot run at the
# torque the six horizontal DOFs are happy with. Bisected on hand_2, lifting
# 30mm -> 35mm and measuring travel after 3s:
#
#     torque  150   200   250   300   350
#     moved  1.50  4.54  4.54  4.54  4.53   (mm, of 5.0 commanded)
#
# The cliff is sharp: 150 stalls outright, 200 tracks fully, and nothing above
# 200 helps. 300 is the measured floor plus margin, because the point of the
# stage is to lift the aux gripper while it is holding something, and the
# bisect above was run unloaded.
#
# Pressing *down* at 50 is fine and tasks rely on it, so this is a floor for
# the lifting direction, not a correction to the whole axis.
STANDARD_TORQUE = [50, 50, 50, 300, 50, 50, 50]

# Creep torque used by zeroing. Separate from the motion gains above because
# zeroing presses each DOF into its hard stop and a stall is the signal, not a
# fault — too much torque binds before the stop, too little stalls short of it.
#
# One table for every hand, deliberately: a per-hand table is a second set of
# numbers to keep measured, and the whole point of seeking a hard stop is that
# it is the same physical feature on both. So the table has to cover the
# stiffer hand, and hand_1 is stiffer than hand_2.
#
# Previous values were [50, 30, 30, 150, 50, 30, 30]. Every one of them was
# bisected against a stall detector that could not tell a slow creep from a
# stop (see wait_for_stall_counts), so the observations they rest on are
# suspect — in particular "the fingers stall short at 50 and reach the end at
# 30", which is the shape of a detector false positive, not of a mechanism that
# binds harder when pushed harder. Doubled from those.
#
# The cost of raising these is the force each DOF ends up pressing into its
# hard stop with, on a printed rack. [200, 100, 100, 500, 200, 100, 100] and
# then z at 350 both seeked audibly hard, so this is the walk back down. z is
# bracketed from both sides now: 350 too hard, 200 too weak, 250 here. It sits
# above the six horizontal DOFs because it carries the aux gripper's weight, but
# below STANDARD_TORQUE's 300: that number is the floor for *lifting* the stage,
# and zeroing presses it down into its stop with gravity helping.
# If a joint still sounds loaded at the end of a seek, come down further.
ZEROING_TORQUE = [150, 50, 50, 250, 150, 50, 50]

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

# Consecutive control steps in which no servo answered before the loop gives up.
# At the default 50Hz this is half a second: long enough to ride out dropped
# frames, short enough that a hand does not spend a second being commanded by a
# loop reading a state vector that stopped updating.
MAX_SILENT_STEPS = 25


class NotZeroedError(RuntimeError):
    """Raised when a motion is commanded before zero offsets are known."""


# ══ Types ═════════════════════════════════════════════════════════════════════

@dataclass
class Dof:
    """One degree of freedom: which servo drives it, and how counts map to mm."""
    servo_id: int
    axis: str
    orientation: int          # +1 or -1: servo count direction vs. positive mm
    min_mm: float = 0.0
    max_mm: float = 60.0
    label: str = ""

    def __post_init__(self):
        if self.axis not in AXES:
            raise ValueError(f"axis must be one of {AXES}, got {self.axis!r}")
        if self.orientation not in (1, -1):
            raise ValueError(f"orientation must be +1 or -1, got {self.orientation}")
        if self.min_mm >= self.max_mm:
            raise ValueError(f"min_mm {self.min_mm} must be < max_mm {self.max_mm}")


@dataclass
class Motion:
    """Servo motion defaults. Torque, speed and acc are raw servo units.

    Each gain is a scalar or a per-DOF sequence of length n_dof. A scalar means
    "same for every DOF" and broadcasts; a sequence lets one axis differ, which
    the z stage needs because it is the only DOF carrying a gravity load.

    Stored as plain ints/lists rather than arrays so the config still
    round-trips through JSON in to_dict/from_dict. CartesianHand broadcasts
    them to arrays once at construction.

    Differing gains are free. SyncWritePosEx carries Speed[]/ACC[]/Torque[] as
    per-servo arrays, because a sync-write is one broadcast packet in which
    each servo reads its own slice, so seven different torques and seven
    identical ones are the same packet and the same time on the wire.
    """
    control_hz: float = 50.0
    torque: object = 50       # 0-1000, scalar or per-DOF sequence
    speed: object = 300
    acc: object = 25


@dataclass
class Geometry:
    """Counts-to-mm conversion.

    counts_per_mm is derived from the rack pitch diameter, but a real gear train
    has backlash and a real pitch diameter is not the nominal one. Set
    counts_per_mm explicitly to override the derived value after measuring a
    known travel distance.
    """
    gear_pitch_diameter_mm: float = 16.0
    counts_per_rev: int = 4096
    counts_per_mm: float = None      # None: derive from the two fields above

    def __post_init__(self):
        if self.counts_per_mm is None:
            if self.gear_pitch_diameter_mm <= 0 or self.counts_per_rev <= 0:
                raise ValueError(
                    f"cannot derive counts_per_mm from pitch diameter "
                    f"{self.gear_pitch_diameter_mm} and {self.counts_per_rev} "
                    f"counts/rev; both must be positive")
            self.counts_per_mm = self.counts_per_rev / (np.pi * self.gear_pitch_diameter_mm)
        # These tables are meant to be edited by hand, and a zero or NaN here
        # would not fail until it had already been divided into a servo command.
        if not np.isfinite(self.counts_per_mm) or self.counts_per_mm <= 0:
            raise ValueError(f"counts_per_mm must be finite and positive, "
                             f"got {self.counts_per_mm}")


@dataclass
class HandConfig:
    """Full description of one physical hand."""
    name: str
    port: str
    dofs: list                                   # list[Dof], index == dof id
    motion: Motion = field(default_factory=Motion)
    geometry: Geometry = field(default_factory=Geometry)

    def __post_init__(self):
        ids = [d.servo_id for d in self.dofs]
        if len(set(ids)) != len(ids):
            raise ValueError(f"[{self.name}] duplicate servo_id in {ids}")
        # Motion cannot check its own vector lengths: it does not know n_dof.
        # Catch a wrong-length gain here rather than at the first servo write.
        for gain in ("torque", "speed", "acc"):
            self.gain_vector(gain)

    def gain_vector(self, gain: str) -> np.ndarray:
        """One motion gain as a per-DOF int array, broadcasting a scalar."""
        value = getattr(self.motion, gain)
        arr = np.asarray(value, dtype=int)
        if arr.ndim == 0:
            return np.full(self.n_dof, arr, dtype=int)
        if arr.shape != (self.n_dof,):
            raise ValueError(
                f"[{self.name}] motion.{gain} has {arr.shape[0]} values, "
                f"expected a scalar or {self.n_dof}")
        return arr.copy()

    def variant(self, **changes) -> "HandConfig":
        """Copy with fields replaced, for one-off overrides like a different port."""
        return replace(self, **changes)

    # ── Introspection ─────────────────────────────────────────────────────────

    @property
    def n_dof(self) -> int:
        return len(self.dofs)

    @property
    def servo_ids(self) -> list:
        return [d.servo_id for d in self.dofs]

    @property
    def lower(self) -> np.ndarray:
        return np.array([d.min_mm for d in self.dofs], dtype=float)

    @property
    def upper(self) -> np.ndarray:
        return np.array([d.max_mm for d in self.dofs], dtype=float)

    @property
    def orientations(self) -> np.ndarray:
        return np.array([d.orientation for d in self.dofs], dtype=int)

    def dofs_on(self, axis: str) -> list:
        """DOF ids driving a given axis. Lets tasks say 'the y jaws' not '[0, 4]'."""
        return [i for i, d in enumerate(self.dofs) if d.axis == axis]

    def clamped_mm(self, dof_id: int, want: float) -> float:
        """A goal in mm, bounded by the DOF's travel table.

        Tasks ask for the clearance they need and get it back capped. They must
        never use `max_mm` itself: the table is CAD, and the one real measurement
        contradicts it -- DOF 0 on hand_2 stalled 29.8mm from zero against the
        table's 50.0. The far end of every rail is open by design, so commanding
        the table's value walks the carriage off its slider, and a clamp cannot
        help there because the clamp uses the same wrong number. Asking for a
        clearance that happens to be under the limit is safe under a right table
        and a wrong one both.

        Lives on the config rather than in a task because every task needs it and
        the second copy is the one that gets it wrong.
        """
        d = self.dofs[dof_id]
        return min(max(want, d.min_mm), d.max_mm)

    def __getitem__(self, dof_id: int) -> Dof:
        return self.dofs[dof_id]

    def __len__(self) -> int:
        return len(self.dofs)

    # ── Unit conversion ───────────────────────────────────────────────────────

    def counts_to_mm(self, dof_id: int, counts, zero_offset) -> float:
        if counts is None:
            return None
        return ((counts - zero_offset) * self.dofs[dof_id].orientation
                / self.geometry.counts_per_mm)

    def mm_to_counts(self, dof_id: int, mm: float, zero_offset) -> int:
        return int(zero_offset
                   + mm * self.geometry.counts_per_mm * self.dofs[dof_id].orientation)

    # Whole-vector forms of the two above, for the control loop. The hand is a
    # robot with a joint vector, so the loop converts all DOFs at once rather
    # than looping. The scalar versions stay for callers holding one DOF.

    def counts_to_mm_all(self, counts, zero_offsets) -> np.ndarray:
        return ((np.asarray(counts, dtype=float) - np.asarray(zero_offsets, dtype=float))
                * self.orientations / self.geometry.counts_per_mm)

    def mm_to_counts_all(self, mm, zero_offsets) -> np.ndarray:
        return (np.asarray(zero_offsets, dtype=float)
                + np.asarray(mm, dtype=float)
                * self.geometry.counts_per_mm * self.orientations).astype(int)

    def clamp(self, positions_mm) -> np.ndarray:
        return np.clip(np.asarray(positions_mm, dtype=float), self.lower, self.upper)

    # ── Normalized action space (the sim/real contract) ───────────────────────
    #
    # Simulation policies are trained in a unitless [-1, 1] box. Hardware speaks
    # millimetres bounded by per-DOF travel. These two functions are the whole
    # translation, so a policy trained against any sim with matching DOF ordering
    # runs here without knowing the hand's dimensions.

    def normalize(self, positions_mm) -> np.ndarray:
        """mm -> [-1, 1], per DOF."""
        span = self.upper - self.lower
        return 2.0 * (np.asarray(positions_mm, dtype=float) - self.lower) / span - 1.0

    def denormalize(self, action) -> np.ndarray:
        """[-1, 1] -> mm, per DOF. Out-of-range actions are clipped rather than
        rejected: a saturating policy should press against the joint limit."""
        span = self.upper - self.lower
        a = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        return self.lower + (a + 1.0) * 0.5 * span

    # ── Policy contract ───────────────────────────────────────────────────────
    #
    # A policy tuned in a digital twin transfers only if the twin and the real
    # hand agree on DOF count, DOF ordering, per-DOF axis and travel, and the
    # control rate. Those fields, and only those, form the contract: a policy
    # works in normalized units, so it never sees the port, the servo IDs, or
    # the count direction. Export the contract for the twin to build against,
    # and compare fingerprints before running a policy on hardware.

    def contract(self) -> dict:
        """The sim/real interface, without hardware plumbing.

        counts_per_mm is deliberately excluded even though it changes what the
        servos are told. It is a per-machine calibration: two units with the
        same 50mm of travel and slightly different gear trains need different
        counts_per_mm to both actually reach 50mm. Folding it into the contract
        would make correctly calibrated hands look incompatible, which is
        backwards. A wrong counts_per_mm is a calibration fault, and the
        fingerprint is not a calibration check.
        """
        return {
            "n_dof": self.n_dof,
            "control_hz": self.motion.control_hz,
            "dofs": [{"axis": d.axis, "min_mm": d.min_mm, "max_mm": d.max_mm}
                     for d in self.dofs],
        }

    def fingerprint(self) -> str:
        """Stable short hash of contract(). Stamp it on anything a twin produces."""
        blob = json.dumps(self.contract(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()[:12]

    # ── Serialization ─────────────────────────────────────────────────────────
    #
    # This file is the source of truth for authoring, but the twin cannot import
    # it, so the full config round-trips through plain dicts as well.

    def to_dict(self) -> dict:
        d = asdict(self)
        d["fingerprint"] = self.fingerprint()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "HandConfig":
        d = dict(d)
        d.pop("fingerprint", None)
        return cls(
            name=d["name"],
            port=d["port"],
            dofs=[Dof(**x) for x in d["dofs"]],
            motion=Motion(**d.get("motion", {})),
            geometry=Geometry(**d.get("geometry", {})),
        )


# ══ The hands ═════════════════════════════════════════════════════════════════

def standard_dofs(first_servo_id: int, travel_mm=STANDARD_TRAVEL) -> list:
    """The 7-DOF LAYOUT above, with servo IDs numbered consecutively.

    travel_mm is per DOF, in LAYOUT order. Checked here because these tables are
    meant to be edited by hand and a short one would otherwise silently give the
    trailing DOFs whatever Dof's own default happens to be.
    """
    if len(travel_mm) != len(LAYOUT):
        raise ValueError(f"travel_mm needs {len(LAYOUT)} values, got {len(travel_mm)}")
    return [
        Dof(servo_id=first_servo_id + i, axis=axis, orientation=orientation,
            min_mm=0.0, max_mm=travel_mm[i], label=label)
        for i, (axis, orientation, label) in enumerate(LAYOUT)
    ]


HAND_1 = HandConfig(
    name="hand_1",
    port="/dev/ttyACM0",
    dofs=standard_dofs(first_servo_id=0),
    motion=Motion(control_hz=50, torque=STANDARD_TORQUE, speed=300, acc=25),
    geometry=Geometry(gear_pitch_diameter_mm=16.0, counts_per_rev=4096),
)

HAND_2 = HandConfig(
    name="hand_2",
    # by-id, not /dev/ttyACM1: ACM numbers are handed out in plug order, so
    # whichever hand enumerates first takes ACM0 and a fixed number silently
    # points at the wrong hand. This path is tied to the adapter's serial.
    port="/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AE6085950-if00",
    dofs=standard_dofs(first_servo_id=7),
    motion=Motion(control_hz=50, torque=STANDARD_TORQUE, speed=300, acc=25),
    geometry=Geometry(gear_pitch_diameter_mm=16.0, counts_per_rev=4096),
)

HANDS = {h.name: h for h in (HAND_1, HAND_2)}


def get_hand(name: str = None) -> HandConfig:
    """Look up a hand by name, defaulting to DEFAULT_HAND."""
    name = name or DEFAULT_HAND
    if name not in HANDS:
        raise KeyError(f"Unknown hand {name!r}. Available: {sorted(HANDS)}")
    return HANDS[name]


def hand_names() -> list:
    return sorted(HANDS)


# ══ Zero-offset persistence ═══════════════════════════════════════════════════
#
# Offsets are measured by the zeroing routine at runtime, not authored, so
# unlike the tables at the top of this file they live in a generated data file.
# Entries are keyed by hand name. An earlier version keyed them by
# `dof_config is config_1` object identity, which mislabelled any hand built
# from a copied dict.

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


# The FT servo driver's Goal_Acc register is one byte (HLSCL.h WritePosEx),
# and the C++ wrapper truncates int -> u8 with no bounds check
# (ft_servo_driver.hpp set_positions: `std::vector<u8> acc_buf(acc.begin(), ...)`),
# so an out-of-range acc silently wraps mod 256 instead of erroring. Caught here,
# at the only two places acc reaches the driver, rather than in the control loop
# where raising would trip the loop's error handler and drop torque.
def _check_acc_range(acc):
    if np.any((np.atleast_1d(np.asarray(acc)) < 0) | (np.atleast_1d(np.asarray(acc)) > 255)):
        raise ValueError(
            f"acc={acc} out of range: the driver's Goal_Acc register is a single "
            f"byte (0-255). A larger value silently wraps mod 256 in the C++ "
            f"driver instead of erroring.")


# ══ The controller ════════════════════════════════════════════════════════════

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
        self._silent_steps = 0

        if load_calibration:
            self.load_calibration()

        if register_signal:
            signal.signal(signal.SIGINT, self._on_sigint)

    # ── Calibration ───────────────────────────────────────────────────────────

    def load_calibration(self) -> bool:
        offsets = load_offsets(self.calib_key, self.n_dof)
        if offsets is None:
            return False
        offsets = self._reconcile_offsets(offsets)
        if offsets is None:
            return False
        self.zero_offset[:] = offsets
        self.is_zeroed = True
        print(f"[{self.name}] loaded zero offsets ({offset_timestamp(self.calib_key)})")
        return True

    # A servo reports (turns since power-up * counts_per_rev) + angle within the
    # current turn. Only the second term is real: it comes off an encoder and is
    # valid the instant power arrives. The turn count starts at zero every
    # power-up, so a saved offset is stale by a whole number of turns, and
    # offset % counts_per_rev is the part of it that survives.
    #
    # That angle is enough to place the joint exactly, but only while its travel
    # is shorter than one turn. Then there is one position in range matching the
    # angle and it can be computed. Once travel exceeds a turn, two positions
    # genuinely share an angle, the servo cannot tell them apart, and neither can
    # we -- so the honest move is to refuse and re-zero rather than pick one.

    def _reconcile_offsets(self, offsets):
        """Re-express saved offsets in the servo's current turn accumulator.

        Returns usable offsets, or None if the calibration cannot be trusted and
        the hand needs zeroing again.
        """
        cfg = self.config
        turn = cfg.geometry.counts_per_rev
        span = (cfg.upper - cfg.lower) * cfg.geometry.counts_per_mm

        raw = self.servo.read_positions(cfg.servo_ids)
        if any(c is None for c in raw):
            # No reading means no check. Not a reason to refuse: a hand we
            # cannot read is a hand we cannot move either, and the first real
            # command will fail on its own with a clearer message than this one.
            print(f"[{self.name}] could not read every servo; using saved "
                  f"offsets unverified")
            return offsets
        raw = np.array(raw, dtype=int)

        if (span < turn).all():
            # How far past its zero each joint sits, in counts, from the encoder
            # angle alone. Whatever the turn counter contributed -- to the
            # reading and to the saved offset alike -- is a whole number of
            # turns and vanishes in the modulo.
            k = ((raw - offsets) * cfg.orientations) % turn
            if (k > span).any():
                out = [d for d in range(self.n_dof) if k[d] > span[d]]
                print(f"[{self.name}] DOFs {out} sit outside their travel on any "
                      f"turn, so the saved offsets do not describe this hand.")
                return None
            rebased = raw - k * cfg.orientations
            shifted = [d for d in range(self.n_dof) if rebased[d] != offsets[d]]
            if shifted:
                print(f"[{self.name}] turn accumulator moved since zeroing; "
                      f"rebased DOFs {shifted}")
            return rebased

        # Travel exceeds one turn, so the angle alone is ambiguous and there is
        # nothing to recompute. All we can do is notice when the saved offsets
        # are obviously stale, which they are whenever a joint reads outside the
        # travel it is declared to have.
        mm = cfg.counts_to_mm_all(raw, offsets)
        bad = [d for d in range(self.n_dof)
               if not (cfg[d].min_mm - 1.0 <= mm[d] <= cfg[d].max_mm + 1.0)]
        if bad:
            print(f"[{self.name}] saved offsets put DOFs {bad} at "
                  f"{[round(mm[d], 1) for d in bad]}mm, outside their travel. "
                  f"The turn accumulator has moved since zeroing.\n"
                  f"  Travel is {span.max() / turn:.2f} turns, longer than one, "
                  f"so the offsets cannot be recovered by arithmetic.\n"
                  f"  Run: python -m cartesian_hand zeroing --hand {self.name}")
            return None
        return offsets

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
            # Commanding a subset is what exposes it: the DOFs left out of the
            # call are not left where they are, they are left at zero.
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
                    _check_acc_range(acc)
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

        Takes either a full vector or a `{dof_id: mm}` mapping. Tasks move two
        or three DOFs out of seven, and a positional list of mostly Nones hides
        the meaning in the index of the one entry that is not:

            hand.set_pos({AUX_JAW: 12.0, Z: 20.0})
            hand.set_pos([None, None, None, 20.0, 12.0, None, None])

        A None in the vector form leaves that DOF's target unchanged, as does
        omitting it from the mapping. speed/acc/torque apply only to the DOFs
        actually commanded in this call, so a gain set for a squeezing jaw
        survives a later move of a different DOF.
        """
        self.require_zeroed()
        if not self.running:
            self.enable()

        if isinstance(positions_mm, dict):
            vector = [None] * self.n_dof
            for dof_id, mm in positions_mm.items():
                vector[dof_id] = mm
            positions_mm = vector

        moved = [d for d, v in enumerate(positions_mm) if v is not None]
        # Each gain is scalar-or-per-DOF, matching Motion. A scalar broadcasts
        # to every moved DOF; a sequence of length n_dof lets one axis differ.
        # An earlier version of this code only took a scalar, and a vector
        # silently broke set_pos. Tasks that pass m.torque get the per-DOF
        # value Motion stores.
        def expand(value):
            if value is None:
                return None
            arr = np.asarray(value)
            if arr.ndim == 0:
                return int(arr)
            if arr.shape != (self.n_dof,):
                raise ValueError(
                    f"per-DOF gain has {arr.shape[0]} values, expected a "
                    f"scalar or {self.n_dof}")
            return arr.astype(int)
        speed_v  = expand(speed)
        acc_v    = expand(acc)
        torque_v = expand(torque)

        with self.lock:
            for d in moved:
                cfg = self.config[d]
                self.target[d] = min(cfg.max_mm, max(cfg.min_mm, float(positions_mm[d])))
                if speed_v is not None:
                    self._speed[d] = int(speed_v) if np.isscalar(speed_v) else int(speed_v[d])
                if acc_v is not None:
                    _check_acc_range(acc_v)
                    self._acc[d] = int(acc_v) if np.isscalar(acc_v) else int(acc_v[d])
                if torque_v is not None:
                    self._torque[d] = int(torque_v) if np.isscalar(torque_v) else int(torque_v[d])

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

    def move(self, action, **kwargs) -> bool:
        """Command a normalized [-1, 1] action vector. The policy-facing move."""
        return self.set_pos(self.config.denormalize(action), **kwargs)

    def run_program(self, motions, timeout: float = 180.0):
        """Drive a `motions.Motions` program to completion. The tensor peer of set_pos.

        The control loop keeps doing the bus traffic; this writes the same two
        standing registers `set_pos` writes -- `target` and `_torque` -- once
        per tick, from the program instead of from an argument. So a program is
        not a second way to drive the hand, it is a different author of the same
        commands, and `_step` needs no changes to support it.

        Positions are copied out per tick rather than aliased into a torch view.
        The copy is seven floats at 50Hz, which is nothing, and it removes the
        whole class of bug where the bus thread writes `actual` underneath a
        tensor the engine is mid-read of.

        Returns the Motions, whose `outcome` says how every row ended. Raises on
        timeout, because a program that has not finished is not a result.
        """
        import torch                 # only programs need torch; set_pos does not

        if not self.running:
            raise RuntimeError(f"[{self.name}] run_program needs a running control loop")
        self.require_zeroed()

        read = lambda: torch.from_numpy(self.positions.astype(np.float32))[None, :]
        with self.lock:
            torque0 = torch.from_numpy(self._torque.astype(np.float32))[None, :]
        motions.start(read(), torque0)

        period = 1.0 / self.control_hz
        next_tick = time.time()
        deadline = next_tick + timeout
        while not motions.done():
            goal, torque = motions.step_once(read())
            with self.lock:
                # Clamped, exactly as set_pos clamps: the travel table is the
                # only thing stopping a carriage from being driven off the open
                # end of its rail, and a program is no more trustworthy about
                # that than a task calling set_pos by hand.
                self.target[:] = self.config.clamp(goal[0].numpy())
                self._torque[:] = torque[0].numpy()

            if self._loop_error is not None:
                raise RuntimeError(f"[{self.name}] control loop died: {self._loop_error}")
            now = time.time()
            if now > deadline:
                raise RuntimeError(
                    f"[{self.name}] program did not finish in {timeout}s "
                    f"(step {motions.step.tolist()} of {motions.K})")
            # Absolute deadline, not `t0 + period - elapsed`: measuring from the
            # top of each tick adds the loop's own overshoot to every period, so
            # the tick rate drifts slow and a timeout denominated in ticks means
            # a different wall-clock duration on every machine. Resync rather
            # than fire a burst of catch-up ticks if a tick ran long.
            next_tick += period
            if next_tick < now:
                next_tick = now
            else:
                time.sleep(next_tick - now)
        return motions

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
            self._silent_steps = 0
            counts = np.array([c if c is not None else 0 for c in raw], dtype=float)
            mm = cfg.counts_to_mm_all(counts, self.zero_offset)
            with self.lock:
                self.actual[ok] = mm[ok]
        else:
            # Nobody answered. Keeping the stale vector is right for one sweep
            # -- interference drops frames -- but a bus that has gone away never
            # answers again, and the loop would keep writing goals into nothing
            # while `actual` sits frozen at the last good reading. Frozen reads
            # as a hand holding position, so this never surfaces on its own.
            # Raising hands it to the loop's error path, which drops torque.
            self._silent_steps += 1
            if self._silent_steps >= MAX_SILENT_STEPS:
                raise RuntimeError(
                    f"[{self.name}] no servo answered for {self._silent_steps} "
                    f"consecutive reads; the bus is gone")

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
    config = get_hand(hand)
    if port:
        config = config.variant(port=port)
    return CartesianHand(config, mock=mock, **kwargs)

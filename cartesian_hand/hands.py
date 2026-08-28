"""What a hand is, and which hands exist. Edit this file to add or retune one.

The top half is the vocabulary: a DOF table, motion defaults, and the geometry
that turns servo counts into millimetres. The bottom half is the actual hand
definitions. They live together because there is only ever one reason to open
this file, and splitting the types away from the values it describes meant
reading two files to answer one question.

Everything dimensional lives here. Nothing downstream carries a hardware
constant of its own, so adding a hand or retuning a gear ratio never means
touching control code.
"""

import hashlib
import json
from dataclasses import asdict, dataclass, field, replace

import numpy as np

AXES = ("x", "y", "z")

DEFAULT_HAND = "hand_2"


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
    """Servo motion defaults. Torque, speed and acc are raw servo units."""
    control_hz: float = 50.0
    torque: int = 50          # 0-1000
    speed: int = 300
    acc: int = 25


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
        # This file is meant to be edited by hand, and a zero or NaN here would
        # not fail until it had already been divided into a servo command.
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

    def dofs_on(self, axis: str) -> list:
        """DOF ids driving a given axis. Lets tasks say 'the y jaws' not '[0, 4]'."""
        return [i for i, d in enumerate(self.dofs) if d.axis == axis]

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
        same 60mm of travel and slightly different gear trains need different
        counts_per_mm to both actually reach 60mm. Folding it into the contract
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


def standard_dofs(first_servo_id: int, travel_mm: float = 60.0) -> list:
    """The 7-DOF layout above, with servo IDs numbered consecutively."""
    return [
        Dof(servo_id=first_servo_id + i, axis=axis, orientation=orientation,
            min_mm=0.0, max_mm=travel_mm, label=label)
        for i, (axis, orientation, label) in enumerate(LAYOUT)
    ]


HAND_1 = HandConfig(
    name="hand_1",
    port="/dev/ttyACM0",
    dofs=standard_dofs(first_servo_id=0),
    motion=Motion(control_hz=50, torque=50, speed=300, acc=25),
    geometry=Geometry(gear_pitch_diameter_mm=16.0, counts_per_rev=4096),
)

HAND_2 = HandConfig(
    name="hand_2",
    port="/dev/ttyACM1",
    dofs=standard_dofs(first_servo_id=7),
    motion=Motion(control_hz=50, torque=50, speed=300, acc=25),
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

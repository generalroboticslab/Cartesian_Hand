"""What a hand is: the shared tables, one dataclass, and the hands themselves.

Nothing downstream holds a hardware constant, so retuning never means opening
control code. What is per-DOF but identical on every unit (axis, count
direction, label) lives in `LAYOUT` once; what differs per unit is the short
list at the bottom.

Pure description -- no port, no threads, no I/O beyond JSON -- so a twin can
import this without compiling the servo extension. Per-DOF quantities answer as
tensors so `device=` is the only sim/hardware difference. The bus is the one
exception: `read_all` returns Python tuples, and its `None` must stay `None`
(a fabricated position reads as a large jump, the opposite of stuck).
"""

import datetime
import hashlib
import json
import math
import os
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import torch

# A gain is one number shared by every DOF, or one per DOF in LAYOUT order --
# both cost the same packet, see `HandConfig`.
Gain = int | Sequence[int]
Device = str | torch.device

# ══ Tunables ══════════════════════════════════════════════════════════════════

AXES = ("x", "y", "z")

DEFAULT_HAND = "hand_3"

# Millimetres are float32 everywhere. Positions are mm off int16 encoders, so
# nothing here needs 53 bits of mantissa, and f32 is what the sim backend wants.
MM = torch.float32

# Physical layout: base gripper (0-2), z stage (3), aux gripper (4-6). Shared,
# not per hand -- the units are two realizations of one design, so a hand
# carries only where its servo IDs start. DOF index addresses the controller;
# the servo ID addresses the bus.
#
# This order is what the sim model must match, and does: axis sequence
# y,x,x,z,y,x,x against the MJCF actuators. Do NOT re-order to
# `source/kinematics.json`'s CAD servo numbering -- a third scheme that
# transposes four DOFs.
#
# `orientation` is servo count direction against positive mm. It CANNOT fix a
# sim/real direction disagreement and must not be reached for when one appears:
# it cancels between `mm_to_counts` and `counts_to_mm`, so flipping a sign moves
# only the hardware. Cost a bench cycle on 2026-09-01.
# 2026-09-01: flipping all seven to +1 reversed five DOFs on the real hand. The
# signs below are right. "sim disagrees" into "hardware is backwards".
LAYOUT = [
    ("y", -1, "base jaw"),
    ("x", -1, "base left finger"),
    ("x", -1, "base right finger"),
    ("z", -1, "vertical translation"),
    ("y", -1, "aux jaw"),
    ("x", -1, "aux left finger"),
    ("x", -1, "aux right finger"),
]

# Read once, so nothing below re-splits the table. Checked here rather than in a
# constructor: LAYOUT is hand-edited and shared, so a bad axis letter is a bug in
# this file that should fail at import, not per hand at first use.
AXIS = tuple(a for a, _o, _l in LAYOUT)
ORIENTATION = tuple(o for _a, o, _l in LAYOUT)
LABELS = tuple(l for _a, _o, l in LAYOUT)
N_DOF = len(LAYOUT)

for _i, (_a, _o, _l) in enumerate(LAYOUT):
    if _a not in AXES:
        raise ValueError(f"LAYOUT[{_i}] axis must be one of {AXES}, got {_a!r}")
    if _o not in (1, -1):
        raise ValueError(f"LAYOUT[{_i}] orientation must be +1 or -1, got {_o}")

# Names for the LAYOUT indices above: tasks address DOFs by mechanical role.
# Here rather than in a task module because they are a reading of LAYOUT and it
# is directly above -- a role map that has drifted from the layout is a mirrored
# policy that still looks plausible on a symmetric gripper.
BASE_JAW, BASE_LEFT, BASE_RIGHT = 0, 1, 2
Z = 3
AUX_JAW, AUX_LEFT, AUX_RIGHT = 4, 5, 6

BASE_FINGERS = [BASE_LEFT, BASE_RIGHT]
AUX_FINGERS = [AUX_LEFT, AUX_RIGHT]

# Travel per DOF, from the v2 CAD. Shared like LAYOUT: a joint limit belongs to
# the design, not the build. Zero is the hard stop zeroing finds; the far end of
# every rail is OPEN, so driving past it walks the carriage off its slider. This
# table is the only thing preventing that -- hence `clamped_mm`, and hence
# nothing should ever command the limit itself.
#
# Tuples, not lists: a list is what dataclasses refuses as a field default.
STANDARD_TRAVEL = (50.0, 55.0, 55.0, 50.0, 50.0, 55.0, 55.0)

# Profile speed per DOF, counts/s. Shared like the travel table -- these are
# properties of the design, not of a build.
#
# **z is 500 because it will not break its own static friction below that, not
# because it needs to be quick.** The servo's `torque` register is a force CAP;
# the effort it actually develops follows position error, so a slow profile
# keeps the commanded setpoint near the stalled joint, the error stays small,
# and the force never rises enough to start the stage moving. Nudging it by hand
# starts it every time (bench, 2026-09-02). Raising the torque cap does nothing
# for this -- 800 was never approached.
#
# 500 is the value the previous implementation used to back off the stop. Its
# *seek* used 60 and warned that a fast creep overshoots the stop and climbs a
# gear tooth; that per-phase split cannot be expressed here, because `Motions`
# carries goal and torque but no speed. If the z stop starts reading long, this
# is the first number to suspect.
STANDARD_SPEED = (300, 300, 300, 500, 300, 300, 300)

# Acceleration ramp, 0-255, shared by every DOF.
#
# 200, not the 25 this shipped with, and it is the other half of the stiction
# story above: `acc` governs how fast the setpoint pulls away from a joint at
# rest, so it -- not the top speed -- is what builds the position error that
# breaks stiction in the first place. Raising `speed` alone to 1200 changed
# nothing on the bench, because at acc=25 the profile is still ramping. Both the
# seek and the park of the previous implementation used 200.
STANDARD_ACC = 200

# ── How to pick `torque_min_to_move` ──────────────────────────────────────────
#
# `torque_min_to_move` is a FLOOR: below it the joint does not move. Pure
# friction on the six horizontal DOFs, friction plus gravity on z, the only axis
# under load. Bisect it by commanding a 5mm move and reading travel after 3s.
#
# It is also what zeroing seeks its hard stops at -- see `tasks/zero.py`. A
# probe wants the lightest push that still travels, because force left over at
# contact deflects the rack instead of stopping the carriage, and the stop is
# recorded that far past where it is. So this number is load-bearing twice: too
# low and a seek stalls mid rail and records that as the datum; too high and
# every stop is recorded long by the deflection. Bisect it, do not pad it.
#
# It is friction, and friction is per unit, so every lab hand below states its
# own. The default is a starting point for a new build, not a measurement: the
# un-bisected values hand_1 and hand_3 run, known to move real hardware with
# margin. A wrong floor fails silently (a stop recorded mid rail), so bisect
# before trusting a zero.
STANDARD_TORQUE_MIN_TO_MOVE = (250, 250, 250, 800, 250, 250, 250)

# Project root, beside the checkout rather than inside the package directory:
# a calibration costs a bench cycle to reproduce and `pip install -e .` wipes
# the package dir. Gitignored -- the numbers describe one physical machine.
CALIB_PATH = os.environ.get(
    "CARTESIAN_HAND_CALIB",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "zero_offsets.json"))


# ══ The type ══════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class HandConfig:
    """Full description of one physical hand.

    Every field is a scalar or a per-DOF sequence in LAYOUT order; a scalar
    means "same for every DOF". Per-DOF costs nothing on the wire -- a sync-write
    is one broadcast packet and each servo reads its own slice. Stored as ints
    and tuples so the tables stay hand-editable; `gain_vector` broadcasts and
    caches the tensor form.

    Frozen because of that cache: setting a gain after `gain_vector` has run
    leaves the old value cached and every later tick keeps commanding it. Use
    `variant()`, which drops the cache.
    """
    name: str
    port: str
    # DOF i is servo `first_servo_id + i`. One consecutive block per hand, and
    # blocks must not overlap between hands -- see `identify`, which reads the
    # block that answers to tell one hand from another.
    first_servo_id: int = 0

    travel_mm: Sequence[float] = STANDARD_TRAVEL      # per DOF, from 0.0
    torque_min_to_move: Gain = STANDARD_TORQUE_MIN_TO_MOVE   # 0-1000; see above
    speed: Gain = STANDARD_SPEED
    acc: Gain = STANDARD_ACC
    control_hz: float = 50.0

    # Counts-to-mm. `counts_per_mm` is derived from the rack pitch diameter, but
    # a real gear train has backlash and a real pitch diameter is not the
    # nominal one. Set it explicitly to override, after measuring a known travel.
    gear_pitch_diameter_mm: float = 16.0
    counts_per_rev: int = 4096
    counts_per_mm: float | None = None       # None: derive from the two above

    # Not part of the config's identity: excluded from comparison, so two hands
    # with the same tables stay equal whichever one has been ticked.
    _cache: dict[tuple[str, str], torch.Tensor] = field(
        default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        if len(self.travel_mm) != N_DOF:
            raise ValueError(f"[{self.name}] travel_mm needs {N_DOF} values, "
                             f"got {len(self.travel_mm)}")
        if any(t <= 0 for t in self.travel_mm):
            raise ValueError(f"[{self.name}] travel must be positive: "
                             f"{list(self.travel_mm)}")
        if self.counts_per_mm is None:
            if self.gear_pitch_diameter_mm <= 0 or self.counts_per_rev <= 0:
                raise ValueError(
                    f"[{self.name}] cannot derive counts_per_mm from pitch "
                    f"diameter {self.gear_pitch_diameter_mm} and "
                    f"{self.counts_per_rev} counts/rev; both must be positive")
            # object.__setattr__ because the class is frozen: derivation in
            # __post_init__ is the one legitimate write, and it happens before
            # any caller can hold a reference.
            object.__setattr__(
                self, "counts_per_mm",
                self.counts_per_rev / (math.pi * self.gear_pitch_diameter_mm))
        # These tables are edited by hand, and a zero or NaN here would not fail
        # until it had already been divided into a servo command.
        if not math.isfinite(self.counts_per_mm) or self.counts_per_mm <= 0:
            raise ValueError(f"[{self.name}] counts_per_mm must be finite and "
                             f"positive, got {self.counts_per_mm}")
        # Catch a wrong-length gain here rather than at the first servo write.
        for gain in ("torque_min_to_move", "speed", "acc"):
            self.gain_vector(gain)

    # ── Per-DOF tensors ───────────────────────────────────────────────────────

    def _vec(self, key: str, values: Sequence[float], dtype: torch.dtype,
             device: Device) -> torch.Tensor:
        """Cache one per-DOF tensor per (quantity, device).

        Returned tensors are shared, so callers must treat them as read-only.
        Everything here is a joint limit or a wiring constant -- nothing that
        should ever be written through -- and the alternative is a fresh 7-float
        allocation on every tick of every env.
        """
        dev = torch.device(device)
        # "cuda" and "cuda:0" are the same device and stringify differently, so
        # the raw argument is not a usable key: the tick passes a resolved
        # `tensor.device` (always indexed) while call sites pass the string, and
        # the cache would hold two copies of every quantity and never hit.
        if dev.type == "cuda" and dev.index is None:
            dev = torch.device("cuda", torch.cuda.current_device())
        slot = (key, str(dev))
        if slot not in self._cache:
            self._cache[slot] = torch.tensor(values, dtype=dtype, device=dev)
        return self._cache[slot]

    def lower(self, device: Device = "cpu") -> torch.Tensor:
        """[J] travel minimum in mm. Zero on every DOF: zero is the hard stop."""
        return self._vec("lower", [0.0] * N_DOF, MM, device)

    def upper(self, device: Device = "cpu") -> torch.Tensor:
        """[J] travel maximum in mm. Never command this -- see `clamped_mm`."""
        return self._vec("upper", list(self.travel_mm), MM, device)

    def orientations(self, device: Device = "cpu") -> torch.Tensor:
        """[J] servo count direction vs. positive mm, +1 or -1.

        Wiring, not a convention still to be chosen: `counts_to_mm` and
        `mm_to_counts` apply it on the way in and out, so above that layer
        positive already means opening on both backends.
        """
        return self._vec("orientations", list(ORIENTATION), MM, device)

    def gain_vector(self, gain: str, device: Device = "cpu") -> torch.Tensor:
        """One gain as a [J] int tensor, broadcasting a scalar.

        Int, because these are raw servo register values and the bus wants
        `.tolist()` of exactly these numbers -- a float here would round at the
        boundary instead of failing here.
        """
        value = getattr(self, gain)
        if isinstance(value, (list, tuple)):
            if len(value) != N_DOF:
                raise ValueError(
                    f"[{self.name}] {gain} has {len(value)} values, "
                    f"expected a scalar or {N_DOF}")
            values = list(value)
        else:
            values = [value] * N_DOF
        return self._vec(f"gain_{gain}", values, torch.int32, device)

    # ── Introspection ─────────────────────────────────────────────────────────

    @property
    def n_dof(self) -> int:
        return N_DOF

    @property
    def servo_ids(self) -> list[int]:
        """Python list: this addresses the bus, which does not speak tensors."""
        return [self.first_servo_id + i for i in range(N_DOF)]

    def clamped_mm(self, dof_id: int, want: float) -> float:
        """A goal in mm, bounded by one DOF's travel. Scalar, for task authoring.

        Tasks ask for the clearance they need and get it capped. They must
        never use the travel limit itself: the table is CAD and the one real
        measurement contradicts it, so a clamp cannot save a task that commands
        the limit -- the clamp uses the same wrong number.
        """
        return min(max(want, 0.0), self.travel_mm[dof_id])

    def travel_budget(self, dof_ids: Sequence[int], margin: float,
                      distance_mm: float | None = None,
                      speed_mm_s: float | None = None) -> float:
        """Seconds to allow a move of `dof_ids`. `distance_mm` defaults to the rail.

        Every deadline a task sets is seconds over millimetres and the `speed`
        gain is servo counts per second, so this conversion sits between them.
        It lives here, and not in the two tasks that need it, because getting it
        wrong is not a crash: a budget that is short by a factor turns every row
        it covers into a timeout, the program still ends tidily, and the task
        reports a failure that reads like a mechanical one. `cap` shipped with
        flat constants and expired 15 of its 18 rows.

        **Slowest member, not the mean.** A group is only as quick as its
        slowest joint, and z alone runs at a different speed (500, not 300), so
        a group holding z and a finger must be budgeted at the finger's rate.

        A budget is a DEADLINE, not a duration -- `stop="goal"` retires on
        arrival and `stop="stuck"` on contact -- so defaulting the distance to
        the full rail over-allows for free, and is what lets a task set a budget
        for a move whose starting position it never reads. It is spent in full
        only by a row that has already failed.
        """
        ids = list(dof_ids)
        # `speed_mm_s` is for a row that overrides the gain (`Step.set`'s own
        # speed channel). Budgeting such a row off this table would time it out
        # by whatever factor the two differ: the zero seek creeps at a fifth of
        # the transit speed, so its budget has to be five times as long.
        mm_per_s = (speed_mm_s if speed_mm_s else
                    self.gain_vector("speed")[ids].min().item() / self.counts_per_mm)
        rail = max(self.travel_mm[d] for d in ids)
        return margin * (rail if distance_mm is None else distance_mm) / mm_per_s

    def clamp(self, positions_mm: torch.Tensor) -> torch.Tensor:
        """[..., J] mm clamped into travel. The whole-vector form of the above."""
        return positions_mm.clamp(self.lower(positions_mm.device),
                                  self.upper(positions_mm.device))

    def variant(self, **changes) -> "HandConfig":
        """Copy with fields replaced, for one-off overrides like a different port."""
        return replace(self, _cache={}, **changes)

    # ── Unit conversion ───────────────────────────────────────────────────────
    #
    # Counts are what the servo speaks; mm are what everything above speaks.
    # Zero is a hard stop found by the zeroing protocol, so `zero_offset` is the
    # count reading at that stop and is meaningless before it runs.

    def counts_to_mm(self, counts: torch.Tensor, zero_offset: torch.Tensor) -> torch.Tensor:
        """[..., J] raw counts -> mm."""
        return (counts - zero_offset) * self.orientations(counts.device) / self.counts_per_mm

    def mm_to_counts(self, mm: torch.Tensor, zero_offset: torch.Tensor) -> torch.Tensor:
        """[..., J] mm -> raw counts, rounded to nearest. Int: the bus takes ints.

        Rounds rather than truncating, which halves worst-case quantization
        error (0.5 counts vs 1.0) and keeps it centred instead of always pulling
        toward the encoder origin.
        """
        counts = zero_offset + mm * self.counts_per_mm * self.orientations(mm.device)
        return counts.round().to(torch.int64)

    # ── Normalized action space (the sim/real contract) ───────────────────────
    #
    # Simulation policies are trained in a unitless [-1, 1] box; hardware speaks
    # millimetres bounded by per-DOF travel. These two are the whole translation,
    # so a policy trained against any sim with matching DOF ordering runs here
    # without knowing the hand's dimensions.

    def normalize(self, positions_mm: torch.Tensor) -> torch.Tensor:
        """[..., J] mm -> [-1, 1], per DOF."""
        lo, hi = self.lower(positions_mm.device), self.upper(positions_mm.device)
        return 2.0 * (positions_mm - lo) / (hi - lo) - 1.0

    def denormalize(self, action: torch.Tensor) -> torch.Tensor:
        """[..., J] [-1, 1] -> mm, per DOF.

        Out-of-range actions clip rather than raise: a saturating policy should
        press against the joint limit, not crash the run.
        """
        lo, hi = self.lower(action.device), self.upper(action.device)
        return lo + (action.clamp(-1.0, 1.0) + 1.0) * 0.5 * (hi - lo)

    # ── Policy contract ───────────────────────────────────────────────────────

    def contract(self) -> dict[str, object]:
        """The sim/real interface, without hardware plumbing.

        A policy tuned in a twin transfers only if the twin and the real hand
        agree on DOF count, ordering, per-DOF axis and travel, and the control
        rate. Those fields and only those: a policy works in normalized units,
        so it never sees the port, the servo IDs, or the count direction.

        `counts_per_mm` is excluded even though it changes what the servos are
        told. It is a per-machine calibration -- two units with the same 50mm of
        travel and slightly different gear trains need different values to both
        actually reach 50mm. Folding it in would make correctly calibrated hands
        look incompatible, which is backwards.
        """
        return {
            "n_dof": N_DOF,
            "control_hz": self.control_hz,
            "dofs": [{"axis": AXIS[i], "min_mm": 0.0, "max_mm": self.travel_mm[i]}
                     for i in range(N_DOF)],
        }

    def fingerprint(self) -> str:
        """Stable short hash of contract(). Record it on whatever a twin produces.

        Also the admission rule for a batch: rows must agree on travel and
        control rate, or a goal in row 0 does not mean what it means in row 1.
        """
        blob = json.dumps(self.contract(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()[:12]


# ══ The hands ═════════════════════════════════════════════════════════════════
#
# The units built and tested in our lab -- not presets. A new build adds its own
# entry (and its own ID block) rather than reusing one of these; see
# docs/hardware.md#configuring-a-new-hand. Everything a unit does not share with the
# others, and nothing else. Anything absent here comes from the tables above.

HAND_1 = HandConfig(
    name="hand_1", port="/dev/ttyACM0",
    first_servo_id=14,

    # Never bisected on this unit -- older hand_2 values, carried over. hand_1
    # is the hand that recorded stops mid rail once, so these are the first
    # thing to measure if it does it again.
    torque_min_to_move=(250, 250, 250, 800, 250, 250, 250),
)

HAND_2 = HandConfig(
    name="hand_2",
    # by-id, not /dev/ttyACM1: ACM numbers are handed out in plug order, so a
    # fixed number silently addresses whichever hand enumerated first.
    port="/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AE6085950-if00",
    first_servo_id=7,

    # Bisected on this unit.
    # torque_min_to_move=(250, 250, 250, 800, 250, 250, 250),
    torque_min_to_move=(100, 100, 100, 400, 100, 100, 100),

    # Transit speed, what the sliders and ordinary task rows run at. The zero
    # seek does NOT use this any more: it asks for its own creep through
    # `Step.set(speed_mm_s=)`, because a seek and a park want different speeds
    # on the same joint and this table can only say one. See `zero.seek_speed`.
    speed=(300, 300, 300, 500, 300, 300, 300),
)

HAND_3 = HandConfig(
    name="hand_3",
    # by-id, not /dev/ttyACM1: ACM numbers are handed out in plug order, so a
    # fixed number silently addresses whichever hand enumerated first.
    port="/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AE6085950-if00",
    first_servo_id=0,

    # NOT bisected on this unit -- it has a newer, higher-friction gripper than
    # hand_2, and hand_2's bisected-low floor (100/400) stalled it before a hard
    # stop (jaws, DOFs 0 and 4, first). Carried over from hand_1's un-bisected
    # values instead: a floor known to move real hardware, at the cost of extra
    # margin over the true minimum. Re-bisect for this unit per the procedure
    # above -- command a 5mm move, read travel after 3s -- rather than trusting
    # this number long-term.
    torque_min_to_move=(250, 250, 250, 800, 250, 250, 250),

    # Transit speed, what the sliders and ordinary task rows run at. The zero
    # seek does NOT use this any more: it asks for its own creep through
    # `Step.set(speed_mm_s=)`, because a seek and a park want different speeds
    # on the same joint and this table can only say one. See `zero.seek_speed`.
    speed=(300, 300, 300, 500, 300, 300, 300),
)

HANDS = {h.name: h for h in (HAND_1, HAND_2, HAND_3)}


def get_hand(name: str = DEFAULT_HAND) -> HandConfig:
    if name not in HANDS:
        raise KeyError(f"unknown hand {name!r}; have {sorted(HANDS)}")
    return HANDS[name]


def identify(bus: Any, port: str | None = None) -> HandConfig:
    """Which hand is on `bus`, from the servo IDs that answer it.

    `bus` is `Any` on purpose: the compiled `FtServo` and `servo.MockServo`
    share a method surface but no base class, and the first is a nanobind
    extension with no stubs, so there is nothing honest to name. Only
    `read_all` and `scan` are used here.

    Every hand's DOFs are one consecutive ID block and the blocks do not overlap,
    so the block that answers names the hand. That is the only distinguishing
    feature: nothing else about a hand is readable over the bus. A new build
    numbered 0-6 therefore matches `hand_3`.

    One sync-read per candidate, not `bus.scan`: a scan pays a reply timeout
    for all 254 unowned IDs to answer the same question. It is used only to
    build the error message, where the extra second buys a listing.

    The caller's port wins when it differs -- an adapter that enumerated
    elsewhere is still that hand. Two blocks answering raises rather than
    guessing: that is two hands on one bus, and either choice moves hardware
    the caller did not name.
    """
    matched = [h for h in HANDS.values()
               if all(r is not None for r in bus.read_all(h.servo_ids))]
    where = f" on {port}" if port else ""
    if len(matched) > 1:
        raise RuntimeError(
            f"{len(matched)} hands answered{where}: {[h.name for h in matched]}"
            " -- say which with --hand")
    if not matched:
        blocks = ", ".join(f"{h.name}={h.servo_ids}" for h in HANDS.values())
        raise RuntimeError(
            f"no known hand{where}: IDs answering are {bus.scan(0, 32)}, "
            f"expected one of {blocks}")
    cfg = matched[0]
    return cfg.variant(port=port) if port and port != cfg.port else cfg


# ══ Saved calibration ═════════════════════════════════════════════════════════

def save_offsets(name: str, offsets: Sequence[int],
                 path: str = CALIB_PATH) -> str:
    """Merge one hand's offsets into the shared file, leaving other hands alone.

    A corrupt file must not cost us a fresh calibration run, so an unreadable
    existing file is renamed `.bad` rather than failing the save. Returns the
    path written to so the caller can log it.
    """
    data = {}
    if os.path.exists(path):
        try:
            with open(path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            backup = path + ".bad"
            os.replace(path, backup)
            print(f"Existing {path} was unreadable, moved to {backup}")
    data[name] = {
        "offsets": [int(v) for v in offsets],
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return path


def append_result(path: str, task_name: str, hand_name: str, data: dict) -> str:
    """Append one bench run's `data` under `[task_name][hand_name]`, a list.

    The generic half of a task's opt-in result log (see `tasks.result_path`):
    nested by task then hand so one file can hold several bench tasks and both
    hands without one run's write clobbering another's, and a list per hand so
    repeated runs accumulate -- the point of a bench log -- rather than
    overwrite. Mirrors `save_offsets`: a corrupt file must not cost a fresh
    bench run, so an unreadable existing file is renamed `.bad` rather than
    failing the write.
    """
    results = {}
    if os.path.exists(path):
        try:
            with open(path) as f:
                results = json.load(f)
        except (json.JSONDecodeError, OSError):
            backup = path + ".bad"
            os.replace(path, backup)
            print(f"Existing {path} was unreadable, moved to {backup}")
    results.setdefault(task_name, {}).setdefault(hand_name, []).append(data)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    return path


def load_offsets(name: str, n_dof: int,
                 path: str = CALIB_PATH) -> list[int] | None:
    """Saved offsets for `name`, or None if missing, wrong length, or unreadable.

    `None` is the signal "run zeroing first"; the caller decides what to do with
    it. A wrong length is treated as missing rather than fatal -- the entry is
    from an older calibration against a different layout and would decode to
    nonsense, but no other hand's data depends on it.
    """
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
    return [int(v) for v in offsets]

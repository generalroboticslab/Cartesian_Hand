# Cartesian Hand

Control stack for a 7-DOF Cartesian gripper driven by Feetech HLS-series serial
servos. Two grippers (a base pair and an auxiliary pair) share a vertical stage,
so the hand can hold one object while turning another. Unscrewing a bottle cap
is the task it was built around.

The stack does three things: it converts servo counts to millimetres against a
measured zero, it runs a fixed-rate position loop over the serial bus, and it
gives a policy trained in a digital twin a defined way to drive the real hand.

## Install

```bash
git submodule update --init --recursive
pip install -e .
```

That builds the nanobind extension (`ft_servo_ext`) around the C++ driver in
the `hardware_bindings` submodule, so the submodule init is not optional. You
need CMake 3.15+ and a C++17 compiler. Python dependencies are numpy and tyro.

If you only want to read the code or develop a policy, you can skip the build
entirely and use the mock backend described below.

## Quick start

```bash
python -m cartesian_hand list                    # hands and tasks
python -m cartesian_hand scan --end-id 20        # what is on the bus
python -m cartesian_hand zeroing                 # find the hard stops, save offsets
python -m cartesian_hand travel                  # measure full travel between stops
python -m cartesian_hand demo                    # sweep the full travel
python -m cartesian_hand publish --hz 2          # watch DOF state
python -m cartesian_hand set-id 7                # rename the one servo on the bus
```

Every command takes `--hand` to pick a hand, `--port` to override its serial
port for one run, and `--mock` to run against a simulated bus with no hardware
attached. `--help` on any subcommand lists its flags.

Zeroing has to run before anything else. Until offsets exist, millimetres have
no meaning and motion commands are refused.

## Setting up a servo

Servos ship with an ID that will collide with the rest of the bus, so each one
is renamed before it goes into a hand. `hand_1` uses IDs 0-6 and `hand_2` uses
7-13, in the DOF order of `LAYOUT`.

```bash
python -m cartesian_hand set-id 7 --port /dev/ttyACM0
```

Connect one servo at a time. Both tools below refuse to run with several on the
bus: the rename would be ambiguous, and the new ID could collide with one
already in use. The old ID is found by scanning, so a servo whose ID nobody
recorded is fine.

The submodule carries a standalone version that also drives the motor
afterwards, which is worth having on the bench: a ping only proves something
answers to the new ID, while motion proves it is the servo in front of you.

```bash
python hardware_bindings/ft_servo/change_id.py /dev/ttyACM0 7
python hardware_bindings/ft_servo/scan.py /dev/ttyACM0
```

Renaming writes to the servo's EPROM and survives power cycles. Once renamed,
the only way to find a servo again is to scan for it.

## Running without hardware

`--mock` swaps the serial driver for a kinematic model with hard stops. Servos
ramp toward their targets and stall at the ends of travel, which is enough to
exercise unit conversion, task sequencing, and policy code:

```bash
python -m cartesian_hand zeroing --mock
python -m cartesian_hand demo --mock
python test_cartesian_hand.py
```

The mock has no friction, no load-dependent stall and no following error. It
tells you whether your control flow is right, not whether your grip will hold.

## What the seven DOFs are

Two grippers share one vertical stage. Each gripper is a parallel jaw plus two
independent finger slides; the stage carries the auxiliary gripper up and down.
That is what lets the hand hold a bottle in one gripper and turn the cap with
the other.

Every DOF is linear and commanded in millimetres, measured from the hard stop
that zeroing finds. There are no rotary joints in the controller's view, even
though the servos themselves rotate — the rack converts turns to travel.

| DOF | Role name | Axis | Orient | What it does | Servo `hand_1` / `hand_2` |
|----:|-----------|:----:|:------:|--------------|:-------------------------:|
| 0 | `BASE_JAW` | y | −1 | base parallel actuation | 0 / 7 |
| 1 | `BASE_LEFT` | x | +1 | base left finger | 1 / 8 |
| 2 | `BASE_RIGHT` | x | −1 | base right finger | 2 / 9 |
| 3 | `Z` | z | −1 | vertical translation | 3 / 10 |
| 4 | `AUX_JAW` | y | −1 | aux parallel actuation | 4 / 11 |
| 5 | `AUX_LEFT` | x | +1 | aux left finger | 5 / 12 |
| 6 | `AUX_RIGHT` | x | −1 | aux right finger | 6 / 13 |

Three things the table hides:

**DOF index is not servo ID.** The index is how the controller addresses a DOF
and is the same on every hand; the servo ID is what answers on the serial bus
and differs per hand. Everything user-facing — actions, observations, `set_pos`,
`normalize` — is in DOF index order. Servo IDs appear only inside the driver.

**`orientation` is which way the servo counts.** `+1` means rising counts are
rising millimetres, `-1` the opposite. It exists because a left and a right
finger are mirror images: the same physical motion is a rising count on one and
a falling count on the other. It describes how the servo was installed, so a
rebuilt hand may need it flipped.

**The ordering is authoritative for the simulation.** DOF order here is the
order the twin's actuators must be in, not the other way round. A twin ordered
differently produces a policy that drives the right values into the wrong
joints, and the contract check will not catch it — the fingerprint hashes travel
limits and rate, not which servo is which.

Names live in [`cartesian_hand/tasks/roles.py`](cartesian_hand/tasks/roles.py),
with the groups tasks actually use: `JAWS`, `BASE_FINGERS`, `AUX_FINGERS`. The
layout itself is `LAYOUT` in `hands.py`, and `standard_dofs(first_servo_id)`
stamps it out with consecutive servo IDs — the only difference between the two
hands.

DOF 3 is the odd one out in every practical sense. It is the only DOF carrying a
gravity load, so it needs its own torque (see Motion gains), it falls to rest
whenever torque drops, and it is the DOF whose zeroing repeats worst between
runs. See Known issues.

## Configuration

Hands are defined in [`cartesian_hand/hands.py`](cartesian_hand/hands.py). It is
a Python module rather than a data file because the hands are near-identical:
they differ only in serial port and servo ID block, so one helper covers both
without repeating the DOF table.

```python
HAND_2 = HandConfig(
    name="hand_2",
    port="/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AE6085950-if00",
    dofs=standard_dofs(first_servo_id=7),
    motion=Motion(control_hz=50, torque=STANDARD_TORQUE, speed=300, acc=25),
    geometry=Geometry(gear_pitch_diameter_mm=16.0, counts_per_rev=4096),
)
```

`hands.py` holds both the types and the values, so configuring a hand means
opening one file. To add a hand, build a `HandConfig` and add it to `HANDS`. To change which hand
commands use by default, edit `DEFAULT_HAND`.

Prefer a `/dev/serial/by-id/` path over `/dev/ttyACM0`. ACM numbers are handed
out in enumeration order, so with two hands plugged in, whichever powers up
first takes `ACM0` and a hardcoded number silently addresses the wrong hand.
The by-id path is tied to the adapter's serial number.

### Motion gains

`torque`, `speed` and `acc` each take a scalar or a per-DOF sequence. A scalar
broadcasts to every DOF; a sequence lets one axis differ:

```python
Motion(torque=50)                          # all DOFs
Motion(torque=[50, 50, 50, 150, 50, 50, 50])   # z stage at 150
```

A wrong-length sequence raises at construction, not at the first servo write.

Mixing values costs nothing. A sync-write is one broadcast packet in which each
servo reads its own slice, so `SyncWritePosEx` takes `Speed[]`, `ACC[]` and
`Torque[]` as per-servo arrays — seven different torques and seven identical
ones are the same packet and the same time on the wire.

The z stage is the only DOF carrying a gravity load, and it needs more torque
than the six horizontal ones. Bisected on `hand_2`, lifting 30mm to 35mm and
measuring travel after 3 seconds:

| torque | 150 | 200 | 250 | 300 | 350 |
|---|---|---|---|---|---|
| moved (of 5.0mm) | 1.50 | 4.54 | 4.54 | 4.54 | 4.53 |

The cliff is sharp: 150 stalls outright, 200 tracks fully, nothing above 200
helps. `STANDARD_TORQUE` uses 300 — the measured floor plus margin, because the
bisect ran unloaded and the stage exists to lift the aux gripper while it is
holding something.

Pressing *down* at 50 works and tasks rely on it, so this is a floor for the
lifting direction rather than a correction to the whole axis.

`counts_per_mm` is derived from the pitch diameter, but a real gear train is not
its nominal drawing. After measuring a known travel distance, set it directly
and the derived value is ignored:

```python
Geometry(counts_per_mm=80.0)
```

Zero offsets are measured rather than authored, so the zeroing task writes them
to `~/.cartesian_hand/zero_offsets.json`, keyed by hand name. They live outside
the package because `pip install -e .` wipes the package directory, and losing a
calibration means re-driving every DOF into its hard stop. Override the location
with `CARTESIAN_HAND_CALIB`. That file is generated; do not edit it by hand.

## Running a policy from a digital twin

The bridge between a twin and the hardware is a fixed contract, defined in
[`cartesian_hand/policy.py`](cartesian_hand/policy.py):

- Actions and observed positions are normalized to `[-1, 1]` per DOF. A policy
  never sees millimetres, serial ports, servo IDs or count directions.
- An action is an absolute position target, not a delta or a velocity.
- DOF ordering is the ordering in `HandConfig.dofs`.
- Steps run at a fixed rate, `motion.control_hz` unless overridden.

Export that contract and build the twin against it:

```bash
python -m cartesian_hand contract --hand hand_2 -o contract.json
```

`fingerprint()` hashes exactly the fields above and nothing else. Two hands on
different ports with different servo IDs but the same kinematics share a
fingerprint, so a policy moves between them. Change a travel limit or the
control rate and the fingerprint changes, because a policy tuned against the old
geometry is no longer valid.

Stamp the fingerprint onto whatever the twin produces. The runner compares it
against the connected hand and aborts on a mismatch rather than driving real
hardware with the wrong assumptions.

A policy is any object with an `act(obs) -> action`:

```python
from cartesian_hand.hand import connect
from cartesian_hand.policy import Policy, run_policy

class Pinch(Policy):
    def __init__(self, config):
        self.fingerprint = config.fingerprint()
        self.jaws = config.dofs_on("y")
        self.n_dof = config.n_dof

    def act(self, obs):
        action = np.zeros(self.n_dof)
        action[self.jaws] = -1.0          # close
        return action

with connect("hand_2") as hand:
    rollout = run_policy(hand, Pinch(hand.config), duration=5.0)
    rollout.save("run.npz")
```

Either form runs from the command line:

```bash
python -m cartesian_hand policy rollout.npz --hand hand_2
python -m cartesian_hand policy my_module:MyPolicy --hand hand_2 --record run.npz
```

See [`examples/twin_policy.py`](examples/twin_policy.py) for a worked version.

### What the runner adds, and why

Hardware is not the twin, so `run_policy` is not a bare loop:

- Slew limiting (`max_delta`, default 0.05 normalized units per step). A twin
  can teleport a joint between steps. A servo answers the same command with
  maximum current. At 60mm of travel and 50Hz the default caps a DOF at 3mm per
  step. Pass `--max-delta 0` to disable it, but measure the current draw first.
- Non-finite actions are rejected. A diverged network should not reach the bus.
- Actions are clipped to real travel limits, so a saturating policy presses
  against the joint limit instead of commanding past it.

Replaying a recorded rollout is the first thing to try when a transfer fails. If
the replay works and the live policy does not, the gap is in the policy. If the
replay also fails, the gap is in the dynamics.

Two cases the contract check treats as errors, not warnings:

- A rollout with no fingerprint. A recorded trajectory is a fixed sequence of
  positions and only means anything on the geometry it came from, so an
  unlabelled one is refused rather than assumed compatible. A hand-written
  policy with no fingerprint is allowed, because it is geometry-agnostic by
  construction.
- Passing `--hz` different from the hand's configured rate. The rate is part of
  the fingerprint, so replaying faster commands proportionally faster motion and
  would quietly invalidate the check that just passed.

`counts_per_mm` is deliberately not in the fingerprint. It is a per-machine
calibration: two units with the same 60mm of travel and slightly different gear
trains need different values to both actually reach 60mm. Including it would
make correctly calibrated hands look incompatible.

## Hardware status

Brought up on `hand_2` (servo IDs 7-13, one CH340 adapter at 1Mbaud). What has
actually run on servos:

- All seven servos enumerate and report plausible voltage and temperature
  (11.3-11.5V, 24-26C).
- Full zeroing completes. Every DOF finds a hard stop and parks at mid travel,
  reading 30.0mm across all seven afterwards.
- The offsets are multi-turn, which settles an open question: `[4293, -1412,
  5608, 5837, 4526, 2012, 6762]` includes values past one 4096-count revolution
  and one negative. These servos are not running inside a single turn, and the
  sign-magnitude encoding in the driver is exercised by real traffic.
- Normalized `move()` actions track. Six DOFs converge to within 0.03mm; the z
  stage needs its raised torque to do so (see Motion gains above).
- Zeroing uses per-DOF creep torque (`hands.ZEROING_TORQUE`) — fingers at 30,
  z at 150, jaws at 50 — instead of a flat 50 that stalled the fingers short of
  their stops and left z in mid-air. After a full zeroing on `hand_2`, all seven
  DOFs park at mid travel and read 29.7-30.0mm against the new offsets. Two
  consecutive runs agree to within 7 counts. See Known issues.
- The control loop is two bus packets per step regardless of gains: one
  sync-read covering all seven servos, one sync-write carrying per-joint
  positions and gains. Measured on `hand_2`:

  | | 7 unicast | 1 sync | |
  |---|---|---|---|
  | read | 1.97ms | 1.47ms | one TX replaces seven, but each servo still replies |
  | write | 2.35ms | ~0ms | broadcast, unacked, so nothing to wait for |

  Sync-read saves less than it looks like it should — it eliminates the seven
  request packets, not the seven replies. Sync-write is nearly free because
  nobody ACKs a broadcast.
- Zero offsets survived a power cycle: after replugging, six DOFs read 30.00mm
  against offsets saved the previous session, and the z stage read 28.73mm,
  having drooped 1.27mm under gravity when torque was cut.

Not yet established:

- **Zeroing does not repeat on all seven DOFs.** Two runs agree to within
  0.05mm on five of them, disagree by 14.2mm on the z stage and 1.6mm on one
  jaw. See Known issues.
- **Total travel is still unmeasured.** Zeroing finds one hard stop per DOF, not
  both, so `max_mm = 60` remains an assumption. See Known issues.
- `counts_per_mm` is still the derived value, never checked against a measured
  distance.
- No task beyond `zeroing` has run on hardware, and no policy has been
  transferred from a twin.
- Grip force has not been characterised.

### Safety notes

- A mock run's offsets are the mock's hard-stop constants, not a measurement, so
  they are saved under a separate `<hand>_mock` key and can never load onto the
  real hand.
- Zeroing drives each DOF into its hard stop at low torque and calls the stop
  wherever motion ceases. A jammed mechanism looks the same as an end stop, so
  watch the first run and check the reported offsets before trusting them.
- If a phase fails partway, offsets roll back to whatever was loaded before.
  A partial calibration is never left in place.
- A DOF that never stalls is reported as a failure rather than zeroed at its
  last position, which would put the origin mid-travel.
- If the control loop hits a driver error it drops torque before exiting. It
  runs on a daemon thread, so without that the servos would sit holding their
  last target with nothing driving them.
- `release()` stops the loop before cutting torque, so the loop cannot
  re-command a servo that is being released.

## Writing a task

A task is a module in `cartesian_hand/tasks/` exposing a `Config` dataclass,
`run(hand, cfg)`, and `DESCRIPTION` for the help text. Dropping a file in that
directory registers it. The CLI is generated by [tyro](https://brentyi.github.io/tyro/)
from `Config`, so the flags, their types and their help text all come from the
dataclass. There is no parser to write and nowhere for the flags and the
function to drift apart:

```python
from dataclasses import dataclass
from .roles import Z

DESCRIPTION = "Wave hello"

@dataclass
class Config:
    reps: int = 3
    """How many times to wave."""

def run(hand, cfg: Config):
    try:
        for _ in range(cfg.reps):
            hand.set_pos({Z: 40.0})
            hand.set_pos({Z: 10.0})
    finally:
        hand.release()
```

That gives you `python -m cartesian_hand wave --reps 5 --mock`, with `--reps`
documented from the field docstring.

Address DOFs by role, not index. `hand.set_pos({AUX_JAW: 12.0})` says what it
does; `set_pos([None, None, None, None, 12.0, None, None])` hides the meaning in
the position of the one entry that is not `None`. Role names are in
[`cartesian_hand/tasks/roles.py`](cartesian_hand/tasks/roles.py).

Gains are per DOF. Setting torque on one DOF does not disturb another, so a jaw
can keep squeezing while a different DOF transits.

Shared motion helpers (`approach`, `squeeze`, `wait_for_stall`) are in
`cartesian_hand/tasks/primitives.py`.

## Layout

```
cartesian_hand/
  hands.py         what a hand is, and which hands exist: HandConfig, unit
                   conversion, the policy contract, and the definitions
  hand.py          CartesianHand: control loop, motion commands, state,
                   zero-offset persistence
  policy.py        Policy contract, run_policy, Rollout, replay
  driver.py        extension loading, MockServo
  __main__.py      the single entry point, a tyro CLI over these dataclasses
  tasks/           one module per task, auto-registered
  ft_servo_python.py   second, unused driver: see below
hardware_bindings/ submodule: IMU, motor and servo bindings. Only ft_servo/ is
                   compiled here, and it is the sole copy of the servo driver.
examples/          twin-to-real worked example
```

Four files, in the order you would read them: `hands.py` says what the hardware
is, `hand.py` how to drive it, `policy.py` how a twin-developed policy reaches
it, and `__main__.py` how to run any of it from a shell.

`ft_servo_python.py` is not one of them. It is a pure-Python reimplementation of
the same SCS wire protocol the C++ extension speaks, over `pyserial` instead of
nanobind. Nothing imports it: `driver.py` knows only the compiled `FtServo` and
`MockServo`, so it is reachable only by hand.

```python
from cartesian_hand.ft_servo_python import FtServo   # needs pip install -e '.[serial]'
```

It existed for two EPROM operations the extension lacked. `write_id` is bound in
C++ now, which leaves `set_position_offset` as the only thing it can do that the
extension cannot, plus raw `unlock_eprom`/`lock_eprom`. That is 340 lines of
second protocol implementation carried for one method — either bind
`set_position_offset` and delete the file, or keep it and accept that two
drivers must stay in agreement with nothing enforcing it.

## Known issues

**`max_mm` is a limit, not a description.** There is one hard stop per DOF, the
one zeroing seeks. The far end of each rail is open by design: drive past it and
the carriage leaves the slider and the servo spins free. `max_mm` is the only
thing that stops that happening, and it has never been measured. The v2 CAD says
50mm on the z stage and jaws and 55mm on the fingers; the config says 60.

It cannot be measured by driving. A `travel` task tried — seek the far stop,
report the span — and the premise is false, because there is no far stop to
find. Run on `hand_2` it took six of seven carriages off their rails. Deleted;
the one measurement it produced before things came apart is that DOF 0 stalled
2428 counts from its zero, 29.8mm at the configured `counts_per_mm`, against a
`max_mm` of 60.

Measure with calipers and type the number in. `counts_per_mm` is hand-wide, so
one axis calibrates all seven; `max_mm` is per-DOF and each rail needs its own.

Everything that commands `max_mm` directly is loaded against this: `demo`
(`hand.config.upper`, all seven DOFs), the quickstart in `__init__.py`, a
normalized action of `+1.0` through `denormalize`, and `caps_contact_based`'s
lift to `cfg[Z].max_mm`.

Travel above or below one 4096-count revolution also decides whether stale zero
offsets can be recovered arithmetically or the hand must be re-zeroed after
every power cycle — see *Saved zero offsets go stale by whole turns* below. 60mm
is 1.19 turns, 50mm is 0.99, and DOF 0's 29.8mm would be 0.59.

**Contact detection never detected contact.** `wait_for_stall` took its
threshold as a distance per poll, defaulting to 0.5mm over a 0.05s poll. That is
10mm/s. The fastest this hand moves is `speed=300`, which is 3.7mm/s, and
`approach` creeps at 50, which is 0.61mm/s — so every DOF measured as stalled on
the third poll, roughly 0.15s in, before it had gone anywhere.

`approach()` therefore returned the position it started from. Since
`caps_contact_based` sizes its grip from that return value, every bottle and cap
radius it has ever printed was the pre-probe jaw position, not a measurement.
Contact-based sizing has never worked; it just never announced the failure.

Fixed by making the threshold a rate. Verified on `hand_2`: a jaw commanded from
29.7mm to 12.0mm now reports 12.0, and closing to the hard stop reports 0.0.
Both used to report 29.7.

The counts-level variant `wait_for_stall_counts`, which zeroing uses, had the
same shape but sat just inside its margin: 5 counts per 0.1s poll against a
creep that covers exactly 5 counts in that time. It worked — zeroing is
reproducible on hardware — but only because the coin landed the right way up. It
now takes counts/sec against a measured interval, defaulting to half the creep
speed, so travelling and stalled are a factor of two apart in each direction.

**Saved zero offsets go stale by whole turns.** A servo reports (turns since
power-up × `counts_per_rev`) + the angle within the current turn. The angle
comes off a magnetic encoder and is right the instant power arrives; the turn
count restarts at zero. So the reading a saved offset was measured against no
longer exists, and every mm command after a power cycle is off by some whole
number of turns — silently, because the numbers stay plausible.

Observed on `hand_2`: four DOFs read 30mm and three read a turn away, from
offsets saved in the previous session.

The angle alone places the joint exactly, provided travel is shorter than one
turn — then only one reachable position matches it. `load_calibration` now
recovers the offsets from that:

```python
k = ((raw - offsets) * orientation) % counts_per_rev
rebased = raw - k * orientation
```

Everything the turn counter contributed, to the reading and to the saved offset
alike, is a multiple of a turn and drops out of the modulo. Arithmetic, not a
fit: there are no candidates to score.

Past one turn of travel two real positions share an angle. The servo cannot tell
them apart and neither can we, so that branch refuses and asks for zeroing.
Which branch runs depends entirely on `max_mm`, and `max_mm` has never been
measured: at the configured 60mm travel is 1.19 turns and offsets are
unrecoverable, at the 50mm the CAD suggests it is 0.99 and the ambiguity does
not exist. Running `travel` decides it.

**A failed read used to decode into a plausible position.** `SCS::readWord`
returns `-1` on any failure — no reply, wrong ID, bad length, CRC mismatch — and
`HLSCL::ReadPos` then ran its sign-magnitude decode over that `-1`. Bit 15 is
set, so the sign branch fired and the error came back out as **+32769 counts,
about 402mm**. The sentinel and real data shared one channel.

Three guards were written against exactly this and were dead code, because the
binding returned `int` and never `None`: `hand.py`'s `if counts is not None`,
and both null-checks in `primitives.py`. Worst case, `wait_for_stall_counts`
uses `confirm_count=2`, so three consecutive dropped frames read as movement 0
and would have registered a false hard stop at 32769 — saved as a zero offset.

Fixed: the driver's reads now return `None` on failure, using `getLastError()`,
which is the channel that was there all along. The vectorized `read_positions`
is immune by construction, since sync-read reports a missing reply through
`syncReadPacketRx`'s return rather than in-band. **This has been reproduced by
arithmetic, not by a live dropped frame** — no read has been observed to fail on
this bus.

**`enable()` used to command every joint to 0mm.** The loop writes the whole
target vector each step and `target` starts as zeros, so starting the loop drove
the hand into its hard stops before any target was set. Commanding a subset is
what exposed it: `set_pos({Z: 35.0})` leaves the other six "unchanged", and
unchanged meant zero. Hit during this bring-up — six joints travelled from 29mm
to their 0mm stops at torque 50. Fixed by seeding the target from the measured
position in `enable()`.

**Zeroing crept at a flat torque that was wrong for three DOFs.** Until this
revision, `zeroing` used `zero_torque=50` for every DOF. Three of the seven need
something different:

- **Fingers (DOF 1, 2, 5, 6)** register a stall short of the end of travel at
  50 and reach it at 30. Lower left finger (DOF 1) was the visible case: it
  stalled around 644 counts at 50, and lands at −102/−108 (two runs) at 30,
  roughly 9mm further into travel.
- **Z stage (DOF 3)** stops in mid-air short of the stop at 50 and reaches it
  at 150. Not the same number as `STANDARD_TORQUE[3] = 300`, which is the
  torque needed to *lift* the stage; this is the seeking direction only.
- **Jaws (DOF 0, 4)** were the only DOFs where flat-50 was right.

The mechanism behind the finger numbers is not established — binding, stiction
and stall-detector sensitivity all fit the observation. The numbers are
measured; the explanation would be a guess.

`hands.ZEROING_TORQUE = [50, 30, 30, 150, 50, 30, 30]` is the per-DOF default.
Pass `--zero-torque N` to broadcast a scalar (legacy behaviour) for any of the
seven that doesn't match its default.

**The encoder turn-origin shift still stands.** A stored offset is not
guaranteed to survive a power cycle: the encoder is absolute within one turn
but the turn number is an accumulator, and a different power-up state lands
at a different turn. `load_calibration()` sets `is_zeroed = True` from a stored
file without checking the turn origin still holds, so a stale calibration
could put a DOF 50mm out with no warning. Two safe moves: re-zero on every
connect, or stamp a turn-origin sentinel into the calibration file and reject
on mismatch. Not done.

**Bisects on individual DOFs can mislead.** Calling `zeroing --dof N` from an
arbitrary starting position can register a friction bind partway into the
mechanism rather than the real hard stop. The full `zeroing` run drives every
DOF from a known starting position and finds the true end of travel. Cross-
check by running `--dof N` after the full zeroing has parked the hand at mid
travel — the result should match. If it doesn't, the joint has multiple stops
or the parallel-phase path is interacting with the mechanism.

**Two servo drivers.** `cartesian_hand/ft_servo_python.py` reimplements the same
protocol as the compiled extension and is unused. See Layout above.

**`set_position_offset` is not bound.** It exists in `ft_servo_python.py` but
not in the C++ extension, so writing a servo's position offset to EPROM means
dropping to the Python driver by hand.

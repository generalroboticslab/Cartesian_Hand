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
python -m cartesian_hand demo                    # sweep the full travel
python -m cartesian_hand publish --hz 2          # watch DOF state
```

Every command takes `--hand` to pick a hand, `--port` to override its serial
port for one run, and `--mock` to run against a simulated bus with no hardware
attached. `--help` on any subcommand lists its flags.

Zeroing has to run before anything else. Until offsets exist, millimetres have
no meaning and motion commands are refused.

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

## Configuration

Hands are defined in [`cartesian_hand/hands.py`](cartesian_hand/hands.py). It is
a Python module rather than a data file because the hands are near-identical:
they differ only in serial port and servo ID block, so one helper covers both
without repeating the DOF table.

```python
HAND_1 = HandConfig(
    name="hand_1",
    port="/dev/ttyACM0",
    dofs=standard_dofs(first_servo_id=0),
    motion=Motion(control_hz=50, torque=50, speed=300, acc=25),
    geometry=Geometry(gear_pitch_diameter_mm=16.0, counts_per_rev=4096),
)
```

`hands.py` holds both the types and the values, so configuring a hand means
opening one file. To add a hand, build a `HandConfig` and add it to `HANDS`. To change which hand
commands use by default, edit `DEFAULT_HAND`.

`counts_per_mm` is derived from the pitch diameter, but a real gear train is not
its nominal drawing. After measuring a known travel distance, set it directly
and the derived value is ignored:

```python
Geometry(counts_per_mm=80.0)
```

Zero offsets are measured rather than authored, so they are written to
`cartesian_hand/zero_offsets.json` by the zeroing task and keyed by hand name.
That file is generated. Do not edit it by hand.

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

## Safety notes for the first hardware run

None of this has been exercised against real servos yet. The behaviour worth
knowing before it is:

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
            hand.set_dofs({Z: 40.0})
            hand.set_dofs({Z: 10.0})
    finally:
        hand.release()
```

That gives you `python -m cartesian_hand wave --reps 5 --mock`, with `--reps`
documented from the field docstring.

Address DOFs by role, not index. `hand.set_dofs({AUX_JAW: 12.0})` says what it
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
  ft_servo_python.py   pure-Python driver, for EPROM writes the extension lacks
hardware_bindings/ submodule: IMU, motor and servo bindings. Only ft_servo/ is
                   compiled here, and it is the sole copy of the servo driver.
examples/          twin-to-real worked example
```

Four files, in the order you would read them: `hands.py` says what the hardware
is, `hand.py` how to drive it, `policy.py` how a twin-developed policy reaches
it, and `__main__.py` how to run any of it from a shell.

## Known issues

`max_mm` is 60mm on every DOF, but one servo revolution at the configured 16mm
pitch diameter covers only 50.3mm. The hands must therefore be running
multi-turn. Either the pitch diameter in `hands.py` is not the real one, or the
travel limits describe more than a single revolution. Measuring a known travel
and setting `counts_per_mm` explicitly would settle it.

`cartesian_hand/ft_servo_python.py` is still a near-copy of the submodule's
`ft_servo_python_only.py`. The two agree on behaviour now, but nothing enforces
that; it is vendored so the package imports without the submodule initialised.

`hardware_bindings/ft_servo/INST.h` and the deleted `src/INST.h` disagreed on
the RESET and CAL opcodes. The submodule's values match the vendor `SCS.cpp`
that consumes them, which is why the submodule is the copy that survived. Those
two instructions are declared in `SCS.h` but never bound to Python, so nothing
sends them today. Check the Feetech datasheet before binding them.

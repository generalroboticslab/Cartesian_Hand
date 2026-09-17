# Cartesian Hand

Control stack for a 7-DOF Cartesian gripper driven by Feetech HLS-series serial
servos. Two grippers, a base pair and an auxiliary pair, share a vertical stage,
so the hand can hold one object while turning another. Unscrewing a bottle cap is
the task it was built around.

A controller opens no serial port, never sleeps, and steps no simulator. It sees
typed observations in millimetres and returns commands; the executor alone owns
I/O and pacing. There are currently two controller forms:

- `Policy.step` is the target for closed-loop manipulation. It runs every tick,
  keeps tensor state, composes reusable primitives, and is batch-friendly.
- `Motions`/`TaskRunner` executes fixed `[N,J,K]` programs. It remains useful for
  zeroing, GUI-authored timelines, and existing task files.

```text
tasks.make(name) ─┬─ Policy ─ PolicyRunner ─┬─ studio.live   servos + viser
                  └─ Task ─── TaskRunner ───┼─ sim.run       CPU MuJoCo
                                            └─ sim.run_warp  batched GPU MuJoCo
```

## Install

```bash
git submodule update --init --recursive
pip install -e .
pip install torch mujoco viser        # not yet declared in pyproject
```

The build compiles the nanobind extension (`ft_servo_ext`) around the C++ driver
in the `hardware_bindings` submodule, so the submodule init is not optional. You
need CMake 3.15+ and a C++17 compiler.

To read the code, write a task, or run everything in simulation, skip the build.
`servo.open_driver` imports the extension inside the call rather than at module
scope, so nothing in this package touches hardware when you import it.

> `pyproject.toml` declares only `numpy` and `tyro` as core dependencies --
> `torch`, `mujoco` and `viser` above are not yet in it. Use the module entry
> points below.

## Quick start

```bash
# hardware
python -m cartesian_hand.studio                          # browser page at :8081
python -m cartesian_hand.studio --hand hand_1            # skip the ID probe
python -m cartesian_hand.studio --task zero              # zero it, headless
python -m cartesian_hand.studio --mock --seconds 5       # no hardware attached
python -m cartesian_hand.studio --teach                  # limp, pose it by hand

# simulation, same task files (object-free tasks only for now)
python -m cartesian_hand.sim --task zero
python -m cartesian_hand.sim --task zero --n-envs 4096 --warp   # GPU, batched
```

Simulation uses the MuJoCo model bundled at `assets/cartesian_hand/` -- no
external checkout needed. `LEGGED_ENV_ROOT` points it at a `legged_env_v2`
checkout instead, for iterating on the model itself; `scripts/sync_sim_asset.py`
copies a regenerated model from there back into `assets/`.

Every entry point is [tyro](https://brentyi.github.io/tyro/) over a function
signature, so `--help` lists the real flags. A tri-state `bool | None` renders as
`--studio {None,True,False}` rather than `--no-studio`.

**Zero the hand before trusting a millimetre.** Without a calibration, zero is
the *startup pose*. The travel clamp still applies, but relative to wherever the
hand happened to be, so starting mid-travel and driving a full stroke can still
run a carriage off its rail.

## The seven DOFs

Two grippers share one vertical stage. Each gripper is a parallel jaw plus two
independent finger slides, and the stage carries the auxiliary gripper up and
down. Every DOF is linear and commanded in millimetres from the hard stop that
zeroing finds. There are no rotary joints in the controller's view; the rack
converts turns to travel.

| DOF | Role | Axis | Orient | What it does | Servo `hand_1` / `hand_2` |
|----:|------|:----:|:------:|--------------|:-------------------------:|
| 0 | `BASE_JAW` | y | −1 | base parallel actuation | 0 / 7 |
| 1 | `BASE_LEFT` | x | −1 | base left finger | 1 / 8 |
| 2 | `BASE_RIGHT` | x | −1 | base right finger | 2 / 9 |
| 3 | `Z` | z | −1 | vertical translation | 3 / 10 |
| 4 | `AUX_JAW` | y | −1 | aux parallel actuation | 4 / 11 |
| 5 | `AUX_LEFT` | x | −1 | aux left finger | 5 / 12 |
| 6 | `AUX_RIGHT` | x | −1 | aux right finger | 6 / 13 |

Names and groups (`BASE_FINGERS`, `AUX_FINGERS`) sit in `config.py` directly
below `LAYOUT`. Putting them in their own file would let the two disagree, and a
role map that disagrees with the layout is a mirrored gripper that still looks
plausible.

Three things the table hides:

**DOF index is not servo ID.** The index is how the controller addresses a DOF
and is the same on every hand. The servo ID is what answers on the serial bus and
differs per hand. Everything above the driver is in DOF index order.

**`orientation` is which way the servo counts.** `+1` means rising counts are
rising millimetres. The current hand wiring has all seven at `-1`, measured on
the bench rather than inferred from the left/right labels.

> **`orientation` cannot fix a sim/real direction disagreement.** It cancels
> between `mm_to_counts` and `counts_to_mm`, so the millimetre the model renders
> does not change when you flip it. Flipping a sign moves only the hardware, and
> turns "the sim disagrees" into "the hardware is backwards". This was confirmed
> on the bench on 2026-09-01, at the cost of a session: flipping all seven to
> `+1` reversed five DOFs on the real hand and left the model where it was.

**The ordering is authoritative for the simulation.** DOF order here is the order
the twin's actuators must be in, not the other way round. `LAYOUT`'s axis
sequence y,x,x,z,y,x,x matches the MJCF actuator order one for one, established
by walking `model.actuator_trnid` rather than by reading the comment. Nothing
asserts it now; see Test suite.

## Writing a task

A task module's `build()` returns a controller. Object manipulation returns a
`primitives.Sequence` — a typed `Policy` whose step machine is declared as a list
of rows (`Move`, `Probe`, `Hold`, `Twist`, `Loop`); fixed timelines and
pre-calibration zeroing return a `Task` generator of `Motions` programs. Both
keep hardware out of task code and are selected by the same task name. There is
no per-task `*Policy` class anymore; the row list is the policy.

A row's `measure={"radius": AUX_JAW}` records what an earlier probe measured
into `SequenceState.measured`, and any later row that needs it names it as
`lambda m: m.radius`. `twist_stroke` still owns its own probe+reset+turn
sequence; that has not changed. Use a generator when the procedure is already a
fixed row schedule whose outcomes do not feed later rows. In that form, each
`yield` evaluates to the `[N, J]` millimetres measured when the program finished:

```python
def build(cfg, start_mm, **kwargs):          # tasks/zero.py, the whole module
    here, stops = start_mm, start_mm.clone()
    alive = torch.ones(len(start_mm), dtype=torch.bool)
    for dof_ids, name in PHASES:
        seek = program({d: Move(here[..., d] - OVERTRAVEL_MM, creep[d],
                                "stuck", SEEK_TIMEOUT_S) for d in dof_ids})
        here = yield seek
        alive &= seek.succeeded()[:, dof_ids, 0].all(dim=1)
        stops[:, dof_ids] = here[:, dof_ids]
        mid = here[:, dof_ids] + travel[dof_ids] / 2   # per DOF, from the stop
        here = yield program({d: Move(where(alive, mid, here)[..., d], PARK_TORQUE,
                                      "goal", PARK_TIMEOUT_S) for d in dof_ids})
    return Result(stops, alive)
```

For a generator task, `build` is the whole procedure rather than a wrapper
around one. Its phases are `yield`s, so splitting them into
`probe_program` / `stroke_program` / `extract_program` called once each from a
fourth function is four names for one procedure and one more place for the
phase order to disagree with itself.

That return value from `yield` is how a measurement gets back into the task, and
no other mechanism was needed for it. The task's `return` arrives as
`StopIteration.value` and ends up in `TaskRunner.result`.

**A task reports failure, it does not raise it.** `Result.ok` is `[N]` — one flag
per env — because `start_mm` is `[N, J]` and the envs are independent. Reducing
the outcome with a bare `.all()` and raising would let one env out of 4096
discard the 4095 that succeeded, and the exception would unwind the generator so
they could not even be recovered from it. Under domain randomisation some envs
are *supposed* to fail, so the flags are data rather than an error condition.

What a failure *means* is the caller's to decide: `sim.run` raises,
`studio.finish` declines to save the calibration, a batched trainer masks the bad
rows. The task's own job is only to keep every goal it derives from a failed
measurement bounded — in zeroing that is exactly one move, the park in
`tasks/zero.py`, which parks a failed env in place rather than at `mid-rail + travel/2`, since the
far end of the rail is open and that is where a carriage leaves its slider.

**Inside a fixed `Motions` program, a measurement can set a value but cannot set
the number of steps.** The Python
loop that emits N strokes runs once, when the program is built. By the time the
program starts ticking, its length is fixed. Environments needing fewer strokes
carry `when=False` on the extra rows and idle through them. That is how
`for i in range(ceil(measurement))` can only be approximated with a fixed upper
bound and masks. This compromise is why feedback-heavy tasks such as `cap` use
direct tensor state instead.

### How a name reaches a file

`--task zero` imports `tasks/zero.py` and calls its `build()`. There is no
registry and nothing registers itself at import time. The file stem is the name
because importing it is the whole dispatch. A task module supplies exactly two
names:

```python
build(hand, start_mm, cfg=None, **kwargs)   # required; returns Policy or Task
Config                                      # optional, dataclass of defaults
```

`build` is the procedure. `Config` is everything else, including the two fields
the dispatch reads:

```python
@dataclass
class Config:
    label: str = "Zero hand"    # puts a button on the page; "" means none
    sets_datum: bool = True     # its result becomes the hand's zero
    overtravel_mm: float = 120.0
    ...
```

Fields rather than `LABEL` / `SETS_DATUM` module constants, so a task is one
configuration object instead of a dataclass plus a scatter of `UPPER_CASE` beside
it. `tasks.buttons()` reads them off `Config()`; building one runs no program, so
the page lays itself out before any task exists.

A variant is a new file next to the one it varies, reaching what it reuses
through the module:

```python
# tasks/cap_gentle.py
"""Squeeze 80 -> 40: hand_2 crushed a PET cap on stroke 2 at 80."""
from . import cap

def build(hand, start_mm, **kwargs):
    return cap.build(hand, start_mm, cfg=cap.Config(squeeze_torque=40.0), **kwargs)
```

That file is reachable as `--task cap_gentle` straight away, and every name in it
is a real import an editor can jump to. `Config` is a dataclass, so its
constructor is the configuration. There is no `configure()` hook to learn, and
`dataclasses.replace` composes one variant onto another.

**`from . import cap`, not `from .cap import Config`.** The second binds `Config`
into the variant's own namespace, which is exactly where `tasks.config()` looks —
so the variant would answer with cap's `label` and put a second "Cycle cap" on
the panel. Reaching through the module is what keeps the button opt-in, and giving
every variant a button would fill the panel with them.

Module docstrings carry the bench log, because a result kept only in working
notes is not versioned beside the task variant it describes.

### The engine

`motions.py` holds a robot protocol written as tensors and advanced one tick at a
time.

```
Program  [N, J, K]   written once by Program.build, read-only during a run
Runtime  [N, J]      plus one step counter per env, [N]

    N  environments in sim, or physical hands on real
    J  joints
    K  steps in the program
```

A **motion** is one cell: one joint, one goal, one torque, one stop rule. Motions
on different joints at the same step run together. Motions on the same joint at
different steps run in order. That ordering is what the cap task needs, because
the aux jaw must release before the fingers slide, and be back on the cap before
they turn.

`Move(goal, torque, stop, timeout_s, when)` has three stop rules. `"goal"`
retires on arrival, `"stuck"` retires on contact, and `"hold"` retires
immediately after leaving a command in place.

**A joint that has finished its move keeps commanding what it last asked for.**
`step_once` returns a goal and torque for every joint, not only for the ones
still running, and the engine calls the value it keeps sending a *standing
order*. This is what makes a grip work. A probe commands the jaw past the object
at reduced torque and lets the object stop it, so the pressure only stays on
because the finished joint goes on commanding that same goal. Resetting a
finished joint's goal to its measured position would release every grip the
moment it was made.

Contact is sensed as **stall**, not as load. The servos do report a load byte,
but it reads zero at rest on every servo on both hands, so a working decode and a
decode returning padding look identical today. Load is also backwards from
intuition: it is drive effort, highest during free motion and near zero at rest,
so a high `load` column on the page does not mean contact.

Deadlines are counted in ticks. Sim has no wall clock and does not run at real
time, so a tick is the only unit that means the same thing on both backends.

**Every torque a task commands on a horizontal DOF is floored at that hand's
`torque_min_to_move`.** Below its own floor a joint does not move at all, so the
row can neither arrive nor stall anywhere but where it started, and the only way
left for it to end is its deadline — the hand standing still between motions,
waiting each timeout out in turn. `tilt` shipped with 50 and 80 against hand_2's
floor of 100 at budgets of about 20 s a row, and `scissors` closed its contact
probe at 150 against hand_1's 250. The floor applies to a probe as much as to
free travel: a contact seek wants the *lightest* push that still travels, and a
below-floor torque does not travel at all.

**z is not excluded, and an earlier version of this file said it was.** The
exemption reasoned that z's resistance is gravity rather than friction and that
a descent should command less, so `Sequence` left `travel_effort[Z]` at the flat
travel torque — 50 against a measured z floor of 800 on both hands. `cap`'s
descent names z with no explicit effort, so it could not move the stage at all,
burned its deadline, and retired the environment, which silently skipped every
row after it. Both halves of the reasoning are wrong: effort is a force *cap*,
so a free descent never approaches it and the exemption bought no gentleness,
and 50 against a floor of 800 is immobile in both directions.
`Sequence.travel_effort` is now `maximum(flat, floor)` on every DOF.

An explicit `effort=` on a row is still not floored, which is deliberate: a row
that presses z into something names its own number, and `lift_effort` and
`TwistPress.effort` already do. The consequence to watch for is a row that names
an explicit z effort *below* the floor, which now stands unchanged and probably
cannot descend.

Nothing checks any of this: mujoco, `MockServo` and the toy hand all ignore the
torque register, so on every backend a below-floor row looks correct and merely
slow.

### Primitives

`primitives.py` serves both controller forms. One legacy helper is left, `tilt`,
which fills a `Step` for the task of the same name; it earns its place because
the four-way joint pairing is what a caller gets wrong, not the `Step.set` call.
A helper that only renames `Step.set` does not belong there, which is why
`twist`, `rotate_in_place` and `move_until_stuck` are gone.

The direct functions are closed-loop behaviors:

- `hold` changes named DOFs and preserves every other standing command;
- `move_to` owns convergence and a tick deadline;
- `close_until_contact` distinguishes contact, timeout, and reaching a closed
  goal with no object;
- `twist_stroke` owns release, finger reset, re-grip, and one twist;
- `strokes_for_revolutions` turns a measured radius into a per-environment
  stroke count.

`twist_stroke` takes two optional per-environment arguments, because three tasks
need the same stroke and only differ in these:

- `reverse` (`[N]` bool) exchanges which finger opens the gap and which closes
  it, which is what runs the stroke the other way round. `cap` unscrews and
  re-threads the same cap in a *single* task, so direction has to live in tensor
  state rather than at the call site.
- `press` (`TwistPress`) drives a third axis to depth after the re-grip, holds
  it through the turn, and backs off before the next release. A cap being
  threaded back onto its bottle and a screw being driven in engage by depth as
  well as rotation. Its `return_effort` is separate from `effort` because
  backing off *raises* the stage and z torque is directional.

An environment that configures no press skips both z phases on the transition
itself rather than idling a tick in each, so a stroke without a press issues
exactly the actions it did before the parameter existed.

The closed-loop primitives accept an `[N]` active mask and return an `Action`,
typed state, and a `PrimitiveResult`; `hold` is the smaller action-update helper.
Results publish `succeeded`, `timed_out`,
`reached_goal_without_contact`, `stopped_at_mm`, and validity tensors.
Information passes between rows via `SequenceState.measured`, which a row that
needs it reads as `lambda m: m.radius`. Primitives do not inspect or mutate one
another's private state.

This explicit result-to-state-to-input path is how closed-loop primitives
exchange information without dictionaries, numbered registers, host-side
branches, or backend objects.

**Toward the stop is negative on every DOF, with no `orientation` term.** The
seek commands `here - overtravel` in millimetres, and `counts_to_mm` has already
applied `orientation` on the way in. A seek written in raw counts has to say
`here - orientation * overtravel`, and that sign comes out wrong about half the
time someone works it out again, as it did twice here. Working in millimetres
removes the mistake instead of correcting it. The cost is that a wrong sign is
**invisible in millimetres** and can only be caught per DOF in counts, which
nothing does today.

## Direct policies

`policy.py` defines the backend-neutral contract. `Observation` contains
`position_mm`, `velocity_mm_s`, `contact`, and `elapsed_ticks`; `Action` contains
`goal_mm`, `max_speed_mm_s`, and normalized `effort_limit`. Every field is a
tensor with a leading environment dimension. Hardware is ordinary `N=1` rather
than a different API.

Policy state and parameters are typed dataclasses with tensor leaves and fixed
primitive-state fields. An optimizer may own a packed `theta[N,P]`, but it
converts that once to a task-specific dataclass such as `CapParameters`; policy
and primitive code never indexes a parameter by string. Measurements may change
values, masks, phases, and tensor counters, but never a Python loop bound or
tensor shape.

Five manipulation tasks are on this path, all transcribed from an earlier
internal implementation (not included in this repo) that ran on real objects:

| task | from | what makes it its own file |
|---|---|---|
| `cap` | `caps_contact_based.py` | the canonical one; probe, twist, extract, re-thread, present |
| `screwdriver` | `manual_screw_driver.py` | both jaws hold **one** tool; the base jaw is taut-contact only and never squeezed. `cw` selects `reverse` + press |
| `pipette` | `pipetting.py` | longest sequence; a twist-lock knob, then plunge and draw as one shared push-to-stall stroke |
| `syringe` | `syringe.py` | no twist at all; the repeating DOF is **z**, and the jaw re-grips the plunger higher each stroke |
| `scissors` | `scissor_type.py` | no twist either; z travel drives the tool's pivot. Both jaws stay squeezed at the end |

`zero` deliberately stays a `Motions` generator: its uncalibrated negative
overtravel must not pass the direct-action clamp that `studio.live` applies to
every policy goal. `tilt` stays one too, since nothing in it is parameterised by
a measurement.

The sequences are validated; these implementations of them have not been run on
the objects.

`Sequence` is the reference direct manipulation policy. Its step machine
runs the tested `cap` sequence as a row list:

```text
Move  height          z down to the cap's top face
Probe probe           close both jaws until stuck; measure the cap's radius
Hold  grip            keep squeezing the bottle, from here to the end
Loop  cycles
  Twist open          unscrew. strokes = ceil(2*pi*r / finger span)
  Move  release       open the aux jaw to radius + clearance
  Move  centre        fingers to mid-span
  Hold  centre hold   0.2 s, so profiled travel cannot overlap the re-grip
  Probe regrip        close the aux jaw onto the cap
  Move  lift          z up, carrying the cap
  Move  cap clear     fingers closed, cap held above the bottle
  Hold  cap clear wait
  Move  cap align     fingers back to mid-span
  Move  put back      z down to thread height
  Twist close         the same stroke mirrored, pressing z down as it turns
  Move  let go        open the aux jaw
Move  present         fingers home
```

Two things in that list carry the argument for this path. `Probe` records what
it measured under a name (`measure={"radius": AUX_JAW}`) and every later row
reads it back as `radius=lambda m: m.radius`; that is the whole measurement
channel. The stroke count is written down nowhere. It is the cap's circumference
over how far one finger can slide, known only once the jaws have closed and felt
how big the cap is, which is exactly what a fixed `[N,J,K]` program cannot
express because its length is fixed before it runs.

A `Probe` latches a `grip` effort the tick contact is confirmed, so the body
of a sequence never re-states the clamp. The base-jaw command remains in
`SequenceState.held_*` while the auxiliary gripper twists and extracts.
Environments may occupy different phases and use different parameter rows in
the same rollout; `Sequence.step` contains no measurement-driven Python
control flow or tensor-to-host conversion.

`tasks/<name>.py` converts its scalar dataclass once into the row list.
Contact closes at the approach speed the task names; free travel uses the
task's own `travel_speed` (or the hand's `speed` gain if the task does not
override). Their deadlines are derived separately, so slow probing does not
make every ordinary move wait on a contact-sized timeout. Horizontal movement
effort is floored at that hand's `torque_min_to_move`, z descent stays light,
and any ascent uses z's measured floor via `primitives.lift_effort`.

The named task is invoked the same way by both CLIs:

```bash
python -m cartesian_hand.studio --hand hand_2 --task cap
python -m cartesian_hand.sim --hand hand_2 --task cap
```

The second command is wired correctly but cannot complete until the model has a
bottle and cap. Programmatically, the same policy object occupies the same
controller argument in both simulation executors:

```python
studio.live(hand="hand_2", policy=policy)
state = sim.run(policy, cfg)
state = sim.run_warp(policy, cfg, n_envs=N, device="cuda")
```

There is no second policy registry or configuration format: `tasks.make("cap",
...)` returns the policy, and `tasks.make("zero", ...)` returns the fixed-program
controller appropriate to zeroing.

## Composing and tuning a task

`python -m cartesian_hand.studio` serves tuning for task dataclasses and a
timeline editor for fixed `Motions` tasks, both in the page's **tasks** window
on the left, beside the task buttons:

- **tune a task** — pick a task, drag its numbers, save the result as a variant.
  The sliders are built from `tasks.tunables()`, which reads
  `field(metadata={"tune": (lo, hi)})` off the task's own `Config`, so the range
  a slider offers is declared beside the value it bounds.
- **TASK TIMELINE** — expanded on every studio page, including watch-only mode.
  A CAD history: select a row and its parameters load into the
  editors, **Apply** an edit in place, **Move up/down**, **Delete**, **Insert
  after**. A row is one action — joints, goal, torque, finish condition (`goal`,
  `stuck`, `wait`, or externally supplied completion), optionally gated on a
  source joint's prior stop outcome or position. Save as a new task file.

**Load task** fills the timeline from the task the tune dropdown names, so an
existing procedure can be opened and not only written. The two halves divide by
what they can reach: tune moves the numbers a task *declared* and keeps its
structure; load reaches everything a row has — its joints, stop rule, frame and
predicate, none of which is a `Config` field — and keeps no structure, because
what comes back is a flat program you save under a new name.

It reads the **built program**, not the source (`compose.rows_from`, the inverse
of `program_source`). Parsing would only ever understand the files `compose.py`
emits — the subset that needed no editor — and would be wrong about every
hand-written task, whose goals are expressions over a `Config` rather than
literals. Two things therefore cannot load, and both say so instead of loading
half a procedure: a direct `Policy` (`cap`) has no steps to show, and only the
*first* program of a multi-program task (`zero`, `scissors`) exists before the
task has run, since the rest are built from measurements it has not taken yet.
Goals come back as the numbers the task computed at the pose it was built at,
which is why the task is built at the hand's current pose.

**Run to row** is the rollback marker, and the reason the panel is worth having:
it executes rows 0..cursor on the hand and leaves it there, so the next row is
authored from the pose the previous ones actually produced. `frame="here"`
distances cannot be predicted from a drawing — you have to be in the pose to
pick one.

It runs the *file*: rows 0..cursor are written to `tasks/_preview.py` and
submitted by name through the same slot the task buttons use. One execution
path, so no class of bug that exists only in a preview. `_preview` is rewritten
every run and gitignored.

Both write `cartesian_hand/tasks/<name>.py` through `compose.py`, which imports
no GUI. **The panel is never in the execution path** — it writes the file and
has no further part in it, so the result runs under `sim.run` at N=4096 and on
the bus with `studio` never imported. The dependency is studio → compose →
tasks, never back.

Feedback stays inside fixed `[N,J,K]` program tensors. Each successful row
records `stopped_at` and `stopped_ok`; later rows evaluate `When(...)` once when
they arm. A timeout invalidates outcome without overwriting last trustworthy
position. Predicate-disabled rows preserve standing command. Each environment
takes its own branch without changing K or returning to Python.

### There is no automatic tuner, on purpose

One shipped (`search.py`, random search over the declared bounds, ranked on
`(row success, ticks)`, winner written as a variant). It was deleted on
2026-09-03 because it optimises against a simulator that cannot see most of what
it samples:

- mujoco drops torque on the floor. `sim.profile` takes the goal and nothing
  else. So every `*_torque` field is invisible there — 1 of `zero`'s 2 knobs, 3
  of `cap`'s 9, and half of every composed task's, since each row emits a
  `goal_i` and a `torque_i`. Sampled anyway, they came back in the winner's
  report as if they were findings.
- It only found the trivial gradient. Score rose monotonically with
  `timeout_margin` and with nothing else, which is a longer budget passing more
  rows rather than a tuned parameter.
- It disagreed with the bench, and the bench was right. Sim scored `zero`'s
  shipped defaults at 0.50; the hand_1 log in `tasks/zero.py` has the same
  defaults reproducing the datum to 0.074 mm with no phase expiring. So
  `--save-as` would have written a variant that slows real zeroing ~70% to fix a
  simulator artifact, and shipped it to hardware.

`tasks.tunables()` stays — it is what the sliders read. **A declared bound is a
range a human may drag, not a claim the number is measurable.** Bringing a tuner
back needs objects in the model, torque that reaches the actuators, and a score
that is task success rather than execution success (`contacts_alive`,
`sweep_at_least`, `contact_force_above` in the `manipulation` specs are the
shape of it). Until then the honest instrument is **Run to row** with a human
watching.

## Execution backends

Each backend can drive either `TaskRunner` or `PolicyRunner`. It reads the hand
or simulation, builds the same millimetre observation, ticks exactly one
controller, and applies the returned command. `servo.py` is only the real/mock
bus adapter; it is not the common sim/real interface.

### `studio.live`, servos and a page to watch them on

```
ctrl   what you asked for    slider -> mm -> counts -> set_positions
qpos   what the hand did     read_all -> counts -> mm -> the model
```

The model's pose is never the slider. It is what came back off the bus, so the
gap between where you asked and where the model is shows the tracking error,
live.
Both directions are one packet, and they are the same two packets any control
loop already sends, so closing the loop costs nothing over watching it.
`read_all` on 7 servos is 1.47 ms and `set_positions` is 0.00 ms, because a
sync-write is a broadcast with no reply. That is a 679 Hz ceiling, 7% of the bus
at 50 Hz.

The page is [viser](https://viser.studio/): ten visual meshes read straight off
the compiled `MjModel`, between two windows.

```
┌──────────────────┐                              ┌──────────────────┐
│ tasks            │                              │ torque armed     │
│  Zero hand       │                              │ mm err load °C   │
│  Cycle cap ...   │       the hand, in 3D        │  (7 rows, 10 Hz) │
│  tune a task ▸   │                              │ 0..6 goal   (mm) │
│  TASK TIMELINE ▾ │                              │ tuning ▸         │
└──────────────────┘                              └──────────────────┘
┌──────────────────┐
│ camera           │
│  [ live frames ] │                              viser's own panel
└──────────────────┘
   floated left by CSS
```

Two windows because the halves are used at different times, and the left one is
used *while watching the right*: `Run to row` is author, run, read the error,
adjust. Tabs were tried first and put the timeline on the tab you cannot see.
Watch-only mode (`--panel False`) builds the left window and nothing else;
editing a task file never needed permission to move the hand.

**The left window is a CSS trick, and a fragile one.** viser serves exactly one
control panel — the theme picks floating/collapsible/fixed for it, and no
message opens a second — so the task folder is built in that panel like
everything else and then taken out of flow by a stylesheet the page serves
itself (`studio.TASK_MENU_CSS`). The selector counts DOM levels from a marker
`<div>` up to the folder's root, so a viser client that adds or drops a wrapper
renders the menu quietly back inside the right-hand panel. That is invisible to
Python: checking it needs a real browser driven against the served page, which
nothing does today.

The panel shows numbers rather than bars: 0.01 mm of tracking error is a real
number and zero pixels, and a bar would have to be built wider than `config`'s
travel to show a rail overrun. There is no GL context in this process.
`--studio False` falls back to MuJoCo's passive viewer, and `--no-viewer`
prints millimetres.

The camera window is the same CSS trick, floated bottom left, and the one thing
in the studio that runs off the control loop. `--camera` defaults to `auto`,
which is whichever USB camera is plugged in and no window when there is none
(`--camera 'OBSBOT Meet 2'`, `/dev/video4`, `0`, a video file, or `none`).
`cartesian_hand/camera.py` opens it MJPG at 720p on a thread of its own and
does capture, resize to 480 px, colour convert and JPEG encode there, handing
viser's server thread nothing but bytes to send at 15 Hz. That split is the
whole design: `cap.read()` blocks for a frame period — 33 ms at 30 fps, longer
than a 20 ms control tick — so a preview on the loop would make every servo
write late. opencv is imported only when a camera is asked for.

The loop writes `qpos` and calls `mj_forward`, never `mj_step`. Stepping would
re-simulate, and gravity and contact would pull `qpos` away from the values the
hand reported, so you would be watching MuJoCo's physics rather than the hand.
`mj_forward` also does not clamp to `jnt_range`, so a real pose outside the
model's declared travel renders as it truly is. With the travel tables
disagreeing by up to 76%, that is the measurement, not a rendering fault.

A page button submits a task, and the loop ticks its `PolicyRunner` or
`TaskRunner` in place of the sliders, still as the only writer on the bus. The
earlier design ran tasks on their own thread, which put two writers on one bus,
so the loop had to stop commanding while a task ran. That gate was a second
control path with its own bugs, and it could not exist in sim at all.

### `sim.run`, MuJoCo

Same task files, run unmodified. What differs is six lines: `mj_step` and a
`qpos` read where the other has `set_positions` and `read_all`.

Two things it does not cover. **Effort is ignored here.** The MJCF's actuators
are `<position>` with a fixed `kp`, so there is no per-joint gain to write it
into, and every `stop="stuck"` fires against a joint limit rather than against a
grip that yields at a tuned force. That is enough to prove a task runs and not
enough to transfer a *force*. The stock MJCF also has no bottle or cap, so it
cannot produce honest object contact or task success. `sim.run` is CPU and N=1;
`sim.run_warp` runs the same task or policy over batched GPU worlds. Batching is
there, but the missing object and effort dynamics still make automatic tuning
misleading.

`mj_step` here and `mj_forward` in the studio, for opposite reasons. In the
studio the hand supplies the physics, and re-simulating would overwrite what it
reported. Here the physics is all there is. Stepping is also what applies the
`<equality>` couplings, so a rack pair's follower moves on its own instead of
needing the explicit write the studio loop does.

Live visualization is the viser page in `studio.py`. There is no shipped offline
video wrapper. If one is needed, use `mujoco.Renderer` on the already-loaded
model; raise `model.vis.global_.offwidth/offheight` before constructing it and
raise the default headlight for this dark model. A second robot_studio/VTK scene
representation is not justified for frame or video output.

## Zeroing

Zeroing turns encoder counts into millimetres with an absolute meaning. Three
phases, in mechanical order: fingers retract before jaws close, jaws clear before
z drops. Run them the other way and a finger is inside a jaw that is closing.

It runs in a relative frame, and needs to. Millimetres are undefined before it
completes, and what the task needs is *distances* rather than absolute
positions. Every goal is `here ± something`, so the frame's origin cancels and
the task is correct
in the startup-relative frame an uncalibrated hand already uses, in the calibrated
frame, and in the sim's where `q = 0` is the rest pose. One task, three frames,
no branch.

The consequence is real: **a zeroing goal is outside the travel table on
purpose.** The seek asks for 120 mm of travel on a 55 mm rail because the stop,
not the number, is meant to end the move. So `studio.live` does not clamp task
goals, only slider goals, on the grounds that a human at a slider can ask for
anything and a program's goals were already bounded when it was built. What keeps
the unclamped path safe is direction: millimetres decrease toward the closed end,
which is a hard stop that physically exists, and the only unbounded request goes
that way. Every outward move a task makes is bounded at build time by
`clamped_mm`.

> Do not "fix" this by clamping in the loop. It quietly turns the seek into a
> no-op when the hand starts near 0.

**A DOF that never stalls aborts the run, and nothing is written.** A joint that
ran out of budget also stopped moving, and where it ended up cannot tell the two
apart, so the outcome is checked rather than the position. Recording a timed-out
DOF puts the origin somewhere mid-travel and makes every later millimetre on that
axis wrong by however far it fell short, without any warning, in the direction of
the open end of the rail. Offsets accumulate in a local tensor and are saved only
after all three phases succeed, so a failed run leaves the previous calibration
intact and needs no rollback branch.

Offsets are written to `zero_offsets.json` at the project root, keyed by hand
name. Project root and not the package directory, because `pip install -e .`
wipes the latter and a lost calibration costs a bench session; gitignored,
because the numbers describe one physical machine. Override the location with
`CARTESIAN_HAND_CALIB`. That file is generated, so do not edit it by hand.

`Config.sets_datum = True` is what routes the result, rather than a
`name == "zero"` comparison, so a retuned zeroing variant in its own file still
installs its calibration.

## Configuration

A hand is one flat frozen dataclass, `HandConfig`, and a hand definition is two
lines:

```python
HAND_1 = HandConfig(name="hand_1", port="/dev/ttyACM0", first_servo_id=0)
```

Everything absent from that call comes from the shared tables at the top of
`cartesian_hand/config.py`. Only what is true of one unit and not the other is
written per hand, which today is the serial port, where its servo IDs start, and
the zeroing creep torque.

An earlier version nested `Dof`, `Motion` and `Geometry` inside `HandConfig`.
Three extra types, a `cfg[dof].max_mm` to read one travel limit, and both hands
filling in identical `Motion` and `Geometry` objects. What is genuinely per-DOF
— axis, count direction, label — is the same on every hand built so far, so it
lives in `LAYOUT` once rather than in seven objects per hand.

`config.py` reads top to bottom: the shared tables (`LAYOUT` and the role names,
`STANDARD_TRAVEL`, `TORQUE_MIN_TO_MOVE`, `TORQUE_STUCK`, `CALIB_PATH`,
`DEFAULT_HAND`), the one dataclass, the two hands, then the calibration file
helpers. Nothing downstream holds a hardware constant of its own, so retuning a
gear ratio or a travel limit never means opening control code.

Prefer a `/dev/serial/by-id/` path over `/dev/ttyACM0`. ACM numbers are handed
out in enumeration order, so with two hands plugged in, a hardcoded number
silently addresses whichever powered up first.

Frozen matters for a concrete reason: the per-DOF tensors are cached per device,
so mutating `cfg.torque` after anything has called `gain_vector` leaves the old
torque in the cache and every later tick keeps commanding it. A docstring saying
"immutable" does not stop that; `FrozenInstanceError` does. Use `variant()`,
which drops the cache.

Everything per-DOF is a `torch.Tensor`, so that `device="cuda"` is the only
difference between the sim backend at N=4096 and the hardware backend at N=1.
numpy is faster at this size, about 0.3 µs against 3 µs for a 7-element convert,
but both are noise against a 20,000 µs tick at 50 Hz. So the tiebreak is having
one array library rather than a boundary to keep straight. The boundary that
remains is the bus, and it is in the right place: `read_all` returns Python
tuples with `None` for a servo that did not answer, and that `None` has to stay
`None`. Writing a placeholder hands the layer above a fabricated position, which
reads as a large jump, the opposite of the stall it is watching for.

### Motion gains

`torque`, `speed` and `acc` each take a scalar or a per-DOF sequence. A wrong
length raises at construction, not at the first servo write.

```python
HandConfig(..., torque=50)                              # all DOFs
HandConfig(..., torque=[50, 50, 50, 300, 50, 50, 50])   # z stage at 300
```

Mixing values costs nothing. A sync-write is one broadcast packet in which each
servo reads its own slice, so seven different torques and seven identical ones
are the same packet and the same time on the wire.

The z stage is the only DOF carrying a gravity load. Bisected on `hand_2`,
lifting 30 mm to 35 mm and measuring travel after 3 s:

| torque | 150 | 200 | 250 | 300 | 350 |
|---|---|---|---|---|---|
| moved (of 5.0 mm) | 1.50 | 4.54 | 4.54 | 4.54 | 4.53 |

150 stalls outright, 200 tracks fully, and nothing above 200 helps. `TORQUE_MIN_TO_MOVE` uses 300, the measured floor plus margin, because the
bisect ran unloaded and the stage has to lift the aux gripper while it is holding
something. Pressing *down* at 50 works and tasks rely on it, so this is a floor
for the lifting direction, not a correction to the whole axis.

`torque_stuck` is separate from `torque`, because zeroing presses each
DOF into its stop and the stall is the signal rather than a fault. Too much
torque binds before the stop, too little stalls short of it, and both read as a
hard stop in the wrong place.

It is per hand rather than one shared table, because the window between those two
failures is set by friction and friction is per unit: raising the number for a
stiff gear train would also push a looser hand's fingers through the stall window
and past their stop. Travel and gearing are shared; this is not. Both hands
currently run `TORQUE_STUCK`, the default, measured on `hand_2`. Retune one
without touching the other:

```python
HAND_1 = HandConfig(name="hand_1", port="/dev/ttyACM0", first_servo_id=0,
                    torque_stuck=(200, 60, 60, 300, 200, 60, 60))
```

The fingers came down from 80 to 50 after they climbed a gear tooth on `hand_2`:
at 80 the creep carried enough momentum that the stall window could not catch it
before it overshot. If a joint still sounds loaded at the end of a seek, come
down further on that hand.

`counts_per_mm` is derived from the pitch diameter, but a real gear train is not
its nominal drawing. After measuring a known travel, set it directly and the
derived value is ignored:

```python
HandConfig(..., counts_per_mm=80.0)
```

## Setting up a servo

Servos ship with an ID that collides with the rest of the bus, so each is renamed
before it goes into a hand. `hand_1` uses IDs 0-6 and `hand_2` uses 7-13, in
`LAYOUT` order. Connect one servo at a time, or the rename is ambiguous and the
new ID could collide with one already in use.

Keep the blocks non-overlapping. They are the only thing that tells one hand from
another over the bus, and `config.identify` uses them: `studio` with no `--hand`
sync-reads each hand's block and opens whichever one answers. Two hands on one
bus, or seven servos that answer where six should, is refused rather than
guessed. A new hand needs a fresh block and an entry in `config.HANDS`.

```bash
python -m hardware_bindings.ft_servo set-id /dev/ttyACM0 7
python -m hardware_bindings.ft_servo scan /dev/ttyACM0
python -m hardware_bindings.ft_servo gui /dev/ttyACM0 --ids 7 8 9
```

The GUI is worth having on the bench: a ping only proves that something answers
to the new ID, while motion proves it is the servo in front of you. These tools
take a device path and know nothing about hands, DOFs or millimetres. They need
`pip install -e 'hardware_bindings[cli]'`, or `[gui]` for the GUI. See
[`hardware_bindings/ft_servo/README.md`](hardware_bindings/ft_servo/README.md).

Renaming writes to the servo's EPROM and survives power cycles. Once renamed, the
only way to find a servo again is to scan for it.

## Running without hardware

`--mock` swaps the serial driver for a kinematic model with hard stops. Servos
ramp toward their targets and stall at the ends of travel, which is enough to
exercise unit conversion, task sequencing, and the studio loop:

```bash
python -m cartesian_hand.studio --mock --task zero
```

> **Redirect `CARTESIAN_HAND_CALIB` before every mock or sim run.** A `zero` on
> `MockServo` writes `zero_offsets.json` with the mock's stops, which overwrites
> the calibration of whatever hand is plugged into the machine. The file is
> gitignored, so the repository does not recover it. This has cost a bench
> calibration once.

The mock has no friction, no load-dependent stall and no following error. It
tells you whether your control flow is right, not whether your grip will hold. It
is also 870 times faster than the real bus, 1.7 µs against 1.47 ms for
`read_all`, so a mock-only profile points at the wrong thing every time.

## Layout

```
cartesian_hand/
  config.py      tunables, then the types, then the hands. Pure description:
                 no port, no threads, no I/O beyond JSON, so a twin can import it
  motions.py     the engine (Motions, Move, Program) plus TaskRunner
  policy.py      typed Observation, Action, Policy protocol, and PolicyRunner
  primitives.py  the tilt Step helper plus the direct closed-loop primitives,
                 the rows, and the Sequence that walks them
  tasks/         one file per task; the file stem is the --task name. A direct
                 task is two things and no more: a human-scale Config, and a
                 build() returning its row list. The state machine is Sequence's
                 and lives in primitives.py, so there is one of it rather than
                 one per task. They were six near-identical controllers in six
                 <name>_policy.py files until 2026-09-04, which is how one
                 deadline bug got copy-pasted into all of them
    zero.py      find every hard stop, report it as the hand's zero
    ready.py     send every DOF to mid travel; the studio's Reset button
    cap.py       probe, strokes, extract, re-thread. A bottle cap
    screwdriver.py  turn a screwdriver either way; cw presses z
    pipette.py   twist-lock knob, then plunge and draw
    syringe.py   clamp the body, draw the plunger, dispense
    scissors.py  two-handle tool; z travel is the pivot
    tilt.py      grip with both stages, pitch the object
  compose.py     writes a task file from a row list. Imports no GUI, so the
                 dependency runs studio -> compose -> tasks and never back
  studio.py      the hardware backend: live loop plus the viser page (WebStudio)
  sim.py         the MuJoCo backend
  mjcf.py        model path and the DOF-to-joint map, split out of studio so
                 sim does not import a web server to find a qpos address
  servo.py       the serial bus, and MockServo for offline runs
tests/           test_trace.py and its recorded traces.json. Plain asserts, no
                 framework. All that is left of the suite; see Test suite
hardware_bindings/  submodule: IMU, motor and servo bindings. Only ft_servo/ is
                    compiled here, and it is the sole copy of the servo driver.
assets/cartesian_hand/  the sim model: cartesian_hand.xml + meshes/, generated
                    and bundled so sim.py/studio.py need no sibling checkout.
                    Refresh with scripts/sync_sim_asset.py.
```

`config.py` says what the hardware is. `motions.py` is the fixed program path,
still used by `zero` and `tilt`; `policy.py` and `primitives.py` are the direct
path every other task takes. `studio.py` and `sim.py` execute both.

(`examples/twin_policy.py` and the root `test_cartesian_hand.py` were removed
2026-09-02: both still imported the deleted `cartesian_hand.hand` and could not
run.)

## Hardware status

Brought up on `hand_2`: servo IDs 7-13, one CH340 adapter at 1 Mbaud.

What has run on servos:

- All seven servos enumerate and report plausible voltage and temperature,
  11.3-11.5 V and 24-26 °C.
- 50 Hz measured in-loop with zero drops. A 5 mm goal on DOF 1 tracked to 0.01 mm
  of error.
- The loop is two bus packets per step regardless of gains: one sync-read
  covering all seven servos, one sync-write carrying per-joint positions and
  gains.

  | | 7 unicast | 1 sync | |
  |---|---|---|---|
  | read | 1.97 ms | 1.47 ms | one TX replaces seven, but each servo still replies |
  | write | 2.35 ms | ~0 ms | broadcast, unacked, so nothing to wait for |

  Sync-read saves less than it looks like it should, because it removes the seven
  request packets and not the seven replies.
- The tick is 89% sleep. Serial I/O is the only real cost in it, and it cannot be
  shrunk from Python.
- `acc` changes lag, not reachability. On the same 5 mm ramp, acc=25 peaks at
  1.37 mm of lag and acc=255 at 0.56 mm. All values arrive.
- Full zeroing completed under the *previous* implementation, with offsets that
  are multi-turn. `[4293, -1412, 5608, 5837, 4526, 2012, 6762]` includes values
  past one 4096-count revolution and one negative, so these servos are not
  running inside a single turn.

Not yet established:

- No direct manipulation policy has completed on its physical object. The
  canonical `--task cap` path does complete through the real `studio.live` bus
  executor with `MockServo` providing bottle/cap stops. This exercises the actual
  command conversion, low-speed contact approach, persistent grip, per-phase
  effort, and extraction sequence; only the mechanics are mocked. The stock
  MuJoCo model has no equivalent objects, so object simulation is not used as a
  completion gate.
- The four tasks ported from an earlier internal implementation are
  transcriptions. `screwdriver`, `pipette`, `syringe` and `scissors` enter the
  real executor path correctly, but the *sequences* are what was validated on
  hardware, not these implementations of them. Each needs its object and a
  calibrated hand.
  Two deliberate divergences from the validated code are recorded in the task
  docstrings: `syringe` opens the aux jaw at entry instead of closing it to
  `aux_min_mm` (the original relied on an unchecked `set_pos` timeout), and every
  ported task floors its horizontal and z-ascent efforts at that hand's measured
  `torque_min_to_move` rather than inheriting a flat travel torque.
- Total travel is unmeasured. Zeroing finds one hard stop per DOF, not both,
  so `STANDARD_TRAVEL` is still CAD. See Known issues.
- `counts_per_mm` is still the derived value, never checked against a measured
  distance.
- Grip force has not been characterised, and no policy has been transferred from
  a twin.

### Test suite

**The unit suite is gone. `tests/` was deleted on 2026-09-04.** Read every claim
in this file as a record of what was measured or reasoned at the time, not as
something a run will catch if it stops being true.

What stands in its place is one golden-trace check:

```bash
python tests/test_trace.py            # check against the recorded traces
python tests/test_trace.py --update   # re-record after an intended change
```

It runs all eight tasks against a toy plant in millimetres, with no bus and no
mujoco, and hashes every goal *and every effort* they command. It pins
behaviour, not the invariants the deleted tests reasoned about, but a change
that alters what a task puts on the wire cannot pass it silently. Traces live in
`tests/traces.json`; a run takes seconds.

Its coverage is worth stating exactly. Mutating `VELOCITY_WINDOW_TICKS` back to
a single tick, the dither bug below, fails all five policy tasks. So does
dropping the `travel_effort` torque floor, and that one is the reason effort is
in the digest at all: a goal-only digest passed that mutation, because no
backend here models force. The toy plant has no friction, no following error and
no encoder quantisation, so it still cannot see a stall that depends on how hard
a joint pushes. Do not read a pass as evidence about grip.

What went with it is worth knowing before trusting a number here. The suite held
the sim-versus-real direction and travel comparison, the dead-bus and 50 Hz
timing checks, the per-row torque floor assertion, and the case where a servo
dithering a single encoder count has to read as stopped. Several of those bugs
are invisible on every backend: mujoco and `MockServo` both ignore the torque
register, and neither quantises position, so a below-floor row or a one-tick
velocity window looks correct in simulation and hangs on the bench.

It is not recoverable from git. The pre-removal commits do not contain every
failing case the suite caught, and the mutation recipes exist nowhere else.

Counting tests measures nothing anyway. Mutate the code and re-run -- including
checking for the case where a test named for the exact bug can't actually see
it, because its body contains a copy of the logic it was meant to be checking.

## Known issues

**`max_mm` is a limit, not a description.** There is one hard stop per DOF, the
one zeroing seeks. The far end of each rail is open by design: drive past it and
the carriage leaves the slider and the servo spins free. `STANDARD_TRAVEL` is the
only thing that prevents that, and it has never been measured. It carries the v2
CAD figures, 50 mm on the jaws and z stage and 55 mm on the fingers.

For the jaw pairs there are four numbers, disagreeing by up to 76%:

| source | mm |
|---|---|
| `config.STANDARD_TRAVEL` | 50.0 |
| the sim MJCF `ctrlrange` | 52.6317 |
| the sim policy specs | 57.0 |
| `hand_2` DOF 0, stalled, measured | 29.8 |

The 29.8 is probably right. `EXPORT_NOTES.md` derives 52.6317 from raw Fusion
limits −30 to +22.6317 with the rack parked hard at +22.6317, assuming the whole
slider span is reachable. If the real stroke is only the 30 mm below the park
point, the measurement is the answer. The 57.0 is stale rather than a fourth
independent number: those specs were written against the pre-sign-flip model and
do not decode against the asset that ships today.

It cannot be measured by driving. A `travel` task tried, seeking the far stop and
reporting the span, and the premise is false because there is no far stop to
find. Run on `hand_2`, it took six of seven carriages off their rails. **Measure
with calipers and type the numbers in.** `counts_per_mm` is hand-wide, so one
axis calibrates all seven, while travel is per-DOF and each rail needs its own.

The disagreement was pinned by a comparator reading both sides live, so
reconciling either one failed deliberately rather than drifting quietly. That
file went with the rest of `tests/`, and the two sides can now diverge in
silence. The direction is not symmetric: **the sim narrows to the hardware,
never the reverse.** `config` is narrower on all seven DOFs and must stay so,
because the sim's extra stroke is not headroom, it is where a carriage leaves
its slider.

**The action offset disagrees, and it is worse than travel.** Scale already
agrees, since both sides move half the travel per unit of action. The offset does
not:

```
sim    mm = default_joint_pos + 0.5*(hi-lo)*a     ->  a=0 is the REST pose
real   mm = midpoint          + 0.5*(hi-lo)*a     ->  a=0 is MID-travel
```

Every MJCF `ctrlrange` starts at 0 and the constants file calls that pose "rest
(jaws shut)", so `a = 0` shuts the jaws in sim and opens them 25 mm on hardware.
The sim's whole negative action half is unused as well, since it clamps against a
range starting at rest. `config`'s convention is the better one and is the one to
keep, so the fix is one offset in the sim. No policy is trained on this hand yet,
so reconciling costs nothing today and invalidates checkpoints after the first
run.

**Saved zero offsets go stale by whole turns.** A servo reports (turns since
power-up × `counts_per_rev`) plus the angle within the current turn. The angle
comes off a magnetic encoder and is right the instant power arrives, but the turn
count restarts at zero. So the reading a saved offset was measured against no
longer exists, and every millimetre command after a power cycle is off by some
whole number of turns, without any warning, because the numbers stay plausible.
Observed on `hand_2`: four DOFs read 30 mm and three read a turn away.

The angle alone places the joint exactly *provided travel is shorter than one
turn*, and then the offsets can be recovered by arithmetic:

```python
k = ((raw - offsets) * orientation) % counts_per_rev
rebased = raw - k * orientation
```

**That recovery is not implemented in the current `config.load_offsets`**, which
returns saved offsets without checking that the turn origin still holds. Even if
it were, one motor turn is 50.27 mm at the derived `counts_per_mm` and the
fingers are configured at 55, so the widest DOF would put all seven in the case
that cannot be recovered. Getting every rail under 50.27 mm removes the ambiguity
outright, which is the same calipers measurement the entry above wants.

**The left and right finger labels are unresolved.** The MJCF calls DOF index 1
`m_right_down_finger`, while `config.LAYOUT` calls it the left one. The sim's
names come from Fusion bodies cross-checked with the designer, so they are the
better evidence, but neither source says which physical finger *servo* 1 drives,
which is the only thing that matters here. It fails without any warning, since a
mirrored policy still looks plausible on a symmetric gripper. Resolve it by
commanding DOF 1 alone and watching which finger moves. `--swap` exchanges DOFs
1 and 2, and 5 and 6, to test it.

**`fingerprint()` has no callers.** It exists to catch the travel mismatch above.
Recording it on whatever a twin produces, and rejecting a mismatch before driving
hardware, is still to be connected up.

**Mechanical, not code: DOF 1 on `hand_2` binds.** At the start of the 2026-08-31
session a 5 mm goal produced 0.09 mm of motion. Driving it ±400 counts a few
times freed it, after which the identical command tracked to 4.99 mm. The symptom
to recognise is full travel in one direction and about 20% in the other.

**Two servo drivers.** `hardware_bindings/ft_servo/ft_servo_python_only.py`
reimplements the same SCS wire protocol as the compiled extension, over
`pyserial`, and now has no importers at all: 404 lines reachable only by typing
its path. `set_position_offset` plus raw `unlock_eprom` and `lock_eprom` are the
only things it can still do that the extension cannot. Either bind those three
and delete the file, or keep it and accept that two drivers have to stay in
agreement with nothing enforcing it.

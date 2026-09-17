# Writing and tuning a task

How a task file is structured, how a `--task` name reaches it, what the direct
policy path offers, and how the studio's panels compose and tune one.

See also [internals.md](internals.md) for the engine those tasks run on, and
[hardware.md](hardware.md) for the hand they run on.

## Writing a task

A task module's `build()` returns a controller. Object manipulation returns a
`primitives.Sequence` — a typed `Policy` whose step machine is declared as a list
of rows (`Move`, `Probe`, `Hold`, `Twist`, `Loop`); fixed timelines and
pre-calibration zeroing return a `Task` generator of `Motions` programs. Both
keep hardware out of task code and are selected by the same task name. There is
no per-task `*Policy` class anymore; the row list is the policy.

The state machine is `Sequence`'s and lives in `primitives.py`, so there is one
of it rather than one per task. The tasks were six near-identical controllers in
six `<name>_policy.py` files until 2026-09-04, which is how one deadline bug got
copy-pasted into all of them.

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

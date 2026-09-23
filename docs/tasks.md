# Writing and tuning a task

A task is one file in `cartesian_hand/tasks/`. `--task squeeze` imports
`tasks/squeeze.py` and calls its `build()`. Start with the example below, then
read the section for the kind of task you are writing. The reasons behind the
design are collected in [Design notes](#design-notes) at the end.

See also [internals.md](internals.md) for the engine those tasks run on, and
[hardware.md](hardware.md) for the hand they run on.

Terms used on this page:

- **row**: one step of a task, such as "move here", "close until contact",
  "hold" or "twist".
- **env**: one of N copies of the hand run as a batch. Hardware is N=1, and the
  GPU sim runs thousands. Every tensor has N as its first dimension.
- **standing order**: the command a joint keeps sending after its row has
  finished. It is what keeps a grip squeezed while later rows run.
- **datum**: the hand's zero, which the `zero` task sets.

## A first task

```python
# cartesian_hand/tasks/squeeze.py
"""Close the base jaw on an object, then hold it 2 mm past contact."""
from dataclasses import dataclass, field

from ..config import BASE_JAW
from ..primitives import Hold, Probe, Sequence


@dataclass
class Config:
    label: str = "Squeeze"          # button text on the studio page; "" for none
    grip_torque: float = 300.0      # 0-1000, like every torque in this repo
    bite_mm: float = field(default=2.0, metadata={"tune": (0.0, 4.0)})  # a slider


def build(hand, start_mm, cfg=None, **kwargs):
    cfg = cfg or Config()
    return Sequence([
        Probe(label="probe", group=BASE_JAW, creep=True,
              measure={"contact_mm": BASE_JAW}),         # close until it stalls
        Hold(label="grip", group=BASE_JAW, effort=cfg.grip_torque / 1000,
             goal=lambda m: m.contact_mm - cfg.bite_mm),  # press bite_mm further
    ], hand=hand, start_mm=start_mm,
       travel_torque=50, approach_torque=150, approach_speed=800)
```

Save the file and it works straight away, with no registration step:

```bash
python -m cartesian_hand.sim --task squeeze                    # runs; fails at the probe (no object in the model)
python -m cartesian_hand.studio --hand my_hand --task squeeze  # zero the hand first
```

The studio page also gets a "Squeeze" button, and `bite_mm` becomes a slider in
the **tune a task** panel. `tasks/base_grasp.py` is the same two rows with the
bench measurements behind each number.

A task module provides two names:

- `build(hand, start_mm, cfg=None, **kwargs)` is required. `start_mm` is
  `[N, J]`, the current position of every joint in millimetres. It returns the
  controller.
- `Config` is an optional dataclass of defaults. `label` puts a button on the
  page (`""` means no button). `sets_datum = True` makes the task's result the
  hand's new zero. Any field with `metadata={"tune": (lo, hi)}` becomes a
  slider with that range.

Goals are absolute in the hand's millimetre frame, so zero the hand before
running a task that uses them. Without calibration, 0 mm is wherever the hand
was at startup.

## Rows

`build` returns a `primitives.Sequence`: a list of rows that runs top to bottom,
separately in every env.

| row | what it does | finishes when |
|---|---|---|
| `Move(goal={dof: mm})` | drives DOFs to a goal | every DOF is within `tolerance_mm` (1 mm) |
| `Probe(group=dofs, measure={"name": dof})` | closes DOFs until they touch, and records where | contact is confirmed. Reaching `goal` (default 0, fully closed) without contact is a failure |
| `Hold(group=dofs, goal=..., seconds=...)` | changes a standing order, or keeps every one for `seconds` | `seconds` have passed, or at once if none are given |
| `Twist(jaw=, left=, right=, radius=, span=, ...)` | one release, reset, re-grip and turn stroke | the stroke is done; `count=` repeats it |
| `Loop(count=..., rows=[...])` | repeats `rows`; `count` may differ per env | |

Fields every row accepts:

- `effort`: torque cap, normalized 0-1 (a torque of 300 is `0.3`). With none,
  the row uses the task's travel effort, or its approach effort for a `Probe`.
- `creep`: move at the slow contact speed instead of full speed.
- `loaded`: this row pushes an object, so its effort never drops below travel
  effort.
- `seconds`: deadline override. By default a deadline is derived from the
  distance and the commanded speed, times `timeout_margin` (1.5).
- `label`: shown in the trace. Nothing reads it.

`Probe(grip=...)` latches that effort the tick contact is confirmed, and the
grip stays on as a standing order, so later rows never restate the clamp. In
`cap` the base jaw keeps squeezing while the aux jaw twists and extracts.

A goal is a number, a tensor, or a function of earlier measurements.
`measure={"contact_mm": BASE_JAW}` on a row stores where that DOF stopped, and a
later row reads it as `lambda m: m.contact_mm`, an `[N]` tensor. Reading a name
that no earlier row stored raises an error that lists the names stored so far.

Rules that catch people out:

- **A row that fails ends the task for that env.** A row fails when it misses
  its deadline, or when a `Probe` closes all the way without touching anything.
  Every later row is skipped for that env.
- **Efforts have a floor.** Travel and approach efforts are raised per DOF to the
  hand's `torque_min_to_move`, because below it a joint does not move at all. An
  explicit `effort=` is used exactly as given (unless the row is `loaded`), so an
  explicit effort below the floor leaves the joint where it is until the
  deadline. See [Effort floors](internals.md#effort-floors).
- **Free moves run at the servo's rated top speed.** `Sequence` ignores
  `travel_speed` and the hand's `speed` gains for them. Use `creep=True` or
  `speed_scale=` to go slower. Contact closes at the task's `approach_speed`,
  and each kind of row gets its own deadline, so slow probing does not make
  ordinary moves wait on a contact-sized timeout.
- **z carries gravity.** To lift the stage while it holds something, use
  `primitives.lift_effort(hand)` as the effort.

`tasks/cap.py` is the full-size example, with probes, a `Loop`, two `Twist`
strokes and a z press. Its outline is under [Direct policies](#direct-policies).

## Variants

To change a task's numbers without copying it, write a new file next to it and
reach what it reuses through the module:

```python
# tasks/cap_gentle.py
"""Squeeze 80 -> 40: hand_2 crushed a PET cap on stroke 2 at 80."""
from . import cap

def build(hand, start_mm, **kwargs):
    return cap.build(hand, start_mm, cfg=cap.Config(squeeze_torque=40.0), **kwargs)
```

`--task cap_gentle` works immediately. Because `Config` is a dataclass,
`dataclasses.replace` stacks one variant onto another.

**Write `from . import cap`, not `from .cap import Config`.** The second form
binds `Config` into the variant's own namespace, which is where `tasks.config()`
looks. The variant would then report cap's `label` and add a second "Cycle cap"
button to the panel. To give a variant its own button, define a `Config` in the
variant, as `tasks/child_safe_cap.py` does.

Record bench results in the module docstring, so each result is versioned next
to the task variant it describes.

## Fixed-program tasks

A task can also be a generator of fixed `Motions` programs instead of a
`Sequence`. Use this form when the procedure is a fixed schedule whose outcomes
do not feed later rows, and for zeroing. Each `yield` hands a program to the
executor and evaluates to the `[N, J]` millimetres measured when it finished.
The task's `return` value ends up in `TaskRunner.result`:

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

`tasks/ready.py` is a shorter example, two steps built with `Program` directly.

**Report failure; do not raise it.** `Result.ok` is `[N]`, one flag per env. The
caller decides what a failure means: `sim.run` raises, `studio.finish` does not
save the calibration, and a batched trainer masks the bad envs. The task's only
job is to keep every goal it derives from a failed measurement bounded. In
`tasks/zero.py` a failed env parks where it is instead of at
`mid-rail + travel/2`, because the far end of the rail is open.

**A measurement can set a value in a later program, but not the number of
steps.** The Python loop that builds a program runs once, before the program
starts, so the program's length is fixed. Envs that need fewer strokes carry
`when=False` on the extra steps and idle through them. A task whose step count
depends on what it measures, such as `cap`, should be a `Sequence`.

## Direct policies

`Sequence` is one implementation of the backend-neutral `Policy` contract in
`policy.py`. Each tick a policy receives an `Observation` (`position_mm`,
`velocity_mm_s`, `contact`, `elapsed_ticks`) and returns an `Action` (`goal_mm`,
`max_speed_mm_s`, and `effort_limit` normalized to 0-1). Every field is a tensor
with N first, and hardware is N=1 on the same API.

Envs may be in different rows and use different parameters in the same
rollout. `Sequence.step` has no Python branch on a measurement and never copies
a tensor to the host.

Policy state and parameters are typed dataclasses with tensor leaves. An
optimizer may own a packed `theta[N,P]`, but it converts that once to a
task-specific dataclass such as `CapParameters`, and policy code never indexes a
parameter by string. Measurements may change values, masks, phases and tensor
counters, but never a Python loop bound or a tensor shape.

Five manipulation tasks are `Sequence`s transcribed from an earlier internal
implementation (not included in this repo) that ran on real objects. The
sequences are validated on hardware. These implementations of them have not
been run on the objects.

| task | from | what makes it its own file |
|---|---|---|
| `cap` | `caps_contact_based.py` | the canonical one; probe, twist, extract, re-thread, present |
| `screwdriver` | `manual_screw_driver.py` | both jaws hold **one** tool; the base jaw is taut-contact only and never squeezed. `cw` selects `reverse` + press |
| `pipette` | `pipetting.py` | longest sequence; a twist-lock knob, then plunge and draw as one shared push-to-stall stroke |
| `syringe` | `syringe.py` | no twist at all; the repeating DOF is **z**, and the jaw re-grips the plunger higher each stroke |
| `scissors` | `scissor_type.py` | no twist either; z travel drives the tool's pivot. Both jaws stay squeezed at the end |

`cap` as rows:

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

The stroke count is the cap's circumference divided by how far one finger can
slide. It is known only after the jaws have closed on the cap, and each env can
have a different one.

Both CLIs run a named task the same way. The sim command is wired correctly but
cannot complete until the model has a bottle and cap:

```bash
python -m cartesian_hand.studio --hand hand_2 --task cap
python -m cartesian_hand.sim --hand hand_2 --task cap
```

In Python, the same policy object goes into every executor.
`tasks.make("cap", hand, start_mm)` builds it, and `tasks.make("zero", ...)`
returns zero's fixed-program controller the same way:

```python
studio.live(hand="hand_2", policy=policy)
state = sim.run(policy, cfg)
state = sim.run_warp(policy, cfg, n_envs=N, device="cuda")
```

## Composing and tuning in the studio

`python -m cartesian_hand.studio` has two task tools, both in the **tasks**
window on the left of the page.

**tune a task**: pick a task, drag its numbers, and save the result as a
variant. The sliders come from the `tune` ranges on the task's `Config` fields.

**TASK TIMELINE**: open on every studio page, including watch-only mode. It is
a list of rows you can edit like a CAD history. Select a row
and its parameters load into the editors. **Apply** an edit in place, **Move
up/down**, **Delete**, or **Insert after**. A row is one action (joints, goal,
torque, and a finish condition of `goal`, `stuck`, `wait` or external
completion), optionally gated on an earlier stop outcome or position of a source
joint. Save the result as a new task file.

**Load task** fills the timeline from the task selected in the tune dropdown,
so you can start from an existing procedure. Tune changes the numbers a task
declared and keeps its structure. Load reaches everything a row has (joints,
stop rule, frame, condition), none of which is a `Config` field, and gives you
back a flat program to save under a new name. Two kinds of task cannot load,
and both say so: a `Sequence` (`cap`) has no fixed steps to show, and a
multi-program task (`zero`, `scissors`) has only its first program before it
runs, because the rest are built from measurements it has not taken yet. Goals come back as the
numbers the task computed at the hand's current pose.

**Run to row** runs rows 0 to the cursor on the hand and leaves it there, so
you write the next row from the pose the earlier rows actually produced. Use it
for `frame="here"` distances, which you can only choose from the pose itself.
It writes `tasks/_preview.py` (rewritten every run, gitignored) and runs that
file by name, like a task button.

Both tools save `cartesian_hand/tasks/<name>.py` through `compose.py`. The saved
file does not depend on the studio, so it runs under `sim.run` and on the bus
without the studio.

Inside these fixed programs, feedback works per env. Each successful row
records `stopped_at` and `stopped_ok`, and a later row evaluates its `When(...)`
condition once, when it starts. A timeout invalidates the outcome without
overwriting the last trustworthy position, and a row disabled by its condition
keeps the standing order.

## Design notes

**One state machine.** The row logic lives once, in `Sequence` in
`primitives.py`. When each task had its own controller file, one deadline bug
got copy-pasted into all six.

**`build` is the whole procedure.** In a generator task the phases are `yield`s
inside `build`. Splitting them into `probe_program`, `stroke_program` and
`extract_program` called from a fourth function would give one procedure four
names and one more place for the phase order to disagree with itself.

**No registry.** The file stem is the task name, and importing the file is the
whole dispatch. `label` and `sets_datum` are `Config` fields rather than
`LABEL` / `SETS_DATUM` module constants, so a task has one configuration object.
Constructing a `Config` runs no program, so the page lays out its buttons before
any task exists. Buttons are opt-in so that variants do not fill the panel.

**Failure is data.** Reducing the outcome with `.all()` and raising would let
one env out of 4096 discard the 4095 that succeeded, and the exception would
unwind the generator so they could not be recovered. Under domain randomisation
some envs are *supposed* to fail.

**`zero` and `tilt` stay generators.** `zero`'s uncalibrated negative overtravel
must not pass through the clamp that `studio.live` applies to every policy goal.
Nothing in `tilt` depends on a measurement.

**Load reads the built program, not the source** (`compose.rows_from`, the
inverse of `program_source`). A parser would only understand files `compose.py`
wrote, and would misread every hand-written task, whose goals are expressions
over a `Config` rather than literals.

**Run to row executes the file.** The preview goes through the same path as
every other task, so no bug can exist only in the preview.

**The panel is not in the execution path.** It writes the file and is done, so
the result runs under `sim.run` at N=4096 and on the bus without importing the
studio. Dependencies go studio → compose → tasks, never back.

**There is no automatic tuner.** The repo used to ship one, `search.py`: random
search over the declared `tune` ranges, ranked on `(row success, ticks)`, with
the winner written as a variant. We deleted it because it optimised against a
simulator that cannot see most of what it sampled:

- MuJoCo ignores torque. `sim.profile` takes the goal and nothing else, so every
  `*_torque` field is invisible there. That is 1 of `zero`'s 2 knobs, 3 of
  `cap`'s 9, and half of every composed task's, since each row emits a `goal_i`
  and a `torque_i`. The search sampled them anyway, and they showed up in the
  winner's report as if they were findings.
- The only gradient it found was trivial. Score rose with `timeout_margin` and
  with nothing else, which only means a longer budget lets more rows pass.
- It disagreed with the bench, and the bench was right. Sim scored `zero`'s
  shipped defaults at 0.50, while the hand_1 log in `tasks/zero.py` shows the
  same defaults reproducing the datum to 0.074 mm with no phase expiring.
  `--save-as` would have written a variant that slows real zeroing by about 70%
  to fix a simulator artifact, and sent it to hardware.

`tasks.tunables()` stays, because the sliders read it. A `tune` range is a
range a person may drag, not a claim that the number can be
measured. A tuner would need objects in the model, torque that reaches the
actuators, and a score based on task success rather than execution success.
`contacts_alive`, `sweep_at_least` and `contact_force_above` in the
`manipulation` specs show what that score would look like. Until then, use
**Run to row** with a person watching.

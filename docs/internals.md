# Engine and execution backends

This page covers the tensor protocol tasks are written against, the primitives
built on it, and the three backends that run them.

See also [tasks.md](tasks.md) for writing a task, and [hardware.md](hardware.md)
for the physical hand.

## The engine

`motions.py` holds a robot protocol written as tensors and advanced one tick at a
time. Tasks written as rows (`primitives.Sequence`, see [tasks.md](tasks.md))
do not build these tensors; fixed-program tasks such as `zero` do.

```
Program  [N, J, K]   written once by Program.build, read-only during a run
Runtime  [N, J]      plus one step counter per env, [N]

    N  environments in sim, or physical hands on real
    J  joints
    K  steps in the program
```

A **motion** is one cell: one joint, one goal, one torque, one stop rule. Motions
on different joints at the same step run together, and motions on the same joint
at different steps run in order. The cap task depends on that ordering, because
the aux jaw must release before the fingers slide and be back on the cap before
they turn.

`Move(goal, torque, stop, timeout_s, when)` has three stop rules. `"goal"`
retires on arrival, `"stuck"` retires on contact, and `"hold"` retires
immediately and leaves its command in place.

**A joint that has finished its move keeps commanding its last goal.**
`step_once` returns a goal and torque for every joint, including the finished
ones, and the engine calls that repeated command a *standing order*. Grips depend
on it. A probe commands the jaw past the object at reduced torque and lets the
object stop it, and the pressure stays on only because the finished joint keeps
commanding that goal. If a finished joint's goal were reset to its measured
position, every grip would release as soon as it was made.

Contact is sensed as **stall**, not as load. The servos do report a load byte,
but it reads zero at rest on every servo on both hands, so we cannot tell a
working decode from one that returns padding. Load is also counterintuitive. It
measures drive effort, which is highest during free motion and near zero at
rest, so a high `load` column on the page does not mean contact.

Deadlines are counted in ticks. Sim has no wall clock and does not run in real
time, so ticks are the only unit that means the same thing on both backends.

### Effort floors

**Every torque a task commands on a horizontal DOF is floored at the hand's
`torque_min_to_move`.** Below that floor a joint does not move, so the row can
neither arrive nor stall anywhere except where it started. It can only end at its
deadline, and the hand stands still between motions, waiting out each timeout in
turn. `tilt` shipped with 50 and 80 against hand_2's floor of 100, with budgets
of about 20 s per row, and `scissors` closed its contact probe at 150 against
hand_1's 250. The floor applies to probes as well as free travel. A contact seek
wants the *lightest* push that still moves, and a torque below the floor does not
move at all.

**z is floored too.** It can seem that z should be exempt, since its resistance
is gravity rather than friction and a descent should push less. But effort is a
force *cap*, so a free descent never reaches it and a lower cap buys no
gentleness. Meanwhile the flat travel torque of 50 against a z floor of 400 to
800 cannot move the stage in either direction. `cap`'s descent names z with no
explicit effort, so with z exempt it could not move the stage at all. It used up
its deadline and retired the environment, which silently skipped every row after
it. `Sequence.travel_effort` is `maximum(flat, floor)` on every DOF.

An explicit `effort=` on a row is not floored, on purpose, unless the row is
`loaded=True`, which raises it to at least travel effort. A row that presses z
into something names its own number, as `lift_effort` and `TwistPress.effort`
do. Watch for a row that names an explicit z effort *below* the floor. It is
used as given and probably cannot descend.

Nothing checks any of this. mujoco, `MockServo` and the toy hand all ignore the
torque register, so on every backend a below-floor row looks correct, only slow.

### Primitives

`primitives.py` serves both controller forms. One legacy helper remains, `tilt`,
which fills a `Step` (one column of a program: goal, torque and stop rule per
joint, `[N, J]`) for the task of the same name. It stays because the
four-way joint pairing is what callers get wrong, not the `Step.set` call. A
helper that only renames `Step.set` does not belong there.

The direct functions are closed-loop behaviors:

- `hold` changes named DOFs and preserves every other standing command;
- `move_to` owns convergence and a tick deadline;
- `close_until_contact` distinguishes contact, timeout, and reaching a closed
  goal with no object;
- `twist_stroke` owns release, finger reset, re-grip, and one twist;
- `strokes_for_revolutions` turns a measured radius into a per-environment
  stroke count.

`twist_stroke` takes two optional per-environment arguments, because three tasks
share the same stroke and differ only in these:

- `reverse` (`[N]` bool) swaps which finger opens the gap and which closes it,
  which runs the stroke in the other direction. `cap` unscrews and re-threads
  the same cap in a *single* task, so direction has to be tensor state rather
  than a choice at the call site.
- `press` (`TwistPress`) drives a third axis to depth after the re-grip, holds
  it through the turn, and backs off before the next release. Threading a cap
  back onto its bottle and driving a screw both engage by depth as well as
  rotation. `return_effort` is separate from `effort` because backing off
  *raises* the stage, and z torque depends on direction.

An environment with no press skips both z phases on the transition itself
instead of idling a tick in each, so a stroke without a press issues exactly the
actions it would if the parameter did not exist.

The closed-loop primitives take an `[N]` active mask and return an `Action`,
typed state, and a `PrimitiveResult`; `hold` is the smaller action-update helper.
Results expose `succeeded`, `timed_out`, `reached_goal_without_contact`,
`stopped_at_mm`, and validity tensors. Information passes between rows through
`SequenceState.measured`, which a later row reads as `lambda m: m.radius`.
Primitives never inspect or change one another's private state. Passing results
through state into the next row's inputs lets closed-loop primitives share
information without dictionaries, numbered registers, host-side branches or
backend objects.

**Toward the stop is negative on every DOF, with no `orientation` term.** The
seek commands `here - overtravel` in millimetres, and `counts_to_mm` has already
applied `orientation` on the way in. A seek written in raw counts would need
`here - orientation * overtravel`, and that sign comes out wrong about half the
time someone works it out again. It happened twice on this project. Working in
millimetres removes the mistake instead of correcting it. The cost is that a
wrong sign is **invisible in millimetres** and can only be caught per DOF in
counts, which nothing does today.

## Execution backends

Each backend can drive either `TaskRunner` or `PolicyRunner`. It reads the hand
or simulation, builds the same millimetre observation, ticks exactly one
controller, and applies the command it returns. `servo.py` is only the real/mock
bus adapter, not the shared sim/real interface.

### `studio.live`, servos and a page to watch them on

```
ctrl   what you asked for    slider -> mm -> counts -> set_positions
qpos   what the hand did     read_all -> counts -> mm -> the model
```

The model's pose never comes from the slider. It comes from the bus, so the gap
between the slider and the model is the live tracking error. Each direction is
one packet, the same two packets any control loop already sends, so closing the
loop costs nothing over watching it. `read_all` on 7 servos takes 1.47 ms and
`set_positions` 0.00 ms, because a sync-write is a broadcast with no reply. That
gives a 679 Hz ceiling, and at 50 Hz the bus is 7% busy.

The page is [viser](https://viser.studio/): ten visual meshes read straight from
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

There are two windows because you use the halves at different times, and you use
the left one *while watching the right*. With `Run to row` you author, run, read
the error and adjust. With tabs, the timeline would sit on the tab you cannot
see. Watch-only mode (`--panel False`) builds only the left window, because
editing a task file does not need permission to move the hand.

**The left window is a fragile CSS trick.** viser serves exactly one control
panel. The theme chooses whether it floats, collapses or stays fixed, and no
message opens a second one. So the task folder is built in that panel like
everything else, then pulled out of the layout by a stylesheet the page serves
(`studio.TASK_MENU_CSS`). The selector counts DOM levels from a marker `<div>` up
to the folder's root, so if a viser release adds or drops a wrapper, the menu
silently renders back inside the right-hand panel. Python cannot see this.
Catching it needs a real browser driven against the served page, and nothing
does that today.

The panel shows numbers, not bars. A tracking error of 0.01 mm is a real number
and zero pixels, and a bar would have to be wider than `config`'s travel to show
a rail overrun. This process has no GL context. `--studio False` falls back to
MuJoCo's passive viewer, and `--no-viewer` prints millimetres.

The camera window uses the same CSS trick, floated bottom left, and it is the one
part of the studio that runs outside the control loop. `--camera` defaults to
`auto`, which picks whichever USB camera is plugged in and shows no window if
there is none. Other values are a name (`--camera 'OBSBOT Meet 2'`),
`/dev/video4`, `0`, a video file, or `none`. `cartesian_hand/camera.py` opens
the camera as MJPG at 720p on its own thread and does capture, resize to 480 px,
colour conversion and JPEG encoding there, so viser's server thread only sends
bytes, at 15 Hz. The separate thread matters because `cap.read()` blocks for a
frame period, 33 ms at 30 fps, which is longer than a 20 ms control tick. A
preview on the loop would make every servo write late. opencv is imported only
when a camera is requested.

The loop writes `qpos` and calls `mj_forward`, never `mj_step`. Stepping would
re-simulate, and gravity and contact would pull `qpos` away from what the hand
reported, so you would be watching MuJoCo's physics instead of the hand.
`mj_forward` also does not clamp to `jnt_range`, so a real pose outside the
model's declared travel renders where it actually is. With the travel tables
disagreeing by up to 76%, that is a measurement, not a rendering fault.

A page button submits a task, and the loop ticks its `PolicyRunner` or
`TaskRunner` in place of the sliders, still as the only writer on the bus.
Running tasks on their own thread would put two writers on one bus, so the loop
would have to stop commanding while a task ran. That gate is a second control
path with its own bugs, and it could not exist in sim at all.

### `sim.run`, MuJoCo

`sim.run` runs the same task files unmodified. The difference is six lines:
`mj_step` and a `qpos` read where the studio has `set_positions` and `read_all`.

It has two gaps. **Effort is ignored.** The MJCF's actuators are `<position>`
with a fixed `kp`, so there is no per-joint gain to write effort into, and every
`stop="stuck"` fires against a joint limit rather than against a grip yielding at
a tuned force. That is enough to show a task runs, but not enough to transfer a
*force*. The stock MJCF also has no bottle or cap, so it cannot produce real
object contact or task success. `sim.run` is CPU with N=1, and `sim.run_warp`
runs the same task or policy over batched GPU worlds. Batching works, but
without objects and effort dynamics, automatic tuning would still be misleading.

The studio uses `mj_forward` and the sim uses `mj_step` for opposite reasons. In
the studio the hand supplies the physics, and re-simulating would overwrite what
it reported. In the sim, the physics is all there is. Stepping also applies the
`<equality>` couplings, so a rack pair's follower moves by itself instead of
needing the explicit write the studio loop does.

Live visualization is the viser page in `studio.py`. There is no offline video
wrapper. If you need one, use `mujoco.Renderer` on the loaded model. Raise
`model.vis.global_.offwidth/offheight` before constructing it, and turn up the
default headlight, because the model is dark. A second robot_studio/VTK scene is
not worth adding just for frames or video.

### Running without hardware

`--mock` swaps the serial driver for a kinematic model with hard stops. Servos
ramp toward their targets and stall at the ends of travel, which is enough to
exercise unit conversion, task sequencing and the studio loop:

```bash
python -m cartesian_hand.studio --mock --task zero
```

> **Redirect `CARTESIAN_HAND_CALIB` before every mock or sim run.** A `zero` on
> `MockServo` writes `zero_offsets.json` with the mock's stops, which overwrites
> the calibration of whatever hand is plugged into the machine. The file is
> gitignored, so git cannot restore it. This has already cost us one bench
> calibration.

The mock has no friction, no load-dependent stall and no following error. It
tells you whether your control flow is right. It cannot tell you whether a grip
will hold. It is also 870 times faster than the real bus (1.7 µs against 1.47 ms
for `read_all`), so a profile taken on the mock always points at the wrong thing.

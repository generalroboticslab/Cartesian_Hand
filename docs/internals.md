# Engine and execution backends

The tensor protocol tasks are written against, the primitives built on it, and
the three backends that tick them.

See also [tasks.md](tasks.md) for writing a task, and [hardware.md](hardware.md)
for the physical hand.

## The engine

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
travel torque — 50 against a z floor of 400 to 800. `cap`'s
descent names z with no explicit effort, so it could not move the stage at all,
burned its deadline, and retired the environment, which silently skipped every
row after it. Both halves of the reasoning are wrong: effort is a force *cap*,
so a free descent never approaches it and the exemption bought no gentleness,
and 50 against that floor is immobile in both directions.
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

### Running without hardware

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

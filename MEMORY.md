# MEMORY

Lossy cache. Code is ground truth; on conflict trust code and fix this file.

## Where things are

- Sim asset: `~/repo/legged_env_v2/asset/cartesian_hand/` — `cartesian_hand.xml`
  (generated, never hand-edit; rebuild with `cartesian_hand_creation.py`),
  `source/kinematics.json` + `source/EXPORT_NOTES.md` for provenance,
  `manipulation/policy/{executor.py,specs/*.json}` for the existing task stack.
- Current architecture: `README.md`. Fixed timelines use
  `Motions.step_once`; closed-loop manipulation uses typed
  `Policy.step(Observation, State) -> (Action, State)`. Both are tensor paths
  executed by the same real, CPU MuJoCo, and Warp loops.
- `cartesian_hand/studio.py` — the live loop (`live`) and the viser page
  (`WebStudio`, renderer + mm sliders) in one file. Reads the hand, drives mujoco
  `qpos` so the model shows the real pose. The instrument for both open ❓ blocks
  in `config.py`. Also the hardware **executor**.
- `cartesian_hand/motions.py` — the fixed-program engine (`Motions`, `Move`,
  `Program`) plus `TaskRunner`. `cartesian_hand/policy.py` — typed direct-policy
  contract plus `PolicyRunner`. `cartesian_hand/primitives.py` — legacy pairing
  helpers, direct `hold`/`move_to`/`close_until_contact`/`twist_stroke`, and the
  `Move`/`Probe`/`Hold`/`Twist`/`Loop` rows that `Sequence` walks. There is no
  `cap_policy.py`; `Sequence` is the one manipulation policy and every task is a
  row table for it. `cartesian_hand/tasks/` — `zero.py`, `ready.py`, `tilt.py`
  (`Motions` generators) and `cap.py`, `scissors.py`, `syringe.py`,
  `screwdriver.py`, `pipette.py` (`Sequence` row tables). No `bulb.py` exists.
  `cartesian_hand/sim.py` — the mujoco executor. `cartesian_hand/mjcf.py` —
  model path + DOF↔joint map, split out of `studio` so `sim` does not import
  viser to find a qpos address.
- Previous implementation: `cartesian_hand_old/`. Reference only.
- Plans: `plan/handoff.md` (canonical direct cap policy),
  `plan/direct_task_ports.md`, `plan/carrier_coordination.md` (design only,
  not started — hand ↔ arm/carrier; gated on three unanswered questions at the
  top of that file).

## Conventions

- **CLI is tyro, never argparse.** `tyro.cli(fn)` off annotations plus a
  Google-style `Args:` block, which tyro renders as per-flag help. Params that
  are not CLI surface (a callable seam like `goal_mm`) get
  `Annotated[..., tyro.conf.Suppress]`. Consequence to expect: a tri-state
  `bool | None` renders as `--studio {None,True,False}`, so `--no-studio`
  becomes `--studio False`. Plain bools still get `--x / --no-x`.

## Tasks build controllers; executors own I/O (2026-09-03)

`tasks.make(name, hand, start_mm)` returns either a typed `Policy` or a fixed
`Motions` task generator. `cap` is the canonical direct policy; `zero` and the
fixed timeline tasks retain `TaskRunner`. This split is intentional: cap keeps
runtime feedback and phase in batched tensor state, while zeroing must issue
uncalibrated negative overtravel that the normal direct-action clamp rejects.
Neither controller form opens a bus, sleeps, or steps a simulator.

The same named invocation selects the same controller file on both executors:

    python -m cartesian_hand.studio --task zero     # headless, real hand
    python -m cartesian_hand.sim --task zero        # same file, mujoco
    python -m cartesian_hand.studio --task cap      # direct policy, real loop
    "Zero hand" / "Open cap" buttons                # same files, page

For fixed-program tasks, `yield`'s return value is the measurement bus: zeroing
seeks a stop, receives where it stopped, and parks relative to that. Direct
policies instead exchange typed `PrimitiveResult` tensors through task state;
cap stores its probe position as radius and passes it to `twist_stroke`.

**Zeroing does not need raw counts, and does not need a calibration to run.** It
needs *distances*: every goal is `here ± something`, so the frame's origin
cancels. Correct in the startup-relative frame, the calibrated frame, and the
sim's — one task, three frames, no branch. The previous implementation drove the
bus in raw counts for this reason and was therefore unsimulatable and
unbatchable; the reason was wrong.

**In millimetres the hard stop is always in the −mm direction, on every DOF,
with no `orientation` term.** A raw-count seek must write
`here - orientation * overtravel` and gets the sign wrong about half the time it
is re-derived (it did, twice). In mm the sign cancels inside `counts_to_mm`.
Working in mm deleted that bug class rather than fixing it. What this costs:
`tests/test_tasks.py` USED TO assert the resulting stall **in counts** per DOF, because
the failure is invisible in millimetres.

**Task goals are deliberately NOT clamped to the travel table; slider goals
are.** Zeroing asks for 120 mm on a 55 mm rail on purpose — the stop, not the
number, ends the move — and it asks in a frame with no relation to the table.
Safety is direction and provenance: the only unbounded ask goes toward the
closed end where a stop physically exists, and every outward move a task makes
is bounded at build time by `clamped_mm`. Do not "fix" this by clamping in the
loop; it silently turns the seek into a no-op when the hand starts near 0.

**The three bugs the blocking zeroing had are structurally gone**, not fixed:
phase-position-vs-DOF indexing (no index into a phase exists any more),
parking against a zero datum (the park in `tasks/zero.py` takes the stop as its
datum),
and one DOF's `max_mm` for a whole phase (per-DOF dict comprehension).

**`STANDARD_TRAVEL` hides the per-DOF park bug.** Every DOF inside a phase
happens to share a travel today — fingers 55, jaws 50, z alone — so
`cfg[dof_ids[0]].max_mm` passes every test against the shipped table. The park
test uses a `variant()` with mixed travel inside a phase for exactly this
reason. Mutation-confirmed against the now-deleted suite: against `STANDARD_TRAVEL` the mutant survives 8/8.

**Mutation results, 2026-09-02** (recipe in "Mutate the code", below):

| mutation | suite |
|---|---|
| never-stalled check removed | 7/8 ✅ caught |
| park datum hoisted to 0 | 7/8 ✅ caught |
| park travel hoisted to `dof_ids[0]` | 7/8 ✅ caught (mixed-travel variant) |
| seek drives `+overtravel` | 5/8 ✅ caught |

**Batch-independence must be checked on engine state, not on where a toy hand
ended up.** A batch runs until its *slowest* env is done, so a small cap sharing
a batch with a large one gets extra ticks to converge that it would not get
alone, and its final position differs by exactly one driver step. Compare
`held_goal` / `held_torque` / `outcome`; `test_batch_rows_never_interact` (deleted) says
so in its docstring.

**Anything that runs `zero` must redirect `CARTESIAN_HAND_CALIB` before
importing `config`.** `CALIB_PATH` is read from the environment at import time,
and `zero` saves on success *whatever backend it ran on*. A run that writes the
default overwrites the calibration of whatever hand is plugged into this
machine — a real one, costing a bench cycle. The previous `tests/test_zeroing.py` (deleted)
did exactly that, including `os.unlink(CALIB_PATH)` in a `finally`.

**`--mock` is NOT a dry run for this. 2026-09-04, cost the bench hand's
calibration.** `studio.live(hand="hand_2", mock=True, task="zero")` from a
scratch bench script wrote `zero_offsets.json` with `MockServo`'s stops,
`[6000]*7` — `STOP_HIGH`, the same on all seven DOFs, which is what the garbage
looks like. The file is **gitignored**, so nothing recovers it from the repo;
this one was reconstructed from a `[hand_2] zeroed:` line still in terminal
scrollback, and its `timestamp` now reads the mock write, not the bench run that
measured it. Export `CARTESIAN_HAND_CALIB=/tmp/…` for *every* mock or sim run,
not only under pytest. Recognising a clobbered entry: real offsets are seven
different numbers and some exceed 6000 (hand_2 DOF 6 is 8591), which the mock
cannot produce.

A stale calibration is also the cheapest explanation for **"a task hangs on its
first row"**: a datum from a different frame puts an ordinary goal outside the
rail, the joint parks against a limit, and the row has nothing to do but wait.
Check `zero_offsets.json` before reading any control code.

## Direct typed policy path (2026-09-03)

`Motions` remains for fixed programs and the GUI timeline. Feedback-heavy object
manipulation now has a direct per-tick path: `policy.py` owns the typed
`Observation`/`Action` contract and `PolicyRunner`; `primitives.py` owns typed
closed-loop behaviors AND the `Sequence`/`Move`/`Probe`/`Hold`/`Twist`/`Loop` rows
that every manipulation task is built from. There is no `cap_policy.py`; the row
table is the policy. Hardware is N=1, CPU MuJoCo is N=1, and Warp is batched;
policy code is unchanged.

Primitive information exchange is explicit. A primitive publishes a
`PrimitiveResult`; the row that follows reads what an earlier one named (e.g.
`measure={"radius": AUX_JAW}`) via a `lambda m:` or `Sequence`'s
`Measures.columns`. No dictionaries, backend objects, numbered registers,
`.item()`, or measurement-dependent Python
control flow belong in the policy contract.

The legacy standing-order bug is fixed: a newly yielded program inherits held
goals and torque, empty programs are skipped, and a `stop="stuck"` row that
reaches its goal retires immediately as a failed `GOAL` rather than waiting for
timeout or pretending it touched an object. The direct cap state separately
keeps base-jaw goal, speed, and effort commanded through twist and extraction.

`tasks/cap.py` is now a row list of `Sequence`; it is no longer a generator and
accepts no `cap_radius` bypass. `studio.submit("cap", ...)` selects
`PolicyRunner`, and the
`MockServo` supplies movable bottle/cap stops. That run covers mm/count
conversion, low-speed approach, velocity-stall contact, persistent base grip,
per-DOF speed/effort, lift, and presentation. Typed N=2 fixtures cover
independent phases and parameter rows.

The real-sequence comparison caught two implementation bugs before the mock bus
run: contact approach had inherited full travel speed instead of the validated
50-count/s default, and re-grip closed at squeeze effort without switching from
approach effort to the standing squeeze afterward. `CapParameters` now carries
the low approach speed, travel effort is `[N,J]`, and `twist_stroke` separates
contact effort from grip effort. Horizontal travel is floored at the selected
hand's `torque_min_to_move`; z descent remains light and z lift uses its own
floor.

This is still not a physical-bottle result. The stock MJCF has no bottle/cap
and fixed-kp actuators ignore per-environment effort, so object simulation is
not a completion gate yet. Do not restore automatic search until objects,
effort, and task-success observations are real.

## Zeroing's park was blind in both sims (2026-09-02)

Reported as "already zeroed at zero position, zeroing gets stuck". **There is no
software hang** — reproduced the start-at-the-stop case three ways (toy hand with
`floor == start`, `sim.run`, `studio.live --mock` with `start_counts=STOP_HIGH`)
and all three complete in seconds and record the stops at 0. The seek retires
`STUCK` on tick 10 exactly as designed. Do not go looking for a generator or
barrier bug there again.

What was actually broken was the **park**, twice, and neither backend can see it:

- **`park_torque = 300` was below the z stage's own `torque_min_to_move`** (350
  on hand_2, 800 on hand_1). Now a per-DOF floor:
  `floor[None, ids].clamp(min=park_torque)`. Necessary but **not** why z stayed
  down — see the speed entry below.
- **Every park budget expired mid-move.** `speed=300` counts/s over
  `counts_per_mm=81.487` is **3.68 mm/s**, so half a finger rail is 7.5 s against
  a fixed `park_timeout_s = 6.0`. Budgets are now derived —
  `zero.seconds_for(hand, mm, margin)` — so they track `HandConfig.speed`.

**Why both sims are blind: `sim.run` drops torque on the floor** (MJCF actuators
are `<position>` with fixed `kp`) **and `MockServo` ignores it** (a joint either
runs at commanded speed or is blocked). A torque that cannot move a real joint is
untestable on either backend. Assert on the *program the task issued*, which is
what `test_the_park_can_actually_move_every_dof_it_commands` (deleted) does. Mutation-
checked: reverting either half drops the suite to 15/16.

**2026-09-04, end of session: whole `tests/` directory was deleted.** No file
was kept, including `test_sim_real_contract.py` and `test_servo.py`. The bench
invariants those two files were guarding -- real-vs-sim direction disagreement,
dead-bus detection, 50 Hz timing, control-tick arithmetic, every load-bearing
number in this file -- were **fought for by those tests, not by hand** and are
no longer pinned. The operator's call and the unit suite is not coming back
from git: pre-removal commits did not include every failing case the suite
caught, and the mutation-recipe tests (which is most of the value here) live
nowhere else. Any claim that was being guarded by a test must be re-derived
end-to-end before it is trusted. A green import is no longer evidence of
correctness. **If something is regressed and you discover it on the bench, this
is why.**

**2026-09-05: `tests/test_trace.py` is the partial replacement.** Runs all eight
tasks against a toy plant in mm (no bus, no mujoco, seconds), hashing every goal
**and every effort** issued, against `tests/traces.json`. `--update` re-records.
It pins behaviour, not the invariants above, but the sim alone could never do
this job: only `zero` completes in `sim.run`, because with no bottle/cap/screw in
the model every manipulation task's first probe closes on air and raises. Trace
covers 8 of 8 where outcomes cover 1 of 8.

Mutation-checked both ways, and the second one is the reason it is worth
anything. `VELOCITY_WINDOW_TICKS` 10 -> 1 fails all five policy tasks. Dropping
`Sequence.travel_effort`'s `maximum(flat, floor)` **passed a goal-only digest** —
no backend here models force, so an immobilising torque moves no joint anywhere
and the trace was identical. Hashing effort as well is what catches it. **Any
future check on this repo owes the same trick**: assert on the command issued,
never on where the toy hand ended up.

Still blind, by construction: no friction, no following error, no encoder
quantisation. A pass says nothing about grip.

Two structural changes in the same session that the now-gone tests had pinned:

- **`Twist.count`** (`primitives.py:840`) carries the loop on the row itself.
  `cap.py` no longer wraps a `Twist` in `Loop(count=lambda m: ..., rows=[Twist(...)])`;
  it writes `Twist(..., count=num_revs)`. The `Loop` row type still exists for
  the other threaded tasks that may want multi-row bodies, but cap, the canonical
  policy, has the simpler form. `Sequence.twist_counts` is the `[N, T]` table;
  `SequenceState.twist_counts` is its per-tick mirror. **If a `Twist` with
  `count is not None` ever needs to live alongside a `Loop` that wraps it, the
  two bookkeeping channels collide -- the row index is the same number.**
- **z torque floor retraction** still stands, see below.

**2026-09-04: the same bug, on the HORIZONTAL DOFs, in `tilt` and `scissors`.**
Reported as "tilt waits out a timeout before every motion". Not the engine —
`travel_torque=50` / `grip_torque=80` against hand_2's floor of 100, and
`scissors`' `approach_torque=150` against hand_1's 250. A joint under its floor
cannot arrive and cannot stall anywhere but where it started, so the row's only
exit is its deadline, and `travel_budget` prices those at ~20 s each. Both files
now floor every horizontal torque at `torque_min_to_move` the way `cap` already
did. The
floor applies to contact probes too: a seek wants the lightest push that *still
travels*. Generalized guard is
`test_no_program_row_is_commanded_below_its_own_torque_floor` (deleted), per row over both
tasks and both hands. **Any new `Motions` task needs this floor** — nothing in
`Step.set` can apply it, since the engine holds no `HandConfig`.

**2026-09-04, later: "z stays excluded from the floor" was WRONG and is now
retracted.** `Sequence.__init__` exempted z from `torque_min_to_move` on the
reasoning above, so `travel_effort[Z]` was `travel_torque/1000` = 0.050 against
z's measured floor of **800 on both hands**. `cap`'s `approach` names z with no
explicit effort, so it could not move the stage, burned its ~6 s deadline, and —
because `Sequence.step` folds `failed` straight into `done` — **retired the
environment, so every row after `approach` silently never ran.** Reported as
"cap skips lines 105-121 on the real hand". It did not fail fast either:
`_stalled`'s `never_moved` guard classes a joint that never moved as *not
started*, so the free-move fast-reject cannot fire and the row always pays the
full clock.

Both halves of the old reasoning are false. Effort is a force **cap**, so a free
descent's error never approaches it and the exemption bought no gentleness; and
50 against a floor of 800 does not make z gentle, it makes it immobile in *both*
directions. The code had already voted four times — `bulb`, `screwdriver`,
`syringe` and `pipette` each copy the exemption and then undo it with
`torch.where(z, lift_effort, travel_effort)`, `lift_effort` being that same
floor, and they do it for **descents** (`entry_effort`) too. `bulb.py:141-145` is
the exemption verbatim, negated at 177 with a comment naming this exact bug.

`Sequence.travel_effort` is now `maximum(flat, floor)` on every DOF. Blast radius
is only rows that name Z and give no `effort=`; across `tasks/` that is exactly
`cap.py:104`. **An explicit effort is still not floored** (`_effort` only raises
a `loaded` row), which is deliberate: "a row that presses z into something names
its own number" is the real contract, and `lift_effort` / `TwistPress.effort`
already are it. Untested consequence to check on the bench: `bulb`'s
`reinsert_z_torque = 150` is an explicit below-floor z press and survives
unchanged, so it probably cannot descend at all — same bug class, different row.

Second reason it stayed hidden: **only the seek's outcome is ever read**
(`seek.all_reached`). A park that times out is silent by construction, and that
is deliberate — see the `alive` mask. So a park bug shows up as a hand in the
wrong place, never as `ok=False`.

**2026-09-04: the policy path could not detect contact on real hardware at
all.** `STUCK_SPEED_MM_S = 0.3` mm/s at 81.5 counts/mm and 50 Hz is **0.49
encoder counts per tick** — under the quantum. `PolicyRunner` differenced
position over a *single* tick, so a servo dithering one count against an object
reported 0.61 mm/s, twice the "not moving" threshold, forever. Contact never
registered, `close_until_contact` ran its full budget, and cap's probe budget is
20.4 s. That is the "waits 20 seconds", "stuck after the second step" report, in
that order, over two sessions.

`motions.py` already had this right and says so at `STUCK_WINDOW_STEPS`: a
tumbling 10-tick window, with a comment recording that ±1 count of dither is
what "recorded hard stops mid-rail on hand_1". The policy path was written with
a one-tick difference and did not inherit it. Fixed in the one place velocity is
*measured* — `policy.VELOCITY_WINDOW_TICKS = 10` — not in the threshold, so
every consumer is fixed at once and `primitives` needed no signature change.

**Neither backend can reproduce it.** `MockServo` integrates a float from
wall-clock time; mujoco reports a float `qpos`. Nothing quantizes, so nothing
dithers, so every simulated run passed while the bench hung. The regression test
(`test_a_joint_dithering_one_encoder_count_reads_as_stopped` (deleted)) feeds counts
directly for that reason. Mutation-checked: restoring the one-tick difference
reproduces `0.61 mm/s` and the test catches it. **Any new code building an
`Observation` by hand owes this window.**

**2026-09-04, same session: a blocked row no longer sits out its
deadline.** `primitives.move_to` measured the stall only when `stall_fallback`
was on, so a *free* move that stopped short — the case where stopping is the
fault — said nothing until the clock expired. `cap`'s approach budget is 6.1 s
and every "the task hangs after the first row" report is that silence. The stall
is now always measured; `stall_fallback` picks the verdict (arrived, or faulted)
and both verdicts land on the tick it is detectable — 0.52 s in the reproduction
above. **Open risk this buys:** a free row now fails if any joint it selects
reads under `STUCK_SPEED_MM_S` for `CONFIRM_TICKS` while more than 1 mm out, so
a servo hesitating 0.2 s mid-travel aborts the row where it used to recover.
Same detector every probe and every loaded move already trusts; if free rows
start failing spuriously on the bench, `confirm_ticks` is the knob, not a revert.

**Speed is the only lever for "zeroing is slow", and the task cannot reach it.**
`Motions` returns `(goal, torque)`; `studio.live` takes speed from `gains()`
(page slider, seeded from `cfg.speed`). Worst-case run at the shipped 300 is
**~67 s**; 600 → ~34 s, 1000 → ~20 s, and the derived budgets shrink with it. Not
raised unattended — it changes how fast the sliders drive a real hand too.

## Torque is a CAP; SPEED is what breaks stiction (2026-09-02, bench)

**z would not rise off its stop. Raising torque was the wrong lever and does not
fix it.** Operator finding: nudging the stage by hand starts it every time, and
it then runs fine. So the joint is not under-powered, it is stuck in *static*
friction and never develops enough force to break out.

The mechanism, and it applies to every DOF on this hand:

    the `torque` register is a FORCE CAP, not a force command.
    effort actually developed  ~  f(position error)
    slow profile -> setpoint stays near the joint -> small error -> small effort

At `speed = 300` (3.68 mm/s) the commanded setpoint creeps away from a stalled z
so gradually that the error never grows enough to exceed stiction, no matter
what the cap says. 800 on hand_1 was never approached. **Raise `speed` and the
setpoint outruns the joint, error grows, effort grows, stiction breaks.**

**`acc` was the lever, not `speed`. Confirmed on the bench 2026-09-02: `acc`
25 → 200 freed the z stage.** `speed` alone at 1200 changed nothing, because at
acc=25 the profile is still ramping and never reaches its top speed — the ramp
rate is what builds position error out of a standstill. `config.STANDARD_ACC`
is now 200, matching what the previous implementation used in **both** its seek
and its park. Do not lower it back toward 25 to make the fingers gentler
without re-testing z.

Consequences already applied:

- **`speed` is now per DOF on the page** (`WebStudio._add_tuning`, paired
  `{dof} speed` above `{dof} torque`). `config.HandConfig.speed` was always
  `int | Sequence[int]` and `studio.live` always read it per DOF — only
  `WebStudio.gains` collapsed it with `[self._speed.value] * n`.
- **The docstring that said "speed and acc are lag, not reachability, so no DOF
  here needs its own" is FALSE and is now corrected in place.** For z, speed is
  reachability.
- `zero.seconds_for` takes the phase's `dofs` and mins over those, so a fast z
  is not paced by slow fingers: z seek 20.4 s → 5.1 s, z park 10.2 s → 2.5 s at
  `speed[3] = 1200`.

**The one thing that cannot be matched: the old code set speed PER PHASE**
(seek 60, park 500) and `Motions` carries goal and torque but no speed, so a DOF
gets one speed for the whole session. z is on the old *park* value, 500, since
the park is the move that was failing. The old seek used 60 and its comment
warns a fast creep overshoots the stop and climbs a gear tooth — **if z's
recorded stop starts reading long, that is the first number to suspect**, and
the real fix is a speed channel in `Motions`, not a compromise value here.

**Suspect every "z did not move" report is this, not torque.** The saved
calibration at 22:38:50 (`hand_1` z stop = count 6457) was written by a run that
reported success, which rules out an early `break` or a failed seek — both
refuse to save. A false stall from the same cause is the open risk: a z that
cannot break stiction reads `< 0.3 mm/s` within 10 ticks and reports STUCK
wherever it was standing. Re-zero from two different z heights and compare
`offsets[3]`; if it tracks the start height, the recorded datum is garbage.

## Sim/real alignment, as of 2026-08-31

**DOF order agrees.** `config.LAYOUT` axis sequence y,x,x,z,y,x,x matches the
MJCF actuator order one-for-one. `base` == sim `down` (base-carried), `aux` ==
sim `up` (bridge-carried). Do **not** re-order to match
`source/kinematics.json`'s `_actuators_note` CAD servo numbering — that is a
third scheme (y,y,z,x,x,x,x) and matching it transposes four DOFs.

Mechanically pinned, not just read off a comment:
`tests/test_studio.py::test_dof_i_drives_actuator_i_s_joint` (deleted) walks
`model.actuator_trnid` and asserts DOF *i* maps to actuator *i*'s joint.

**`orientation` cannot fix a sim/real direction disagreement. Do not reach for
it (2026-09-01, cost a bench cycle).** DOFs 0, 2, 3, 4, 6 rendered backwards in
`studio` against a correctly-moving hand — exactly the `orientation == -1` set,
which is the only table in the repo with that pattern. Flipping all seven to
`+1` **reversed the real hand** and left the model where it was.

The reason is structural: `orientation` cancels between `mm_to_counts` and
`counts_to_mm`, so the millimetre the model renders is *invariant* under it. It
moves the hardware and nothing else. Any fix for a render-direction bug has to
be on the sim side of `counts_to_mm`. `test_config.py` (deleted) now pins the signs
literally; the pre-existing assert compared `sign(counts)` with
`cfg.orientations()` — the table against itself — and passed for any table.

Sim direction is not in doubt and does not need re-deriving: measured through
the real `WebStudio.push` path, `qpos=0` is jaws **fully shut** (inner gap
0.0000 m) and `+0.05` is a 0.100 m gap; `+mm` extends every finger `+x` and
raises the bridge. All seven joints, positive = more. So the disagreement is
**not** a sign anywhere between the slider and `data.qpos` — still open, and
the next thing to check is what the page actually draws (screenshot it), not
what `geom_xpos` says it should.

**Sim qpos order is not actuator order.** It is `bridge_z, left_up_y,
left_up_finger_x, right_up_y, right_up_finger_x, left_down_y,
left_down_finger_x, right_down_y, right_down_finger_x`. Always resolve with
`mj_name2id` + `jnt_qposadr`; positional indexing scrambles four joints and
still animates plausibly.

**9 qpos, 7 actuators, 2 equalities.** The rack-pair followers (`right_up_y`,
`right_down_y`) are driven by `<equality><joint>`, which the constraint solver
applies during `mj_step` only. Anything that writes `qpos` and calls
`mj_forward` must set the follower explicitly or one side of each jaw stays at
zero. Three renderers hit this trap. The viser path cannot: it renders from
`data.geom_xpos` after the loop's own `qpos` write, so there is no second place
to get it wrong.

**Travel does not agree, on any DOF.** Jaw pairs (DOF 0, 4) have four numbers:
50.0 (config), 52.6317 (MJCF), 57.0 (policy specs' ctrlrange span), 29.8
(measured, hand_2 DOF 0). Fingers: 55 config vs 60 sim. Z: 50 vs 60. A
normalized action therefore means a different millimetre on each side. Blocked
on calipers (plan item 2).

**The 29.8mm measurement is probably right.** `EXPORT_NOTES.md` derives 52.6317
from raw Fusion limits −30..+22.6317 with the rack parked at +22.6317, assuming
the full slider span is reachable. If the real stroke is only the 30mm below the
park point, 29.8 is the answer and the sim is ~76% too wide. Not settled.

**The policy specs are stale against the committed asset.** `specs/*.json` and
`executor.py` were authored pre-sign-flip (negative travel, asymmetric
ctrlranges `m_up_pair -0.02..0.035`, `m_down_pair -0.04..0.017`). The committed
MJCF is `[0, travel]` for all seven. `frac` values do not decode against it.

**Sim has no control rate.** `timestep=0.0005` (2000 Hz physics), executor writes
`ctrl` and steps; nothing declares a 50 Hz control loop. `control_hz` is in
`fingerprint()` with no counterpart on the sim side — a decimation of 40 has to
be asserted somewhere or the fingerprint match is theatre.

**Nothing calls `fingerprint()`.** It exists exactly to catch the travel
mismatch above and has zero callers. env.py's job.

**The action OFFSET disagrees too, and it is worse than travel (2026-09-01).**
Scale already agrees — both sides move half the travel per unit action (mjlab
`make_range_action_scale`, `DEFAULT_RANGE_ACTION_FRACTION = 0.5`, against
`config.denormalize`). The offset does not:

    sim   mm = default_joint_pos + 0.5*(hi-lo)*a      -> a=0 is the REST pose
    real  mm = midpoint          + 0.5*(hi-lo)*a      -> a=0 is MID-travel

Every MJCF ctrlrange starts at 0 and the constants file calls that pose
"export/rest (jaws shut)", so `a=0` shuts the jaws in sim and opens them
**25.0mm** (27.5 on fingers) on hardware. The sim's whole negative action half
is dead as well — it clamps against a ctrlrange starting at rest, so a policy
trains on half its output range. `config`'s convention is the better one and is
the one to keep. The fix is one offset in the sim, not a rewrite of either
action space.

*Caveat:* `default_joint_pos` comes from the entity init_state at graft time,
and no task wires this hand up (grep `mj_envs/tasks/` for "cartesian" — only a
handoff markdown answers). Rest-pose is inferred from the ctrlranges, not read.

**Direction is NOT symmetric — sim narrows to hardware, never the reverse.**
config is narrower on all 7 DOFs and must stay so: the far end of each rail is
open (`config.py:104`), so the sim's extra stroke is not headroom, it is where a
carriage leaves its slider and the servo spins free. Adopting the sim's numbers
on hardware is a hardware-damage bug. Recommended backwards once in review
before the rail-end note was re-read — check direction before repeating it.

**No policy is trained on this hand yet**, so reconciling costs nothing today
and invalidates checkpoints after the first run. Cheapest it will ever be.

**There is now a comparator: `tests/test_sim_real_contract.py` (deleted).** Reads both
sides live (config + the compiled MJCF, no mjlab import) and pins the recorded
disagreement, so reconciling either side fails it deliberately instead of
drifting silently. Mutation-checked both ways. It does **not** close the gap —
that edit lands in `legged_env_v2`, which 5 grafting robots import.

## Verified facts worth not re-deriving

- `FtServo` is thread-safe (`bus_mutex_`, documented lock order) and releases
  the GIL on every blocking call. A 50 Hz control thread does not block Python.
- `read_all` raises while `start_poll` runs (shared sync-read buffer).
  Unreachable through `servo.py`, which does not wrap `start_poll`.
- The `get_*` poll-cache family returns **zeros** when the poll thread is not
  running — indistinguishable from a bus at origin. Deliberately not wrapped.
- `set_positions(speed=0)` means *as fast as it can*, not "do not move".
  `speed=None` raises `TypeError`. Both pinned in `tests/test_servo.py` (deleted).
- numpy beats torch ~4-30x on 7-element converts, but both are noise against a
  20,000us tick. Tensors chosen for one array library, not speed.
- **Per-tick cost, measured 2026-09-01 at N=7 (20,000us budget @50Hz).** config's
  tensor work is 18us total = 0.09%: `counts_to_mm` 4.24, `clamp` 2.19,
  `mm_to_counts().tolist()` 6.70, `gain_vector`x3 2.78, cached `lower()` 0.66.
  The `_vec` cache saves 0.88us/call (1.54 build vs 0.66 hit) — nearly pointless
  on CPU, it survives on the plan's device requirement, not on speed.
- **The tick is 89% sleep. Nothing in `studio.live` is slow.** Measured
  2026-09-01 against hand_2 on `/dev/ttyACM0`, 20,000us budget @50Hz:

  | | us | % tick |
  |---|---|---|
  | `bus.read_all(7)` — **real hardware**, 1 Mbps | **1473** | 7.4% |
  | `web.push` + `report`, browser attached | ~600 | 3.0% |
  | config tensor work (all of it) | 18 | 0.09% |
  | `mj_forward` | 13 | 0.06% |
  | torch scalar scatter of the bus reply | 17 | 0.085% |
  | `time.sleep(period - elapsed)` (`studio.py:492`) | ~17,900 | ~89% |

  Serial I/O is the only real cost and it is not shrinkable from Python. Before
  optimizing anything in this loop, re-read this table.
  `bus.read_all` on the **mock** is 1.7us — 870x faster than the real bus, so a
  mock-only profile will point at the wrong thing every time.
- Fusing the precomputed `2/(hi-lo)` into `normalize`/`denormalize` is ~2x at
  N=4096 CUDA. **Not done** — 1-3% of a `mj_step`. Do not redo this measurement.
- **Commanding is free.** `set_positions` measures **0.00 ms** — a sync-write is
  a broadcast with no reply, so it returns as soon as it is queued. `read_all`
  on 7 servos is **1.47 ms**. Write+read is still 1.47 ms: a 679 Hz ceiling,
  ~7% of the bus at 50 Hz. Closing the loop costs nothing over watching it.
- Verified on hardware 2026-08-31: 50 Hz measured in-loop, zero drops; a 5 mm
  goal on DOF 1 tracked to **err 0.01 mm**.
- `acc` changes lag, not reachability. Same 5 mm ramp: acc=25 peaks at 1.37 mm
  of lag, acc=255 at 0.56 mm. All values arrive.
- **`mj_forward`, never `mj_step`.** Stepping re-simulates and gravity/contact
  drag `qpos` off the reported values; `mj_forward` also does not clamp to
  `jnt_range`, so a real pose outside the model's travel renders as it truly is.
  Both properties are the measurement, not an optimization. `mj_forward` is
  cheap here anyway: 12.8us against `mj_kinematics`' 2.8us.

## This box has a display and the viewer works

`DISPLAY=:1`, real Xorg on vt2, `XAUTHORITY=/run/user/1001/gdm/Xauthority`.
`launch_passive` measured 2026-09-01: 283 frames over 6 s, window alive
throughout. On-screen VTK works too (pyvista 0.45.2).

*Do not conclude a GL path is broken from a probe that opens and closes in the
same breath.* An earlier claim that `launch_passive` fails with `GLXBadDrawable`
was the probe's own fault — it called `close()` immediately after a single
`sync()`, so the error was a buffer swap on a drawable being torn down. That
wrong claim cost a planning session.

## viser is the render path, and the GUI (2026-09-01)

`WebStudio` in `studio.py`. One browser page: the model's ten visual meshes and
seven mm sliders. Replaced VtkScene + dearpygui, which were two native windows
docked side by side. Measured with a real browser attached (playwright driving
`/usr/bin/google-chrome`; viser **rejects a raw websocket on a version
handshake**, so a probe without a real client silently measures nothing and
reports `clients connected 0`):

```
viser server up      0.035 s      (VtkScene build 0.32 s)
10 meshes uploaded   0.003 s      34,088 verts
pose push            0.60 ms/frame, 3% of a 20 ms tick   (VTK 0.79 ms)
module import        0.81 s       (1.24 s with pyvista + dearpygui)
```

**Geometry comes straight off the compiled `MjModel`.** `mesh_vert`/`mesh_face`
per visual geom (`geom_group == 2`, which is the 10; the 448 CoACD hulls are
group 0), posed from `data.geom_xpos`/`geom_xmat`. No importer, no genotype, no
second scene representation — so no second copy of the DOF map to drift.

- **Do not apply `mesh_pos`/`mesh_quat`.** They read up to 88 mm here and look
  exactly like a forgotten transform. mujoco already baked them into the stored
  vertices. Settled by measurement, not argument: `mjv_updateScene` places every
  visual geom at `data.geom_xpos` to 1.3e-9 m (that residue is `mjvGeom.pos`
  being float32). Pinned by
  `test_a_geom_is_placed_at_geom_xpos_with_no_further_transform` (deleted).
- **Colour is `mat_rgba[geom_matid]`, never `geom_rgba`** — same trap as the
  offscreen path below.
- **A browser tab is not a window.** Closing one does not and must not end the
  session; the page reopens, and dropping torque because someone closed a tab
  drops whatever is held. Ctrl-C or `--seconds` ends a run.

**Defaults:** the page is the renderer and its sliders drive. `--studio False`
falls back to the mujoco viewer, `--panel False` is watch-only, `--no-viewer` is
headless. `studio=None`/`panel=None` mean "decide"; `studio` follows `viewer`.

Sliders are **on by default in every windowed mode** — millimetres, labelled per
DOF, bounded by `config`'s travel. They shipped opt-in once and that was a bug:
`--studio` sets `viewer=False`, closing the mujoco Control panel the sliders
lived in, so **`--studio` alone commanded nothing** — `data.ctrl` frozen on the
startup pose, tracking error a perfect 0.00. Not a crash, not visibly wrong.
`wants_panel` is its own function so the rule is testable without serving a page
per case.

Under `--studio False` the goal source is mujoco's own Control panel, read off
`data.ctrl`. That is safe — `narrow_ctrlrange` has already bounded it to
`config`'s travel — just metres and unlabelled. *An earlier version of this file
called it a rail-crash risk; wrong, that is what `narrow_ctrlrange` is for.*

`WebStudio.goal` is a `goal_mm` callable, so it plugs into the seam a policy
will later arrive through and `live()`'s loop needed no change.

**The page carries a readout and the gains too** (2026-09-01, after `ft_servo`'s
bench GUI). Both were missing for the same reason: the numbers existed and only
stdout ever saw them.

**Layout follows `hardware_bindings/ft_servo/__main__.py`'s ordering** — same
tool at a different level (that one addresses servo IDs and counts, this one
DOFs and millimetres). `speed` and `acc` at the top, then one block per DOF:
bold header, `goal`, `torque`. viser lays out in call order, so the gains must
be built before the blocks even though `gains()` reads a torque handle each
block creates. Two earlier cuts were rejected: a table above the sliders (check
whether DOF 3 tracked → scroll up, match by index) and a markdown row per DOF.

**The read-only bars are gone (2026-09-02, operator request)** — the `mm` and
`load` bar rows were 14 of the panel's ~35 rows and controlled nothing. The
readout now prints inside each block's header line: `{name} {mm:.2f} err {±.2f}
load {int}`. That is the earlier "markdown row per DOF" rejection come back
with a new premise (the panel was mostly read-only rows); the
table-above-the-sliders cut stays rejected — these numbers sit one row from
their slider, no index matching. Numbers also never clamp, which retires
`MEASURED_MARGIN_MM`: a reading past `config`'s travel prints as it is, where
the bar needed the ±10 mm margin to show the same. `LOAD_SPAN` left with the
load bar.

- **`err` is signed and lives in the header** — which way the hand lags is the
  question on a rack that binds in one direction, and 0.01 mm of it is a real
  number and no pixels of bar; the pose itself is in the 3D view. `&nbsp;`
  between fields: markdown collapses runs of spaces.
- **`load` prints signed and inverted** — sign-magnitude drive effort, highest
  in free motion and near zero at rest; a big number is motion, not contact.
- Refreshed at `READOUT_HZ` = 10, not the control rate; the meshes still move
  every tick. Seven blocks changing 50 times a second is unreadable.
- **Gains** — `speed`, `acc`, per-DOF `torque`, read every tick through
  `WebStudio.gains`. They were hoisted `.tolist()`s outside the loop, so finding
  the z stage's 300 meant editing `config.TORQUE_MIN_TO_MOVE` and restarting — which
  returns the hand to its startup pose to test a number that only means anything
  mid-move. Unlike the bench tool there is no target to re-send by hand: the loop
  re-sends a goal 50 times a second already.
- Both free: **0.005 ms and 0.0005 ms a tick**, browser attached.
- `report(mm, load, want)` takes the goal the loop actually commanded, not
  `_want`: under `--panel False` something else is the goal source, and under
  `--teach` there is none, so `err` is dropped rather than shown against a goal
  that was never sent. Those modes have no slider above a row either, so the row
  names its own DOF.

**Frame the camera, or the hand is a speck.** `CAMERA_POS`/`CAMERA_LOOK_AT`, set
per client in an `on_client_connect` — viser has no global initial camera and
each tab gets its own. It opens metres out, which suits a room; this model's
visual meshes span **0.27 m** corner to corner, so the default view drew the
hand ~25 px wide in a 1200 px viewport and the render path shipped a day with
its subject invisible. Aim at the mesh bbox centre `(0.016, 0, 0.003)`, not the
world origin — the model is built around the base, so the origin is off one edge.
`scene.set_up_direction("+z")` too: mujoco is Z-up.

*A screenshot test only catches what you look at.* This survived a browser
verification because that run was checked for finger colours, and the colours
were fine.

**A dead bus now ends the run.** `MAX_SILENT_STEPS = 25` — half a second at
50 Hz — consecutive ticks in which *no* servo answered raises. Holding the last
good position on a dropped read is right for one frame and wrong forever: a hand
holding still and an unplugged adapter produce identical `counts`, so the page
showed a frozen pose while the loop kept commanding a bus that was gone.
`hand.py` counted this before it was deleted and nothing had since. One silent
servo is still a skipped frame, not a dead bus.

*Ctrl-C works on this path* — measured, gone in 1 s. `ft_servo`'s GUI needs
explicit SIGINT/SIGTERM handlers, but that is a `ViserServer` inside a `tyro`
subcommand; here neither importing viser nor constructing a server replaces
`default_int_handler`. **A background process that has exited but not been
reaped still answers `kill -0`**, which is how this first read as a hang. Check
`ps -o stat=` for `Z` before believing a shell-level liveness probe.

## Offscreen mujoco rendering works (kept for video)

`mujoco.Renderer` needs no window, 0.89 ms at 1280x720. `imageio` +
`imageio_ffmpeg` installed, so frames go straight to mp4. Chosen over
robot_studio (would need a generated genotype and a second scene
representation) and over viser (deferred — live remote interaction only).

Three settings, none discoverable from a useful error:

- `model.vis.global_.offwidth/offheight` before constructing the `Renderer`.
  Default framebuffer is 640 wide; anything larger raises at construction.
- Default headlight (`ambient 0.1, diffuse 0.4`) renders this model at mean
  pixel **0.9/255** — effectively black, and a black frame passes any test that
  only checks a frame came back. `ambient 0.45, diffuse 0.7` is legible.
- `mesh_texcoordnum 0` on all ten visual geoms, so colour is the flat
  `mat_rgba[geom_matid[g]]` path — **reading `geom_rgba` instead gives every
  body default grey**, which renders plausibly and destroys the left/right
  measurement.

**The asset was built to answer the left/right question.** Ten deliberately
distinct materials — `left_up_finger` yellow, `right_up_finger` cyan,
`left_down_finger` light blue, `right_down_finger` light purple. Confirmed
rendering correctly.

## Load is not a contact signal — it is inverted

Measured on hand_2 DOF 1, 2026-08-31:

| state | load |
|---|---|
| moving freely | **1044 – 1168** |
| settled at goal | 0 – 100 |
| stalled against a hard stop (torque 300) | 66 – 315 |

Load is drive effort. It is **highest during free motion** and low at rest, so
`load > threshold` fires constantly while the hand is simply moving and stays
quiet when it jams. A naive force trigger built on it is backwards.

This confirms the plan's design: contact is *commanded to move and not moving*
(position differencing), with load at most a secondary vote. Do not re-propose a
load-magnitude threshold for `stop_on_force`.

## Unexplained, now moot: the VTK-path segfaults

Two hardware runs of `studio.py` on the VTK path died `segmentation fault (core
dumped)` a few seconds in, 2026-09-01. Never reproduced under `--mock` and never
root-caused. The in-process GL context was the prime suspect — VTK there had a
documented core-dump mode with two on-screen windows — but that was never shown.
The viser path has no GL in this process at all, so the failure cannot recur;
**it is closed by construction, not by diagnosis.** If a segfault reappears,
this was not it.

## Hardware gotcha

DOF 1 on hand_2 was mechanically bound at the start of the 2026-08-31 session:
a 5 mm goal produced 0.09 mm of motion. Not a code fault and it did not
reproduce — driving it ±400 counts a few times freed it, after which the
identical command tracked to 4.99 mm. Symptom to recognise: full travel in one
direction, ~20% in the other. `config.py`'s `TORQUE_STUCK` note records
fingers climbing a gear tooth on this same hand.

## The bench hand answers to IDs 0-6, on the port config calls hand_2 (2026-09-02)

Measured, not inferred. `bus.scan(0, 32)` on
`/dev/serial/by-id/usb-1a86_..._5AE6085950-if00` returns `[0,1,2,3,4,5,6]`; IDs
7-13 drop every reply. That by-id path resolves to `/dev/ttyACM0`, and
`/dev/ttyACM0` is the only ACM device present — so `HAND_1.port` and
`HAND_2.port` are two spellings of the same adapter, and only one hand has ever
been plugged into this box.

Consequences:
- `studio.live`'s old `hand="hand_2"` default addressed IDs 7-13 and would have
  failed with "servos [7,...,13] did not answer" against what is actually
  connected. `--hand hand_1` was the working invocation and nothing said so.
- Unresolved: whether the unit on the bench *is* hand_1, or is hand_2 with
  hand_1's ID block still flashed. `config.HAND_2`'s comment claims IDs 7-13.
  One of the two is wrong and the bus cannot say which — it reports IDs, not
  serial numbers. Resolve by reading the label on the unit, not from software.
- Do not "fix" `config.HAND_2` by renumbering it to 0-6. That would make the two
  entries indistinguishable and break `config.identify`.

## Mutate the code to check a test, before adding another one

Counting tests measures nothing. 2026-09-01: mutated `studio.live`'s dropped-reply
branch to `counts[i] = 0.0` (fabricate instead of hold) and **all 32 studio tests
passed** — including `test_a_dropped_reply_holds_instead_of_fabricating` (deleted), which
re-implemented the loop body inside the test and asserted against its own copy.
A test named for the exact bug, that could not see it.

Two traps found doing this, both worth knowing:
- **A test that never calls the code under test.** Symptom: the test body
  contains a copy of the logic. Grep for loops in tests that mirror source.
- **A mutant hidden by the fixture's own datum.** The replacement test gagged a
  servo at t=0 and still passed: startup counts *are* the zero datum, so a
  fabricated `0.0` decodes to 0.0 mm — where the hand already was. Had to let the
  hand travel first, then drop, then assert the held value equals the **last good
  one** (not merely that it is constant; the mutant is constant too).

Recipe: `cp` the source, `perl -0pi -e 's/old/new/'` one branch, run the file,
`cp` back. Ten seconds, and it is the only thing that tells you which tests are
load-bearing.

2026-09-02, again worth its cost: three mutants against the new per-env failure
code, and only ONE of the three was caught by the test written for it. Dropping
the `alive` mask from zeroing's park (the unbounded-park safety fix) and gutting
the `not alive.any()` early return both passed a green suite. The test asserted
the *surviving* env's park and the failing env's flags — never the failing env's
**position**, which is the thing the fix exists to bound. Write the assertion
against the state the bug would corrupt, not against the state you were thinking
about.

## An exception cannot be a per-env result (2026-09-02)

`tasks/zero.py` and `tasks/cap.py` reduced their outcome with a bare `.all()`
over envs and raised `RuntimeError`. At N=1 that is right; at N=4096 one env that
timed out discards the 4095 that succeeded, and raising from inside the generator
means they cannot be recovered from the exception either. Tasks now return
`motions.Result(value, ok, why)` with `ok` `[N]`; policy moved to the callers
(`sim.run` raises, `studio.finish` declines to save, a trainer masks).

The non-obvious part, and the reason this is not a two-line change: **the raise
was load-bearing for safety, not only for reporting.** Zeroing's park commands
`stop + travel_mm/2`, and for an env whose seek timed out the "stop" is arbitrary
mid rail — so that goal aims travel/2 past the *open* end, which is where a
carriage leaves its slider. Removing the raise without bounding that move trades
a spurious abort for a mechanical failure. Failed envs now park **in place**
(`torch.where(alive, ...)`), bounded by construction.

Second thing that had to be preserved: both tasks early-return on
`not alive.any()` / `not ok.any()`. At N=1 that is every failure, and without it
a failed zeroing goes on to drive the jaws and z stage for another 40s, and a
cap probe that closed on air unscrews nothing for a minute before saying so.
Cheap line, entirely about the operator, and a mutation check is what showed it
was untested.

No batched executor exists yet (`sim.py`: "No batching yet. N=1"), so this buys
nothing today — it removes the reason a task could not be run under one.

## Dead levers / rejected

- **Decomposing `studio.live` (2026-09-05, considered and declined).** 434 lines
  in one function, easily the worst readability problem left, and it was left
  alone on purpose. Its loop threads about twelve mutable variables — `counts`,
  `zero`, `want`, `armed`, `runner`, `pending`, `silent`, `task_ticks` and the
  rest — so any extraction either passes a dozen arguments or invents a class,
  and both are how a control-loop bug gets introduced. `--mock` cannot see the
  bug classes that matter here (no friction, no quantisation, torque ignored),
  so a green mock run would not be evidence the decomposition was correct. Do
  this one *at the bench*, in front of a hand, or not at all. `sim.py`'s two
  backends were the safe version of the same instinct and were deduplicated
  instead.
- **Trimming `config.normalize`/`denormalize` (again, 2026-09-05).** A dead-code
  scan flags them as the only unreferenced methods left in the package, exactly
  as it did on 2026-09-01. Their consumer was `test_sim_real_contract.py`, now
  deleted, and they remain the documented seam for the still-open sim/real
  action-offset reconciliation. The scan will keep flagging them. Leave them.
- **Multi-location `ft_servo_ext` import search.** Had three paths; verified only
  the first ever fires. Removed — the fallbacks mutated `sys.path` as an import
  side effect and turned a missing build into a confusing three-path report.
- **VTK + dearpygui as the GUI at all.** Superseded by viser 2026-09-01, after
  three attempts at one window: `vtkSliderWidget`s (work, look bad — 3D scene
  actors with fuzzy vector text), blitting VTK into a dearpygui texture
  (robot_studio measured it: 10.7 ms RGBA conversion + 16.4 MB upload per frame
  against 1.07 ms to draw, `native.py:15`), and docking two native windows
  (an arrangement, not a fix). viser has the problem by construction. Do not
  re-attempt any of the three, and do not reintroduce an in-process GL context
  for this tool.
- **robot_studio as the render path.** `VtkScene` worked and was the default for
  one day. Dropped with the GL: it needed an MJCF import to a genotype, a second
  scene representation, and its own copy of the DOF map, to render the same ten
  meshes `MjModel` already carries. `native.py` was never liftable either — every
  callback routes through `RobotStudio.set_joint_value` on the genotype/state
  layer, and this tool must never write a model file.
- **Running a task on its own thread with its own bus handle (2026-09-02).**
  Built it as the "Zero hand" button, shipped it, replaced it the same week. The
  shape looks reasonable and is not: two writers on one bus forced the control
  loop to *stop commanding* while a task ran, which is a second control path
  with its own bugs (`_task_running`, `_new_zero`, a rebaseline that had to be
  deferred a tick); it makes "task" mean something different from "policy"; and
  none of it can exist in sim, so it split the two backends at exactly the seam
  this project exists to close. A button now writes a name into `web.pending`
  and the loop is the only writer at every instant. Do not reintroduce a thread
  for a task.
- **Porting `cartesian_hand_old/tasks/zeroing.py` and `primitives.py`
  forward (2026-09-02).** Done once, and it was a regression even after three
  real bugs were fixed out of it: 189 lines of blocking bus polling
  (`wait_for_stall_counts`, `wait_until_counts`) reimplementing what
  `stop="stuck"` and `stop="goal"` already are. The engine already existed
  (`cartesian_hand_old/motions.py`) and `cartesian_hand_old/tasks/sim_zeroing.py`
  already proved zeroing was expressible as a `Program` in ~25 lines. Before
  porting anything else from `cartesian_hand_old/tasks/`, check whether the
  motions form of it is already sitting next to it.
- **Load-magnitude threshold for contact.** See above; inverted signal.
- **Lifting robot_studio's `native.py` pose panel.** See above; genotype-coupled.
- **Fusing the normalize scale constant.** Measured, ~2x, still noise.
- **Vectorizing / de-torching the bus-reply scatter in `studio.live`
  (2026-09-01).** Built it, benchmarked it, reverted it same session. `counts`
  and `load` became numpy arrays with `torch.from_numpy` zero-copy views, so the
  seven scalar writes hit numpy instead of paying a torch dispatch each:

  | | us/tick |
  |---|---|
  | torch scalar scatter (kept) | 17.00 |
  | numpy scatter + `from_numpy` view | 0.46 |
  | `fromiter` substitution | 2.14 |
  | mask + whole-vector write | 3.14 |
  | Python list + one convert | 3.29 |

  Two findings survive the revert:
  1. **Fully vectorizing is *slower*** at N=7 — masking and `fromiter` spend more
     building index arrays than seven scalar stores cost. "Vectorize everything"
     is wrong at this width.
  2. **37x on a path worth 0.085% of the tick is nothing.** See the tick table
     above: it saved 17us against ~17,900us of idle sleep, and cost 14 lines of
     comment plus a test guarding a silent-aliasing failure the change itself
     introduced. Reverted on those grounds, not on correctness — it worked.

  Numpy-backed buffers with torch views belong only where a measured batched
  path needs them, not retrofitted into `studio.live` at N=1.
- **Trimming `config.py`'s "unused" surface (2026-09-01).** Grepping
  `cartesian_hand/` finds no caller for `normalize`/`denormalize`, the
  `_cache`/`_vec` device machinery, `contract`/`fingerprint`, `clamped_mm`,
  `NotZeroedError`, `CALIB_PATH`, `TORQUE_STUCK`. Nearly all are load-bearing
  anyway — three of the four cuts proposed off that grep were wrong:
  - `denormalize` has 7 uses in `tests/test_sim_real_contract.py` (deleted), which pins the
    sim/real action-offset blocker. That test file *is* the consumer.
  - the device cache supports the shared tensor contract (`device="cuda"` must
    be the only difference between backends).
  - `fingerprint()` is intended as the batch admission rule.
  - the zeroing constants and `clamped_mm` are used by the current task and
    studio paths even when package-only caller searches make them look sparse.

  Only `to_dict`/`from_dict` was genuinely dead — sole consumer was its own
  round-trip test. Removed, 533→500 lines. Before proposing another trim here,
  grep tests and task call sites, not just the package.
- **Reading `config.py`'s prose:code ratio as over-engineering.** 280 prose to
  146 code. It is measured data — the z-stage torque bisect, the four-way travel
  disagreement, the left/right UNRESOLVED note. Expensive to re-derive, cheap to
  carry. The code was the only part worth cutting. 2026-09-02: the module
  docstring's design-history sections ("One dataclass, not four", "Tensors, not
  numpy") were compressed ~44→22 lines; every measured-data comment untouched.
  The `LAYOUT` comment at "orientation below alternates +1/-1" is stale against
  an all-`-1` table (mid-edit by another session, `test_config` red on it) —
  left for that session to settle.

## Dead files still importing deleted modules — cut 2026-09-02

`examples/twin_policy.py` and root `test_cartesian_hand.py` did
`from cartesian_hand.hand import ...` and died with `FileNotFoundError`; `hand.py`,
`policy.py` and `tasks/` now live in `cartesian_hand_old/`. Both removed at the
operator's "cut if not used" — they were unrecoverable as code (dead imports) and
tracked in git, so history keeps them. Consequence, recorded then and still true:
`examples/twin_policy.py` was the only non-test caller of `fingerprint()`, so
`fingerprint`'s callers are now `tests/test_config.py` (deleted) and the plan's future env.
`cartesian_hand_old/` stays — untracked reference, deliberate keep, deleting it is
permanent (no git history).

## Zeroing reproduces its datum to 0.074 mm on the bench hand (2026-09-03)

Ran `studio.live(hand="hand_1", task="zero")` against the real bus and diffed
the saved offsets against the calibration already on disk from an earlier run:

    old   [2285, 5037, 5784, 6602, 5794, 3843, 4808]
    new   [2291, 5038, 5782, 6603, 5796, 3842, 4811]
    delta [  +6,   +1,   -2,   +1,   +2,   -1,   +3]  counts

Worst case 6 counts = 0.074 mm at 81.5 counts/mm, i.e. an order of magnitude
inside the 1.0 mm `POSITION_TOLERANCE_MM` the parks retire on. Zeroing is
repeatable; a datum shift larger than ~10 counts is a real change, not noise.

Three claims fell out of the same run, all visible in the console trace:
- Parks land on each DOF's *own* mid travel — 27.4/27.2 mm fingers (rail 55),
  24.6/24.7 jaws and 24.4 z (rail 50). A hoisted travel would have put them all
  on one number.
- z's park saturates the load channel at 1000 while every other DOF sits near
  280. That is `park_torque` being clamped up to z's 800 floor, and z being the
  only DOF under gravity — the fix that made phase 3 actually lift.
- No phase expired, so `timeout_margin=1.5` covers the real creep speed.

Bench state, unchanged by the run and normal for this hand: z (servo 3) idles at
48-51 C holding the stage, load ~30-150; every other servo 29-35 C at load 0.
**Do not release torque to cool it** — z drops ~24 mm onto its hard stop.

Same `tasks/zero.py` also ran CPU mujoco at N=1 and `mujoco_warp` at N=4096 in
this session, landing on the same park targets. Hardware / 1 sim / 4096 sims,
nothing swapped.

## The sim had no speed limit, so time in sim meant nothing (2026-09-03)

mujoco's `<position>` actuators are handed a target and driven to it as hard as
`kp` allows. The servos do not work that way: `set_positions(..., speed=300)`
means the setpoint *walks* toward the goal at 300 counts/s. Nothing modelled
that, so a sim joint crossed 55 mm in the same tick it crossed 5.

Measured before the fix: **zeroing finished in 45 ticks in sim against ~4000 on
the bench.** Every distance and every timeout cost the same, and a 12-trial
search over `scissors` returned 209 ticks for all 12 samples plus the defaults —
a perfectly flat objective, which is what sent us looking.

Fix is `sim.step_limit_mm` / `sim.profile`, applied to the *command* in `run`
and `run_warp` (and, while it existed, `search.Arena`). Two lines, and they are the two lines
`tests/test_tasks.py::toy_run` (deleted) had been using all along — the toy hand in the
test suite modelled the servo correctly and the backend claiming to be physical
did not.

What it exposed, and this is the part worth carrying:

- **Every park in `zero` times out in sim.** Wants GOAL, gets TIMEOUT, all three
  phases. The hand ends at 15.0 mm where mid travel is 25.0. It had always done
  this; nothing could see it because `zero`'s `Result.ok` reports on its *seeks*
  and never looks at its parks.
- **z never leaves its stop in sim** (-0.3 mm). A profiled setpoint stays
  `limit` mm ahead of the joint, so the position error — and therefore the force
  the actuator develops — is bounded by `kp * 0.074 mm`, which does not lift the
  stage. On hardware the same task lifts z fine. This is the sim/real gap now
  standing where "torque is dropped on the floor" already stood, and it is the
  same phenomenon `studio._add_tuning` records from the bench: a slow profile
  keeps the setpoint close enough that error never grows enough to break static
  friction. Do not "fix" it by removing the profile — that restores a sim where
  time is meaningless.
- Raising `timeout_margin` 1.5 -> 2.57 takes row success 0.50 -> 0.93 and the
  parks from 15.0 mm to 24.0 mm. **This is a sim artifact, not a finding — do
  not apply it to the hand.** See the next entry.

## REJECTED LEVER: automatic tuning against the current sim (2026-09-03)

**Do not rebuild `search.py` until all three preconditions below hold.** It was
written and deleted the same day. It looked like it worked — `zero` row success
0.50 -> 0.93 unattended — and that number is the artifact, not the result.

**1. Most sampled fields are invisible to the backend.** `sim.profile(step[0],
...)` takes the goal; torque is dropped on the floor. So every `*_torque` field
is unobservable in mujoco: 1 of `zero`'s 2 knobs, 3 of `cap`'s 9, and **half of
every composed task's**, since `compose.program_source` emits `goal_i` *and*
`torque_i` per row. Sampled anyway, they appeared in the winner's report looking
like findings (`park_torque = 312.319`, drawn from `uniform(100, 800)`,
affecting nothing).

**2. Only the trivial gradient was findable.** Score monotone in
`timeout_margin`, flat in everything else. "Longer budget passes more rows."

**3. The sim disagreed with the bench and the bench was right.** Sim scores
`zero`'s shipped defaults 0.50; the hand_1 log in `tasks/zero.py` has those same
defaults reproducing the datum to **0.074 mm with no phase expiring**.
`--save-as` would have written `timeout_margin=2.568` into a variant — ~70%
slower real zeroing to fix a simulator artifact — with a docstring claiming an
improvement. Worse than no tuner: it manufactures an artifact carrying its own
justification.

Preconditions, in order: (a) task-success scoring, not execution success —
`manipulation`'s `success` predicates are the shape; (b) torque reaching the
actuators in sim, **or** a sampler that refuses fields the backend cannot
observe (one function, needs no object scene); (c) only then a sampler.

**Worth keeping from it.** The scoring rule was sound as far as it went and
should be revived when scoring is: `row_success = (succeeded() & acts).sum() /
acts.sum()` over every program a task issued. Task-agnostic, needs no
cooperation from the task, cannot drift from the engine because it *is* the
engine's own comparison, and — the point — cannot be *narrower* than what the
task did. `Result.ok` can: `zero` checks seeks and never its parks, so ranking
on it scores "gave up sooner" as "finished faster". Rank lexicographically
`(-ok_rate, ticks)`, never a weighted sum; a weight between "worked" and "was
quick" is a number nobody can defend, and picking one lets a fast failure
outrank a slow success.

`tasks.tunables` survives — the studio's sliders read it. **A declared
`{"tune": (lo, hi)}` bound is a range a human may drag with the hand in front of
them, not a claim the number is measurable by whatever reads it.**

## z cooks itself holding position, and it does not need to (2026-09-03)

Servo 3 (z) on the bench hand idles hot while energized: **48 C when found, 51 C
after a zeroing run, 60 C an hour later**, load climbing 28 -> 148. Everything
else sits at 29-36 C with load 0. Feetech over-temperature protection trips near
70 C, and a trip releases torque — which is the worst way for the stage to come
down, because it is uncontrolled.

Measured after lowering z onto its own hard stop and calling
`enable_torques(ids, False)`: **60 C -> 48 C in 90 seconds, and the stage did not
move** (6395 counts, held across a 90 s window with no power). The rack is
self-locking, so z holds its height mechanically and the holding current was
buying nothing.

Consequences:
- Leaving a session with z energized mid-travel is a slow thermal ramp toward a
  trip. Park it low and release torque instead.
- Do **not** release torque on z while it is high without lowering it first. The
  self-locking was measured near the bottom stop; nothing here establishes it
  holds a loaded gripper at 25 mm, and being wrong drops the stage.
- The load channel is the tell. 28 vs 148 on the same joint at the same height
  is the difference between resting and fighting; the other six read 0.

## Composed tasks consume contact outcomes in fixed tensors (2026-09-03)

`Motions` now records latest executed-row result per environment and joint:
`stopped_at [N,J]`, `stopped_ok [N,J]`, `stopped_valid [N,J]`, plus
`executed [N,J,K]`. `When(kind, source_dof, threshold)` compiles into program
tensors and resolves once in `_arm`; no Python branch, no K change, each env
chooses independently. Supported gates: previous stop succeeded/failed and last
successful stop >=/<= a position threshold. A timeout sets `stopped_ok=False`
but never overwrites `stopped_at` with a plausible mid-rail contact.

`stop="external"` consumes explicit optional `[N,J]` bool passed through
`TaskRunner.tick`; both sim backends and hardware `studio.live` expose callbacks.
No producer means all false and timeout. Do not infer it from servo load: that
channel remains unverified and inverted. Stock MJCF remains objectless, so it
cannot honestly supply physical-object success yet.

Studio composer is now unconditional (exists in watch-only mode), named `TASK
TIMELINE`, expanded by default, and includes `run when`, source joint, threshold,
and `finish when`. User feedback established that collapsed/slider-only composer
was effectively absent, not merely hard to find.

## `cap` is contact-based. There is no measured variant (2026-09-03, user)

Verbatim: *"there is only one cap.py that is contact based, that's it."* Said in
response to a proposal to split out a `cap_fixed.py` that took the radius as
config so it would be a plain composable row list.

`cap.build(..., cap_radius=...)` was removed when cap became the canonical direct
policy. There is one contact-based task and no measured-radius escape hatch.
The objectless simulator therefore cannot complete it yet; that is an honest
missing environment feature, not a reason to add a second task protocol.

Cap is the task whose phases and repetition depend on contact measurement. The
fixed row table *can* express it cleanly with the `Twist(count=..., measure=...)`
inlined form and a `Loop` for the capture/extract cycle — it does, now; see
`tasks/cap.py`. Phases and measurement-driven transitions live inside
`Sequence` itself, not in a per-task policy class.

## Any row that LIFTS z at travel torque is a silent no-op (2026-09-03)

Found on the bench: `cap`'s extraction lift ran and the cap never left the
bottle, and the task reported success. The row existed; the stage did not move.

**z torque is directional and the code cannot see the direction.** hand.py's
bisect (lift 30 -> 35 mm, travel after 3 s): `150 -> 1.50mm`, `200 -> 4.54mm`,
nothing above 200 helps. The old working hardware carried travel torque as a
per-DOF table with z alone raised — `STANDARD_TORQUE = [50,50,50,300,50,50,50]`
— and its comment is explicit: *"Pressing down at 50 is fine and tasks rely on
it, so this is a floor for the lifting direction, not a correction to the whole
axis."* `cap.py` had flattened that table to one scalar `travel_torque = 50`.

So: **a descending z row at travel torque is correct; an ascending one is a
row that runs, expires, and reports nothing wrong.** Fixed in `cap.py` by giving
the lift `torque_min_to_move[Z]` (per-hand measured floor, what `zero`'s park
already lifts z with) and a budget derived from the goal height rather than the
flat `move_timeout_s` — 3 s does not cover a 50 mm lift at ~6 mm/s, which is
most of `lift_mm`'s own declared tuning range.

**Nothing automated can catch this class.** mujoco drops torque on the floor,
`MockServo` ignores it, and an expired budget still ends a program tidily. So
the guard is a static one: `tests/test_tasks.py::test_the_cap_lift_is_commanded_hard_ (deleted)
enough_and_long_enough_to_happen` asserts the commanded torque against
`config`'s floor (not a literal, so raising the floor per hand raises the row)
across `lift_mm`'s whole tuning range. Mutation-tested both halves.

**Check any new task for this.** The question is not "does the row exist" but
"does this row raise z, and if so is it commanded above the floor with a budget
that covers the distance". `cap`'s *entry* descent is deliberately left at 50.

### The same file's budgets were flat constants, and that was worse

Found while answering "what acceleration and velocity does this task use?" —
neither, and that is the point: `Motions` carries goal and torque only, so speed
and acc are per-hand config applied by the executor (`studio.live` passes them
to `set_positions`; sim models speed via `sim.profile` and **ignores acc
entirely**). A task's only lever on time is its deadline.

`zero` derives every deadline from `hand.gain_vector("speed")` and works. `cap`
shipped with `move_timeout_s = 3.0` / `probe_timeout_s = 5.0`. At 3.68 mm/s a
3 s deadline buys **11 mm**, and the twist sweeps 40 mm. Driven through mujoco:

    before   rows=18  failed=15  row_success=0.17
    after    rows=18  failed=5   row_success=0.72

Every `stop="goal"` row was expiring. The twist got 3 s of a 10.9 s move, so
each stroke turned ~27% of its span while the stroke count assumed 100% —
`num_revs=1` delivered about a quarter turn. The 5 that remain are 4 `stuck`
rows with no object to stall on and the z lift hitting the sim stiction gap
above; none is a budget bug.

Fixed by `HandConfig.travel_budget(dof_ids, margin, distance_mm=None)`, now
shared by both tasks so the counts -> mm -> seconds conversion exists once.
Defaults to the full rail: **a budget is a DEADLINE, not a duration** (goal rows
retire on arrival, stuck rows on contact), so over-allowing is free and lets a
task budget a move whose start it never reads. Slowest member of the group,
because z alone runs at 500 where everything else runs at 300.

**Neither half of this is visible to any automated check.** An expired budget
still ends a program tidily; only `succeeded()` per row shows it. Grade a task
with `(succeeded() & executed)` over every program it issued, never `Result.ok`.

## A row list must be a CAD history timeline, not an append log (2026-09-03)

User, twice: the GUI is not intuitive; think Fusion 360 / SolidWorks — history
timeline, steps executed interactively, edited/moved/removed easily.

The panel shipped append-only (add row / undo last). Wrong shape: a procedure is
authored by *revising* it, so changing one number of row 1 meant deleting rows
2..n and retyping. Now: select a row and its parameters load into the editors,
**Apply to row**, **Move up/down**, **Delete row**, **Insert after row**.

**`Run to row` is the part that matters** and is the answer to the user's "if I
cannot verify live, what is the use of it". It executes rows 0..cursor and
leaves the hand there, so the next row is authored from the pose the previous
ones actually produced. `frame="here"` distances cannot be predicted — you have
to be in the pose to pick one. Assessed honestly: the row *editors* are marginal
(one row is one line of `Step.set`, faster to type), the interactive *prefix run*
is the only capability the panel adds that nothing else has.

Implementation, and it is deliberate: rows 0..cursor are written to
`tasks/_preview.py` and submitted **by name** through the same `WebStudio.pending`
slot the task buttons use. So the timeline runs the *file*, not a preview of it —
one execution path, no bug class that exists only in the artifact you did not
keep. Rejected: interpreting `Row`s in the studio (a second definition of what a
row means, guaranteed to drift from the codegen), and making `pending` carry a
`Task` object (ripples into `submit`, `finish` and `sets_datum`, all keyed on a
task *name*). `_preview` is gitignored and rewritten every run.

**Latent bug this surfaced, and it bites any overwrite path.**
`importlib.import_module` returns the *cached* module for a rewritten file;
`invalidate_caches()` does not help, it only rescans directories. So every
preview after the first silently executed the previous edit. `compose._write`
now reloads via `sys.modules`. Mutation-tested — reverting the reload fails
`test_web_studio.py::test_the_timeline_edits_reorders_and_runs_what_it_shows`.

## Temperature belongs in GUI telemetry, polled slowly (2026-09-03)

`studio.live` polls one servo temperature every 0.25 s, round-robin, after the
normal position read. Each of 7 servos refreshes ~1.75 s. One bus owner, no
telemetry thread, no catch-up burst. `WebStudio.report` shows a `°C` column and
renders unknown as `--`. This follows the measured z thermal ramp (48 -> 60 C)
without adding seven synchronous reads to every 20 ms control tick.

## Five validated tasks ported to the direct policy path (2026-09-04)

`bulb`, `screwdriver`, `pipette`, `syringe` and `scissors` now build typed
policies, from `cartesian_hand_old_validated_real/tasks/`. `scissors` was
converted *off* `Motions`; `zero` and `tilt` stay generators. Suite 110 -> 137.

**Transcriptions, not bench results.** The sequences were validated on hardware;
these implementations have not touched the objects. Do not report otherwise.

`twist_stroke` grew two optional per-env arguments instead of being copied three
times — `bulb`, `screwdriver` and `pipette`'s knob all run the identical
release/reset/re-grip/turn:

- `reverse [N] bool` swaps which finger opens the gap. It must be a **tensor**,
  not a call-site choice: `bulb` unscrews and re-threads in one task.
- `press: TwistPress` drives a third axis to depth after the re-grip, holds it
  through the turn, backs off before the next release. `return_effort` is a
  separate field from `effort` because backing off *raises* the stage.

Phase enum is now `RELEASE, RESET_FINGERS, REGRIP, PRESS, TURN, RETRACT,
STROKE_DONE`. **A no-press env skips both z phases on the transition itself**,
not by idling a tick in each — that is what kept `cap` bit-identical (its
mock-bus completion test asserts final counts and efforts and never moved).

**The old code's z ascents were the documented silent no-op, everywhere.** Every
ported task now floors its z-ascent effort at `torque_min_to_move[Z]`:
`syringe`'s pulls, `pipette`'s rise between plunge/draw strokes, `bulb`'s
extraction lift *and* per-stroke back-off, `screwdriver`'s press return,
`scissors`' stroke. Descents deliberately keep the light number. Five mutants
(each floor -> travel torque): 5/5 caught. Mutants on `reverse`, the press gate,
`return_effort`, and press-held-through-turn: 4/4 caught.

**`syringe`'s entry diverges from the validated code on purpose.** The original
closed the aux jaw to `aux_min_mm` at entry and then "approached" that same
target; it only worked because `set_pos`'s timeout was unchecked — with a plunger
present the jaw stalls on it and the move never converges. Entry now *opens* both
jaws and lets the pinch phases find contact, which makes
`reached_goal_without_contact` mean "gripped air" instead of being unreachable.

Also added: `strokes_for_revolutions` (four callers; `cap` was inlining it).
`joint_mask` was `cap_policy._includes`; it now has no callers but is kept
because README documents it.

## Uncommitted work was destroyed mid-session by a concurrent agent (2026-09-04)

`cartesian_hand/tasks/tilt.py` and `tasks/scissors.py` were overwritten with
their **git HEAD** contents at 00:37:38, between a green baseline run and the
next test run, discarding uncommitted work. Not caused by this session's edits.
`ps` showed three other `claude` processes started 9-20 min earlier, and
`README.md` was written at 00:38:16 by nothing in this session.

Recovered because both files had been read in full earlier and were still in
context. `plan/handoff.md` explicitly says to preserve those two files' edits,
which is what settled that restoring was right rather than presumptuous.

**How to tell a revert from someone else's new work, before overwriting
anything:** diff the suspect file against `git show HEAD:<path>`. Byte-identical
to HEAD means a checkout/revert destroyed local work; different means it is
somebody's edit and must not be clobbered. That check is one command and is the
whole difference between recovering work and destroying it.

Partial loss is still possible and happened here: the restored files predated a
torque-flooring change that the (also uncommitted)
`test_no_program_row_is_commanded_below_its_own_torque_floor` (deleted) required, so both
needed their horizontal row torques floored again by hand. **A worktree this
dirty with concurrent writers has no recovery story** — the tree was tarred to
`/tmp/ch_recover/` afterwards, which is a workaround, not a fix. Commit.

Untracked-and-deleted has no recovery path but a snapshot, so the lesson stands:
commit, or a loss is total. But **do not read every disappearance as an
attack** — see the next entry, where `tests/` going missing was the user's own
deliberate removal and was initially misdiagnosed as this.

## `tests/` was removed on purpose; verify on the bench (2026-09-04, user)

The whole `tests/` directory was deleted by the **user**, deliberately: the
suite made conversations slow, and the hand is verified on real hardware
instead. Not a concurrent-agent revert, though it was first diagnosed as one
because the prior entry had primed for exactly that and five other
`claude --dangerously-skip-permissions` processes were in fact running.

**Do not restore `tests/` and do not propose pytest as the verification story.**
A snapshot from 13:04 that day exists at `/tmp/ch_recover/mine_130442.tgz` if a
specific old test is ever wanted, but restoring wholesale re-adds what the user
removed. `plan/handoff.md`'s `pytest -q tests` (`109 passed`) is historical.

Consequences to carry:

- Verification means driving the real hand — `python -m cartesian_hand.studio
  --hand hand_2 --task <name>` — and reading the phase/timing trace, not a green
  suite.
- The five hardware-validated policy tasks (`bulb`, `pipette`, `syringe`,
  `screwdriver`, `scissors`) now have **no automated regression net**. Any port
  of them onto `Sequence` is riskier than it was, not safer; the only check is a
  bench run per task.
- Lesson from the misdiagnosis: before calling something destruction, account
  for the user having done it on purpose. `git status` looks identical either
  way.

## Four hand-written policies became `Sequence` row tables (2026-09-04)

`scissors`, `syringe`, `screwdriver`, `pipette` were each a bespoke tensor state
machine — `*Parameters` dataclass, `*State` dataclass, phase enum, a chain of
`torch.where` transitions, a `_check_parameters` wall. 1955 lines became 697.
`primitives.Sequence` already was that machine; nothing new was written to do it.
The `*Policy` classes and their per-task `_check_parameters` are **gone** — if a
plan, docstring or README paragraph names `ScissorsPolicy` or friends, it is
stale. `primitives.joint_mask` now has no callers (`Sequence._mask` replaced it)
and is kept only because README documents it.

`zero`, `ready` and `tilt` stay `Motions` generators. `tilt` deliberately: it is
the last task the studio timeline can open, because `compose.rows_from` reads a
built `Motions` and a `Policy` has no rows to read. Porting it would silently
cost that.

Traps this rewrite has already paid for, in the order they bite:

- **A `Move`'s `effort` is one number for the whole row.** Every port built a
  per-DOF `torch.where(z, lift, travel)` entry effort; a row cannot. Splitting
  into "jaws and fingers" then "z" is mandatory, not tidiness — merged, you
  either cap a free jaw sweep at z's ~0.8 gravity effort (a jaw that can crush)
  or expire the z row under travel effort (a stage that never moves and reports
  nothing wrong).
- **`Move.goal` dict keys must be hashable**, so `tuple(AUX_FINGERS)`, never the
  list `AUX_FINGERS`.
- **A `Loop` always runs at least once**; `count=0` still executes the body.
- **Ordering inside a `Loop` decides what the last iteration leaves behind.**
  `syringe`'s pull loop releases and lowers at its *start*, so the final pull
  leaves the plunger drawn at `pull_z` where the push half re-pinches it. Both
  rows are no-ops on the first pass. Trailing them puts z back at `clearance_z`
  and the next pinch closes on barrel.
- **A capped push that parks short is normal, and a plain `Move` calls it a
  fault.** `syringe`'s push needed `accept_stall=True` (the port lacked it):
  without it the row burns its deadline, retires the whole `Sequence`, and the
  contact-based seat that would finish the dispense never runs. `accept_stall`
  and not `loaded` — a descent's effort must not be raised to the travel floor.

Two of `cap`'s bench fixes were carried to all four: `approach_speed` 50 -> 300
counts/s (at 0.61 mm/s the creep ran at twice `STUCK_SPEED_MM_S`, so hesitation
read as contact and a 20 mm close took 33 s), and an explicit `travel_speed` on
the two that sweep fingers through a `Twist` (`screwdriver`, `pipette`).
`scissors.min_handle_radius` was deleted: `Probe`'s
`reached_goal_without_contact` is the same 1 mm gate, one phase earlier.

`primitives.Probe` gained `goal: Value = 0.0` for this — `syringe` pinches to
`aux_min_mm` and seats to `clearance_z`, and closing past either hits structure
before the object. Default keeps every other probe identical.

**Verification status: none of the four has run on hardware in this form.** What
was checked is that all nine tasks build for both hands at N=1 and N=4, and that
each `Sequence` ticks to `done` with no failure against a kinematic stub
(position walks toward the goal, contact always true). That proves row wiring,
deadlines and transitions, and proves nothing about force or contact. Per the
entry above, the bench run is the check.

Every shipped task now names a `Config.label`, so all eight have a studio
button. Before, `syringe`/`screwdriver`/`pipette` had `label = ""` and were
reachable only through the collapsed **tune a task** folder, whose **Run tuned**
button does submit the plain task when no slider has moved — findable by nobody.
Generated variants still clear their label on purpose; that is what keeps the
panel from becoming a wall of buttons.

## A free `Move` rejects a stall on sight, and its gate is 1 mm, not `tolerance_mm` (2026-09-04, bench)

`cap` failed in phase 9 (`cap clear`) at 1.2 s of a 6.1 s budget, fingers parked
at 2.7 and 0.5 mm against a goal of 0. This reads like a timeout and is not one.

`Move` is free unless `loaded` or `accept_stall` is set, so `move_to` runs with
`stall_fallback=False` and routes `_stalled` into `_closed_loop`'s **reject**
channel (`reached_goal_without_contact`, misnamed for this use). That fires on
the tick it becomes detectable, which is the whole point — a blocked approach
used to sit quiet for its full deadline. One rejected joint retires the whole
`Sequence`, so phases 10-14 never ran and the trace showed only phase 9.

**The trap:** `_stalled`'s "away from goal" test is a hardcoded `> 1.0` mm, and
it does not track `Move.tolerance_mm`. The right finger at 0.5 mm was inside
that gate and passed; the left at 2.7 was not. Raising `tolerance_mm` still
works, because `_closed_loop` computes `rejected_now = ... & ~satisfied` and
`reached` (which does use the tolerance) wins — but the two thresholds are
independent and reading only one of them predicts the wrong outcome.

Fix: `cap.CARRY_TOL_MM = 4.0` on the two `AUX -> 0.0` rows (`cap clear`,
`present`). The fingers close onto the cap's own stop while the aux jaw clamps
it; the last few mm are not available. More torque would drive them harder into
that stop — hand_2's finger floor is already 100 and the bench read load 124 —
so tolerance is the knob, not effort. Not `loaded=True`: that accepts *any*
confirmed stall, so a finger stopping at 15 mm would read as arrival.

Confirmed visually on the bench. `python -m cartesian_hand.sim --task cap`
cannot check this: it dies at row 0 (`height FAILED 0.5s vertical translation
-0.3->10.0`) because sim starts unzeroed and has no cap body, so phase 9 is
unreachable there. Pre-existing, unrelated to this fix.

## The predicted false stall landed: `move_to` had no startup grace (2026-09-04, bench)

The `Torque is a CAP` entry (2026-09-02) closes by naming an open risk: "a joint
that cannot break stiction reads `< 0.3 mm/s` within 10 ticks and reports STUCK
wherever it was standing." That happened, on `cap` phase 11 (`cap align`).

Symptom: `cap align FAILED 0.2s  aux left finger 2.8->22.5  aux right finger
1.8->22.5`, and 0.2 s is exactly `CONFIRM_TICKS` at 50 Hz. **The tell that this
is not a mechanical fault: the very next trace line reads `22.5 22.5`.** The
standing command carried both fingers the full distance after the `Sequence` had
already retired. The move was always possible; it had not started yet.

Mechanism, straight from the 2026-09-02 finding: the torque register is a force
cap, developed effort follows position error, and a profile ramping out of a
standstill needs time before it pushes hard enough to break stiction. `_stalled`
counted quiet ticks from the row's first tick, so the verdict landed mid-ramp.
Reversing two *loaded* fingers (they were holding the cap against their stop) is
the slowest possible start, which is why this row found it first.

Fix: `primitives.START_GRACE_TICKS = 25` (0.5 s), a new `grace_ticks` argument
to `_stalled` that suppresses counting for a row's opening ticks.

**It applies only where a stall is a FAULT** -- a free `move_to`. Not to
`stall_fallback` rows, not to `close_until_contact`, where the same measurement
means *arrival*. That split is the whole design and the first cut got it wrong
by applying grace everywhere: a twist's PRESS, TURN and RETRACT end by stalling
on nearly every stroke, so blanket grace was 0.5 s of dead time per phase per
stroke across a multi-stroke cap cycle. For a probe it is also 1.8 mm of extra
creep at 3.68 mm/s, past the 1 mm tolerance those work to.

So this costs no motion time. A row that reaches its goal finishes exactly when
it always did; only *declaring a fault* slows, 0.2 s -> 0.7 s.

**Not fixed, not observed, predates this:** the mirror bug on the arrival side.
A joint that has not started moving reads quiet, and under `stall_fallback` that
is arrival -- a probe records wherever it was standing as a radius every later
row inherits, and a turn can "arrive" at tick 10 having barely rotated, which
with `stop_on_stall` ends a closing twist with the cap under-tightened. Same
root cause, opposite sign. Giving that side a grace is what the paragraph above
rejects, so a fix would need a different mechanism.

Checked with a two-case harness against a synthetic joint (slow start must
succeed; never-moving joint must still be rejected, at tick 34 = 25 + 10 - 1),
plus all eight tasks building for both hands at N=1 and N=4. Bench confirmation
of phase 11 is still owed.

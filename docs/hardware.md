# Hardware, calibration and configuration

Zeroing, the `HandConfig` tables, motion gains, bringing a new servo onto the
bus, what has been measured on real hardware, and what is still wrong.

See also [tasks.md](tasks.md) for writing a task, and [internals.md](internals.md)
for the engine and backends.

## DOF indexing and orientation

The DOF table in the [README](../README.md#the-hand) hides three things.

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

**The ordering is authoritative for the simulation.** DOF order is the order the
twin's actuators must be in, not the other way round. `LAYOUT`'s axis sequence
y,x,x,z,y,x,x matches the MJCF actuator order one for one, established by walking
`model.actuator_trnid` rather than by reading the comment. Nothing asserts it
now; the comparator that once pinned it against the hardware is gone.

Names and groups (`BASE_FINGERS`, `AUX_FINGERS`) sit in `config.py` directly
below `LAYOUT`. Putting them in their own file would let the two disagree, and a
role map that disagrees with the layout is a mirrored gripper that still looks
plausible.

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

**`hand_1`, `hand_2` and `hand_3` are the three units built in our lab, not
presets.** Every number measured on them in this file (torque floors, the z
bisect, travel, zero offsets) is that unit's own. If you built your own hand,
add an entry for it rather than borrowing one of ours:

```python
MY_HAND = HandConfig(name="my_hand", port="/dev/serial/by-id/usb-...", first_servo_id=0,
                     torque_min_to_move=(250, 250, 250, 800, 250, 250, 250),
                     torque_stuck=(400, 400, 400, 800, 400, 400, 400))
HANDS = {h.name: h for h in (HAND_1, HAND_2, HAND_3, MY_HAND)}
```

Then run with `--hand my_hand`, or set `DEFAULT_HAND = "my_hand"`. The torque
tables above are `hand_3`'s un-bisected values: a floor known to move real
hardware, not a measurement of yours. Bisect `torque_min_to_move` on your unit
(see [Motion gains](#motion-gains)), and measure travel with calipers before
trusting `STANDARD_TRAVEL` (see [Known issues](#known-issues)).

**Add the entry before the first run.** With no `--hand`, `studio` picks the
hand by which servo-ID block answers, so a fresh build numbered 0-6 without an
entry of its own answers as `hand_3` and runs on `hand_3`'s torques, with no
error. Once `my_hand` shares that block, the probe refuses with "2 hands
answered", so pass `--hand my_hand` (or delete our entries from `HANDS`).

A hand is one flat frozen dataclass, `HandConfig`, and the definition above is
all of one.

The two torque tables have no default, on purpose: they are friction, friction is
per unit, and a default is what tunes two hands with one edit.

Everything absent from that call comes from the shared tables at the top of
`cartesian_hand/config.py`. Only what is true of one unit and not the other is
written per hand, which today is the serial port, where its servo IDs start, the
torque floor and the transit speed. `DEFAULT_HAND` is `hand_3`, which is also
what `--mock` and `sim` use when no `--hand` is given; offline, only its travel
and speed tables matter. `hand_2` and `hand_3` share a port path because it
names one USB adapter that moved between them.

An earlier version nested `Dof`, `Motion` and `Geometry` inside `HandConfig`.
Three extra types, a `cfg[dof].max_mm` to read one travel limit, and both hands
filling in identical `Motion` and `Geometry` objects. What is genuinely per-DOF
— axis, count direction, label — is the same on every hand built so far, so it
lives in `LAYOUT` once rather than in seven objects per hand.

`config.py` reads top to bottom: the shared tables (`LAYOUT` and the role names,
`STANDARD_TRAVEL`, `STANDARD_SPEED`, `STANDARD_ACC`, `CALIB_PATH`,
`DEFAULT_HAND`), the one dataclass, the three hands, then the calibration file
helpers. Nothing downstream holds a hardware constant of its own, so retuning a
gear ratio or a travel limit never means opening control code.

Prefer a `/dev/serial/by-id/` path over `/dev/ttyACM0`. ACM numbers are handed
out in enumeration order, so with two hands plugged in, a hardcoded number
silently addresses whichever powered up first.

Frozen matters for a concrete reason: the per-DOF tensors are cached per device,
so mutating `cfg.speed` after anything has called `gain_vector` leaves the old
speed in the cache and every later tick keeps commanding it. A docstring saying
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

`torque_min_to_move`, `torque_stuck`, `speed` and `acc` each take a scalar or a
per-DOF sequence. A wrong length raises at construction, not at the first servo
write.

```python
HandConfig(..., torque_min_to_move=250)                              # all DOFs
HandConfig(..., torque_min_to_move=[250, 250, 250, 800, 250, 250, 250])  # z at 800
```

Mixing values costs nothing. A sync-write is one broadcast packet in which each
servo reads its own slice, so seven different torques and seven identical ones
are the same packet and the same time on the wire.

The z stage is the only DOF carrying a gravity load. Bisected on `hand_2`,
lifting 30 mm to 35 mm and measuring travel after 3 s:

| torque | 150 | 200 | 250 | 300 | 350 |
|---|---|---|---|---|---|
| moved (of 5.0 mm) | 1.50 | 4.54 | 4.54 | 4.54 | 4.53 |

150 stalls outright, 200 tracks fully, and nothing above 200 helps. The hands
set z's `torque_min_to_move` well above that, 400 on `hand_2` and 800 on the
others, because the bisect ran unloaded and the stage has to lift the aux gripper while it is holding
something. Pressing *down* at 50 works and tasks rely on it, so this is a floor
for the lifting direction, not a correction to the whole axis.

The zero seek presses each DOF into its stop at that hand's
`torque_min_to_move`, the lightest push that still travels. Too much torque
deflects the rack and records the stop long; too little stalls mid rail and
records that as the stop. It is per hand because friction is per unit.
`torque_stuck` is still set on every hand but nothing reads it; it is kept as a
record of what each unit needed when the seek ran at a multiple of the floor.

`counts_per_mm` is derived from the pitch diameter, but a real gear train is not
its nominal drawing. After measuring a known travel, set it directly and the
derived value is ignored:

```python
HandConfig(..., counts_per_mm=80.0)
```

## Setting up a servo

Servos ship with an ID that collides with the rest of the bus, so each is renamed
before it goes into a hand. `hand_3` uses IDs 0-6, `hand_2` 7-13 and `hand_1`
14-20, in `LAYOUT` order. Connect one servo at a time, or the rename is ambiguous and the
new ID could collide with one already in use.

Keep the blocks non-overlapping. They are the only thing that tells one hand from
another over the bus, and `config.identify` uses them: `studio` with no `--hand`
sync-reads each hand's block and opens whichever one answers. Two hands on one
bus, or seven servos that answer where six should, is refused rather than
guessed. A new hand needs a fresh block and an entry in `config.HANDS` (see
[Configuration](#configuration)).

```bash
python scripts/ft_servo_tools/cli.py set-id /dev/ttyACM0 7
python scripts/ft_servo_tools/cli.py scan /dev/ttyACM0
python scripts/ft_servo_tools/cli.py gui /dev/ttyACM0 --ids 7 8 9
```

The GUI is worth having on the bench: a ping only proves that something answers
to the new ID, while motion proves it is the servo in front of you. These tools
take a device path and know nothing about hands, DOFs or millimetres. The GUI needs
`pip install -e '.[gui]'`. See
[`cartesian_hand/src/ft_servo/README.md`](../cartesian_hand/src/ft_servo/README.md).

Renaming writes to the servo's EPROM and survives power cycles. Once renamed, the
only way to find a servo again is to scan for it.

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

- No direct manipulation policy has completed on its physical object, and
  none completes offline either. `--mock` has no object in the jaws, so
  `studio --mock --task cap` fails at its first probe; `MockServo.set_stops`
  can add one from Python, but no CLI flag does. The stock MuJoCo model has no
  objects, so `sim --task cap` prints `finished` without having touched
  anything.
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

The disagreement was once pinned by a comparator reading both sides live, so
reconciling either one failed deliberately rather than drifting quietly. That
file is gone, and the two sides can now diverge in silence. The direction is
not symmetric: **the sim narrows to the hardware, never the reverse.** `config`
is narrower on all seven DOFs and must stay so, because the sim's extra stroke
is not headroom, it is where a carriage leaves its slider.

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

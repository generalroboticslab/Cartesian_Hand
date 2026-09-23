# Hardware, calibration and configuration

Start with [Configuring a new hand](#configuring-a-new-hand), then
[Zeroing](#zeroing). The rest of the page is reference: DOF order, the
`HandConfig` tables and motion gains, adding a servo to the bus, what has been
measured on real hardware, and what is still wrong.

Terms used on this page:

- **DOF**: one of the hand's seven motorized joints, each a servo driving a
  carriage along a rail.
- **base / aux**: the hand's two grippers. Each has a jaw and a pair of
  fingers. The aux gripper rides the z stage.
- **datum**: the hand's zero, the per-DOF encoder offsets that zeroing measures.

For writing a task see [tasks.md](tasks.md). For the engine and backends see
[internals.md](internals.md).

## Configuring a new hand

`config.py` ships `hand_1`, `hand_2` and `hand_3`, the three units built in our
lab. They are not presets. Each measured number in their entries belongs to that
unit. Give a new build its own entry, in this order:

1. **Give the servos IDs** as one consecutive block in DOF order, so DOF *i* is
   servo `first_servo_id + i` (see [Setting up a servo](#setting-up-a-servo)):

   | DOF | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
   |---|---|---|---|---|---|---|---|
   | joint | base jaw | base left finger | base right finger | z stage | aux jaw | aux left finger | aux right finger |
   | axis | y | x | x | z | y | x | x |
2. **Add the entry** at the bottom of the hands in `cartesian_hand/config.py`:

   ```python
   MY_HAND = HandConfig(name="my_hand", port="/dev/serial/by-id/usb-...",
                        first_servo_id=0)
   HANDS = {h.name: h for h in (HAND_1, HAND_2, HAND_3, MY_HAND)}
   ```

   Everything else comes from the shared tables at the top of the file. From
   here on, every command takes `--hand my_hand`.
3. **Check directions.** Run `python -m cartesian_hand.studio --hand my_hand`,
   drag each DOF's slider a few mm, and confirm the real joint and the 3D model
   move the same way. Keep the moves small, because before zeroing, 0 mm is
   wherever the hand was at startup. If a joint moves the wrong way, it is
   mounted differently from ours; read
   [DOF indexing](#dof-indexing-and-orientation) before changing anything.
4. **Measure the torque floor.** The default `torque_min_to_move` is an
   unmeasured starting point that moves our hands with some margin. Bisect yours
   per DOF with a 5 mm move read after 3 s ([Motion gains](#motion-gains)). The
   bench GUI has a torque slider per servo. Set the result on the entry.
5. **Measure travel** with calipers and set `travel_mm`. The default is the CAD
   value, and the far end of each rail has no stop
   ([Known issues](#known-issues)).
6. **Zero** with `python -m cartesian_hand.studio --hand my_hand --task zero`.

Once you have measured a known distance, `counts_per_mm=` overrides the value
derived from the gear.

Always pass `--hand`. Without it, `studio` picks the hand whose servo-ID block
answers. A build numbered 0-6 with no entry of its own answers as `hand_3` and
runs on `hand_3`'s numbers without any error. If it has an entry, the probe
refuses because two hands answered.

## Zeroing

Zero after every power cycle, before running any task:

```bash
python -m cartesian_hand.studio --hand my_hand --task zero
```

or press **Zero hand** on the studio page. Saved offsets go stale by whole
turns across a power cycle ([Known issues](#known-issues)), so a calibration
from an earlier session can be off by a full revolution.

Zeroing drives each DOF into its closed-end hard stop and records where it
stalled, which turns encoder counts into millimetres with an absolute meaning.
It runs three phases in mechanical order: fingers retract before the jaws
close, and the jaws clear before z drops. In the other order a finger would be
inside a closing jaw. Offsets are saved to `zero_offsets.json` at the project
root, or to `$CARTESIAN_HAND_CALIB` if set. If any DOF fails to stall, the run
aborts and the previous calibration stays.

Zeroing works in a relative frame. Millimetres have no absolute meaning until it
finishes, and the task only needs distances anyway. Every goal is
`here ± something`, so the origin cancels. The same task is correct in the
startup-relative frame of an uncalibrated hand, in the calibrated frame, and in
the sim, where `q = 0` is the rest pose.

**A zeroing goal is deliberately outside the travel table.** The seek asks for
120 mm on a 55 mm rail because the hard stop is supposed to end the move, not the
number. For that reason `studio.live` clamps slider goals but not task goals. A
person at a slider can ask for anything, while a program's goals were bounded
when it was built. The unclamped path is safe because of direction. Millimetres
decrease toward the closed end, which has a physical hard stop, and that is the
only direction in which a task sends an unbounded request. Every outward move a
task makes is bounded at build time by `clamped_mm`.

> Do not clamp task goals in the loop. When the hand starts near 0, the clamp
> turns the seek into a no-op.

**If a DOF never stalls, the run aborts and nothing is written.** A joint that
ran out of time has also stopped moving, and its final position looks the same
as a real stop, so zeroing checks the outcome instead of the position. Recording
a timed-out DOF would put the origin mid-travel. Every later millimetre on that
axis would then be off by the shortfall, toward the open end of the rail, with
no warning. Offsets collect in a local tensor and are saved only after all three
phases succeed, so a failed run leaves the previous calibration in place.

Offsets are written to `zero_offsets.json` at the project root, keyed by hand
name. The file sits at the project root and not in the package directory
because `pip install -e .` wipes the package directory, and a lost calibration
costs a bench session. It is gitignored because the numbers describe one
physical machine. `CARTESIAN_HAND_CALIB` overrides the location. The file is
generated, so do not edit it by hand.

A task's result becomes the calibration when its `Config.sets_datum` is `True`.
The check is on that flag, not on `name == "zero"`, so a retuned zeroing variant
in its own file also installs its calibration.

## DOF indexing and orientation

The DOF table in [Configuring a new hand](#configuring-a-new-hand) leaves out
three details.

**DOF index is not servo ID.** The controller addresses a DOF by its index,
which is the same on every hand. The servo ID is what answers on the serial bus,
and it differs per hand. Everything above the driver works in DOF index order.

**`orientation` is the direction the servo counts.** `+1` means rising counts
are rising millimetres. On the current hand wiring all seven are `-1`. We
measured that on the bench; it is not inferred from the left/right labels.

> **`orientation` cannot fix a sim/real direction disagreement.** It cancels
> between `mm_to_counts` and `counts_to_mm`, so flipping it leaves the
> millimetre the model renders unchanged and moves only the hardware. If the sim
> disagrees with the hardware and you flip the sign, the hardware ends up
> backwards. We lost a bench session to this on 2026-09-01: flipping all seven
> to `+1` reversed five DOFs on the real hand and left the model where it was.

**The simulation follows DOF order, not the other way round.** The twin's
actuators must be in DOF order. `LAYOUT`'s axis sequence y,x,x,z,y,x,x matches
the MJCF actuator order one for one. We checked this by walking
`model.actuator_trnid`, not by reading the comment. Nothing asserts it.

Names and groups (`BASE_FINGERS`, `AUX_FINGERS`) are in `config.py` directly
below `LAYOUT`. They share a file so the two cannot drift apart: a role map that
disagrees with the layout gives you a mirrored gripper that still looks
plausible.

## Configuration reference

### How it is laid out

A hand is one flat frozen dataclass, `HandConfig`. Anything a hand's entry
leaves out comes from the shared tables at the top of
`cartesian_hand/config.py`. An entry holds only what differs between units,
which today is the serial port, the first servo ID, the torque floor and the
transit speed. `DEFAULT_HAND` is `hand_3`, and `--mock` and `sim` use it when no
`--hand` is given. Offline, only its travel and speed tables matter. `hand_2` and
`hand_3` share a port path because it names one USB adapter that moved between
them.

Per-DOF properties (axis, count direction, label) are the same on every hand
built so far, so they live once in `LAYOUT` rather than in seven objects per
hand.

`config.py` reads top to bottom: the shared tables (`LAYOUT` and the role names,
`STANDARD_TRAVEL`, `STANDARD_SPEED`, `STANDARD_ACC`,
`STANDARD_TORQUE_MIN_TO_MOVE`, `CALIB_PATH`, `DEFAULT_HAND`), the dataclass, the
three hands, then the calibration file helpers. No code downstream keeps its own
hardware constant, so retuning a gear ratio or a travel limit never means
editing control code.

Use a `/dev/serial/by-id/` path, not `/dev/ttyACM0`. ACM numbers are assigned in
enumeration order, so with two hands plugged in, a hardcoded number addresses
whichever one powered up first.

The dataclass is frozen for a practical reason. Per-DOF tensors are cached per
device, so if you changed `cfg.speed` after something had called `gain_vector`,
the cache would keep the old speed and every later tick would command it.
`FrozenInstanceError` prevents that. Use `variant()`, which also drops the cache.

Every per-DOF value is a `torch.Tensor`, so the only difference between the sim
backend at N=4096 and the hardware backend at N=1 is `device="cuda"`. numpy is
faster at this size (about 0.3 µs against 3 µs to convert 7 elements), but both
are negligible next to a 20,000 µs tick at 50 Hz, so we chose one array library
over a numpy/torch boundary. The one boundary left is the bus. `read_all`
returns Python tuples with `None` for a servo that did not answer, and that
`None` has to stay `None`. A placeholder would hand the layer above a made-up
position, which looks like a large jump, the opposite of the stall that layer is
watching for.

### Motion gains

`torque_min_to_move`, `speed` and `acc` each accept a scalar or a per-DOF
sequence. A sequence of the wrong length raises at construction, not at the
first servo write.

```python
HandConfig(..., torque_min_to_move=250)                              # all DOFs
HandConfig(..., torque_min_to_move=[250, 250, 250, 800, 250, 250, 250])  # z at 800
```

Different values per DOF cost nothing. A sync-write is one broadcast packet in
which each servo reads its own slice, so seven different torques take the same
packet and the same time on the wire as seven identical ones.

Only the z stage carries a gravity load. We bisected its torque on `hand_2`,
lifting from 30 mm to 35 mm and measuring travel after 3 s:

| torque | 150 | 200 | 250 | 300 | 350 |
|---|---|---|---|---|---|
| moved (of 5.0 mm) | 1.50 | 4.54 | 4.54 | 4.54 | 4.53 |

At 150 it stalls, at 200 it tracks fully, and more than 200 does not help. The
hands still set z's `torque_min_to_move` well above that, 400 on `hand_2` and
800 on the others, because the bisect ran unloaded and the stage has to lift the
aux gripper while it holds something. Pressing *down* works at 50 and tasks rely
on it, so this floor applies to lifting only.

The zero seek presses each DOF into its stop at the hand's `torque_min_to_move`,
the lightest push that still moves the joint. Too much torque deflects the rack
and records the stop too far out. Too little stalls mid-rail and records that as
the stop. The value is per hand because friction differs between units.

`counts_per_mm` is derived from the pitch diameter, and a real gear train does
not match its drawing exactly. After measuring a known travel, set it directly
and the derived value is ignored:

```python
HandConfig(..., counts_per_mm=80.0)
```

## Setting up a servo

New servos ship with an ID that collides with the rest of the bus, so rename
each one before it goes into a hand. `hand_3` uses IDs 0-6, `hand_2` 7-13 and
`hand_1` 14-20, in `LAYOUT` order. Connect one servo at a time. Otherwise the
rename is ambiguous and the new ID could collide with one already in use.

The ID blocks are the only way to tell hands apart over the bus, and
`config.identify` relies on them: `studio` with no `--hand` sync-reads each
hand's block and opens whichever one answers. It refuses to guess when two hands
answer, or when seven servos answer where six should. A block that overlaps one
of ours therefore works only with `--hand`. Pick IDs from 21 up if you want the
probe to find it (see [Configuring a new hand](#configuring-a-new-hand)).

```bash
python scripts/ft_servo_tools/cli.py set-id /dev/ttyACM0 7
python scripts/ft_servo_tools/cli.py scan /dev/ttyACM0
python scripts/ft_servo_tools/cli.py gui /dev/ttyACM0 --ids 7 8 9
```

Use the GUI on the bench. A ping only proves that something answers to the new
ID; moving it proves it is the servo in front of you. These tools take a device
path and know nothing about hands, DOFs or millimetres. The GUI needs
`pip install -e '.[gui]'`. See
[`cartesian_hand/src/ft_servo/README.md`](../cartesian_hand/src/ft_servo/README.md).

Renaming writes to the servo's EPROM, so the new ID survives power cycles. After
a rename, scanning is the only way to find the servo.

## Hardware status

Brought up on `hand_2`: servo IDs 7-13, one CH340 adapter at 1 Mbaud.

What has run on servos:

- All seven servos enumerate and report plausible voltage and temperature,
  11.3-11.5 V and 24-26 °C.
- The loop measured 50 Hz with zero drops. A 5 mm goal on DOF 1 tracked to
  0.01 mm of error.
- Each step is two bus packets regardless of gains: one sync-read covering all
  seven servos, and one sync-write carrying per-joint positions and gains.

  | | 7 unicast | 1 sync | |
  |---|---|---|---|
  | read | 1.97 ms | 1.47 ms | one TX replaces seven, but each servo still replies |
  | write | 2.35 ms | ~0 ms | broadcast, unacked, so nothing to wait for |

  Sync-read saves less than you might expect because it removes the seven
  request packets but not the seven replies.
- The tick is 89% sleep. Serial I/O is the only real cost in it, and Python
  cannot make it smaller.
- `acc` affects lag only. On the same 5 mm ramp, acc=25 peaks at 1.37 mm of lag
  and acc=255 at 0.56 mm, and every value arrives.
- Full zeroing has completed, but only with the *previous* implementation. The
  offsets it found, `[4293, -1412, 5608, 5837, 4526, 2012, 6762]`, include values
  past one 4096-count revolution and one negative value, so these servos do not
  stay within a single turn.

Not yet established:

- No direct manipulation policy has completed on its physical object, and none
  completes offline either. `--mock` has no object in the jaws, so
  `studio --mock --task cap` fails at its first probe. `MockServo.set_stops` can
  add one from Python, but there is no CLI flag for it. The stock MuJoCo model
  has no objects, so `sim --task cap` prints `finished` without touching
  anything.
- The four tasks ported from an earlier internal implementation are
  transcriptions. `screwdriver`, `pipette`, `syringe` and `scissors` run through
  the real executor path correctly, but what was validated on hardware is the
  *sequences*, not these implementations of them. Each needs its object and a
  calibrated hand. The task docstrings record two deliberate differences from
  the validated code. `syringe` opens the aux jaw at entry instead of closing it
  to `aux_min_mm`, because the original relied on an unchecked `set_pos`
  timeout. And every ported task floors its horizontal and z-ascent efforts at
  the hand's measured `torque_min_to_move` instead of using one flat travel
  torque.
- Total travel is unmeasured. Zeroing finds only one hard stop per DOF, so
  `STANDARD_TRAVEL` is still the CAD value. See Known issues.
- `counts_per_mm` is still the derived value and has never been checked against
  a measured distance.
- Grip force has not been characterised, and no policy has been transferred from
  a twin.

## Known issues

**`max_mm` is a safety limit, not a measured length.** Each DOF has one hard
stop, the one zeroing seeks. The far end of each rail is open by design. Drive
past it and the carriage leaves the slider and the servo spins freely.
`STANDARD_TRAVEL` is the only thing that prevents this, and it has never been
measured. It holds the v2 CAD figures: 50 mm on the jaws and z stage, 55 mm on
the fingers.

For the jaw pairs, four sources give four numbers, up to 76% apart:

| source | mm |
|---|---|
| `config.STANDARD_TRAVEL` | 50.0 |
| the sim MJCF `ctrlrange` | 52.6317 |
| the sim policy specs | 57.0 |
| `hand_2` DOF 0, stalled, measured | 29.8 |

The 29.8 is probably right. `EXPORT_NOTES.md` derives 52.6317 from raw Fusion
limits of −30 to +22.6317 with the rack parked hard at +22.6317, which assumes
the whole slider span is reachable. If the real stroke is only the 30 mm below
the park point, the measured value is correct. The 57.0 is stale rather than
independent: those specs were written against the model before the sign flip
and do not decode against the asset that ships today.

Travel cannot be measured by driving the hand. A `travel` task tried to seek the
far stop and report the span, but there is no far stop to find. Run on `hand_2`,
it took six of seven carriages off their rails. **Measure with calipers and type
the numbers in.** `counts_per_mm` is shared by the whole hand, so measuring one
axis calibrates all seven. Travel is per DOF, so each rail needs its own
measurement.

No test compares the sim and `config` travel tables, so they can drift apart
unnoticed. When you reconcile them, **narrow the sim to the hardware, never the
reverse.** `config` is narrower on all seven DOFs and must stay that way, because
the sim's extra stroke is where a carriage leaves its slider.

**The action offset also disagrees, and that is worse than travel.** Scale
already agrees, since both sides move half the travel per unit of action. The
offset does not:

```
sim    mm = default_joint_pos + 0.5*(hi-lo)*a     ->  a=0 is the REST pose
real   mm = midpoint          + 0.5*(hi-lo)*a     ->  a=0 is MID-travel
```

Every MJCF `ctrlrange` starts at 0, and the constants file calls that pose "rest
(jaws shut)", so `a = 0` shuts the jaws in sim and opens them 25 mm on hardware.
The sim also never uses its negative action half, because it clamps against a
range that starts at rest. `config`'s convention is the better one to keep, so
the fix is one offset in the sim. No policy has been trained on this hand yet, so
fixing it now costs nothing. After the first training run, it would invalidate
checkpoints.

**Saved zero offsets go stale by whole turns.** A servo reports
(turns since power-up × `counts_per_rev`) plus the angle within the current
turn. The angle comes from a magnetic encoder and is correct as soon as power
arrives, but the turn count restarts at zero. After a power cycle, the reading a
saved offset was measured against no longer exists, and every millimetre command
is off by some whole number of turns. Nothing warns you, because the numbers
still look plausible. On `hand_2` we saw four DOFs read 30 mm and three read a
turn away.

If travel is shorter than one turn, the angle alone places the joint exactly,
and the offsets can be recovered arithmetically:

```python
k = ((raw - offsets) * orientation) % counts_per_rev
rebased = raw - k * orientation
```

**`config.load_offsets` does not implement that recovery.** It returns the saved
offsets without checking that the turn origin still holds. Even with the
recovery, one motor turn is 50.27 mm at the derived `counts_per_mm` and the
fingers are configured at 55 mm, so the widest DOF puts all seven in the case
that cannot be recovered. Getting every rail under 50.27 mm removes the
ambiguity, and that takes the same calipers measurement as the travel issue
above.

**The left and right finger labels are unresolved.** The MJCF calls DOF index 1
`m_right_down_finger`, while `config.LAYOUT` calls it the left one. The sim's
names come from Fusion bodies cross-checked with the designer, so they are the
better evidence. Neither source says which physical finger *servo* 1 drives,
though, and that is what matters here. Getting it wrong raises no error, because
a mirrored policy still looks plausible on a symmetric gripper. To resolve it,
command DOF 1 alone and watch which finger moves. `--swap` exchanges DOFs 1 and
2, and 5 and 6, for testing.

**`fingerprint()` has no callers.** It is meant to catch the travel mismatch
above. Recording it on whatever a twin produces, and rejecting a mismatch before
driving hardware, is not wired up yet.

**Mechanical: DOF 1 on `hand_2` binds.** At the start of the 2026-08-31 session
a 5 mm goal produced 0.09 mm of motion. Driving it ±400 counts a few times freed
it, and the same command then tracked to 4.99 mm. The symptom is full travel in
one direction and about 20% in the other.

# ft_servo

Driver for Feetech HLS-series serial bus servos (HLS3915 and relatives). C++
with nanobind bindings, plus a bench CLI under `scripts/ft_servo_tools/`.

## Install

```bash
pip install -e ".[studio]"               # builds the extension
pip install -e ".[gui]"                  # adds viser for the bench GUI
```

Importing the bindings needs none of those extras:

```python
from cartesian_hand.servo import open_driver

drv = open_driver("/dev/ttyACM0")
```

---

## Quick start

```bash
python scripts/ft_servo_tools/cli.py /dev/ttyACM0
```

Opens the GUI. It scans on open, so you need not know what is on the bus, and it
can rename a servo, so you need not know its ID either. **It energizes the
servos**: torque comes on at start and goes off at exit.

The subcommands below do the same jobs one at a time, for scripts and for
renaming without a browser.

```bash
python scripts/ft_servo_tools/cli.py scan   /dev/ttyACM0
python scripts/ft_servo_tools/cli.py set-id /dev/ttyACM0 7
python scripts/ft_servo_tools/cli.py gui    /dev/ttyACM0 --ids 7 8 9
```

---

## CLI

### scan: what is on the bus

```
scanning IDs 0-20: found [7, 8, 9]
  ID   7: pos=-1621 volt=11.5V temp=28C
  ID   8: pos=6424  volt=11.4V temp=28C
```

One ping per ID, so keep `--end` low (default 20). Negative positions are
normal. `?` means the read failed, not that the value is zero.

### set-id: rename a servo

**Connect one servo only.** This subcommand scans to discover which servo to
rename, so two candidates make the choice ambiguous. The GUI's rename has no
such restriction: you pick the servo from a list, so a crowded bus is fine.

**The servo moves.** It drives 300 counts (~2.5mm) out and back to prove the new
ID responds, so leave the output free to turn. It refuses under 9.0V or over
65°C, because a servo that browns out mid-write answers to no ID at all
afterwards. The ID goes to EPROM and survives power cycles, so after a rename
the only way to find the servo is to scan for it.

### gui: drive them by hand

Scans, takes whatever answers, prints the URL it actually bound. `--ids 7 8 9`
drives a subset instead.

Per servo: live position bar, live load bar, target slider, torque slider.
Above them `span` and `speed` apply bus-wide, plus a folded-away **change a
servo ID** panel. Everything but the serial port is live.

Renaming refuses an ID already on the bus, and refuses under 9.0V or over 65°C.
On success it releases torque, rescans, and rebuilds the panel. A new ID
invalidates every label on screen at once, so the panel is replaced rather than
patched.

| flag | default | |
|---|---|---|
| `--span` | 4000 | counts a slider may travel either side of the startup position (~49mm) |
| `--torque` | 50 | force cap, 0-1000; every servo's slider starts here |
| `--speed` | 300 | speed for slider moves |
| `--web-port` | 8080 | falls through to the next free port if taken |

`span` and `torque` are safety guards, because this tool cannot know what a
servo drives or how hard it should push. On the Cartesian hand only *one* end of
each rail has a stop, so a wide slider is a one-drag route to pushing a carriage
off. Torque 50 is what six of the seven DOFs run at; the z stage needs 300 to
lift.

Torque only reaches a servo attached to a goal, so both sliders re-send the
current target. A servo stalled by too low a cap moves as soon as you raise it.
`span` re-anchors on the **startup** position, not the current one, or a servo
walks its own window along one drag at a time.

**Torque comes on at start and off at exit.** Ctrl+C and `kill` both release. If
it dies some other way:

```python
open_driver("/dev/ttyACM0").enable_torques([7, 8, 9], False)
```

---

## Python

```python
from cartesian_hand.servo import open_driver

drv = open_driver("/dev/ttyACM0")
ids = drv.scan(0, 20)

drv.read_positions(ids)             # [-1621, 6424, -1170], or None per servo
drv.set_positions(ids, [0] * len(ids), speed=300, acc=50, torque=50)
drv.enable_torques(ids, False)
drv.close()
```

`scripts/ft_servo_tools/example.py` is that code as a script: scan, read, move
the whole bus 5mm and back, release. **It moves the hand.**

```bash
python scripts/ft_servo_tools/example.py /dev/ttyACM0
```

```
[7, 8, 9, 10, 11, 12, 13] at [2678, 2496, 2989, 3852, 2762, 371, 975]
moved to [3277, 3096, 3589, 4451, 3358, 971, 1575]
back at [2683, 2496, 2989, 3855, 2762, 371, 975]
```

### The rest of the surface

Every method has a docstring, so `help(FtServo)` is the reference. This table
is an overview.

| | |
|---|---|
| **motion** | |
| `set_positions(ids, pos, speed, acc, torque)` | goal position; gains are ints or per-servo lists |
| `set_speeds(ids, speeds, acc, torque)` | constant-speed mode; needs `set_modes` first |
| `set_modes(ids, mode)` | 0 position, 1 constant speed, 2 constant torque |
| `enable_torques(ids, on)` | energize or release; released servos backdrive |
| **live reads** (hit the bus, `None` when no reply arrives) | |
| `read_positions(ids)` | whole bus, one packet |
| `read_position(id)` `read_speed(id)` `read_load(id)` | one servo, one round trip each |
| `get_voltage(id)` | tenths of a volt: `115` is 11.5V |
| `get_temperature(id)` | Celsius |
| `ping(id)` | the ID, or `-1` on timeout |
| **cached reads** (free, but **zeros unless `start_poll` runs**) | |
| `get_positions(ids)` `get_speeds(ids)` `get_loads(ids)` | whole bus |
| `get_position(id)` `get_speed(id)` `get_load(id)` | one servo |
| **bus and lifecycle** | |
| `scan(start_id=0, end_id=253)` | IDs that answer; one ping each, so keep it tight |
| `start_poll(ids, interval_us=5000)` / `stop_poll()` | background refresh; owns the sync-read buffer |
| `poll_silence()` | consecutive sweeps nobody answered; 0 while healthy |
| `write_id(id, new_id)` | EPROM; use the `set-id` CLI instead |
| `close()` | closes the port; **does not release torque** |

Each plural has a singular (`set_position`, `set_speed`, `enable_torque`,
`set_mode`). They buy nothing over a length-1 list and exist because the vendor
SDK has them.

Not bound: `HLSCL_TORQUE_LIMIT` (register 48, a device-level force cap
independent of the per-goal `torque`), `MIN`/`MAX_ANGLE_LIMIT` (registers 9-12,
travel limits enforced in firmware), `set_position_offset`.

### Batch or not

hand_2, 7 servos, 1Mbaud:

| | batched | one at a time |
|---|---|---|
| read | 1.47ms | 1.84ms |
| write | **0.003ms** | 2.32ms |

Almost all of the difference is in writes. `set_positions` is a broadcast nobody acknowledges,
so it costs one buffered write no matter how many servos; `set_position` is a
unicast that waits for a status reply from each in turn.

Reads scale with servo count either way, ~0.2ms each. Sync-read broadcasts one
request, but every servo still transmits its own reply. Batching saves the
per-servo request and turnaround, nothing more.

Per-servo gains cost nothing: same packet, same time on the wire, because
`INST_SYNC_WRITE` broadcasts once and each servo reads its own slice.

```python
drv.set_positions(ids, targets, [300] * 3, [50] * 3, [50, 50, 300])
```

---

## Gotchas

**A failed read is `None`, never a number.** `SCS::readWord` returns `-1` on
failure and `HLSCL::ReadPos` sign-magnitude-decodes that into `+32769`, a
plausible position about 400mm from zero. The driver catches it via
`getLastError()`. `read_positions` cannot hit it at all: sync-read signals a
missing reply out of band.

**`get_*` returns zeros without `start_poll`**, which reads exactly like a bus
parked at origin. This has caused a real bug here. Conversely `read_positions`
throws while the poll thread runs, because they share one buffer. Use one or
the other.

**A cache cannot tell you the bus went away.** When the poll thread stops
getting replies it keeps the last good numbers, so `get_positions` reads the
same whether the hand is holding still or the adapter has been unplugged. A GUI
polling that cache spins forever on a hand that is not there. `poll_silence()`
tells the two apart. It counts consecutive sweeps in which *no* servo
replied. One or two is interference; a climbing count is a dead bus. The GUI releases and exits after two
seconds of it.

The same trap without a cache: `read_positions` returning all-`None` looks like
a bad frame, so a control loop that keeps the previous vector will happily
command a dead bus while `actual` sits frozen. `hand.py` counts consecutive
all-silent reads and raises at `MAX_SILENT_STEPS` (half a second at 50Hz), which
routes into the loop's existing drop-torque path.

**Position, speed and load are sign-magnitude** (bit 15 = sign), not two's
complement, so negative values are real data. Voltage and temperature cannot be
negative, so `-1` there *is* an error.

**Saved zero offsets go stale by whole turns.** The within-turn encoder angle is
absolute and survives power loss; the turn counter is RAM and restarts at zero.
Positions stay plausible while being wrong. Recovery arithmetic is in
[`docs/hardware.md`](../../../docs/hardware.md#known-issues).

**±32767 counts is ±8 turns** at 4096 counts/rev.

**A stale `.so` can shadow the installed one.** `cartesian_hand.servo` imports
`from .ft_servo_ext import FtServo`, so it only finds the extension installed
inside the `cartesian_hand/` package. A CMake `build/` tree left in the repo
root can still shadow the installed one on `sys.path`-by-`cwd` imports. Symptom:
`AttributeError` for a method you can see in the source. Fix: delete the
stale build tree.

---

## Files

| | |
|---|---|
| `ft_servo_driver.hpp` | the driver: locking, poll thread, sync-read/write |
| `ft_servo_ext.cpp` | nanobind bindings, one docstring per method |
| `INST.h`, `SCS.*`, `SCSerial.*`, `HLSCL.*` | vendor SDK, left alone |

The bench CLI lives at `scripts/ft_servo_tools/`; this directory is just the
C++ side.

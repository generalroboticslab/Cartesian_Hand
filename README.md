# The Cartesian Hand

**In-hand manipulation with all-linear fingers.**

Boxi Xia†, Bokuan Li†, Ryan Shin, Zijiang Yang, Jiaxun Liu,
[Boyuan Chen](http://boyuanchen.com/)

Duke University, [General Robotics Lab](https://generalroboticslab.com/).
† equal contribution, co-first authors.

[Project page](https://generalroboticslab.com/cartesian_handv1) with full-length video of every object.

Two parallel grippers hold different parts of an object. Four translating
fingertips move those parts relative to each other. Every joint slides in a
straight line, so a fingertip travels along a fixed axis in any configuration
and the hand has no singular poses.

This repository is the control stack: task authoring, the tensor engine that
runs a task, and three backends (`cartesian_hand.studio` for hardware,
`cartesian_hand.sim` for CPU MuJoCo, `cartesian_hand.sim --warp` for batched
GPU MuJoCo) that execute it.

## Objects

<div align="left">
  <img src="media/objects.webp" width="820">
</div>

35 objects, across laboratory, manufacturing and household settings. Objects
that share a mechanism share a procedure. The same procedures also transfer to
a humanoid with one hand on each arm.

| Mechanism | `--task` | n |
|---|---|--:|
| Cap | `cap` | 15 |
| Two-handle | `scissors` | 6 |
| Trigger | `triggers` | 5 |
| Pump | `pump`, `syringe` | 3 |
| Screwdriver | `screwdriver` | 3 |
| Pipette | `pipette` | 2 |
| Reorientation | `tilt` | 1 |

The task files in `cartesian_hand/tasks/` are transcriptions of an earlier
internal implementation. They have not been re-run on the objects yet.

<table>
  <tr>
    <td><img src="media/hero.webp" width="640"></td>
    <td><img src="media/humanoid-bimanual.webp" width="640"></td>
  </tr>
  <tr>
    <td align="center"><sub>Bimanual manipulation</sub></td>
    <td align="center"><sub>Same procedures on a humanoid</sub></td>
  </tr>
</table>

## The hand

<div align="left">
  <img src="media/design-cad.jpg" width="380">
</div>

Four actuated fingertips (P1 to P4), a parallel-jaw gripper on each side, and
the auxiliary gripper rides the vertical stage. Joint speed is around 60 mm/s,
mass is 850 g, and a single gripper holds 2 kg statically. Travel is unenforced
by hardware: `config.STANDARD_TRAVEL` is the only thing between a command and a
carriage off its rail. Read
[Known issues](docs/hardware.md#known-issues) before driving any DOF to a limit.

## How it fits together

A controller opens no serial port, never sleeps, and steps no simulator. It
sees typed observations in millimetres and returns commands. The executor alone
owns I/O and pacing.

```text
tasks.make(name) ─┬─ Policy ─ PolicyRunner ─┬─ studio.live   servos + viser
                  └─ Task ─── TaskRunner ───┼─ sim.run       CPU MuJoCo
                                            └─ sim.run_warp  batched GPU MuJoCo
```

Two controller forms share the pipeline. `Policy.step` is the closed-loop
target, runs every tick, keeps tensor state, and is batch-friendly.
`Motions`/`TaskRunner` executes fixed `[N,J,K]` programs, still used for
zeroing and GUI-authored timelines.

## Installation

```bash
git clone https://github.com/generalroboticslab/Cartesian_Hand.git
cd Cartesian_Hand
pip install -e ".[sim,studio]"
```

`torch` is a core dependency. The extras: `sim` pulls MuJoCo, `studio` pulls
viser for the browser page, `camera` pulls OpenCV for the studio's camera
window, and `gui` pulls viser for the bench CLI's GUI subcommand. The batched
GPU path additionally needs `warp` and `mujoco_warp`, which are not declared
because they are not on PyPI under stable names.

Needs Python 3.10+ on Linux (macOS untested; Windows does not build, the serial
driver uses `termios`). Every `pip install` compiles the nanobind extension
(`ft_servo_ext`) from the C++ driver under `cartesian_hand/src/ft_servo/`, so
CMake 3.15+ and a C++17 compiler are required even without hardware.
`servo.open_driver` imports the extension inside the call, so importing the
package never touches a serial port.

## Quick start

No hardware? Run the studio on a fake servo bus and open http://localhost:8081.
The 3D hand follows the sliders and the task buttons. `sim` runs the same task
files in MuJoCo, headless, and prints the final millimetres.

```bash
export CARTESIAN_HAND_CALIB=/tmp/zero_offsets.json   # keep mock zeroing off a real hand's file
python -m cartesian_hand.studio --mock                        # browser page at :8081
python -m cartesian_hand.studio --mock --task zero            # zero the fake hand
python -m cartesian_hand.sim --task zero                      # CPU MuJoCo
python -m cartesian_hand.sim --task zero --n-envs 4096 --warp  # GPU, batched
```

Neither has an object to hold. The fake bus has stops at the ends of travel and
nothing else, so `zero` completes and manipulation tasks such as `cap` fail at
their first probe. The MuJoCo model has no objects either, so a manipulation
task there reports `finished` without having held anything.

With a hand plugged in. `config.py` ships the three hands built in our lab
(`hand_1` to `hand_3`); add an entry for yours first, see
[Configuration](docs/hardware.md#configuration):

```bash
python -m cartesian_hand.studio --hand my_hand   # your entry in config.HANDS
python -m cartesian_hand.studio                  # or pick the hand by servo IDs
python -m cartesian_hand.studio --task zero      # zero it, headless
python -m cartesian_hand.studio --teach          # limp, pose it by hand
python -m cartesian_hand.release_torque          # emergency torque cut; port is hardcoded, edit it
```

Zero the hand before trusting a millimetre. Without calibration, zero is the
startup pose, so starting mid-travel and driving a full stroke can run a
carriage off its rail. See [Zeroing](docs/hardware.md#zeroing).

Simulation uses the MuJoCo model bundled at `assets/cartesian_hand/`, so no
external checkout is needed. Every entry point is
[tyro](https://brentyi.github.io/tyro/) over a function signature, so `--help`
lists the real flags.

## Status

This repository is the control stack, rewritten. It is not the code that
produced the object results above.

Working on hardware: all seven servos enumerate, the loop holds 50 Hz with
zero drops, and a 5 mm goal tracks to 0.01 mm of error.

Not established here: no manipulation task in `cartesian_hand/tasks/` has
completed on its physical object. Total travel is CAD rather than measured,
and `counts_per_mm` has never been checked against a measured distance. Full
measurements are in
[Hardware status](docs/hardware.md#hardware-status); each gap has an entry
under [Known issues](docs/hardware.md#known-issues).

## Documentation

| | |
|---|---|
| [docs/tasks.md](docs/tasks.md) | Writing a task, how a `--task` name reaches a file |
| [docs/internals.md](docs/internals.md) | The `[N,J,K]` engine, primitives, three backends |
| [docs/hardware.md](docs/hardware.md) | Zeroing, `HandConfig`, gains, servo setup, status, known issues |

## Citation

```bibtex
@misc{xia2026cartesianhandinhandmanipulation,
  title         = {The Cartesian Hand: In-Hand Manipulation with All-Linear Fingers},
  author        = {Boxi Xia and Bokuan Li and Ryan Shin and Zijiang Yang
                   and Jiaxun Liu and Boyuan Chen},
  year          = {2026},
  eprint        = {2609.25696},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO},
  url           = {https://arxiv.org/abs/2609.25696},
}
```

## Acknowledgements

Supported by DARPA FoundSci under award HR00112490372, DARPA TIAMAT under award
HR00112490419, ARO under award W911NF2410405, and ARL STRONG under awards
W911NF2320182, W911NF2220113 and W911NF242021.

## License

Apache License 2.0. See [LICENSE](LICENSE).

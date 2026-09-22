> **In-repo note (2026-08-29):** these are the ORIGINAL export notes from the `mini_gripper` pipeline and
> describe its sign convention (travel negative from the export pose, `polycoef -1` followers). The committed
> model FLIPS that: every axis points in its useful direction, all ranges are `[0, travel]`, followers are
> `polycoef +1`. See `kinematics.json` `_ranges_note` / `_couplings_note` and the module README.

# cartesian_hand_sim — CAD export

What was pulled out of the Fusion 360 design `cartesian_hand_sim` on 2026-08-28,
and what it means. Exported on Windows; the MJCF build happens on Ubuntu.

## Files in `step/`

| File | What it is |
|---|---|
| `cartesian_hand_sim.step` | 14.5 MB. Geometry + assembly tree, written by `ExportManager` from the **root** component, so occurrence paths are complete. |
| `component_kinematics.json` | 31 joints, 4 motion links, 7 rigid groups. Lengths in mm, angles in rad. |
| `component_physics.json` | 178 leaf occurrences, 824.26 g total. **Placeholder** — no physical materials were assigned in Fusion, so these are the default material, not the real gripper. Re-run the exporter once materials are set; nothing else has to be redone. |

All three were written by `common/fusion_export_all/fusion_export_all.py`
(a Fusion add-in script — it runs inside Fusion, not under the project venv).

## Pose at export

The mechanism was posed with all four jaws retracted before exporting. Because
the pipeline keeps mesh vertices in the Fusion world frame, **that pose is
`q=0`** — see `common/SETUP_FROM_FUSION.md`. Joint ranges below are written
relative to it. `ref` in `kinematics.json` can move `q=0` later without
re-exporting.

## Topology — 10 bodies, 9 slide DOFs

Same architecture as the active Cartesian Hand (`grasp_suite/gripper.xml`):
10 bodies, 9 slide joints, zero revolute DOFs.

```
base                                         world-fixed
├── bridge              bridge_z     Z  -60..0
│   ├── left_up_rack    left_up_y    Y  -52.63..0       driven
│   │   └── left_up_finger    left_up_finger_x    X  -60..0
│   └── right_up_rack   right_up_y   Y  0..+52.63       coupled to left_up_y, ratio -1
│       └── right_up_finger   right_up_finger_x   X  -60..0
├── left_down_rack      left_down_y  Y  -52.63..0       driven
│   └── left_down_finger      left_down_finger_x  X  -60..0
└── right_down_rack     right_down_y Y  0..+52.63       coupled to left_down_y, ratio -1
    └── right_down_finger     right_down_finger_x X  -60..0
```

The four `*_finger_x` joints are independent — one servo each. Only the two
rack pairs are coupled. 7 actuators total.

Ranges above are in **mm relative to the export pose**; MJCF wants metres, so
divide by 1000. They are **not** the raw Fusion limits — see the next section.

## Joint ranges are rebased onto the export pose

Fusion states a joint's limits relative to **its own joint zero**. MuJoCo's
`q=0` is whatever pose the mechanism was in when the STEP was written. Those
two origins only coincide if the joint happened to be sitting on its Fusion
zero at export time, so the MuJoCo range is

```
range = [limit_min − value_mm, limit_max − value_mm]
```

using the `value_mm` this export recorded per joint. The offsets here:

| Joint | `value_mm` at export | Raw Fusion limits | MuJoCo range (mm) |
|---|---|---|---|
| `bridge_z` | 0 | −60 .. 0 | −60 .. 0 |
| all four `*_finger_x` | 0 | −60 .. 0 | −60 .. 0 |
| `left_down_y` | **+22.6317** | −30 .. +22.6317 | **−52.6317 .. 0** |
| `left_up_y` | **+22.6317** | −30 .. +22.6317 | **−52.6317 .. 0** |

Five joints sit exactly on their Fusion zero and pass through unchanged. Both
rack drivers are parked hard against their upper limit, so `q=0` is the fully
closed pose and the jaws only ever travel negative. Using their raw limits
instead would put the jaws ~22 mm off the geometry.

The exporter reports this directly — its summary popup names every DOF that is
not on its Fusion zero, with the offset. Read it before trusting a range.

**This is the second export.** The first one was posed by hand and the up pair
came out 0.2156 mm short of the limit, so its range disagreed with the down
pair's by that much. Re-posing with both drivers pushed to a hard stop removed
the asymmetry. If you re-pose again, every mesh and collision piece has to be
regenerated too: vertices are frozen at the export pose, so geometry and range
must come from the same export.

## Name mapping — Fusion to MJCF

Left/right follows the existing model's convention: **left = −Y, right = +Y**
(verified against `grasp_suite/step/body_inertia.json`: `left_up_rack` sits at
Y=−55.1 mm, `right_up_rack` at Y=+55.1 mm).

### Bodies

| Fusion occurrence | MJCF body | COM Y (mm) | COM Z (mm) |
|---|---|---|---|
| `base_gripper (1):1/base_gripper:1` | `base` | +0.9 | 13.7 |
| `aux_gripper (1):1/aux_gripper:1` | `bridge` | +1.4 | 26.8 |
| `fingers:1/finger (1):1/finger:1` | `left_down_rack` | −10.8 | 2.7 |
| `fingers:1/finger (1):5/finger:1` | `right_down_rack` | +10.8 | −2.7 |
| `fingers:1/finger (1):6/finger:1` | `left_up_rack` | −11.1 | 34.7 |
| `fingers:1/finger (1):7/finger:1` | `right_up_rack` | +11.1 | 29.3 |
| `finger_tips:1/finger_tip (1):1/finger_tip:1` | `left_down_finger` | −5.2 | 0.7 |
| `finger_tips:1/finger_tip (1):2/finger_tip:1` | `right_down_finger` | +5.2 | −0.7 |
| `finger_tips:1/finger_tip (1):3/finger_tip:1` | `left_up_finger` | −5.4 | 32.7 |
| `finger_tips:1/finger_tip (1):4/finger_tip:1` | `right_up_finger` | +5.4 | 31.3 |

### Joints

| Fusion joint | Connects | Axis | Range (mm) | MJCF joint |
|---|---|---|---|---|
| `Slider 1` | bridge ← base | −Z | −60 .. 0 | `bridge_z` |
| `Slider 6` | left_down_rack ← base | +Y | −30 .. 22.63 *(rebased: −52.63 .. 0)* | `left_down_y` |
| `Slider 7` | right_down_rack ← base | +Y | *(follower)* | `right_down_y` |
| `Slider 8` | left_up_rack ← bridge | +Y | −30 .. 22.63 *(rebased: −52.63 .. 0)* | `left_up_y` |
| `Slider 9` | right_up_rack ← bridge | +Y | *(follower)* | `right_up_y` |
| `Slider 10` | left_down_finger ← left_down_rack | −X | −60 .. 0 | `left_down_finger_x` |
| `Slider 11` | right_down_finger ← right_down_rack | −X | *(blank — use −60 .. 0)* | `right_down_finger_x` |
| `Slider 12` | left_up_finger ← left_up_rack | −X | −60 .. 0 | `left_up_finger_x` |
| `Slider 13` | right_up_finger ← right_up_rack | −X | −60 .. 0 | `right_up_finger_x` |

The 7 `Revolute 2` joints are the gear pinions. They are unlimited free spins
and carry no useful DOF — the active model drops its gears the same way
(`gripper.xml` has zero revolute joints).

### Motion links

| Fusion | Joint pair | Meaning |
|---|---|---|
| `Motion Link 26` | Slider 6 ↔ 7 | **real** — down rack pair, opposite directions |
| `Motion Link 27` | Slider 9 ↔ 8 | **real** — up rack pair, opposite directions |
| `Motion Link 28` | Slider 10 ↔ 11 | **CAD mistake** — added accidentally, couples the two down fingers. Delete in Fusion. |
| `Motion Link 29` | Slider 12 ↔ 13 | **CAD mistake** — same, for the two up fingers. |

## Servos — 7 of them, and what each drives

Derived from per-occurrence COMs in `component_physics.json`, not from any
CAD annotation. Each servo has a matching gear.

| Servo instance | COM [X, Y, Z] mm | Mounted on | Drives |
|---|---|---|---|
| `FEETECH_HL_3915_Servo_Motor:3` | [−17.0, +21.5, 7.5] | base | `bridge_z` |
| `:1` | [−12.1, −6.0, 0.0] | base | down jaw pair Y |
| `:2` | [−12.0, −6.0, 32.0] | bridge | up jaw pair Y |
| `:5` | [97.9, −27.9, 0.0] | left_down_rack | `left_down_finger_x` |
| `:6` | [97.9, +27.9, 0.0] | right_down_rack | `right_down_finger_x` |
| `:4` | [97.9, −28.2, 32.0] | left_up_rack | `left_up_finger_x` |
| `:7` | [97.9, +28.2, 32.0] | right_up_rack | `right_up_finger_x` |

Three servos in the body, one riding each of the four racks.

## Articulation — confirmed by the designer, 2026-08-28

Two of the four motion links are real; two are not. Confirmed directly against
the design, and consistent with the exported geometry:

**The four racks form two coupled pairs, moving in opposite directions.**
Lower pair `Slider 6 / 7` sits at Z≈0, upper pair `Slider 8 / 9` at Z≈30. Within
each pair the two racks sit symmetrically about Y=0 (∓10.8 mm and ∓11.1 mm) and
both carry a `+Y` axis, so opposite-sign travel is the only way they close on
each other. Hence `polycoef = [0, -1, 0, 0, 0]` for both couplings — the same
two-coupling structure the active Cartesian Hand uses, but with the sign flipped
because that model mirrors its axes instead.

**The four fingers are independent.** Each fingertip rides its own rack
(tip:1→rack:1, tip:2→rack:5, tip:3→rack:6, tip:4→rack:7 — no crossing) and has
its own servo.

`Motion Link 28` and `29` are a **CAD mistake**, confirmed by the designer: they
were added accidentally and couple the two down fingers and the two up fingers
respectively. They do not correspond to anything in the mechanism. Ignore them
when building the model, and delete them in Fusion.

Their one visible side effect in this export is that `Slider 11` carries no
range — Fusion treated it as `Motion Link 28`'s follower and derived its travel
from `Slider 10` instead of storing its own. Nothing else is contaminated: the
STEP geometry, the mass properties and the other eight joints are unaffected by
a motion link.

That gives 9 joints − 2 couplings = **7 actuators**, matching both the 7 servos
in the CAD and the active Cartesian Hand.

### Consequence for the joint ranges

Fusion derives a follower joint's limits from its driver, so three joints
exported with no range of their own. When writing `kinematics.json`:

| Joint | Exported range | Range to write | Why |
|---|---|---|---|
| `right_down_finger_x` (Slider 11) | *(none)* | −60 .. 0 mm | Independent — same travel as its three siblings. Its blank range is an artifact of the discarded `Motion Link 28`. |
| `right_down_y` (Slider 7) | *(none)* | −22.63 .. 30 mm | Follower of `left_down_y` at ratio −1, so its range is the driver's negated and reversed. |
| `right_up_y` (Slider 9) | *(none)* | −22.63 .. 30 mm | Same, follower of `left_up_y`. |

## Config files — written and validated on Windows

Both files that carry human decisions are done. Everything downstream of them
is mechanical.

`config/groups.json` — 178 leaf occurrences across 10 bodies. Verified with
`assembly_tree.py --groups`: 0 orphans, 0 overlaps,
`ALL LEAVES COVERED EXACTLY ONCE`, exit 0. Per-body leaf counts: base 49,
bridge 25, each rack 25, each finger 1.

Kept deliberately comment-free — `combine_inertia.py` does not filter
`_`-prefixed keys and dies with `AttributeError` on a string value. That is a
documented trap, not an oversight.

`config/kinematics.json` — 10 bodies, 9 joints, 2 couplings, 7 actuators.
Checked: body sets agree with `groups.json`, exactly one `world` root, every
parent resolves, every coupling and actuator references a real joint, the
coupled ranges are exact `-1` mirrors of each other, and 9 − 2 = 7 matches the
actuator count. Collision is **disabled** in this first build — a visual-only
model is enough to confirm the mechanism, ranges and couplings in the viewer,
and it skips the multi-minute CoACD stage.

Three values in it are inherited from the sibling Cartesian Hand rather than
derived for this design, and are flagged inline: the `±87 N` force limit, the
per-joint `damping`/`armature`, and the actuator `kp`/`kv`.

## Not done yet — four commands, on Ubuntu

The CAD stages need the Linux `LD_PRELOAD` libexpat workaround, so they do not
run on Windows. See `common/SETUP_FROM_FUSION.md`. From `cartesian_hand_sim/`:

```bash
# 1. STEP -> 10 merged OBJs, world-frame, mm
python3 ../common/pipeline/step_to_obj.py \
    step/cartesian_hand_sim.step config/groups.json --out-dir meshes

# 2. decimate the oversized ones (in place)
python3 ../common/pipeline/decimate.py meshes/base.obj meshes/bridge.obj \
    meshes/left_up_rack.obj meshes/right_up_rack.obj \
    meshes/left_down_rack.obj meshes/right_down_rack.obj \
    meshes/left_up_finger.obj meshes/right_up_finger.obj \
    meshes/left_down_finger.obj meshes/right_down_finger.obj

# 3. placeholder inertia -- uniform density, no Fusion needed
python3 ../common/pipeline/mesh_inertia.py \
    --bodies base bridge left_up_rack right_up_rack left_down_rack \
             right_down_rack left_up_finger right_up_finger \
             left_down_finger right_down_finger \
    --mesh-dir meshes --density 1100 --out step/body_inertia.json

# 4. emit the MJCF
python3 ../common/pipeline/build_mjcf.py \
    --inertia step/body_inertia.json --kinematics config/kinematics.json \
    --mesh-dir meshes --out gripper.xml
```

Then load `gripper.xml` in the viewer and check the mechanism: `bridge_z`
raises the upper assembly, each rack pair closes symmetrically, and the four
fingers extend independently.

Afterwards, when the model needs real physics rather than a look:

- `decompose_collision.py` for CoACD pieces, then flip `collision.enabled` to
  `true` in `kinematics.json` and re-run step 4
- assign physical materials in Fusion, re-run the export script, then
  `combine_inertia.py` for real masses in place of the density placeholder

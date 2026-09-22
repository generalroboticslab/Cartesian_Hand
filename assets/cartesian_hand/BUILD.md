# Building `cartesian_hand.xml`

The model is a build artifact. Never hand-edit `cartesian_hand.xml`; the four
scripts here read `source/` + `meshes/` and emit it bit-identically.

## Regenerate the XML (no extra dependencies)

```bash
python assets/cartesian_hand/cartesian_hand_creation.py            # write
python assets/cartesian_hand/cartesian_hand_creation.py --check    # exit 1 on drift
```

`--check` rebuilds each variant in memory, byte-compares against the committed
file, and compiles it with MuJoCo. `nu == 7`, `njnt == 9`, `neq == 2`,
`ngeom == 498` is what "OK" looks like for the graft-ready module.

Inputs it reads:

- `source/kinematics.json` — body tree, 9 slide joints, 2 couplings, 7 actuators
- `source/body_inertia.json` — per-body mass / CoM / inertia (placeholder density)
- `meshes/` — 22 visual OBJs, named `{body}__{material}.obj`; 477 CoACD colliders
  under `meshes/collision_pieces/`

## Where the hardware is

- **CAD** — `assets/cartesian_hand/source/cartesian_hand_sim.step` (13.6 MB).
  Fusion re-posed export of the Cartesian Hand, the only record of the
  geometry. `step_to_obj.py` reads this file and rewrites the 10 (now 22
  split by material) visual OBJs.
- **OBJ meshes** — `assets/cartesian_hand/meshes/`. The split visual meshes
  (`base__steel.obj`, `base__nylon.obj`, `base__red.obj`, ...) and the CoACD
  collision pieces under `meshes/collision_pieces/`. Loaded directly by
  MuJoCo via the `meshdir="meshes"` line in `cartesian_hand.xml`.

Optional cross-check against `mj_envs/asset_zoo/cartesian_hand/cartesian_hand_constants.py`
(the mjlab config module, not in this repo) is skipped when that file is absent.

## Regenerate the meshes (needs extra dependencies)

The committed meshes and CoACD pieces stay as inputs because the upstream tools
are not deterministic across versions. Only re-run the stages below when the
geometry actually changed.

```bash
# 1. Re-tessellate meshes/ from the Fusion STEP. Needs OCP (OpenCascade Python).
python assets/cartesian_hand/step_to_obj.py

# 2. Recompute source/body_inertia.json from the meshes. Needs trimesh + numpy.
python assets/cartesian_hand/mesh_inertia.py

# 3. Re-decompose collision geometry. Needs coacd + trimesh. Run only for bodies
#    whose geometry changed -- refreshing unchanged bodies churns the diff for
#    no gain, since CoACD is not piece-for-piece reproducible.
python assets/cartesian_hand/decompose_collision.py base bridge

# 4. Regenerate the XML.
python assets/cartesian_hand/cartesian_hand_creation.py
```

`source/cartesian_hand_sim.step` is the only record of the Fusion export and is
the input to stage 1. Without it, only stages 2–4 can run.

## Layout

```
assets/cartesian_hand/
  cartesian_hand.xml              the graft-ready model (BUILD ARTIFACT)
  cartesian_hand_creation.py      XML builder, stdlib-only + mujoco
  step_to_obj.py                  STEP -> per-body OBJs (needs OCP)
  mesh_inertia.py                 meshes -> body_inertia.json (needs trimesh)
  decompose_collision.py          meshes -> CoACD pieces (needs coacd)
  meshes/                         22 visual + 477 CoACD colliders, committed
  meshes_lowres/                  superseded 10-mesh build; kept as the
                                  --verify reference, loaded by nothing
  source/                         CAD provenance + builder inputs
    kinematics.json               body tree, joints, couplings, actuators
    body_inertia.json             per-body physics
    groups.json                   Fusion-occurrence -> body assignment
    cartesian_hand_sim.step       Fusion re-posed export 2026-08-28 (13.6 MB)
    component_*.json              per-component kinematics and physics (context)
    EXPORT_NOTES.md               Fusion rebase + range derivation
```

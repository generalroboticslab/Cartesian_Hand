# Building `cartesian_hand.xml`

The XML is generated. Do not hand-edit it; the scripts in this directory
read `source/` and `meshes/` and emit it bit-identically.

## Regenerate the XML (no extra dependencies)

```bash
python assets/cartesian_hand/cartesian_hand_creation.py            # write
python assets/cartesian_hand/cartesian_hand_creation.py --check    # exit 1 on drift
```

`--check` rebuilds each variant in memory, byte-compares against the committed
file, and compiles it with MuJoCo. For the graft-ready module, an OK line
reports `nu == 7`, `njnt == 9`, `neq == 2`, `ngeom == 498`.

The builder reads:

- `source/kinematics.json` -- body tree, 9 slide joints, 2 couplings, 7 actuators
- `source/body_inertia.json` -- per-body mass, CoM, inertia at placeholder density
- `meshes/` -- 22 visual OBJs named `{body}__{material}.obj`, plus 477 CoACD
  colliders under `meshes/collision_pieces/`

It also tries to cross-check against
`mj_envs/asset_zoo/cartesian_hand/cartesian_hand_constants.py`, the mjlab config
module. That file does not live in this repo, so the check is skipped when it is
absent.

## Where the geometry lives

- **CAD**: `assets/cartesian_hand/source/cartesian_hand_sim.step` (13.6 MB).
  The Fusion re-posed export of the hand. `step_to_obj.py` reads this file and
  rewrites the visual OBJs (10 bodies, each split by CAD material into
  `{body}__{material}.obj`).
- **OBJs**: `assets/cartesian_hand/meshes/`. The split visual meshes and the
  CoACD pieces under `meshes/collision_pieces/`. MuJoCo loads them via the
  `meshdir="meshes"` line in `cartesian_hand.xml`.

## Regenerate the meshes (extra dependencies)

The committed meshes and CoACD pieces stay as inputs because the upstream tools
are not deterministic across versions. Re-run the stages below only when the
geometry has actually changed.

```bash
# 1. Re-tessellate meshes/ from the Fusion STEP. Needs OCP (OpenCascade Python).
python assets/cartesian_hand/step_to_obj.py

# 2. Recompute source/body_inertia.json from the meshes. Needs trimesh + numpy.
python assets/cartesian_hand/mesh_inertia.py

# 3. Re-decompose collision geometry. Needs coacd + trimesh. Run only for bodies
#    whose geometry changed; refreshing unchanged bodies churns the diff for no
#    gain, since CoACD is not piece-for-piece reproducible.
python assets/cartesian_hand/decompose_collision.py base bridge

# 4. Regenerate the XML.
python assets/cartesian_hand/cartesian_hand_creation.py
```

Stage 1 needs `source/cartesian_hand_sim.step`. Without it, only stages 2 to 4
can run.

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

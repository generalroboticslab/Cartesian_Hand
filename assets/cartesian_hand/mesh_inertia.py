"""Recompute ``source/body_inertia.json`` from ``meshes/`` at a uniform density.

The third vendored stage of the ``mini_gripper`` CAD pipeline, after ``step_to_obj.py``. Upstream
this was ``common/pipeline/mesh_inertia.py``, invoked as (source/EXPORT_NOTES.md:239)

    mesh_inertia.py --mesh-dir meshes --density 1100 --out step/body_inertia.json

and, like the STEP stage, it was never vendored -- so a change to ``source/groups.json`` silently
left the physics describing the OLD body assignment. That is exactly the bug this file exists to
close: moving the Z servo from ``base`` to ``bridge`` moves ~35 g and its whole inertia
contribution between two bodies, and nothing else in the repo would have noticed.

Density is a PLACEHOLDER (1100 kg/m3 uniform, "CAD properties unreliable" per the file's own
``source`` field) -- no Fusion materials were assigned at export. It is read back from the file
being rewritten, so this script never silently redefines it. Once real materials land, this stage
is what consumes them.

The mesh is the UNION of a body's ``{body}__{material}.obj`` groups: the material split is a
rendering concern (see step_to_obj.py), and physics is the whole body.

Non-watertight bodies are expected and are not an error -- 6 of the 10 are open shells because OCC
emits what the B-rep has. trimesh's divergence-theorem integration handles them the same way the
upstream stage did; ``watertight`` is recorded per body so the caveat travels with the numbers.

    python asset/cartesian_hand/mesh_inertia.py                  # rewrite every body
    python asset/cartesian_hand/mesh_inertia.py base bridge      # rewrite only these
"""

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh

HERE = Path(__file__).resolve().parent
MESHES = HERE / "meshes"
OUT = HERE / "source" / "body_inertia.json"
ENC = "utf-8"


def body_mesh(body: str) -> trimesh.Trimesh:
    """A body's visual meshes concatenated into one soup, in millimetres."""
    parts = sorted(MESHES.glob(f"{body}__*.obj"))
    assert parts, f"no meshes/{body}__*.obj -- run step_to_obj.py first"
    return trimesh.util.concatenate(
        [trimesh.load(p, process=False, force="mesh") for p in parts])


def inertia_of(body: str, density_kg_m3: float) -> dict:
    """Mass properties in the MuJoCo <inertial> form: mass, CoM, principal moments + orientation.

    Returns both mm and m/SI flavours of each quantity because the builder reads the SI ones and a
    human cross-checking against Fusion reads the mm ones.
    """
    mesh = body_mesh(body)
    mesh.density = density_kg_m3 * 1e-9        # kg/m3 -> kg/mm3, matching the mesh's own units
    tensor = mesh.moment_inertia               # kg.mm2 about the CoM, in body (world-aligned) axes
    diag, rot = np.linalg.eigh(tensor)         # ascending eigenvalues; columns are principal axes
    if np.linalg.det(rot) < 0:                 # eigh may hand back a reflection; MuJoCo needs a
        rot[:, 0] *= -1                        # proper rotation to convert to a quaternion
    quat = trimesh.transformations.quaternion_from_matrix(
        np.pad(rot, ((0, 1), (0, 1))) + np.diag([0, 0, 0, 1]))
    com = np.asarray(mesh.center_mass)
    return {
        "com_m": (com * 1e-3).tolist(),
        "com_mm": com.tolist(),
        "diaginertia_kg_m2": (diag * 1e-6).tolist(),
        "diaginertia_kg_mm2": diag.tolist(),
        "inertia_tensor_kg_mm2": tensor.tolist(),
        "mass_kg": float(mesh.mass),
        "n_occurrences": 1,
        "quat_wxyz": quat.tolist(),
        "volume_cm3": float(mesh.volume) / 1000.0,
        "watertight": bool(mesh.is_watertight),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("bodies", nargs="*", help="bodies to rewrite (default: all in the file)")
    args = ap.parse_args()

    data = json.loads(OUT.read_text(encoding=ENC))
    bodies = args.bodies or sorted(data["bodies"])
    unknown = set(bodies) - set(data["bodies"])
    assert not unknown, f"not bodies in {OUT.name}: {sorted(unknown)}"

    for body in bodies:
        before = data["bodies"][body]["mass_kg"]
        data["bodies"][body] = inertia_of(body, data["density_kg_m3"])
        after = data["bodies"][body]["mass_kg"]
        print(f"{body:20s} mass {before * 1e3:7.2f} -> {after * 1e3:7.2f} g")

    OUT.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding=ENC, newline="\n")
    total = sum(b["mass_kg"] for b in data["bodies"].values())
    print(f"wrote {OUT.relative_to(HERE)}  (total mass {total:.4f} kg)")


if __name__ == "__main__":
    main()

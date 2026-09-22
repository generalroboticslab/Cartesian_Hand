"""Regenerate a body's CoACD convex-hull colliders in ``meshes/collision_pieces/``.

The fourth vendored stage of the ``mini_gripper`` CAD pipeline (source/EXPORT_NOTES.md:243).
Deliberately NOT wired into ``cartesian_hand_creation.py``: CoACD is not deterministic across
versions, so the hulls stay COMMITTED INPUTS and the builder's bit-identical guarantee holds. Run
this by hand, only for the bodies whose geometry actually changed, and expect a real diff.

    python asset/cartesian_hand/decompose_collision.py base bridge
    python asset/cartesian_hand/decompose_collision.py            # all 10 (rarely what you want)

WHY IT HAD TO BE VENDORED: these hulls are what contact runs on (visual geoms are group-2
``contype=0``), so a body-assignment change in ``source/groups.json`` that is not followed here
ships a physically wrong model -- the Z servo would keep colliding as part of ``base`` while
visually riding ``bridge``. Nothing else in the repo can detect that.

VERSION CAVEAT: the committed set was produced by the upstream pipeline; this runs whatever CoACD
is installed here (1.0.7 at the time of writing). Regenerating one body therefore mixes producers
within one directory. The parameters below are the library defaults, which is also the strongest
evidence the upstream stage used them: re-running an UNTOUCHED body (``left_down_rack``) at these
settings yields 47 pieces against the 50 committed -- close enough to confirm the settings, far
enough to confirm the hulls are not reproducible piece-for-piece. Do not use this script to
"refresh" bodies that did not change; the churn buys nothing and costs the provenance.

Input is the union of a body's ``{body}__{material}.obj`` visual meshes (the material split is a
rendering concern; collision is the whole body), in millimetres, matching the builder's scale.
"""

import argparse
from pathlib import Path

import coacd
import trimesh

from mesh_inertia import MESHES, body_mesh

PIECES = MESHES / "collision_pieces"
SEED = 0  # CoACD's own default; pinned so a re-run of the same body is at least self-consistent


def decompose(body: str) -> int:
    """Replace this body's pieces with a fresh decomposition. Returns the new piece count."""
    mesh = body_mesh(body)
    parts = coacd.run_coacd(coacd.Mesh(mesh.vertices, mesh.faces), seed=SEED)

    # Drop the old set first: the count changes, and a shrinking body would otherwise leave a
    # tail of stale high-index pieces that the builder happily picks up as real colliders.
    for old in PIECES.glob(f"{body}_[0-9][0-9].obj"):
        old.unlink()
    for i, (vertices, faces) in enumerate(parts):
        piece = trimesh.Trimesh(vertices, faces, process=False)
        (PIECES / f"{body}_{i:02d}.obj").write_text(
            trimesh.exchange.obj.export_obj(piece, include_normals=False))
    return len(parts)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("bodies", nargs="*", help="bodies to redo (default: every body with meshes)")
    args = ap.parse_args()

    bodies = args.bodies or sorted({p.name.split("__")[0] for p in MESHES.glob("*__*.obj")})
    for body in bodies:
        before = len(list(PIECES.glob(f"{body}_[0-9][0-9].obj")))
        print(f"{body:20s} {before:3d} -> {decompose(body):3d} pieces", flush=True)
    print("re-run cartesian_hand_creation.py to refresh the XMLs and the piece manifest")


if __name__ == "__main__":
    main()

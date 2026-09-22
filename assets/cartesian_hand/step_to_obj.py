"""Re-tessellate ``source/cartesian_hand_sim.step`` into the 10 per-body visual OBJs.

Replacement for the ``mini_gripper`` ``common/pipeline/step_to_obj.py`` + ``decimate.py`` pair,
which produced ``meshes/*.obj`` but was never vendored into this repo (see
``source/EXPORT_NOTES.md`` "Not done yet"). Only the STEP and ``source/groups.json`` were
committed, so the meshes were reproducible in principle but not in practice. This restores that.

Why re-tessellate instead of re-decimating: the pre-decimation OBJs were never committed. Six of
the ten meshes sit at exactly 10000 faces -- the old decimator's cap -- so their detail ceiling is
already spent. Raising it has to go back to the CAD.

Input / output
  in   source/cartesian_hand_sim.step   geometry + assembly tree + appearances, written from the
                                        Fusion root so occurrence paths are complete
       source/groups.json               occurrence-path prefix -> body, 178 leaves over 10 bodies
  out  meshes/<body>__<material>.obj    22 meshes over 10 bodies, Fusion WORLD frame, millimetres

Bodies are split by CAD APPEARANCE so each group can carry its own MJCF material and the hand can
be rendered in real steel/nylon rather than the DoF-identifying hues. All 178 leaves resolve to
exactly 3 appearances (CAD_COLORS); an unrecognised one is a hard error, never a default bucket.
The 4 fingers are pure nylon (1 mesh each), the other 6 bodies are steel+nylon+red (3 each).

The previous build's meshes are kept at ``meshes_lowres/`` -- not loaded by anything, but they are
what ``--verify`` diffs against, and the only record of what shipped before. ``meshes/`` also holds
``collision_pieces/``, which this script never touches.

Assumptions
  - The STEP carries Fusion's occurrence names verbatim in NEXT_ASSEMBLY_USAGE_OCCURRENCE, which
    XCAF surfaces as component label names. The mangled PRODUCT names (``125-3PINSEAT_1_ASM_7``)
    are not usable and are never read.
  - Vertices stay in the world frame at the export pose: that pose IS q=0, and the joint ranges in
    kinematics.json were rebased onto it. Emitting geometry in any other frame silently desyncs the
    model (EXPORT_NOTES.md, "If you re-pose again, every mesh ... has to be regenerated too").
    --verify guards this by diffing each body's bounding box against the existing mesh.
  - STEP length unit is mm and MESH_SCALE in the builder is 0.001, so no unit conversion happens
    here; --verify would catch a 1000x drift as a bbox mismatch.

Design decisions
  - Deflection is absolute (millimetres of chord error), not relative-to-size. Relative deflection
    scales the tolerance by each face's own size, so the small parts -- fingertips, bearings, screws
    -- come out coarser than the big shells; at the setting that gave a good base, the fingers
    tessellated to FEWER faces than the committed meshes they were meant to improve on. An absolute
    tolerance gives every part the same surface accuracy, and the big shells are then decimated back
    down, which is the right order: tessellate accurately, reduce deliberately.
  - Decimation is a cap, not a target -- meshes already under it (the fingers, native 2314 faces)
    pass through untouched, exactly as the old decimate.py left them.
  - Decimation uses MeshLab's CAD-tuned quadric edge collapse, not plain QEM. Plain quadric
    collapse has no feature constraint: it treats a 90-degree machined corner as ordinary error
    to spend, so corners get shaved off first. Measured on the base body at the same budget,
    fast_simplification pulled a solid's bounding box in by 2.7 mm (max surface deviation 5.0 mm,
    mean 1.2 mm) and missed the face target by 3x; planar-quadric collapse with normal and
    boundary preservation holds the box to 0.28 mm (max 1.8, mean 0.25) AND hits the target,
    which is why the whole set got SMALLER while getting more accurate. PART_BBOX_TOL_MM guards
    against a regression here -- the body-level --verify did NOT catch it, because a shrunk
    solid is usually interior and some other solid still defines the body's extremes.
  - Collision is NOT regenerated here. The CoACD hulls in meshes/collision_pieces/ are committed
    inputs (CoACD is not deterministic across versions) and are what contact actually runs on;
    visual geoms are group-2 contype=0, so changing VISUAL resolution cannot alter sim behaviour.
    Changing groups.json can: run decompose_collision.py for the affected bodies afterwards, and
    mesh_inertia.py for all of them.

Usage
    python asset/cartesian_hand/step_to_obj.py --verify     # defaults reproduce the committed set
"""

import argparse
import json
from pathlib import Path

import numpy as np
from OCP.BRep import BRep_Tool
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.STEPCAFControl import STEPCAFControl_Reader
from OCP.TCollection import TCollection_ExtendedString
from OCP.TDataStd import TDataStd_Name
from OCP.TDF import TDF_Label, TDF_LabelSequence
from OCP.TDocStd import TDocStd_Document
from OCP.TopAbs import TopAbs_FACE
from OCP.TopExp import TopExp_Explorer
from OCP.TopLoc import TopLoc_Location
from OCP.Quantity import Quantity_Color
from OCP.TopoDS import TopoDS
from OCP.XCAFDoc import XCAFDoc_ColorType, XCAFDoc_DocumentTool

HERE = Path(__file__).parent
STEP = HERE / "source" / "cartesian_hand_sim.step"
GROUPS = HERE / "source" / "groups.json"

# Relative linear deflection: chord error as a fraction of each face's bounding-box diagonal.
# 1e-3 is ~4x the face budget the old 10000-face cap allowed on the base.
LIN_DEFLECTION = 0.01  # mm of chord error; absolute, so small parts get the same surface accuracy
ANG_DEFLECTION = 0.2  # radians between adjacent facet normals
MIN_PART_FACES = 500  # decimation floor per solid; below this a part stops reading as its shape
CREASE_ANGLE_DEG = 30  # normals are averaged across edges gentler than this, split across sharper
# Per-solid bounding-box shrink a decimation pass is allowed to cause. A corner-eating decimator
# shows up here first and by a wide margin (2.7 mm measured), long before it moves a body's own
# bbox. Sized just above the 0.3 mm that CAD-tuned collapse costs at the shipped budget.
PART_BBOX_TOL_MM = 0.5

# The STEP's appearances, keyed by the LINEAR rgb OCC reports (the file stores sRGB:
# 0.627 -> 0.352, 0.247 -> 0.05). Rounded to 3 decimals because OCC's sRGB->linear conversion is
# not bit-stable across versions.
#
# MANY-TO-ONE ON PURPOSE. The 2026-09-06 re-export carries FOUR distinct greys that all name the
# same Fusion material, 'Nylon 12 (with Formlabs Fuse 1 3D Printer)' -- the designer tinted the
# printed parts by role (shells darkest, then racks, then fingertips lightest) to tell the
# mechanism apart in CAD. That is a CAD-viewing aid, not what the hardware looks like: one SLS
# print in one powder is one colour. This scheme's whole job is the hardware appearance (the DoF
# hue scheme in cartesian_hand_creation.MATERIALS already exists for telling parts apart), so all
# four fold into 'nylon'. Drop the extra keys if a figure ever wants the CAD tints back.
CAD_COLORS = {
    (0.352, 0.352, 0.352): "steel",  # 'Steel - Satin' -- servos, screws, bearings, rack stock
    (0.05, 0.05, 0.05): "nylon",     # Nylon 12, sRGB 0.247 -- racks, gears
    (0.08, 0.08, 0.08): "nylon",     # Nylon 12, sRGB 0.314 -- base_gripper / aux_gripper shells
    (0.188, 0.188, 0.188): "nylon",  # Nylon 12, sRGB 0.471 -- the four fingers
    (0.578, 0.578, 0.578): "nylon",  # Nylon 12, sRGB 0.784 -- the four fingertips
    # 'Opaque(255,0,0)': a Fusion appearance SWATCH with no material behind it, almost certainly a
    # leftover default on the servo middle case -- the least trustworthy entry here.
    (1.0, 0.0, 0.0): "red",          # WJ-WK00-0413-MIDDLECASE servo case
}


def _name(label: TDF_Label) -> str:
    attr = TDataStd_Name()
    if not label.FindAttribute(TDataStd_Name.GetID_s(), attr):
        return ""
    return attr.Get().ToExtString()


def material_of(color_tool, label, proto) -> str | None:
    """CAD_COLORS name for a leaf, or None if it carries no colour.

    Colour sits on the component label for an overridden instance and on the referred prototype
    otherwise, so both are tried; ColorSurf before ColorGen because a surface colour is the more
    specific of the two. Takes LABELS, not shapes -- XCAFDoc_ColorTool has no shape overload.
    """
    rgb = Quantity_Color()
    for lab in (label, proto):
        for kind in (XCAFDoc_ColorType.XCAFDoc_ColorSurf, XCAFDoc_ColorType.XCAFDoc_ColorGen):
            if color_tool.GetColor_s(lab, kind, rgb):
                key = (round(rgb.Red(), 3), round(rgb.Green(), 3), round(rgb.Blue(), 3))
                return CAD_COLORS.get(key, key)
    return None


def collect_leaves(shape_tool, label, path, loc, out, color_tool=None):
    """Walk the XCAF assembly, recording (occurrence_path, world-located shape, material) per leaf.

    Follows references into their prototype but keeps the *component's* name and location, which is
    what makes repeated parts (7 identical servos) resolve to distinct occurrence paths.
    ``material`` is None when ``color_tool`` is not supplied.
    """
    proto = label
    if shape_tool.IsReference_s(label):
        proto = TDF_Label()
        shape_tool.GetReferredShape_s(label, proto)
        loc = loc.Multiplied(shape_tool.GetLocation_s(label))

    children = TDF_LabelSequence()
    shape_tool.GetComponents_s(proto, children)
    if children.Length() == 0:
        material = material_of(color_tool, label, proto) if color_tool else None
        out.append((path, shape_tool.GetShape_s(proto).Moved(loc), material))
        return

    for i in range(1, children.Length() + 1):
        child = children.Value(i)
        ref = TDF_Label()
        # name lives on the referred prototype for Fusion exports; fall back to the component
        label_for_name = child
        if shape_tool.IsReference_s(child):
            shape_tool.GetReferredShape_s(child, ref)
        child_name = _name(child) or _name(ref)
        collect_leaves(shape_tool, child, f"{path}/{child_name}" if path else child_name, loc, out,
                       color_tool)


def tessellate(shape, lin_deflection):
    """Triangulate a shape and return (vertices Nx3, faces Mx3) in the shape's own placed frame."""
    BRepMesh_IncrementalMesh(shape, lin_deflection, False, ANG_DEFLECTION, True)
    verts, faces, offset = [], [], 0
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        face = TopoDS.Face_s(explorer.Current())
        loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation_s(face, loc)
        explorer.Next()
        if tri is None:
            continue
        trsf = loc.Transformation()
        n = tri.NbNodes()
        pts = np.empty((n, 3))
        for i in range(1, n + 1):
            p = tri.Node(i).Transformed(trsf)
            pts[i - 1] = (p.X(), p.Y(), p.Z())
        idx = np.empty((tri.NbTriangles(), 3), dtype=np.int64)
        reversed_face = face.Orientation() == face.Orientation().TopAbs_REVERSED
        for i in range(1, tri.NbTriangles() + 1):
            a, b, c = tri.Triangle(i).Get()
            idx[i - 1] = (a - 1, c - 1, b - 1) if reversed_face else (a - 1, b - 1, c - 1)
        verts.append(pts)
        faces.append(idx + offset)
        offset += n
    if not verts:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64)
    return np.vstack(verts), np.vstack(faces)


def decimate(vertices, faces, target):
    """Quadric edge collapse tuned for machined geometry. Returns (vertices, faces).

    ``planarquadric`` adds a quadric term per planar region so a flat machined face resists
    being eaten from its rim inward, ``preservenormal`` rejects collapses that flip a facet, and
    ``preserveboundary`` pins open edges (this hand's parts are not all watertight -- OCC emits
    what the B-rep has). Together these are what keep a 90-degree corner; without them the
    collapse spends its error budget on exactly those corners first, because a corner vertex has
    the largest quadric and removing it "buys" the most reduction per collapse.
    """
    import pymeshlab

    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(vertices, faces))
    ms.meshing_decimation_quadric_edge_collapse(
        targetfacenum=target, planarquadric=True, preservenormal=True,
        preserveboundary=True, preservetopology=True)
    out = ms.current_mesh()
    return out.vertex_matrix(), out.face_matrix()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default="meshes", help="output dir, relative to this script")
    ap.add_argument("--max-faces", type=int, default=40000,
                    help="per-body face target. Soft: MIN_PART_FACES can overshoot it on bodies "
                         "made of many small solids, and bodies already under it pass through")
    ap.add_argument("--deflection", type=float, default=LIN_DEFLECTION,
                    help="relative linear deflection (fraction of face bbox diagonal)")
    ap.add_argument("--verify", action="store_true",
                    help="diff each body's bbox against meshes/<body>.obj and fail on drift")
    args = ap.parse_args()

    out_dir = HERE / args.out_dir
    out_dir.mkdir(exist_ok=True)
    groups = json.loads(GROUPS.read_text())

    doc = TDocStd_Document(TCollection_ExtendedString("step"))
    reader = STEPCAFControl_Reader()
    reader.SetNameMode(True)
    reader.SetColorMode(True)
    reader.ReadFile(str(STEP))
    assert reader.Transfer(doc), f"STEP transfer failed: {STEP}"
    shape_tool = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())
    color_tool = XCAFDoc_DocumentTool.ColorTool_s(doc.Main())

    roots = TDF_LabelSequence()
    shape_tool.GetFreeShapes(roots)
    leaves = []
    for i in range(1, roots.Length() + 1):
        collect_leaves(shape_tool, roots.Value(i), "", TopLoc_Location(), leaves, color_tool)
    print(f"{len(leaves)} leaf occurrences")

    # groups.json lists occurrence-path prefixes; longest match wins so a specific servo path
    # overrides the assembly prefix that would otherwise swallow it.
    prefixes = sorted(((inc, body) for body, g in groups.items() for inc in g["include"]),
                      key=lambda kv: -len(kv[0]))
    # (body, material) buckets: a body's solids are split by CAD appearance so each group can carry
    # its own MJCF material. Material ordering within a body follows CAD_COLORS, not first-seen, so
    # the emitted file set does not depend on assembly traversal order.
    by_body, unassigned, uncolored = {b: {} for b in groups}, [], []
    for path, shape, material in leaves:
        for prefix, body in prefixes:
            if path == prefix or path.startswith(prefix + "/"):
                if material is None or not isinstance(material, str):
                    uncolored.append(f"{path} ({material})")
                by_body[body].setdefault(material, []).append(shape)
                break
        else:
            unassigned.append(path)
    assert not unassigned, f"{len(unassigned)} leaves matched no group, e.g. {unassigned[:5]}"
    # Loud, not a default bucket: an unrecognised appearance means the CAD gained a material and
    # the MJCF side (CAD_MATERIALS in cartesian_hand_creation.py) has to gain one too. Silently
    # folding it into 'steel' would ship a part painted as the wrong material forever.
    assert not uncolored, (f"{len(uncolored)} leaves have no known CAD colour -- add it to "
                           f"CAD_COLORS and to CAD_MATERIALS in the builder: {uncolored[:5]}")
    for body in by_body:
        by_body[body] = {m: by_body[body][m] for m in CAD_COLORS.values() if m in by_body[body]}

    import trimesh

    failures, files = [], []
    for body, shapes_by_material in by_body.items():
        # Weld before decimating, per part. OCC emits every B-rep face as its own vertex block, so
        # an unwelded solid is a pile of disconnected patches whose every edge is a free boundary
        # -- quadric collapse then eats each patch into slivers and the solid's outer shell
        # disappears, leaving only its internals visible. Welding first restores the shared edges
        # the collapse needs to work across. Welding per part rather than after the merge keeps
        # distinct solids from fusing where they touch.
        parts, part_material = [], []
        for material, shapes in shapes_by_material.items():
            for shape in shapes:
                part = trimesh.Trimesh(*tessellate(shape, args.deflection), process=False)
                part.merge_vertices()
                parts.append(part)
                part_material.append(material)
        raw = sum(len(p.faces) for p in parts)

        # Split the budget per solid by SURFACE AREA, never by face count and never by decimating
        # the merged soup against one global target. Face count tracks thread and knurl density,
        # not how much of the render a solid occupies: in `base`, base_gripper is 64.7% of the
        # area but only 8.0% of the faces, so a face-proportional split handed most of the budget
        # to servo internals and squeezed the one part that carries the silhouette 15:1. Area
        # halves the resulting corner damage (0.53 -> 0.28 mm) and the file is SMALLER, because
        # crease-split normals cost less on big flat shells than on thread-dense screws.
        # MIN_PART_FACES keeps a small solid recognisable when its area share rounds to nothing --
        # without a floor the MF85ZZ bearing setting a rack's inner Y extreme vanished entirely.
        # The budget spans the WHOLE BODY, not each material group: splitting the output by
        # material must not change how many faces the body gets, or a 3-way split would triple it.
        area = np.array([p.area for p in parts])
        budget = args.max_faces * area / area.sum() if area.sum() else np.zeros(len(parts))

        groups_out, worst_part_bbox = {}, 0.0
        for part, share, material in zip(parts, budget, part_material):
            v, f = np.asarray(part.vertices), np.asarray(part.faces)
            target = max(MIN_PART_FACES, int(share))
            if len(f) > target:
                before = np.array([v.min(axis=0), v.max(axis=0)])
                v, f = decimate(v, f, target)
                shrink = np.abs(np.array([v.min(axis=0), v.max(axis=0)]) - before).max()
                worst_part_bbox = max(worst_part_bbox, shrink)
            verts, faces = groups_out.setdefault(material, ([], []))
            faces.append(np.asarray(f) + sum(len(x) for x in verts))
            verts.append(v)

        # Ship explicit vertex normals, split at sharp edges. Without a `vn` block MuJoCo derives
        # one averaged normal per vertex, which smooths straight across the 90-degree edges of a
        # CAD part: flat panels pick up a shading gradient and the underlying triangulation shows
        # through as soft facets. That artifact is independent of resolution -- it looked WORSE at
        # 60k faces than the old 10k build, which shipped normals. Splitting at CREASE_ANGLE_DEG
        # keeps genuinely curved surfaces (servo barrels, fillets) smooth while restoring hard
        # edges. Costs ~2x the vertices, since a vertex on a crease is duplicated per smoothing
        # group; face count, and therefore collision and physics, is unchanged.
        # Filename ALWAYS carries the material, even for a single-material body, so the file set
        # alone records which appearance each mesh is -- the builder needs that to bind the CAD
        # material scheme, and a bare `{body}.obj` would lose it. The "one material keeps the old
        # name" rule that protects plot_workspace.py applies to the GEOM name, not the file, and
        # lives in the builder (visual_pieces).
        written, bounds, total_faces = [], [], 0
        for material, (verts, faces) in groups_out.items():
            mesh = trimesh.Trimesh(np.vstack(verts), np.vstack(faces), process=False)
            # Ship explicit vertex normals, split at sharp edges. Without a `vn` block MuJoCo
            # derives one averaged normal per vertex, which smooths straight across the 90-degree
            # edges of a CAD part: flat panels pick up a shading gradient and the underlying
            # triangulation shows through as soft facets. That artifact is independent of
            # resolution -- it looked WORSE at 60k faces than the old 10k build, which shipped
            # normals. Splitting at CREASE_ANGLE_DEG keeps genuinely curved surfaces (servo
            # barrels, fillets) smooth while restoring hard edges. Costs ~2x the vertices, since a
            # vertex on a crease is duplicated per smoothing group; face count, and therefore
            # collision and physics, is unchanged.
            mesh = trimesh.graph.smooth_shade(mesh, angle=np.radians(CREASE_ANGLE_DEG))
            stem = f"{body}__{material}"
            # digits=4 -> 0.1 um, still 100x finer than the 0.01 mm tessellation tolerance, and a
            # third smaller on disk than trimesh's default 8 digits of nanometre noise.
            (out_dir / f"{stem}.obj").write_text(
                trimesh.exchange.obj.export_obj(mesh, include_normals=True, digits=4))
            written.append(stem)
            bounds.append(np.asarray(mesh.bounds))
            total_faces += len(mesh.faces)

        if worst_part_bbox > PART_BBOX_TOL_MM:
            failures.append(f"{body}: decimation shrank a solid's bbox by {worst_part_bbox:.3f} mm "
                            f"(> {PART_BBOX_TOL_MM} mm) -- the decimator is eating corners")

        note = f"  part bbox {worst_part_bbox:.3f} mm"
        if args.verify:
            # meshes_lowres/ predates the material split and holds one unsplit mesh per body, so
            # the comparison is against the UNION of this body's groups, not each group alone.
            union = np.array([np.min([b[0] for b in bounds], axis=0),
                              np.max([b[1] for b in bounds], axis=0)])
            old = trimesh.load(HERE / "meshes_lowres" / f"{body}.obj", process=False)
            drift = np.abs(union - np.asarray(old.bounds)).max()
            note += f"  bbox drift {drift:.4f} mm vs lowres"
            # Guards the frame and the units, not the fidelity: a mm/m slip is 1000x and a wrong
            # export pose tens of mm, both far past this. Legitimate sub-mm differences are
            # expected and are usually the OLD mesh being wrong -- its 46x decimation pulled some
            # rack extremes in by up to 1 mm, which the re-tessellation restores.
            if drift > 2.0:
                failures.append(f"{body}: bbox drift {drift:.3f} mm vs meshes_lowres/{body}.obj")
                note += "  <-- FAIL"
        print(f"{body:20s} {len(parts):3d} solids  {raw:7d} -> {total_faces:6d} faces  "
              f"[{'+'.join(m for m in groups_out)}]{note}")
        files.extend(written)

    assert not failures, "geometry check failed:\n  " + "\n  ".join(failures)

    # Drop visual OBJs this run did not write. Splitting `base` into base__steel/nylon/red leaves
    # the old unsplit base.obj behind, and the builder's discovery treats a body having BOTH forms
    # as an error -- correctly, since it cannot tell which is current. Only top-level *.obj is
    # touched; collision_pieces/ is a subdirectory and the committed hulls are never generated
    # here. Nothing outside this script's own output is at risk.
    stale = sorted(p for p in out_dir.glob("*.obj") if p.stem not in files)
    for path in stale:
        path.unlink()
    if stale:
        print(f"removed {len(stale)} superseded: {', '.join(p.name for p in stale)}")
    print(f"\nwrote {len(files)} meshes for {len(by_body)} bodies to {out_dir}")


if __name__ == "__main__":
    main()

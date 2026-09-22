"""cartesian_hand — declarative MJCF builder (regenerates the committed XMLs).

Same contract as ``asset/duke_v2/parallel_gripper/parallel_gripper_creation.py`` and
``asset/duke_v2/humanoid_v21/humanoid_v21_creation_v3.py``: the XML is a pure BUILD ARTIFACT,
never hand-edited. Run this to regenerate it; a non-empty ``git diff`` afterwards means something
drifted (``CONTRIBUTING.md``: generated artifacts must regenerate bit-identically).

    python asset/cartesian_hand/cartesian_hand_creation.py            # regenerate all
    python asset/cartesian_hand/cartesian_hand_creation.py --check    # exit 1 on drift

``--check`` rebuilds in memory, byte-compares every generated file with what is committed,
COMPILES each XML with MuJoCo, and cross-checks the DoF order against the mjlab config module.

WHAT IT BUILDS (three variants of the SAME model, differing only in scene furniture):
  - ``cartesian_hand.xml``                                  graft-ready module: no floor /
                                                                  lights / <visual> / <statistic>
  - ``manipulation/gripper_final/cartesian_hand.xml``        module + <size memory> (the
                                                                  manipulation scenes add many
                                                                  object colliders)
  - ``manipulation/gripper_final/cartesian_hand_scene.xml``  + lights, headlight, a solid
                                                                  low floor: the default gripper
                                                                  the manipulation demos run on
  - ``meshes/collision_pieces/_piece_counts.json``                 per-body collider count
                                                                  manifest, derived from disk

INPUTS (all committed, all under this folder — nothing is fetched from the external
``mini_gripper`` pipeline that first produced this gripper):
  - ``source/kinematics.json``    body tree, 9 slide joints (axis / rebased range / damping /
                                  armature), 2 couplings, actuator gains + force limit,
                                  collision class, solver options.  See its ``_*_note`` keys.
  - ``source/body_inertia.json``  per-body mass / CoM / principal inertia + orientation quat
                                  (PLACEHOLDER Fusion default density — no materials assigned).
  - ``meshes/<body>__<material>.obj``           the 22 visual meshes (10 bodies split by CAD
                                  appearance), discovered from disk
  - ``meshes/collision_pieces/<body>_NN.obj``   the CoACD convex-hull colliders, discovered from
                                  disk (zero-padded, contiguous NN from 00). Regenerating the
                                  pieces themselves (CoACD) is deliberately NOT part of this
                                  script: CoACD is not deterministic across versions, which would
                                  break the bit-identical guarantee, so the .obj files are
                                  committed inputs — exactly as parallel_gripper treats its meshes.
  - the material colours, coupling solver params, DoF order and scene furniture below (small,
    hand-authored, and the only things not read from ``source/``).

Every input inconsistency the script can detect fails LOUDLY (assert) rather than emitting a
plausible-looking but wrong model: a parent typo, a joint on an unknown body, two joints on one
body, a coupling / actuator naming a missing joint, a body with no colliders, a stray or
un-padded piece file, a range that does not survive 6-significant-digit formatting.

WHY NOT ``asset/create/robot_builder`` (which the sibling creation scripts use)
  Its emitter hardcodes the collision default class to ``contype=1 conaffinity=1`` and has no
  ``condim`` at all (this hand needs ``conaffinity=0`` + condim / friction / solref so the jaws
  collide with objects but never with each other — expressible only via ~450 per-geom overrides),
  flattens mesh paths to the bare filename (breaking ``collision_pieces/``), and its ``Actuator``
  has no kp / kv / forcerange for a <position> servo. Emitting this model faithfully through it
  would mean changing the shared builder, so this script emits the MJCF directly with
  ElementTree, data-driven from the JSON above. If robot_builder later grows those knobs, porting
  this file onto it is mechanical.

DESIGN INVARIANTS (enforced by --check where possible)
  - Body / joint / actuator names are referenced by name from
    ``mj_envs/asset_zoo/cartesian_hand/cartesian_hand_constants.py`` and by every
    manipulation demo (``manipulation/demos/*.py``). Do not rename.
  - ``ACTUATOR_ORDER`` is the DoF / ctrl-index layout (``d.ctrl[i]``); the constants module's
    ``POSITION_ACTUATORS`` / ``HAND_DRIVEN_JOINT_RANGES`` must mirror it — ``--check`` (and every
    export) parses that module and fails if they disagree.
  - Every body sits at ``pos="0 0 0"``: the meshes are exported in the Fusion WORLD frame, so
    q=0 IS the export pose (both rack pairs parked shut) and the joint ranges are rebased onto it.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SRC = HERE / "source"
MESHES = HERE / "meshes"
PIECES = MESHES / "collision_pieces"
MANIFEST = PIECES / "_piece_counts.json"
MANIP = HERE / "manipulation" / "gripper_final"
CONSTANTS = REPO / "mj_envs" / "asset_zoo" / "cartesian_hand" / "cartesian_hand_constants.py"

MODEL_NAME = "cartesian_hand"
MESH_SCALE = "0.001 0.001 0.001"          # meshes are authored in mm
ENC = "utf-8"

# ── hand-authored inputs (everything else comes from source/) ──────────────────────────────────

# Part colours (one material per body). Order = emitted <material> order.
MATERIALS = {
    "base":              "0.85 0.40 0.40 1",
    "bridge":            "0.40 0.75 0.40 1",
    "left_down_rack":    "0.20 0.40 0.65 1",
    "left_down_finger":  "0.45 0.75 0.95 1",   # blue finger
    "right_down_rack":   "0.40 0.20 0.55 1",
    "right_down_finger": "0.80 0.50 0.95 1",   # purple finger
    "left_up_rack":      "0.65 0.50 0.10 1",
    "left_up_finger":    "0.95 0.85 0.30 1",   # yellow finger
    "right_up_rack":     "0.20 0.55 0.55 1",
    "right_up_finger":   "0.50 0.90 0.90 1",   # cyan finger
}

# Paint: the saturated hues above stay the source of truth -- they are what the actuator table
# below, the constants module and the README mean by "blue"/"purple"/"yellow"/"cyan" finger, and
# they are what keeps the ten parts apart at a glance. But at full saturation they render as toy
# plastic, so each is blended most of the way toward a dark charcoal before it reaches the XML.
# Same technique as argus_mini's foot paint (`_FOOT_TINT`), at a much higher tint share: that
# palette only needs six feet to differ against a dark chassis, this one needs ten parts to stay
# separable, and below ~0.4 the blue and cyan pairs start to collide at figure scale.
# Edit MATERIALS to change WHICH hue a part is; edit PAINT_TINT to change how loud all of them are.
PAINT_TINT = 0.55                    # fraction of the saturated hue kept; rest is PAINT_BASE
PAINT_BASE = (0.20, 0.21, 0.23)      # dark charcoal


def _paint(rgba: str) -> str:
    """Blend a saturated `r g b a` material string toward PAINT_BASE, preserving alpha."""
    *rgb, alpha = (float(v) for v in rgba.split())
    muted = (PAINT_TINT * c + (1.0 - PAINT_TINT) * b for c, b in zip(rgb, PAINT_BASE))
    return " ".join(f"{c:.3f}" for c in muted) + f" {alpha:g}"


# The SECOND colour scheme: what the hardware actually looks like, keyed by CAD appearance rather
# than by body. step_to_obj.py splits each body's mesh by these, so the two schemes differ only in
# which material each visual geom binds -- the geometry is shared. NOTHING binds these by default;
# the committed XMLs stay on the hues above so the viewer, the actuator table and every "purple
# finger" reference keep meaning what they say. Opt in at load time with
# `cartesian_hand_constants.apply_cad_materials(model)` (or `render_hand.py --cad-colors`).
#
# rgba is the sRGB the STEP stores. specular/shininess follow argus_mini's material classes
# (create_argus_mini.py:1118-1150), which is the repo's paper-render reference and the reason this
# scheme exists: it names materials by physical class, not part identity.
CAD_MATERIALS = {
    # 'Steel - Satin'. Metal: specular is above the 0.55 conductor threshold that
    # blender_view._METALLIC_SPECULAR keys on, so it renders as metal rather than grey plastic.
    # Matches BEARING_MATERIAL, whose shininess is deliberately low for a satin (not mirror) finish.
    "steel": dict(rgba="0.627 0.627 0.627 1", specular="0.60", shininess="0.42"),
    # 'Nylon 12 (with Formlabs Fuse 1 3D Printer)'. Matte SLS print: matches PLA_MATERIAL, whose
    # rgba (0.20 0.21 0.23) is what PAINT_BASE above already borrows.
    "nylon": dict(rgba="0.247 0.247 0.247 1", specular="0.05", shininess="0.05"),
    # 'Opaque(255,0,0)' on the servo middle case. UNLIKE the two above this is a Fusion APPEARANCE
    # swatch, not a material -- almost certainly a leftover default, and the least trustworthy
    # entry here. Kept at the literal CAD value so the scheme is honest to its source; tone it here
    # if a figure needs it, since saturated red is the toy-plastic look this scheme exists to avoid.
    "red": dict(rgba="1 0 0 1", specular="0.15", shininess="0.15"),
}

# Mimic-coupling solver params (right jaw follows left, ratio -1; see kinematics.json).
EQ_SOLREF = "0.005 1"
EQ_SOLIMP = "0.95 0.99 0.001"

# DoF / ctrl-index order of the 7 actuators (0-based). This IS the <actuator> element order.
ACTUATOR_ORDER = (
    "m_down_pair",          # 0  both lower jaws open/close (Y)
    "m_right_down_finger",  # 1  purple finger extend (X)
    "m_left_down_finger",   # 2  blue finger extend (X)
    "m_bridge_z",           # 3  whole upper stage lift (Z)
    "m_up_pair",            # 4  both upper jaws open/close (Y)
    "m_right_up_finger",    # 5  cyan finger extend (X)
    "m_left_up_finger",     # 6  yellow finger extend (X)
)

# Scene furniture for the manipulation variants (natural lighting copied from the old grasp_suite
# scene; a solid muted floor instead of its checker so the tools read clearly).
MANIP_MEMORY = "512M"
SCENE_HEADLIGHT = dict(diffuse="0.6 0.6 0.6", ambient="0.35 0.35 0.35", specular="0.2 0.2 0.2")
SCENE_GLOBAL = dict(azimuth="-30", elevation="-20")
SCENE_STATISTIC = dict(center="0.05 0 0.03", extent="0.30", meansize="0.01")
SCENE_FLOOR = dict(name="floor", type="plane", size="0 0 0.05", pos="0 0 -0.12",
                   rgba="0.28 0.30 0.34 1")
SCENE_LIGHTS = (
    dict(name="overhead", pos="0 0 0.6", dir="0 0 -1", directional="true", diffuse="0.85 0.85 0.85"),
    dict(name="fill", pos="0.4 -0.4 0.4", dir="-1 1 -1", directional="true", diffuse="0.45 0.45 0.45"),
)

HEADER = {
    "module": (
        "Cartesian Hand - FINAL version. GENERATED by cartesian_hand_creation.py from "
        "source/kinematics.json + source/body_inertia.json + meshes/ (mini_gripper "
        "cartesian_hand_sim build, Fusion re-posed export 2026-08-28, git 93c7ed8: double-helix "
        "racks + real servos, CoACD collision). Graft-ready: no scene furniture. DO NOT HAND-EDIT."
    ),
    "manip": (
        "cartesian_hand for the manipulation demos (module + <size memory>). GENERATED by "
        "../../cartesian_hand_creation.py. DO NOT HAND-EDIT."
    ),
    "scene": (
        "cartesian_hand + SCENE furniture (natural lights + solid low floor) for the "
        "manipulation-demo viewers; the committed module stays graft-ready. GENERATED by "
        "../../cartesian_hand_creation.py. DO NOT HAND-EDIT."
    ),
}

# Only the graft-ready module is shipped with this repo. The `manip` and `scene` variants live
# next to the manipulation demos in `legged_env_v2/asset/cartesian_hand/manipulation/` and are
# regenerated from there -- they are not part of the public release.
TARGETS = {
    "module": HERE / "cartesian_hand.xml",
}


# ── helpers ────────────────────────────────────────────────────────────────────────────────────

def g6(v) -> str:
    """MuJoCo-style compact number (6 significant digits, no trailing zeros)."""
    return f"{float(v):.6g}"


def vec(values) -> str:
    return " ".join(g6(v) for v in values)


def exact(values, what: str) -> str:
    """``vec`` for values that other code copies verbatim (joint ranges -> ctrlranges -> the
    constants module): refuse to silently round them."""
    for v in values:
        assert float(g6(v)) == float(v), f"{what}: {v!r} does not survive 6-significant-digit formatting"
    return vec(values)


def load_inputs():
    kin = json.loads((SRC / "kinematics.json").read_text(encoding=ENC))
    inertia = json.loads((SRC / "body_inertia.json").read_text(encoding=ENC))["bodies"]
    tree, joints, col = kin["tree"], kin["joints"], kin["collision"]

    # bodies: same set everywhere, one root, every parent resolvable
    assert set(tree) == set(inertia) == set(MATERIALS), (
        "body set disagrees between kinematics.json / body_inertia.json / MATERIALS")
    roots = [b for b, s in tree.items() if s["parent"] == "world"]
    assert len(roots) == 1, f"expected exactly one root body (parent 'world'), got {roots}"
    bad = {b: s["parent"] for b, s in tree.items() if s["parent"] != "world" and s["parent"] not in tree}
    assert not bad, f"bodies whose parent is not in the tree: {bad}"

    # joints: on known bodies, at most one per body, slide only
    unknown = {j: s["body"] for j, s in joints.items() if s["body"] not in tree}
    assert not unknown, f"joints on bodies not in the tree: {unknown}"
    bodies_with_joint = [s["body"] for s in joints.values()]
    dup = {b for b in bodies_with_joint if bodies_with_joint.count(b) > 1}
    assert not dup, f"more than one joint declared on body: {dup}"
    assert all(s["type"] == "slide" for s in joints.values()), "this builder only emits slide joints"

    # couplings + actuators must name existing joints; every actuator must be ordered
    for c in kin["couplings"]:
        assert c["joint1"] in joints and c["joint2"] in joints, f"coupling names a missing joint: {c}"
    for a, s in kin["actuators"].items():
        assert s["joint"] in joints, f"actuator {a} drives a missing joint {s['joint']!r}"
    assert tuple(sorted(ACTUATOR_ORDER)) == tuple(sorted(kin["actuators"])), (
        f"ACTUATOR_ORDER != kinematics.json actuators: {set(ACTUATOR_ORDER) ^ set(kin['actuators'])}")

    # collision class + exclusions
    assert col["enabled"], "kinematics.json collision.enabled must be true for the shipped model"
    assert isinstance(col["friction"], str) and isinstance(col["solref"], str), (
        "collision.friction / solref must be MJCF attribute strings")
    for pair in col.get("exclude_pairs", []):
        assert len(pair) == 2 and all(b in tree for b in pair), f"bad exclude pair {pair}"
    return kin, inertia


_PIECE_RX = re.compile(r"^(?P<body>.+)_(?P<idx>\d{2})\.obj$")


def collision_pieces(body: str) -> list[Path]:
    """This body's CoACD pieces from disk: zero-padded, contiguous from 00, at least one, and no
    stray files that merely start with the body name."""
    mine, stray = [], []
    for p in PIECES.iterdir():
        if not p.name.startswith(body + "_") or not p.name.endswith(".obj"):
            continue
        m = _PIECE_RX.match(p.name)
        if m and m.group("body") == body:
            mine.append((int(m.group("idx")), p))
        elif _PIECE_RX.match(p.name) is None:
            stray.append(p.name)          # e.g. left_up_finger_1.obj (un-padded)
    assert not stray, f"{body}: stray / un-padded collision piece files: {stray}"
    mine.sort()
    idx = [i for i, _ in mine]
    assert idx and idx == list(range(len(idx))), (
        f"{body}: collision pieces must be contiguous from 00 and non-empty, got {idx}")
    return [p for _, p in mine]


def visual_pieces(body: str) -> list[tuple[str, str, str]]:
    """This body's visual meshes from disk as (material, file stem, geom suffix).

    step_to_obj.py writes one `{body}__{material}.obj` per CAD appearance the body contains: 3 for
    the six multi-material bodies, 1 for each pure-nylon finger.

    The geom suffix is where the single-material rule lives: a body with ONE material gets the
    bare `{body}_visual` it has always had, a split body gets `{body}_visual_{material}`. This is
    load-bearing, not cosmetic -- plot_workspace.py selects finger geoms by the exact `_visual`
    suffix (and only ever consumes the four fingers), so keeping those four names intact is what
    lets the split land without touching that figure. Split bodies end in `_visual_steel` etc.,
    which fails that filter and is skipped.

    Ordering follows CAD_MATERIALS, not the filesystem, so the XML does not depend on directory
    iteration order.
    """
    found = {p.name[len(body) + 2:-4]: p.stem for p in MESHES.glob(f"{body}__*.obj")}
    assert found, f"missing visual mesh(es) meshes/{body}__<material>.obj -- run step_to_obj.py"
    unknown = set(found) - set(CAD_MATERIALS)
    assert not unknown, f"{body}: mesh materials {sorted(unknown)} are not in CAD_MATERIALS"
    solo = len(found) == 1
    return [(m, found[m], "visual" if solo else f"visual_{m}") for m in CAD_MATERIALS if m in found]


def children_of(tree: dict, parent: str) -> list[str]:
    return [b for b, spec in tree.items() if spec["parent"] == parent]


def constants_dof_layout() -> tuple[tuple[str, ...], dict[str, tuple[float, float]]]:
    """(POSITION_ACTUATORS, HAND_DRIVEN_JOINT_RANGES) read from the mjlab config module by AST —
    no mjlab import needed, so this stays runnable in a bare environment."""
    mod = ast.parse(CONSTANTS.read_text(encoding=ENC))
    acts, ranges = None, None
    for node in mod.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            target = node.targets[0] if isinstance(node, ast.Assign) else node.target
            if not isinstance(target, ast.Name):
                continue
            if target.id == "POSITION_ACTUATORS":
                acts = tuple(ast.literal_eval(node.value))
            elif target.id == "HAND_DRIVEN_JOINT_RANGES":
                ranges = dict(ast.literal_eval(node.value))
    assert acts is not None and ranges is not None, f"could not read the DoF layout from {CONSTANTS}"
    return acts, ranges


def check_constants_in_sync(kin: dict) -> None:
    # Optional: the constants module lives in mjlab's `mj_envs/asset_zoo/cartesian_hand/`, which is
    # not part of this repo. Skip the cross-check when the module is absent -- the actuator order is
    # self-checked against kinematics.json below either way, so the export is still safe.
    if not CONSTANTS.exists():
        return
    acts, ranges = constants_dof_layout()
    assert acts == ACTUATOR_ORDER, (
        f"DoF order drift: constants POSITION_ACTUATORS {acts} != creation ACTUATOR_ORDER {ACTUATOR_ORDER}")
    expect_joints = [kin["actuators"][a]["joint"] for a in ACTUATOR_ORDER]
    assert list(ranges) == expect_joints, (
        f"DoF order drift: constants HAND_DRIVEN_JOINT_RANGES keys {list(ranges)} != {expect_joints}")
    for j, rng in ranges.items():
        assert tuple(float(x) for x in rng) == tuple(float(x) for x in kin["joints"][j]["range"]), (
            f"range drift for {j}: constants {rng} != kinematics.json {kin['joints'][j]['range']}")


# ── build ──────────────────────────────────────────────────────────────────────────────────────

def build(variant: str) -> ET.Element:
    kin, inertia = load_inputs()
    check_constants_in_sync(kin)
    tree, joints, col = kin["tree"], kin["joints"], kin["collision"]
    joint_of = {spec["body"]: name for name, spec in joints.items()}

    root = ET.Element("mujoco", model=MODEL_NAME)
    ET.SubElement(root, "compiler", angle="radian", meshdir="meshes", autolimits="true")
    if variant in ("manip", "scene"):
        ET.SubElement(root, "size", memory=MANIP_MEMORY)
    opt = kin["options"]
    ET.SubElement(root, "option", timestep=g6(opt["timestep"]), iterations=str(int(opt["iterations"])),
                  integrator=opt["integrator"])
    if variant == "scene":
        vis = ET.SubElement(root, "visual")
        ET.SubElement(vis, "headlight", **SCENE_HEADLIGHT)
        ET.SubElement(vis, "global", **SCENE_GLOBAL)
        ET.SubElement(root, "statistic", **SCENE_STATISTIC)

    # defaults: visual = mesh, non-colliding, group 2; collision = group 3 with the hand's contact
    # class (contype 1 / conaffinity 0 -> touches objects and the world, never itself).
    default = ET.SubElement(root, "default")
    ET.SubElement(ET.SubElement(default, "default", {"class": "visual"}), "geom",
                  type="mesh", contype="0", conaffinity="0", group="2")
    ET.SubElement(ET.SubElement(default, "default", {"class": "collision"}), "geom",
                  contype=str(int(col["contype"])), conaffinity=str(int(col["conaffinity"])), group="3",
                  condim=str(int(col["condim"])), friction=col["friction"], solref=col["solref"])

    # assets: per body in tree order, visual mesh(es) then its collision pieces; then materials.
    asset = ET.SubElement(root, "asset")
    pieces, visuals = {}, {}
    for body in tree:
        visuals[body] = visual_pieces(body)
        for _material, stem, suffix in visuals[body]:
            ET.SubElement(asset, "mesh", name=f"{body}_{suffix}_mesh", file=f"{stem}.obj",
                          scale=MESH_SCALE)
        pieces[body] = collision_pieces(body)
        for i, p in enumerate(pieces[body]):
            ET.SubElement(asset, "mesh", name=f"{body}_col_{i:02d}_mesh",
                          file=f"collision_pieces/{p.name}", scale=MESH_SCALE)
    for body, rgba in MATERIALS.items():
        ET.SubElement(asset, "material", name=f"{body}_mat", rgba=_paint(rgba))
    # The CAD scheme's materials are DEFINED but bound by nothing -- apply_cad_materials() in the
    # constants module re-points geom_matid at load time. Defining them here is what makes that a
    # pure rebind instead of a second set of XMLs to keep in sync.
    for material, attrs in CAD_MATERIALS.items():
        ET.SubElement(asset, "material", name=f"{material}_mat", **attrs)

    # worldbody: (scene furniture) + the kinematic tree
    world = ET.SubElement(root, "worldbody")
    if variant == "scene":
        ET.SubElement(world, "geom", **SCENE_FLOOR)
        for light in SCENE_LIGHTS:
            ET.SubElement(world, "light", **light)

    emitted: list[str] = []

    def emit_body(parent_el: ET.Element, body: str) -> None:
        emitted.append(body)
        el = ET.SubElement(parent_el, "body", name=body, pos="0 0 0")
        ine = inertia[body]
        ET.SubElement(el, "inertial", mass=g6(ine["mass_kg"]), pos=vec(ine["com_m"]),
                      diaginertia=vec(ine["diaginertia_kg_m2"]), quat=vec(ine["quat_wxyz"]))
        if body in joint_of:
            jn = joint_of[body]; j = joints[jn]
            ET.SubElement(el, "joint", name=jn, type=j["type"], axis=vec(j["axis"]),
                          range=exact(j["range"], f"joint {jn} range"),
                          damping=g6(j["damping"]), armature=g6(j["armature"]))
        # One visual geom per CAD material. All of them bind the body's HUE material -- the CAD
        # scheme is a runtime rebind (see CAD_MATERIALS), so the committed XML looks exactly as it
        # did before the mesh split.
        for _material, _stem, suffix in visuals[body]:
            ET.SubElement(el, "geom", name=f"{body}_{suffix}", **{"class": "visual"},
                          mesh=f"{body}_{suffix}_mesh", material=f"{body}_mat")
        for i in range(len(pieces[body])):
            ET.SubElement(el, "geom", name=f"{body}_col_{i:02d}", **{"class": "collision"},
                          type="mesh", mesh=f"{body}_col_{i:02d}_mesh")
        for child in children_of(tree, body):
            emit_body(el, child)

    (root_body,) = children_of(tree, "world")
    emit_body(world, root_body)
    assert sorted(emitted) == sorted(tree), (
        f"bodies silently dropped from the tree walk: {sorted(set(tree) - set(emitted))}")

    # contact exclusions (none today; emitted if kinematics.json ever lists any)
    if col.get("exclude_pairs"):
        contact = ET.SubElement(root, "contact")
        for b1, b2 in col["exclude_pairs"]:
            ET.SubElement(contact, "exclude", body1=b1, body2=b2)

    # couplings
    eq = ET.SubElement(root, "equality")
    for c in kin["couplings"]:
        ET.SubElement(eq, "joint", joint1=c["joint1"], joint2=c["joint2"],
                      polycoef=vec(c["polycoef"]), solref=EQ_SOLREF, solimp=EQ_SOLIMP)

    # actuators, in DoF order; ctrlrange = the driven joint's (exactly-representable) range
    acts, adef = kin["actuators"], kin["actuator_defaults"]
    act = ET.SubElement(root, "actuator")
    for name in ACTUATOR_ORDER:
        a = acts[name]
        ET.SubElement(act, adef["type"], name=name, joint=a["joint"], ctrllimited="true",
                      ctrlrange=exact(joints[a["joint"]]["range"], f"actuator {name} ctrlrange"),
                      kp=g6(a["kp"]), kv=g6(a["kv"]),
                      forcelimited="true", forcerange=vec(adef["forcerange"]))
    return root


def render(variant: str) -> str:
    header = HEADER[variant]
    assert "--" not in header and not header.endswith("-"), "'--' is illegal inside an XML comment"
    root = build(variant)
    ET.indent(root, space="  ")
    return f"<!-- {header} -->\n" + ET.tostring(root, encoding="unicode") + "\n"


def render_manifest() -> str:
    kin, _ = load_inputs()
    counts = {body: len(collision_pieces(body)) for body in sorted(kin["tree"])}
    return json.dumps(counts, indent=2) + "\n"


def compile_check(variant: str, text: str) -> str:
    """Compile a rendered XML with MuJoCo (next to meshes/ so meshdir resolves); returns a summary."""
    import mujoco  # local import: --check stays importable without MuJoCo, compiling needs it
    tmp = HERE / f"_check_{variant}.xml"
    tmp.write_text(text, encoding=ENC, newline="\n")
    try:
        m = mujoco.MjModel.from_xml_path(str(tmp))
    finally:
        tmp.unlink()
    return f"nbody={m.nbody} njnt={m.njnt} nu={m.nu} neq={m.neq} ngeom={m.ngeom} nmesh={m.nmesh}"


# ── entry points ───────────────────────────────────────────────────────────────────────────────

def export() -> int:
    for variant, path in TARGETS.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(render(variant), encoding=ENC, newline="\n")
        print(f"wrote {path.relative_to(HERE)}")
    MANIFEST.write_text(render_manifest(), encoding=ENC, newline="\n")
    print(f"wrote {MANIFEST.relative_to(HERE)}")
    return 0


def check() -> int:
    """Rebuild in memory, compare with the committed files, compile each XML. 0 = all good."""
    drifted = []
    for variant, path in TARGETS.items():
        text = render(variant)
        ok = path.exists() and path.read_text(encoding=ENC) == text
        summary = compile_check(variant, text)
        print(f"  [{'OK  ' if ok else 'DRIFT'}] {path.relative_to(HERE)}   ({summary})")
        if not ok:
            drifted.append(variant)
    ok = MANIFEST.exists() and MANIFEST.read_text(encoding=ENC) == render_manifest()
    print(f"  [{'OK  ' if ok else 'DRIFT'}] {MANIFEST.relative_to(HERE)}")
    if not ok:
        drifted.append("manifest")
    if drifted:
        print(f"DRIFT: {drifted} differ from a fresh build -- regenerate with this script, never hand-edit.")
        return 1
    print("all generated files bit-identical to a fresh build; DoF order in sync with the constants module")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true",
                    help="do not write; exit 1 if any committed generated file differs from a fresh build")
    args = ap.parse_args()
    sys.exit(check() if args.check else export())


if __name__ == "__main__":
    main()

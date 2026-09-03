"""Where the sim model is, and how its joints line up with this hand's DOFs.

Split out of `studio.py` so the simulation backend does not have to import a
web server to find out which qpos address DOF 4 writes. `studio` re-exports
everything here, so nothing that already imported it from there had to change.

The mapping is the load-bearing part. A wrong entry does not raise -- it
produces a perfectly plausible animation of the wrong joint moving, which is
the failure mode that justifies a lookup table over an index.
"""
import os

import mujoco

from .config import HandConfig

# The sim asset is a sibling checkout, not an installed package. Relative to this
# file, so the pair of repos moves together and no username appears in a path.
LEGGED_ENV = os.environ.get(
    "LEGGED_ENV_ROOT",
    os.path.normpath(os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "..", "legged_env_v2")))

DEFAULT_MJCF = os.path.join(
    LEGGED_ENV, "asset", "cartesian_hand", "cartesian_hand.xml")

MM_PER_M = 1000.0

# Hardware DOF index -> the sim joints it drives.
#
# Index, not name: `config.LAYOUT`'s axis order (y,x,x,z,y,x,x) matches the MJCF
# actuator order one-for-one -- asserted against `model.actuator_trnid` in
# `test_dof_i_drives_actuator_i_s_joint`, not just read off a comment. That is
# also what lets `data.ctrl` be used as a goal vector directly. The left/right
# *labels* disagree between the two files; that is the open ❓, and `--swap` is
# how you test it.
#
# Two joints where the hardware has one servo: a rack pair is one servo driving
# both sides, which the model expresses as an `<equality><joint>` coupling.
# Equalities are applied by the constraint solver during `mj_step`, and the
# studio loop only ever calls `mj_forward`, so the follower has to be written
# explicitly or one side of each jaw sits at zero while the other moves.
DOF_TO_JOINTS = [
    ("left_down_y", "right_down_y"),      # 0  base pair
    ("right_down_finger_x",),             # 1  base finger
    ("left_down_finger_x",),              # 2  base finger
    ("bridge_z",),                        # 3  z stage
    ("left_up_y", "right_up_y"),          # 4  aux pair
    ("right_up_finger_x",),               # 5  aux finger
    ("left_up_finger_x",),                # 6  aux finger
]

SWAP_PAIRS = ((1, 2), (5, 6))             # what --swap exchanges


def dof_mapping(swap: bool = False) -> list[tuple[str, ...]]:
    """`DOF_TO_JOINTS` with `--swap` applied."""
    mapping = list(DOF_TO_JOINTS)
    if swap:
        for a, b in SWAP_PAIRS:
            mapping[a], mapping[b] = mapping[b], mapping[a]
    return mapping


def mjcf_path(xml: str | None = None) -> str:
    """The model to load: explicit, then $CARTESIAN_HAND_MJCF, then the sibling
    checkout."""
    return xml or os.environ.get("CARTESIAN_HAND_MJCF", DEFAULT_MJCF)


def qpos_addrs(model: mujoco.MjModel, swap: bool = False) -> list[list[int]]:
    """[[qpos index, ...], ...] per hardware DOF. Resolved once, outside the loop.

    By name, never by position: the model's qpos order is `bridge_z, left_up_y,
    left_up_finger_x, right_up_y, ...`, which is not the actuator order.
    Indexing positionally scrambles four joints and still produces a perfectly
    plausible-looking animation, which is the failure mode worth a lookup.
    """
    addrs = []
    for names in dof_mapping(swap):
        row = []
        for name in names:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise KeyError(f"model has no joint {name!r}")
            row.append(int(model.jnt_qposadr[jid]))
        addrs.append(row)
    return addrs


def narrow_ctrlrange(model: mujoco.MjModel, cfg: HandConfig) -> None:
    """Clamp the actuators to `config`'s travel instead of the model's.

    The MJCF is wider on every DOF -- 52.63mm against 50 on the jaws, 60 against
    55 on the fingers -- and the far end of each rail is open, so the extra
    stroke is not headroom, it is where a carriage leaves its slider. Narrowing
    the model means the UI cannot ask for a goal the loop would then silently
    refuse: the difference between a slider that feels stuck and one that stops
    where the hardware stops.
    """
    model.actuator_ctrlrange[:, 0] = cfg.lower().numpy() / MM_PER_M
    model.actuator_ctrlrange[:, 1] = cfg.upper().numpy() / MM_PER_M
    model.actuator_ctrllimited[:] = 1

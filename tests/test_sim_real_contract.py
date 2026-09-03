"""Do the sim and this hardware agree on what a policy's action means?

    python tests/test_sim_real_contract.py

A policy emits [-1, 1] per DOF and something turns that into millimetres. That
conversion is written twice -- here in `config.denormalize`, and in the sim as
mjlab's `raw * scale + offset` -- and nothing compared the two until this file.

They disagree today. So these tests **pin the disagreement** instead of
asserting it away: each reads both sides live and checks the gap against a
recorded table. Reconciling either side makes them fail, which is the point.
The tables below are a fact about two repositories, and moving one is a
deliberate act that should have to touch this file.

That is the whole job here. The root cause of both blockers is not a wrong
number, it is that two sources of truth existed with nothing between them.

Skips whole when the sim MJCF is absent: `config` is standalone by design --
the digital twin can import it without a compiler and the hand can run without
a sim checkout -- and that has to stay testable both ways.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from cartesian_hand import mjcf, studio
from cartesian_hand.config import HAND_1

MM_PER_M = mjcf.MM_PER_M

# Recorded 2026-09-01, both columns read live below.
#
# `(config max_mm, MJCF ctrlrange hi)` per DOF, in LAYOUT order. The two files
# are `cartesian_hand/config.py:STANDARD_TRAVEL` and
# `legged_env_v2/asset/cartesian_hand/cartesian_hand.xml`, whose numbers
# `mj_envs/.../cartesian_hand_constants.py:HAND_DRIVEN_JOINT_RANGES` copies
# again as the source for both the actuators and the action scale.
#
# **Direction is not symmetric.** config is narrower on every DOF and must stay
# that way. The far end of each rail is open by design (`config.py:104`), so the
# sim's extra stroke is not headroom -- it is where a carriage leaves its slider
# and the servo spins free. The sim comes down to these numbers; this table
# never comes up to the sim's.
#
# The jaw pairs stay at 50.0 rather than the measured 29.8: one stalled reading
# on one hand may under-report the stroke, and a sim workspace too small to hold
# the task is its own failure. See `config.py`'s UNRESOLVED note -- calipers, not
# this table, settle it.
EXPECTED_TRAVEL_MM = [
    (50.0, 52.6317),   # 0  base parallel actuation
    (55.0, 60.0),      # 1  base left finger
    (55.0, 60.0),      # 2  base right finger
    (50.0, 60.0),      # 3  vertical translation
    (50.0, 52.6317),   # 4  aux parallel actuation
    (55.0, 60.0),      # 5  aux left finger
    (55.0, 60.0),      # 6  aux right finger
]

# What the sim adds to a scaled action, per mjlab
# (`mjlab/envs/mdp/actions/actions.py:157`, `use_default_offset=True` by
# default): `processed = raw * scale + default_joint_pos`.
#
# **This one is an assumption, not a read.** `default_joint_pos` comes from the
# entity's init_state at graft time, and no task wires this hand up yet -- grep
# `mj_envs/tasks/` for "cartesian" and only a handoff markdown answers. It is
# taken to be the rest pose because every ctrlrange in the MJCF starts at 0 and
# the constants file calls that pose "0 = export/rest (bridge down, fingers
# retracted, jaws shut)". If a task later sets a different init_state, this
# assumption is what broke, and `test_action_zero_means_two_different_poses`
# is where it surfaces.
SIM_OFFSET_IS_CTRLRANGE_LOW = True

# mjlab `make_range_action_scale` with `DEFAULT_RANGE_ACTION_FRACTION = 0.5`:
# |action| = 1 travels half the range. `config.denormalize` uses the same half
# span, so scale already agrees -- the offset is the entire disagreement.
SIM_ACTION_FRACTION = 0.5


def _model():
    """The sim model, or None when the sibling checkout is absent.

    Loaded fresh and never passed to `mjcf.narrow_ctrlrange`, which mutates
    `actuator_ctrlrange` in place -- running it first would narrow the sim to
    config's travel and make every test here pass by erasing what it measures.
    """
    path = mjcf.mjcf_path()
    if not os.path.exists(path):
        return None
    import mujoco
    return mujoco.MjModel.from_xml_path(path)


def _sim_travel_mm(model):
    """[J] (lo, hi) ctrlrange in mm, in actuator order.

    Actuator order is DOF order: asserted in
    `test_studio.test_dof_i_drives_actuator_i_s_joint`, not assumed here.
    """
    return [(lo * MM_PER_M, hi * MM_PER_M) for lo, hi in model.actuator_ctrlrange]


# ── Travel: two tables, no comparison until now ───────────────────────────────

def test_sim_travel_agrees_with_recorded_table_and_stays_wider():
    """Both columns read live against the recorded table -- reconciling either
    side breaks the recorded values, deliberately rather than silently.

    Also the invariant that survives the reconciliation: the sim stays wider
    than `config` on every DOF, since `config`'s table is the rail-end guard and
    the sim being narrower would just be fine, but being *narrower without
    notice* is what we cannot have.
    """
    model = _model()
    if model is None:
        return
    cfg, sim = HAND_1, _sim_travel_mm(_model())
    assert len(sim) == cfg.n_dof, f"{len(sim)} actuators, {cfg.n_dof} DOFs"
    for dof, (want_cfg, want_sim) in enumerate(EXPECTED_TRAVEL_MM):
        assert abs(cfg.travel_mm[dof] - want_cfg) < 1e-6, (
            f"DOF {dof}: config travel moved, {cfg.travel_mm[dof]} not {want_cfg}; "
            f"re-record EXPECTED_TRAVEL_MM")
        assert abs(sim[dof][1] - want_sim) < 1e-4, (
            f"DOF {dof}: sim ctrlrange moved, {sim[dof][1]} not {want_sim}; "
            f"re-record EXPECTED_TRAVEL_MM")
        assert sim[dof][0] <= 0.0 + 1e-4, \
            f"DOF {dof}: sim floor {sim[dof][0]} above config"
        assert sim[dof][1] >= cfg.travel_mm[dof] - 1e-4, (
            f"DOF {dof}: sim ceiling {sim[dof][1]} below config "
            f"{cfg.travel_mm[dof]} -- if the sim was deliberately narrowed, "
            f"this test is the one to delete")


# ── The action map: the same number means two different poses ─────────────────

def test_action_zero_means_two_different_poses():
    """The blocker, as a number.

    sim:  mm = default_joint_pos + 0.5*(hi-lo)*a   -> a=0 lands on the rest pose
    real: mm = midpoint          + 0.5*(hi-lo)*a   -> a=0 lands mid-travel

    Same scale, different offset. A policy trained in sim to hold `a=0` shuts
    the jaws there and opens them 25mm here. Half the sim's action range is also
    dead -- negative actions clamp against a ctrlrange that starts at rest.

    Recorded as the gap rather than asserted away, for the same reason as the
    travel table: the fix is in the other repository.
    """
    model = _model()
    if model is None:
        return
    cfg = HAND_1
    zero = torch.zeros(cfg.n_dof)
    real_mm = cfg.denormalize(zero)
    for dof, (lo, hi) in enumerate(_sim_travel_mm(model)):
        sim_mm = lo if SIM_OFFSET_IS_CTRLRANGE_LOW else 0.5 * (lo + hi)
        gap = abs(real_mm[dof].item() - sim_mm)
        assert gap > 1.0, (
            f"DOF {dof}: the offsets now agree ({gap:.3f}mm apart). If the sim "
            f"set its action offset to mid-travel, delete this test -- the "
            f"blocker is fixed.")
        assert abs(gap - 0.5 * cfg.travel_mm[dof]) < 1e-4, (
            f"DOF {dof}: gap {gap:.3f}mm is neither zero nor half of config's "
            f"travel; one of the two conventions changed shape")


def test_scale_already_agrees_so_only_the_offset_is_wrong():
    """Narrows the fix. Both sides move half the travel per unit action, so the
    reconciliation is one offset, not a rewrite of either action space."""
    cfg = HAND_1
    span = cfg.denormalize(torch.ones(cfg.n_dof)) - cfg.denormalize(torch.zeros(cfg.n_dof))
    for dof in range(cfg.n_dof):
        assert abs(span[dof].item() - SIM_ACTION_FRACTION * cfg.travel_mm[dof]) < 1e-4


def test_config_uses_the_whole_action_range():
    """Why the sim should move and not this side.

    `denormalize` spends all of [-1, 1] inside travel. The sim's offset puts
    rest at one end, so everything below zero clamps and a policy learns on half
    its output. That asymmetry is the argument for which convention wins.
    """
    cfg = HAND_1
    assert torch.allclose(cfg.denormalize(-torch.ones(cfg.n_dof)), cfg.lower())
    assert torch.allclose(cfg.denormalize(torch.ones(cfg.n_dof)), cfg.upper())


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  ok    {t.__name__}")
        except BaseException as e:
            failed += 1
            print(f"  FAIL  {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

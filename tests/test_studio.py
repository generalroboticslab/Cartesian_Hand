"""Self-check for the live studio.

    python tests/test_studio.py

Most of this is the joint mapping, because that is where the mirror can lie
without looking wrong: a scrambled map still animates smoothly. The mujoco
tests skip if the model is not on this machine.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from cartesian_hand import mjcf, studio
from cartesian_hand.config import get_hand


def _model():
    path = os.environ.get("CARTESIAN_HAND_MJCF", mjcf.DEFAULT_MJCF)
    if not os.path.exists(path):
        return None
    import mujoco
    return mujoco.MjModel.from_xml_path(path)


# ── The joint mapping: where the mirror can silently lie ──────────────────────

def test_dof_mapping_covers_all_joints_and_matches_actuators():
    """Two checks in one: every named joint resolves and is unique (a missing or
    duplicated joint crashes or scrambles), and each DOF drives its actuator's
    joint. If wrong the mirror still animates, plausibly, with four joints
    scrambled.
    """
    model = _model()
    if model is None:
        return
    import mujoco
    names = [n for row in mjcf.DOF_TO_JOINTS for n in row]
    assert len(names) == model.njnt == 9, f"{len(names)} mapped, {model.njnt} in model"
    assert len(set(names)) == 9, "a joint is driven by two DOFs"
    for n in names:
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) >= 0, n
    for dof, names in enumerate(mjcf.DOF_TO_JOINTS):
        driven = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT,
                                   model.actuator_trnid[dof, 0])
        assert driven == names[0], f"DOF {dof}: actuator drives {driven}, mapped {names[0]}"


def test_the_rack_pairs_carry_their_coupled_follower():
    """Equalities are applied by the constraint solver during mj_step; the
    mirror only calls mj_forward. Without the follower in the map, one side of
    each jaw sits at zero while the other moves."""
    model = _model()
    if model is None:
        return
    import mujoco
    coupled = set()
    for i in range(model.neq):
        for obj in (model.eq_obj1id[i], model.eq_obj2id[i]):
            coupled.add(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, obj))
    pairs = {n for row in mjcf.DOF_TO_JOINTS if len(row) > 1 for n in row}
    assert coupled == pairs, f"model couples {coupled}, map pairs {pairs}"


def test_addresses_are_resolved_by_name_not_position():
    """qpos order is bridge_z, left_up_y, left_up_finger_x, right_up_y, ...  --
    not the actuator order. Positional indexing scrambles four joints and still
    looks fine on screen, so pin the one address that proves a lookup happened:
    DOF 3 is the z stage, which is actuator 3 but qpos 0."""
    model = _model()
    if model is None:
        return
    addrs = mjcf.qpos_addrs(model)
    assert addrs[3] == [0], f"bridge_z at qpos {addrs[3]}, expected [0]"
    assert sorted(a for row in addrs for a in row) == list(range(9))


def test_swap_exchanges_only_the_finger_dofs():
    """--swap is how the left/right disagreement gets tested against the real
    hand, so it must move the fingers and nothing else."""
    model = _model()
    if model is None:
        return
    plain, swapped = mjcf.qpos_addrs(model), mjcf.qpos_addrs(model, swap=True)
    for dof in (0, 3, 4):
        assert plain[dof] == swapped[dof], f"swap moved DOF {dof}"
    for a, b in mjcf.SWAP_PAIRS:
        assert swapped[a] == plain[b] and swapped[b] == plain[a]


# ── Control ───────────────────────────────────────────────────────────────────

def test_a_goal_actually_moves_the_hand():
    """The whole point of control mode. Mock integrates toward its target, so a
    goal that is never written shows up as a hand that never moves."""
    if _model() is None:
        return
    out = _capture(lambda: studio.live(
        mock=True, viewer=False, seconds=0.6,
        goal_mm=lambda mm, load: [20.0] * 7))
    last = [float(x) for x in out.strip().split("[")[-1].split("]")[0].split()]
    assert min(last) > 1.0, f"hand did not move toward the goal: {last}"


def test_a_goal_past_travel_is_clamped_to_config_not_the_model():
    """The MJCF is wider than `config` on every DOF, and the far end of each
    rail is open. Clamping to the model's ctrlrange would command a stroke that
    walks a carriage off its slider."""
    cfg = get_hand("hand_2")
    want = cfg.clamp(torch.full((cfg.n_dof,), 999.0))
    assert want.tolist() == cfg.upper().tolist()
    assert want[0].item() == 50.0, "jaw clamped to something other than config's 50mm"
    assert want[0].item() < 52.6317, "clamped to the MJCF travel, not config's"


def test_the_sliders_are_narrowed_to_hardware_travel():
    """A slider that can ask for 52.63mm when the loop will only pass 50 feels
    broken at the top of its range. Narrow the model instead."""
    model = _model()
    if model is None:
        return
    cfg = get_hand("hand_2")
    assert abs(model.actuator_ctrlrange[0, 1] - 0.0526317) < 1e-9, "fixture changed"
    mjcf.narrow_ctrlrange(model, cfg)
    for dof in range(cfg.n_dof):
        assert abs(model.actuator_ctrlrange[dof, 1] * 1000 - cfg.travel_mm[dof]) < 1e-6, dof
        assert model.actuator_ctrllimited[dof] == 1


def test_torque_is_enabled_only_after_the_present_position_is_written():
    """Enabling torque snaps a servo to whatever goal is still in its register.
    Written second, the hand lurches to a stale goal the moment the tool starts."""
    if _model() is None:
        return
    import cartesian_hand.studio as mod
    real, order = mod.open_driver, []

    class Spy:
        def __init__(self):
            self.inner = real("mock", mock=True)

        def __getattr__(self, k):
            return getattr(self.inner, k)

        def set_positions(self, *a, **kw):
            order.append("goal")
            return self.inner.set_positions(*a, **kw)

        def enable_torques(self, *a, **kw):
            order.append("torque")
            return self.inner.enable_torques(*a, **kw)

    mod.open_driver = lambda *a, **kw: Spy()
    try:
        # Named, not discovered: `open_driver` is patched, so discovery would
        # ask a mock-backed spy which hand it is and get both of them.
        studio.live(hand="hand_2", viewer=False, seconds=0.05)
    finally:
        mod.open_driver = real
    assert order[:2] == ["goal", "torque"], f"startup order was {order[:2]}"


def test_teach_mode_releases_torque_and_commands_nothing():
    """Teach is the read-only half. A stray goal would fight the hand being
    posed, and re-enabling torque would freeze it."""
    if _model() is None:
        return
    import cartesian_hand.studio as mod
    real, calls = mod.open_driver, []

    class Spy:
        def __init__(self):
            self.inner = real("mock", mock=True)

        def __getattr__(self, k):
            return getattr(self.inner, k)

        def set_positions(self, *a, **kw):
            calls.append("goal")
            return self.inner.set_positions(*a, **kw)

        def enable_torques(self, ids, on):
            calls.append(f"torque={bool(on)}")
            return self.inner.enable_torques(ids, on)

    mod.open_driver = lambda *a, **kw: Spy()
    try:
        _capture(lambda: studio.live(hand="hand_2", viewer=False, seconds=0.2,
                                     teach=True))
    finally:
        mod.open_driver = real
    assert calls == ["torque=False"], f"teach mode did more than release: {calls}"


# A dropped reply mid-run is pinned in `test_web_studio.py`, which drives the
# real loop. The version that lived here re-implemented the loop body inside the
# test and passed with `studio.live` mutated to fabricate a zero.


def test_a_silent_servo_at_startup_is_an_error_not_a_zero():
    """Startup defines the origin. A servo that did not answer would silently
    anchor the model at whatever counts happened to be in the buffer."""
    import cartesian_hand.studio as mod
    if _model() is None:
        return
    real = mod.open_driver

    class Silent:
        def __init__(self, *a, **kw):
            self.inner = real("mock", mock=True)

        def __getattr__(self, k):
            return getattr(self.inner, k)

        def read_all(self, ids):
            out = self.inner.read_all(ids)
            out[2] = None
            return out

    mod.open_driver = lambda *a, **kw: Silent()
    try:
        studio.live(hand="hand_2", viewer=False, seconds=0.1)
        raise AssertionError("a silent servo at startup was accepted")
    except RuntimeError as e:
        assert "did not answer" in str(e), e
    finally:
        mod.open_driver = real


def _capture(fn):
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        fn()
    return buf.getvalue()


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

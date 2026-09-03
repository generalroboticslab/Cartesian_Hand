"""Self-check for the viser page: render path and slider panel.

    python tests/test_web_studio.py

Headless: `ViserServer` accepts scene and GUI calls with no client attached.
What can lie silently here: wrong transform on a geom, colour read off the
wrong array, which slot a slider writes.
"""
import sys, os, socket
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mujoco
import numpy as np
import torch

from cartesian_hand import compose, mjcf, servo, studio, tasks
from cartesian_hand.config import get_hand
from cartesian_hand.studio import VISUAL_GROUP, WebStudio, wants_panel


def _free_port():
    """Each WebStudio binds a port, and these tests build several."""
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _model():
    if not os.path.exists(mjcf.mjcf_path()):
        return None
    return mujoco.MjModel.from_xml_path(mjcf.mjcf_path())


def _web(sliders=True):
    """A real page, or None if the model is not on this machine."""
    model = _model()
    if model is None:
        return None, None
    cfg = get_hand("hand_2")
    return model, WebStudio(model, cfg, [0.0] * cfg.n_dof,
                            port=_free_port(), sliders=sliders)


# ── Geometry: where this can lie and still render something ───────────────────

def test_only_the_ten_visual_geoms_are_built():
    """The model carries 458 geoms, 448 of them CoACD collision hulls. Building
    them would be 45x the meshes for a question this tool never asks."""
    model, web = _web()
    if web is None:
        return
    try:
        assert model.ngeom == 458, model.ngeom
        assert len(web._handles) == 10, len(web._handles)
        for geom, _ in web._handles:
            assert model.geom_group[geom] == VISUAL_GROUP
    finally:
        web.close()


def test_a_geom_is_placed_at_geom_xpos_with_no_further_transform():
    """`mesh_pos` reads up to 88mm on this model, which looks exactly like a
    transform someone forgot to apply. It is not -- mujoco bakes it into the
    stored vertices. mujoco's own visualizer is the reference: if it places a
    geom somewhere other than `geom_xpos`, `push` is wrong."""
    model = _model()
    if model is None:
        return
    data = mujoco.MjData(model)
    data.qpos[:] = 0.01
    mujoco.mj_forward(model, data)
    scene = mujoco.MjvScene(model, 500)
    mujoco.mjv_updateScene(model, data, mujoco.MjvOption(), None,
                           mujoco.MjvCamera(), mujoco.mjtCatBit.mjCAT_ALL, scene)
    assert scene.ngeom == 10, scene.ngeom
    for i in range(scene.ngeom):
        offset = np.linalg.norm(scene.geoms[i].pos - data.geom_xpos[scene.geoms[i].objid])
        # A micron, not zero: `mjvGeom.pos` is float32, so this reads ~1e-9 even
        # when the two agree exactly. The error being ruled out is 88mm.
        assert offset < 1e-6, f"geom {scene.geoms[i].objid} off by {offset}"


def test_pushing_a_pose_moves_the_handles_and_only_the_moved_bodies():
    model, web = _web()
    if web is None:
        return
    try:
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        web.push(data)
        before = {g: np.array(h.position) for g, h in web._handles}

        bridge = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "bridge_z")
        data.qpos[model.jnt_qposadr[bridge]] = 0.02
        mujoco.mj_forward(model, data)
        web.push(data)
        moved = {g for g, h in web._handles
                 if np.linalg.norm(np.array(h.position) - before[g]) > 1e-9}
        assert moved, "the z stage moved nothing"
        assert len(moved) < len(web._handles), "the fixed base moved too"
    finally:
        web.close()


def test_the_four_finger_colours_stay_distinct():
    """The left/right question is answered by looking at which colour moved, and
    by nothing else. `geom_rgba` would give every body the same default grey --
    it renders perfectly plausibly and destroys the measurement."""
    model = _model()
    if model is None:
        return
    fingers = ["left_up_finger", "right_up_finger",
               "left_down_finger", "right_down_finger"]
    colours = set()
    for name in fingers:
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        assert body >= 0, name
        for geom in range(model.ngeom):
            if model.geom_bodyid[geom] == body and model.geom_group[geom] == VISUAL_GROUP:
                colours.add(tuple(model.mat_rgba[model.geom_matid[geom]][:3]))
    assert len(colours) == 4, f"{len(colours)} distinct finger colours, want 4"


# ── Sliders ───────────────────────────────────────────────────────────────────

class _Event:
    """What viser hands a gui callback: the handle it fired for."""
    def __init__(self, value):
        self.target = type("T", (), {"value": value})()


def test_a_slider_writes_its_own_dof_and_no_other():
    """The per-slider closure is the whole mapping between a slider and a servo.
    Get it wrong and the page drives the wrong joint while looking fine."""
    _, web = _web()
    if web is None:
        return
    try:
        before = list(web.goal(None, None))
        web._setter(3)(_Event(12.5))
        after = web.goal(None, None)
        assert after[3] == 12.5, after
        for dof in range(len(after)):
            if dof != 3:
                assert after[dof] == before[dof], f"DOF {dof} moved too"
    finally:
        web.close()


def test_the_goal_seam_hands_back_the_live_list():
    """`live` calls `goal_mm(measured, load)` every tick and expects the newest
    slider values. A copy taken at construction would freeze the page."""
    _, web = _web()
    if web is None:
        return
    try:
        web._setter(0)(_Event(7.0))
        assert web.goal(None, None)[0] == 7.0
        web._setter(0)(_Event(9.0))
        assert web.goal(None, None)[0] == 9.0, "goal returned a stale copy"
    finally:
        web.close()


# ── Gains ─────────────────────────────────────────────────────────────────────

def test_gains_come_off_the_page_and_open_on_the_config():
    """The gains used to be hoisted out of the loop at startup, so finding the
    z stage's 300 meant editing `config` and restarting with the hand back at
    its startup pose."""
    _, web = _web()
    if web is None:
        return
    try:
        cfg = get_hand("hand_2")
        speed, acc, torque = web.gains()
        assert torque == cfg.gain_vector("torque_min_to_move").tolist(), torque
        # z is the loaded axis and the only DOF whose value differs, so this is
        # what says the page opened per DOF instead of on one broadcast number.
        assert torque[3] != torque[0], "z stage did not open on its own torque"
        assert speed == cfg.gain_vector("speed").tolist(), speed
        assert acc == cfg.gain_vector("acc").tolist(), acc

        # Speed is per DOF for the same reason torque is, and it is the fix for
        # z refusing to rise off its stop: `torque` is a CAP, and the effort a
        # servo develops follows position error, so a slow profile keeps the
        # setpoint close enough to the joint that the error never grows enough
        # to break z's static friction. One speed broadcast across the bus made
        # that untunable from the page.
        web._torques[3].value = 120
        web._speeds[3].value = 1200
        assert web.gains() == (speed[:3] + [1200] + speed[4:], acc,
                               torque[:3] + [120] + torque[4:])
    finally:
        web.close()


def test_the_gains_are_all_lists_because_a_mix_matches_no_overload():
    """This one reached hardware. `FtServo.set_positions` is nanobind-overloaded
    on three sequences or three scalars; `(int, int, list)` matches neither and
    raised `TypeError` on the first real run, after passing every offline test
    because the mock broadcast each gain independently."""
    _, web = _web()
    if web is None:
        return
    try:
        assert all(isinstance(g, list) and len(g) == 7 for g in web.gains())
        # And the mock now refuses the mix, so this cannot pass offline again.
        bus = servo.open_driver("mock", mock=True)
        try:
            bus.set_positions([7], [0], 300, 25, [50])
        except TypeError as e:
            assert "all sequences or all scalars" in str(e), e
            return
        finally:
            bus.close()
        assert False, "the mock accepted a gain mix hardware rejects"
    finally:
        web.close()


# ── Readout ───────────────────────────────────────────────────────────────────

def test_the_readout_carries_an_error_and_refreshes_at_10hz():
    """The tool's claim is that tracking error is visible live, and 0.01mm of it
    is a real number and zero pixels of geometry."""
    _, web = _web()
    if web is None:
        return
    try:
        mm = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        load, want = torch.full((7,), 42.0), mm + 0.25

        assert web._every == 5, web._every       # 50 Hz control, 10 Hz readout
        for _ in range(web._every - 1):
            web.report(mm, load, want)
            assert "+0.25" not in web._readout.content, "refreshed every tick"
        web.report(mm, load, want)
        text = web._readout.content
        assert "+0.25" in text, text
        row = next(l for l in text.splitlines() if l.startswith("2 "))
        assert "3.00" in row and "42" in row, row

        # teach commands nothing, so an error against it reads as one ignored.
        for _ in range(web._every):
            web.report(mm, load, None)
        assert "+0.25" not in web._readout.content, web._readout.content
    finally:
        web.close()


def test_the_readout_shows_a_pose_outside_config_travel():
    """`mj_forward` deliberately does not clamp to `jnt_range`, and the four
    travel tables disagree by up to 76%. A readout bounded by `config`'s travel
    would hide the one reading that settles which of them is right -- which is
    why the header carries numbers with no range at all, not bars."""
    _, web = _web()
    if web is None:
        return
    try:
        # 29.8 measured against 52.63 in the MJCF: both must be showable, and a
        # bar bounded by config's 0..50 could show neither past its end.
        for value in (-5.0, 29.8, 52.63):
            for _ in range(web._every):
                web.report(torch.full((7,), value), torch.zeros(7))
            assert f"{value:.2f}" in web._readout.content, (
                value, web._readout.content)
    finally:
        web.close()


def test_every_dof_gets_a_readout_row_and_the_commanding_sliders():
    """One readout row per DOF in the single block, plus a goal and a torque
    slider each when the sliders drive. Watch-only keeps the scene and the whole
    readout, drops the two that command.

    Column alignment is asserted because it is the point of the block: seven
    markdown components became seven lines of one, and lines only read as
    columns if the DOF with the longest name does not shift its own row.
    """
    _, web = _web()
    if web is None:
        return
    try:
        assert (len(web._goal), len(web._torques)) == (7, 7)
        for _ in range(web._every):
            web.report(torch.zeros(7), torch.zeros(7))
        rows = web._readout.content.splitlines()[1:-1]     # drop the ``` fences
        assert len(rows) == 8, rows                        # header + 7 DOFs
        assert "vertical z" in rows[4], rows[4]
        assert len({len(r) for r in rows}) == 1, [len(r) for r in rows]
    finally:
        web.close()

    _, watch = _web(sliders=False)
    if watch is None:
        return
    try:
        assert len(watch._handles) == 10, "watch dropped the scene"
        for _ in range(watch._every):
            watch.report(torch.zeros(7), torch.zeros(7))
        assert len(watch._readout.content.splitlines()) == 10
        assert watch._torques == []
    finally:
        watch.close()


# ── The bus going away ────────────────────────────────────────────────────────

def _gag(one=False):
    """A `goal_mm` that makes the mock's `read_all` stop answering.

    Through the goal seam because it is the one callable `live` invokes per tick
    with the loop already running: gagging before the call would trip the
    startup read instead, which is a different path with a different message.
    The mock deliberately never drops a reply on its own -- fabricating drops on
    a timer would make every other test using it intermittently flaky.
    """
    real = servo.MockServo.read_all
    gagged = ((lambda self, sids: [None] + real(self, sids)[1:]) if one else
              (lambda self, sids: [None] * len(sids)))

    def goal(measured_mm, load):
        servo.MockServo.read_all = gagged
        return measured_mm.tolist()

    goal.restore = lambda: setattr(servo.MockServo, "read_all", real)
    return goal


def test_a_dropped_reply_holds_the_last_position_instead_of_fabricating():
    """A fabricated position is a jump the hand never made; motion is the only
    thing this tool measures.

    Travel *before* the drop: gagging at t=0 hides the bug, since the startup
    datum is 0.0 mm and a fabricated 0.0 decodes to where the hand already was.
    Assertion is against the last good reading, not just constant.
    """
    real = servo.MockServo.read_all
    seen, tick = [], [0]
    DROP_AT = 6

    def goal(measured_mm, load):
        tick[0] += 1
        seen.append(measured_mm[0].item())
        if tick[0] == DROP_AT:            # servo 0 goes silent, the rest answer
            servo.MockServo.read_all = lambda s, ids: [None] + real(s, ids)[1:]
        return [20.0] * 7                 # drive away from the zero datum

    try:
        studio.live(mock=True, viewer=False, seconds=0.8, goal_mm=goal)
    finally:
        servo.MockServo.read_all = real

    before, after = seen[:DROP_AT], seen[DROP_AT + 1:]
    assert before[-1] > 1.0, f"hand never left the datum, drop is invisible: {before}"
    assert len(after) > 3, f"loop barely ran past the drop: {seen}"
    assert abs(after[0] - before[-1]) < 0.5, (
        f"fabricated a position: held {before[-1]:.2f} then reported {after[0]:.2f}")
    assert max(after) - min(after) < 1e-6, f"silent servo kept moving: {after[:6]}"


def test_a_bus_that_stops_answering_ends_the_run():
    """Holding the last good position is right for one frame and wrong forever:
    a hand holding still and an unplugged adapter produce identical counts, so
    without a counter the page shows a frozen pose while the loop keeps
    commanding a bus that is gone."""
    gag = _gag()
    try:
        studio.live(mock=True, viewer=False, seconds=30, goal_mm=gag)
    except RuntimeError as e:
        assert "no servo answered" in str(e), e
        return
    finally:
        gag.restore()
    assert False, "a dead bus ran to --seconds"


def test_one_silent_servo_is_a_dropped_frame_not_a_dead_bus():
    """A single DOF that stops answering while the rest do is a wiring fault
    worth watching, not a reason to drop whatever the hand is holding."""
    gag = _gag(one=True)
    try:
        studio.live(mock=True, viewer=False, seconds=0.8, goal_mm=gag)
    finally:
        gag.restore()


# ── Which mode gets sliders ───────────────────────────────────────────────────

def test_which_modes_get_sliders():
    """One truth table for one 4-argument boolean, rather than a test per row.

    The bug this pins actually shipped: `--studio` opened a view with no way to
    command, and it does not look broken -- `data.ctrl` holds the startup pose
    and the tracking error reads a perfect 0.00.
    """
    drives = lambda m, l: []
    for panel, windowed, teach, goal, expect, why in [
        (None,  True,  False, None,   True,  "windowed default is on"),
        (None,  True,  True,  None,   False, "teach sends no goals"),
        (None,  True,  False, drives, False, "caller is already the goal source"),
        (None,  False, False, None,   False, "--no-viewer: nothing is served"),
        (False, True,  False, None,   False, "--no-panel ignored"),
        (True,  False, False, None,   True,  "explicit panel ignored"),
    ]:
        assert wants_panel(panel, windowed, teach, goal) is expect, why


def test_panel_and_teach_are_refused_together():
    """teach sends no goals at all, so sliders under it do nothing -- a silently
    wasted session, which is worth two lines."""
    try:
        studio.live(mock=True, teach=True, panel=True, viewer=False, seconds=0.1)
    except ValueError as e:
        assert "teach" in str(e), e
        return
    assert False, "--panel --teach was accepted"


def test_the_timeline_edits_reorders_and_runs_what_it_shows():
    """The four things an append-only row list could not do, plus the run.

    Guards two failures that are invisible on the page: a cursor left past the
    end of a shortened list (the next click edits the wrong row), and `Run to
    row` submitting a stale `_preview` -- the module is cached by name, so
    without `compose._write`'s reload the second run of an edited timeline
    silently executes the first.
    """
    model, web = _web()
    if web is None:
        return
    try:
        c = web.composer
        for goal in (10.0, 20.0, 30.0):                  # insert, three times
            c._goal.value = goal
            c._insert_row(None)
        assert [r.goal for r in c.rows] == [10, 20, 30], c.rows
        assert c._cursor.value == "2", c._cursor.value   # follows the new row

        c._cursor.value = "0"                            # select loads the row
        assert c._goal.value == 10.0, c._goal.value
        c._goal.value = 11.0
        c._apply_row(None)                               # edit in place
        assert [r.goal for r in c.rows] == [11, 20, 30], c.rows

        c._move_down(None)
        assert [r.goal for r in c.rows] == [20, 11, 30], c.rows
        assert c._cursor.value == "1", c._cursor.value   # cursor follows
        c._move_up(None)
        c._move_up(None)                                 # no-op at the top
        assert [r.goal for r in c.rows] == [11, 20, 30], c.rows

        c._cursor.value = "2"
        c._delete_row(None)                              # delete off the end
        assert [r.goal for r in c.rows] == [11, 20], c.rows
        assert c._cursor.value == "1", c._cursor.value   # clamped, not stale

        c._run_to_row(None)                              # rows 0..1
        assert web.pending == compose.PREVIEW, web.pending
        assert len(tasks.tunables(compose.PREVIEW)) == 4, "goal+torque per row"

        c._cursor.value = "0"
        c._run_to_row(None)                              # rows 0..0
        assert len(tasks.tunables(compose.PREVIEW)) == 2, "stale _preview module"
    finally:
        (compose.tasks_dir() / f"{compose.PREVIEW}.py").unlink(missing_ok=True)
        web.close()


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

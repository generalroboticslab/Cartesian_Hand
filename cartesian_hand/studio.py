"""Live loop: command the hand from mujoco, show what the hand actually did.

    python -m cartesian_hand.studio                  # browser page, mm sliders in it
    python -m cartesian_hand.studio --studio False   # the mujoco viewer instead
    python -m cartesian_hand.studio --teach          # limp, pose it by hand
    python -m cartesian_hand.studio --no-viewer      # print mm, no window

Drag a millimetre slider and the hand follows. The model's pose is never the
slider -- it is what came back off the bus -- so the gap you see between where
you asked and where the model is *is* the tracking error, live.

    ctrl   what you asked for      slider -> mm -> counts -> set_positions
    qpos   what the hand did       read_all -> counts -> mm -> the model

Both directions are one packet, and they are the same two packets any control
loop already sends, so control costs nothing over watching. Measured on hand_2:
`read_all` 1.47 ms, `set_positions` 0.00 ms (a broadcast with no reply), so
write+read is 1.47 ms -- a 679 Hz ceiling, 7% of the bus at 50 Hz.

Safety
------
This moves the hand.

* At startup the goal is set to the *present* position and only then is torque
  enabled, because enabling torque snaps a servo to whatever goal is still in
  its register. Nothing moves until you drag something.
* Goals are clamped to `config`'s travel, not the model's `ctrlrange` -- the two
  disagree and the far end of each rail is open. See `narrow_ctrlrange`.
* **That clamp is relative, not absolute** when no calibration exists. Zero is
  the startup pose, not a hard stop, so starting mid-travel and driving a full
  stroke can still run a carriage off its rail. Press **Zero hand** on the page
  first, or start near the closed end.
* **Task goals are not clamped at all**, deliberately. See the command block.
* **torque armed**, the page's first row, drops torque and stops commanding.
  Unchecking it makes the hand limp -- a loaded z stage falls. Re-checking
  writes the present pose before energizing, so the hand holds where you left
  it rather than snapping back to the goal from before.

Tasks
-----
The loop is an executor. A page button builds a controller from `tasks/` and
ticks either its direct policy or its fixed motion program in place of the
sliders, still as the only writer on the bus. Nothing is threaded and nothing
takes the bus away, which is what lets the same task file run against MuJoCo
under `sim.run` with this module not imported at all.

Kinematic, not dynamic
----------------------
The loop writes `qpos` and calls `mj_forward`, never `mj_step`. Stepping
re-simulates, and gravity and contact would pull `qpos` off the values the hand
reported -- you would be watching mujoco's opinion, not the hand's. `mj_forward`
also does not clamp to `jnt_range`, so a real pose outside the model's declared
travel renders as it truly is. With the two travel tables disagreeing by up to
76%, that is the measurement, not a rendering bug.

`--swap` exchanges finger DOFs 1<->2 and 5<->6, which is how the left/right
disagreement between `config.LAYOUT` and the MJCF gets settled against the real
hand.

What lives here
---------------
`live` is the loop. `WebStudio` is both the renderer and the goal source: a
viser page serving the model's ten visual meshes between two windows. On the
right, viser's own panel: the arm switch, one readout table, seven goal
sliders, and a collapsed `tuning` folder of speed, acc and the seven torques.
On the left, a floating **tasks** window -- the task buttons, the tune folder
and the timeline -- which is one folder in that same panel, taken out of flow
by `TASK_MENU_CSS`. `--studio False` falls back to mujoco's passive viewer.

Two windows because the halves are used at different times and the left one is
used *while watching the right*: `Run to row` is author, run, read the error,
adjust. Tabs were tried first and put the timeline on the tab you cannot see.

Everything not touched during a run is one line of text or behind a fold: the
readout is a single markdown code block of seven aligned rows rather than
seven components, and tuning is nine rows found once and then left. That is
~11 rows against the ~35 the panel opened with, nothing removed.

The readout carries live millimetres, signed `err` and load as numbers rather
than the two read-only bar rows it started as -- 14 rows, none of them
controls. Numbers win on every count here: 0.01 mm of tracking error is a real
number and zero pixels of bar, the pose itself is already in the 3D view, and a
number never clamps, where a bar had to be built wider than `config`'s travel
to show a rail overrun (the tables disagree by up to 76%). The gains are live
because the z stage's torque was otherwise an edit to
`config.TORQUE_MIN_TO_MOVE` and a restart, which puts the hand back at its
startup pose to test a number that only means anything mid-move. All of it
costs 0.005 ms a tick, measured with a browser attached.

**One page, and no GL context in this process.** Two native windows were tried
first and the sliders never had a good home in them: `vtkSliderWidget`s are 3D
scene actors and look it, blitting VTK into a dearpygui texture costs 10.7 ms
and 16.4 MB a frame against 1.07 ms to draw (robot_studio measured it), and
docking two windows side by side is an arrangement, not a fix. viser has neither
problem -- the sliders are page furniture and the rendering is three.js in the
browser. Measured here with a real client attached: **0.60 ms a frame, 3% of a
20 ms tick**, against VTK's 0.79 ms, and a 0.035 s server against a 0.32 s scene
build.

Moving the GL out of this process also deletes a class of failure rather than
debugging it: no `__GL_SYNC_TO_VBLANK`, no interactor to pump, no context bound
to its creating thread, no "two on-screen VTK windows core-dumps", and none of
the extra threads those forced. The control loop is the only thread this module
starts; viser runs its own server.

Geometry comes **straight off the compiled `MjModel`** -- `mesh_vert` and
`mesh_face` per visual geom, posed each tick from `data.geom_xpos`/`geom_xmat`.
No importer, no genotype, no second scene representation to keep in step with
the model, and so no second copy of the DOF map that could drift from it. That
last point was a real trap and is now structurally gone: the render reads the
same `data` the loop just wrote, so it cannot disagree about which finger is
which. Verified rather than assumed -- mujoco's own visualizer places every
visual geom at `geom_xpos` to 1.3e-9 m, so `mesh_pos`/`mesh_quat` are a record
of what was baked into the vertices, not a transform still to apply.

Importing this module requires the `legged_env_v2` sibling checkout for the
default model path only. Set `LEGGED_ENV_ROOT` if it is not beside this repo.
"""

import os
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Any, Callable

import mujoco
import mujoco.viewer
import numpy as np
import torch
import tyro
import viser

from . import compose, motions
from .config import (AUX_FINGERS, AUX_JAW, BASE_FINGERS, BASE_JAW, CALIB_PATH,
                     DEFAULT_HAND, HANDS, LABELS, Z, HandConfig, get_hand,
                     identify, load_offsets, save_offsets)
from .mjcf import MM_PER_M, mjcf_path, narrow_ctrlrange, qpos_addrs
from .policy import Policy, PolicyRunner
from .servo import open_driver
from . import tasks


# 8081, not viser's default 8080: this box already serves something else on 8080,
# and viser does not fail on a busy port -- it takes the next free one and prints
# it, so a page opened at the port you asked for shows an unrelated application
# while the loop runs, invisibly, somewhere else.
VISER_PORT = 8081
SLIDER_STEP_MM = 0.1

# The readout text refreshes at this rate, not at the control rate. The meshes
# do move every tick -- they are 0.60 ms and motion at 10 Hz looks like motion at
# 10 Hz -- but seven rows of numbers changing 50 times a second is unreadable,
# and a number you cannot read is not a measurement. Same reason the headless
# `_printer` prints every fifth tick.
READOUT_HZ = 10

# Consecutive ticks in which *no* servo answered before the loop gives up. Half a
# second at 50 Hz.
#
# The loop holds the last good position on a dropped read, which is right for one
# bad frame and wrong forever: a hand holding still and an adapter that was
# unplugged produce identical `counts`, so without this the page shows a frozen
# pose while the loop keeps commanding a bus that is gone. `hand.py` counted the
# same thing before it was removed; nothing has since.
MAX_SILENT_STEPS = 25

# Loopback, not viser's own 0.0.0.0 default. These sliders energize servos and
# there is no auth in front of them, so the default must not be "anyone who can
# reach this box can move the hand". `--web-host 0.0.0.0` opts into the remote
# viewing this render path makes possible, as a decision rather than a surprise.
VISER_HOST = "0.0.0.0"

# Where a connecting browser's camera is put, in metres, and what it looks at.
#
# viser opens metres away from the origin, which suits a room and not a hand:
# this model's visual meshes span 0.27 m corner to corner, so the default view
# renders it about 25 px wide in a 1200 px viewport. A render path whose whole
# job is showing where the hand is opened with the hand invisible.
#
# 0.34 m out against a 0.27 m diagonal leaves margin at viser's default fov, and
# the aim point is the mesh bounding box centre rather than the world origin --
# the model is built around the base, so the origin is off to one edge of it.
CAMERA_POS = (0.22, -0.22, 0.16)
CAMERA_LOOK_AT = (0.016, 0.0, 0.003)

# The task menu is its own window on the left of the page, and this is the whole
# mechanism: viser serves exactly one control panel (a `ThemeConfigurationMessage`
# picks floating/collapsible/fixed for it and there is no message that opens a
# second), so the folder is built in that panel like everything else and then
# taken out of flow by CSS the page serves itself.
#
# Why bother: the panel had grown to the buttons, seven goal sliders, two
# collapsed folders and a timeline, and the two halves are used at different
# times -- a task owns every joint while it runs, so a goal slider is dead
# furniture beside it. Tabs were tried first and are worse for the one case that
# matters: watching the readout while a task runs means the timeline is on the
# tab you cannot see, and `Run to row` is an author-run-adjust loop against a
# number in the other half of the panel.
#
# **The selector is positional and will not survive a viser client rewrite.**
# `add_html` renders inside `<div dangerouslySetInnerHTML>` (viser's Html.tsx),
# so the marker sits five levels under the folder's `mantine-Paper-root`:
# Paper > Collapse > pad > pad > html-div > marker. `:has()` walks back up. To
# re-derive it after an upgrade, print the marker's ancestor chain -- that is
# what `tests/test_web_studio.py::test_the_task_menu_is_a_window_on_the_left`
# does in a real browser, and it fails rather than silently rendering the menu
# back inside the right-hand panel.
#
# Style notes: `--mantine-color-body` rather than white, so dark mode follows;
# z-index stays under viser's own notifications, which are also top-left and
# will briefly sit on top of this (they are dismissable, and the only one that
# fires unprompted is the software-WebGL warning).
TASK_MENU_CLASS = "cartesian-hand-task-menu"
TASK_MENU_CSS = f"""<style>
.mantine-Paper-root:has(> div > div > div > div > .{TASK_MENU_CLASS}) {{
  position: fixed; left: 1em; top: 1em; width: 24em;
  max-height: calc(100vh - 2em); overflow-y: auto; z-index: 5;
  background: var(--mantine-color-body);
  box-shadow: 0 2px 12px rgba(0, 0, 0, 0.25);
  border-radius: 0.5em; padding: 0.5em 0.7em;
}}
</style><div class="{TASK_MENU_CLASS}"></div>"""

# The MJCF puts visual geoms in group 2 and the 448 CoACD collision hulls in 0.
# Building only group 2 is why the scene is ten meshes and not 458; MuJoCo's own
# renderer excludes them the same way.
VISUAL_GROUP = 2

def wants_panel(panel: bool | None, windowed: bool, teach: bool,
                goal_mm: Callable | None) -> bool:
    """Whether the page's sliders drive the hand, when the caller did not say.

    On in every windowed mode: millimetres, labelled per DOF and bounded by
    `config`'s travel, against a mujoco Control panel that is metres, unlabelled
    and bounded by the model. No mode wants the worse one.

    Its own function because getting this wrong is silent -- `--studio` shipped
    with the panel opt-in and the result was not an error but a window watching
    a hand that could no longer be asked to do anything.

    Off in the cases where it could not work rather than would not help: `teach`
    sends no goals, a caller passing `goal_mm` already is the goal source, and
    not `windowed` means nothing is being served.
    """
    if panel is not None:
        return panel
    return windowed and not teach and goal_mm is None


def submit(name: str, cfg: HandConfig,
           measured_mm: torch.Tensor) -> motions.TaskRunner | PolicyRunner:
    """Build the named task's runner at the hand's current pose.

    The starting position is an argument rather than something the task reads,
    because a task has no way to read anything -- that is what makes it run on
    both backends. Zeroing needs it to set its first goals; the cap task does
    not, and asks for nothing.

    Direct tasks and fixed-program tasks share discovery and submission. Only
    their runner differs. `hold_torque` seeds untouched joints in the legacy
    runner; direct policies carry their own standing action.
    """
    controller = tasks.make(name, cfg, measured_mm[None, :])
    if isinstance(controller, Policy):
        return PolicyRunner(controller, cfg.control_hz)
    return motions.TaskRunner(
        controller,
        hold_torque=cfg.gain_vector("torque_min_to_move").to(torch.float32)[None, :])


def report_task_rate(name: str, cfg: HandConfig, started_at: float,
                     ticks: int) -> None:
    """Report when tick deadlines and room time materially disagree."""
    real_hz = ticks / max(time.time() - started_at, 1e-9)
    if abs(real_hz - cfg.control_hz) > 0.1 * cfg.control_hz:
        print(f"[{cfg.name}] {name}: control loop ran at {real_hz:.1f} Hz, "
              f"not the configured {cfg.control_hz:.0f} Hz -- every task "
              f"deadline is off by {cfg.control_hz / real_hz:.2f}x")


def finish(name: str, runner: motions.TaskRunner, cfg: HandConfig,
           zero: torch.Tensor, counts: torch.Tensor,
           web: "WebStudio | None") -> tuple[torch.Tensor, bool]:
    """Apply a finished task's result. Returns `(datum, ok)`.

    Only zeroing changes the datum, and changing it is the entire point of it:
    every millimetre reported before it ran was relative to the startup pose,
    and every one after is measured from a hard stop. The sliders are moved to
    match in the same breath -- a slider left where the old datum put it would
    command the difference between the two frames as a move, which is most of a
    stroke.

    `Config.sets_datum`, not `name == "zero"`: a retuned zeroing variant lives in its
    own file under its own name and still reports hard stops, and a name
    compare would silently drop its result on the floor.

    A task reports failure per env (`motions.Result`) rather than raising, so
    this is where the policy lives, and the policy is: any env failed, nothing is
    written. On hardware N is 1 so that is the only reachable rule anyway, and
    `save_offsets` is the one irreversible step in the loop -- a stop recorded
    from a DOF that timed out mid rail puts the origin at an arbitrary place and
    costs a bench cycle to notice.
    """
    value, ok, why = runner.result
    # Always, not only on failure. A task can return ok=True having spent a
    # minute on rows that never met their condition -- `Result.ok` is whatever
    # that task chose to check, and it is usually less than what it did. Those
    # rows are exactly the seconds an operator watches the hand stand still, so
    # they are worth a line whether or not the task calls the run a success.
    for line in runner.slow_rows(LABELS):
        print(f"[{cfg.name}] {name}: {line}")
    if not bool(ok.all()):
        print(f"[{cfg.name}] {name}: {why}")
        return zero, False
    if not tasks.sets_datum(name):
        print(f"[{cfg.name}] {name}: done, {value.tolist()}")
        return zero, True
    offsets = cfg.mm_to_counts(value, zero)[0]
    save_offsets(cfg.name, offsets.tolist())
    new_zero = offsets.to(torch.float32)
    if web:
        web.rebaseline(cfg.counts_to_mm(counts, new_zero).tolist())
    print(f"[{cfg.name}] zeroed: {offsets.tolist()} -> {CALIB_PATH}")
    return new_zero, True


def open_hand(hand: str | None = None, port: str | None = None,
              mock: bool = False) -> tuple[HandConfig, Any]:
    """The bus, and which hand is on it: `(cfg, bus)`.

    The bus is `Any`: the compiled `FtServo` and `servo.MockServo` share the
    five-method surface `servo.py` documents but no base class, and the real one
    is a nanobind extension with no stubs, so there is nothing honest to name.

    A named `hand` skips discovery -- the caller already knows, and probing
    would only be a chance to disagree with them.

    Otherwise every port `config.HANDS` lists is opened in turn and asked which
    servo IDs answer. That is what lets `python -m cartesian_hand.studio` with
    no flags open whichever hand is plugged in, instead of opening `hand_2`'s
    port and reporting seven silent servos when it was the other one. A path
    that does not exist is a hand that is not plugged in, so it is skipped, and
    a port that answers with IDs nobody claims is reported rather than used.

    Under `--mock` there is nothing to discover: the fake bus answers to every
    ID, so it would match both hands and every future one. It opens
    `DEFAULT_HAND` instead, which is what a fake bus has always been.
    """
    if hand:
        cfg = get_hand(hand)
        return (cfg.variant(port=port) if port else cfg), open_driver(
            port or cfg.port, mock=mock)
    if mock:
        cfg = get_hand(DEFAULT_HAND)
        return cfg, open_driver(cfg.port, mock=True)

    problems = []
    for p in [port] if port else sorted({h.port for h in HANDS.values()}):
        if not os.path.exists(p):
            problems.append(f"{p}: no such device (hand not plugged in?)")
            continue
        bus = open_driver(p, mock=False)
        try:
            cfg = identify(bus, p)
        except RuntimeError as e:
            bus.close()
            problems.append(str(e))
            continue
        print(f"[studio] found {cfg.name} on {p}")
        return cfg, bus
    raise RuntimeError("no hand found. Name one with --hand, or --port to "
                       "open a device this table does not list.\n  "
                       + "\n  ".join(problems))


def live(hand: str | None = None,
         xml: str | None = None,
         swap: bool = False,
         teach: bool = False,
         zero_raw: bool = False,
         calib: bool = True,
         port: str | None = None,
         mock: bool = False,
         viewer: bool = True,
         task: str | None = None,
         seconds: float | None = None,
         goal_mm: Annotated[Callable | None, tyro.conf.Suppress] = None,
         external_signal: Annotated[Callable | None, tyro.conf.Suppress] = None,
         policy: Annotated[Policy | None, tyro.conf.Suppress] = None,
         studio: bool | None = None,
         panel: bool | None = None,
         web_port: int = VISER_PORT,
         web_host: str = VISER_HOST) -> None:
    """Command a hand from mujoco and show what it actually did.

    Returns on `seconds`, on Ctrl-C, or when the mujoco viewer's window closes.
    A browser tab is not a window and closing one does **not** end the session:
    the page can be reopened, and dropping torque because someone closed a tab
    would drop whatever the hand is holding.

    Raises `RuntimeError` if no servo answers for `MAX_SILENT_STEPS` ticks. A
    single servo going quiet is a frame to skip; the whole bus going quiet is the
    adapter being unplugged, and it is indistinguishable from a hand holding
    perfectly still unless something counts.

    `teach` is a command, not an observation, and a limp z stage drops the aux
    gripper under its own weight -- support the hand before passing it.

    Under `--studio False` the goal source is mujoco's own Control panel, read
    straight off `data.ctrl`. It is metres and unlabelled, but `narrow_ctrlrange`
    has already bounded it to `config`'s travel, so it is safe -- just worse to
    read than the page's.

    Torque is not restored or released on exit. Whoever takes the hand back owns
    that: releasing would drop whatever is held, and leaving it enabled holds
    the last goal, which is where the hand already is.

    Args:
        hand: which entry in `config.HANDS` to open. Default None discovers it
            from the servo IDs that answer -- see `open_hand`.
        xml: MJCF path. Default $CARTESIAN_HAND_MJCF, then the sibling checkout.
        swap: exchange finger DOFs 1<->2 and 5<->6, to test the left/right
            disagreement between `config.LAYOUT` and the MJCF.
        teach: drop torque and send no goals; pose the hand by hand.
        zero_raw: use the servo count origin instead of the startup pose. Moves
            every reported millimetre for the rest of the run.
        calib: load saved offsets from `config.CALIB_PATH` and use them as
            zero. False falls back to the startup pose -- the clamp becomes
            relative, not absolute.
        port: serial port to open. Overrides the hand's configured one, and
            with no `hand` it is the only port discovery looks at.
        mock: fake bus, no serial port needed.
        viewer: False prints millimetres instead of opening any window.
        task: run this task at startup, as if its page button had been clicked
            -- a file stem under `tasks/`, e.g. `zero` or `cap`. How a hand gets
            zeroed with no browser open, and deliberately the same code path
            the button uses rather than a second one that could drift from it.
            An unknown name raises with the registry listed.
        seconds: stop after this long.
        goal_mm: callable `(measured_mm, load) -> [J] mm` replacing the sliders
            as the goal source. How the loop is exercised without a display, and
            the seam the control loop will command through. Not a CLI flag.
        policy: typed direct policy submitted once for this run. It receives the
            same batched tensor observation as simulation and owns goal, speed,
            and effort until its state reports done. Not a CLI flag.
        studio: serve the viser page; False for the mujoco passive viewer. None
            follows `viewer` -- the page whenever there is a view at all.
        panel: the page's millimetre sliders drive the hand, as `goal_mm`. False
            is watch-only. None means *decide*, and decides on -- see
            `wants_panel`. Moot under `--studio False`, where mujoco's own
            Control panel is the goal source.
        web_port: port for the viser page.
        web_host: interface to serve it on. Loopback by default -- these sliders
            move a real hand and nothing authenticates them. `0.0.0.0` to view
            or drive from another machine.
    """
    if policy is not None and (task is not None or goal_mm is not None or teach):
        raise ValueError("policy conflicts with task, goal_mm, and teach")
    if policy is not None and panel:
        raise ValueError("policy conflicts with a commanding slider panel")
    if studio is None:
        studio = viewer
    path = mjcf_path(xml)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"no mujoco model at {path}\n"
            "Pass --xml, or set CARTESIAN_HAND_MJCF to cartesian_hand.xml.")
    model = mujoco.MjModel.from_xml_path(path)
    # After the model, before anything is commanded: a missing MJCF should not
    # cost a bus open, and `narrow_ctrlrange` needs the config discovery returns.
    cfg, bus = open_hand(hand, port, mock)
    narrow_ctrlrange(model, cfg)
    data = mujoco.MjData(model)

    # Flattened once so the per-tick pose write is a single fancy-index
    # assignment rather than a nested Python loop over nine joints. `dof_of`
    # repeats a rack pair's DOF twice, which is how the follower gets written.
    addrs = qpos_addrs(model, swap)
    qpos_of = np.array([a for row in addrs for a in row])
    dof_of = np.array([dof for dof, row in enumerate(addrs) for _ in row])

    ids = cfg.servo_ids
    period = 1.0 / cfg.control_hz
    # Hoisted: gains are static, and .tolist() per tick is a conversion the bus
    # never needed repeated.
    speed = cfg.gain_vector("speed").tolist()
    acc = cfg.gain_vector("acc").tolist()
    torque = cfg.gain_vector("torque_min_to_move").tolist()

    first = bus.read_all(ids)
    if any(r is None for r in first):
        silent = [i for i, r in zip(ids, first) if r is None]
        raise RuntimeError(f"servos {silent} did not answer on {cfg.port}")
    counts = torch.tensor([r[0] for r in first], dtype=torch.float32)
    load = torch.zeros(cfg.n_dof, dtype=torch.float32)
    temperatures: list[int | None] = [None] * cfg.n_dof
    temperature_dof = 0
    next_temperature_at = time.monotonic()
    # Saved offsets make millimetres absolute; the startup pose is relative.
    # Order matters: `calib=False` skips the load, `zero_raw=True` discards it
    # even if loaded, and the fall-through to `counts.clone()` is the same
    # relative-zero behaviour a pre-zeroing hand has always had.
    saved = load_offsets(cfg.name, cfg.n_dof) if calib and not zero_raw else None
    if saved is not None:
        zero = torch.tensor(saved, dtype=torch.float32)
        print(f"[{cfg.name}] zeroed: {saved}")
    elif zero_raw:
        zero = torch.zeros(cfg.n_dof)
        print(f"[{cfg.name}] zero: raw count origin (uncalibrated)")
    else:
        zero = counts.clone()
        print(f"[{cfg.name}] zero: startup pose (no calibration -- "
              f"clamp is relative, not absolute)")

    if teach:
        bus.enable_torques(ids, False)
    else:
        # Order matters and is not interchangeable: enabling torque snaps a
        # servo to whatever goal is still in its register, which after a power
        # cycle or a previous run is anywhere at all. Write where it is now,
        # then energize, and the hand holds instead of lurching.
        bus.set_positions(ids, counts.int().tolist(), speed, acc, torque)
        bus.enable_torques(ids, True)
        # Park the sliders on the current pose for the same reason: a slider
        # sitting at 0 while the hand is at 30mm would command a full-stroke
        # move the moment the loop starts.
        data.ctrl[:] = cfg.counts_to_mm(counts, zero).numpy() / MM_PER_M

    # `viewer` is read before the studio block below flips it.
    sliders = (False if policy is not None else
               wants_panel(panel, viewer or studio, teach, goal_mm))
    if sliders and teach:
        raise ValueError("--panel and --teach conflict: teach sends no goals")

    # The page replaces the mujoco window rather than joining it: two views of
    # the same pose is two things to arrange and one of them is always behind
    # the other. Built after `zero` so its sliders open on the pose the hand is
    # already in.
    web = None
    if studio:
        web = WebStudio(model, cfg, cfg.counts_to_mm(counts, zero).tolist(),
                        port=web_port, host=web_host, sliders=sliders)
        if sliders:
            goal_mm = web.goal
        viewer = False

    # A seam, not a branch, for the same reason `goal_mm` is one: the gains are
    # static when nothing is serving them and live when the page is, and the loop
    # should not have to know which. Under `--panel False` this closes over the
    # hoisted lists and the page never touches them.
    gains = web.gains if (web and sliders) else lambda: (speed, acc, torque)

    if viewer:
        ctx, report = mujoco.viewer.launch_passive(model, data), None
    else:
        ctx, report = _NoViewer(), _printer()

    t0 = time.time()
    task_t0, task_ticks = t0, 0
    silent = 0
    armed = not teach    # tracks the bus, so the page's switch is an edge
    want = None          # stays None under teach, which commands nothing
    runner = PolicyRunner(policy, cfg.control_hz) if policy is not None else None
    running = type(policy).__name__ if policy is not None else None
    # A `--task` is submitted exactly like a click, and the loop returns when it
    # finishes: with no page open there is nothing left to watch, and a headless
    # zeroing run that kept ticking would hold the hand until Ctrl-C.
    pending, exit_when_done = task, task is not None or policy is not None
    try:
        with ctx as view:
            while view.is_running():
                tick = time.time()
                if seconds is not None and tick - t0 >= seconds:
                    break

                # ---- command ---------------------------------------------
                # First, so the servos are moving while the read that follows
                # is on the wire. Free: a sync-write is a broadcast with no
                # reply, so it returns as soon as it is queued.
                # The page's arm switch, applied on the edge only -- torque is a
                # register, and re-sending it every tick is a packet per tick for
                # a value that did not change. Re-arming repeats the startup
                # order (write present position, then energize) because a servo
                # snaps to whatever goal is still in its register, and the pose
                # was posed by hand while it was limp. A running task is dropped
                # rather than paused: it timed its rows against a hand that was
                # moving, and resuming would replay stale deadlines.
                arm = (not teach) and (web.armed if web else True)
                if arm != armed:
                    if arm:
                        bus.set_positions(ids, counts.int().tolist(),
                                          speed, acc, torque)
                    bus.enable_torques(ids, arm)
                    armed = arm
                    print(f"[{cfg.name}] torque {'on' if arm else 'off'}")
                    if arm and web:
                        web.rebaseline(cfg.counts_to_mm(counts, zero).tolist())
                    if not arm:
                        runner, pending = None, None

                if not armed:
                    want = None          # disarmed: nothing is commanded
                    if web:
                        web.pending = None   # a click while limp is not a queue
                else:
                    measured = cfg.counts_to_mm(counts, zero)

                    # A click submits a task; the loop executes it. Only when
                    # nothing is running -- a task owns every joint, so a
                    # second one would be two goals for one servo. A click that
                    # lands mid-task is dropped rather than queued: by the time
                    # the first finished the hand would be somewhere the second
                    # was not submitted from.
                    if web and web.pending:
                        if runner is None:
                            pending = web.pending
                        web.pending = None
                    if pending and runner is None:
                        runner, running = submit(pending, cfg, measured), pending
                        task_t0, task_ticks = time.time(), 0
                        print(f"[{cfg.name}] {running}: started")
                    pending = None
                    if runner is not None:
                        task_ticks += 1

                    step = None
                    direct_action = None
                    direct_done = False
                    if runner is not None:
                        try:
                            external = (external_signal(measured[None, :], load[None, :])
                                        if external_signal else None)
                            if isinstance(runner, PolicyRunner):
                                contact = (torch.as_tensor(external, dtype=torch.bool)
                                           if external is not None else None)
                                direct_action = runner.tick(
                                    measured[None, :], contact)
                                direct_done = runner.finished()
                            else:
                                step = runner.tick(measured[None, :], external)
                        except RuntimeError as e:
                            # NOT the path a failed task takes any more -- a DOF
                            # that never found its stop or a probe that closed on
                            # air comes back as `Result.ok` and is handled in
                            # `finish`. What is left here is the program itself
                            # being malformed. Either way: drop it and hand the
                            # joints back to the sliders, which still hold the
                            # pose from before the click. That releases a joint
                            # leaning on a stop, and it is the only direction
                            # that is safe without knowing why the task failed.
                            print(f"[{cfg.name}] {running}: {e}")
                            runner, failed = None, True
                        else:
                            failed = False
                        if (step is None and runner is not None
                                and isinstance(runner, motions.TaskRunner)):
                            report_task_rate(running, cfg, task_t0, task_ticks)
                            zero, done_ok = finish(running, runner, cfg, zero,
                                                   counts, web)
                            runner, failed = None, not done_ok
                        if step is None and runner is None and exit_when_done:
                            if failed:
                                raise RuntimeError(f"task {running!r} failed")
                            break

                    if direct_action is not None:
                        want = cfg.clamp(direct_action.goal_mm[0])
                        speed_now = (direct_action.max_speed_mm_s[0]
                                     * cfg.counts_per_mm).round().clamp_min(1)
                        speed_now = speed_now.to(torch.int32).tolist()
                        tq = (direct_action.effort_limit[0].clamp(0.0, 1.0)
                              * 1000).round().to(torch.int32).tolist()
                        _, acc_now, _ = gains()
                    elif step is not None:
                        # A task's goals are NOT clamped to the travel table.
                        # Zeroing deliberately asks for 120mm on a 55mm rail so
                        # that the hard stop, not the number, ends the move --
                        # and it works in whatever millimetre frame the hand is
                        # in, which before a calibration is relative to the
                        # startup pose and has no relation to the table. What
                        # keeps it safe is direction and provenance: the only
                        # unbounded ask goes toward the closed end, where a
                        # stop physically is, and every outward move a task
                        # makes is bounded at build time by `clamped_mm`. A
                        # slider is a human and gets clamped; a program is not.
                        want, tq = step[0][0], step[1][0].round().int().tolist()
                        # Per-DOF speed from the row where it asked for one, the
                        # page's gain where it did not (0). A seek and a park
                        # want different speeds on the same joint, and before
                        # `Motions` carried this the only place to say so was
                        # `config.speed`, which also governs the sliders.
                        speed_now, acc_now, _ = gains()
                        speed_now = [round(float(s) * cfg.counts_per_mm) if s > 0
                                     else g
                                     for s, g in zip(step[2][0].tolist(), speed_now)]
                    else:
                        want = (torch.as_tensor(goal_mm(measured, load),
                                                dtype=torch.float32)
                                if goal_mm else
                                torch.from_numpy(data.ctrl).float() * MM_PER_M)
                        # config's travel, not the model's: see narrow_ctrlrange.
                        want = cfg.clamp(want)
                        speed_now, acc_now, tq = gains()

                    bus.set_positions(ids, cfg.mm_to_counts(want, zero).tolist(),
                                      speed_now, acc_now, tq)
                    data.ctrl[:] = want.numpy() / MM_PER_M
                    if web is not None and (direct_action is not None or step is not None):
                        # A task's last commanded pose is the right next pose
                        # too: nothing in the loop has said otherwise, the
                        # sliders are still parked where the user clicked the
                        # button, and falling back to them here snaps the hand
                        # back to whatever they happened to be holding -- a
                        # ready pose nothing in the task commanded. One copy
                        # per tick a task runs, no websocket traffic unless
                        # rebaseline's `.value = ...` actually changes it.
                        web.rebaseline(want.tolist())
                    if direct_done and runner is not None:
                        report_task_rate(running, cfg, task_t0, task_ticks)
                        failed = runner.failed()
                        state = runner.state
                        if failed:
                            phase = getattr(state, "phase", None)
                            detail = ("" if phase is None else
                                      f" in phase(s) {phase.tolist()}")
                            print(f"[{cfg.name}] {running}: failed{detail}")
                        else:
                            print(f"[{cfg.name}] {running}: done")
                        runner = None
                        if exit_when_done:
                            if failed:
                                raise RuntimeError(f"task {running!r} failed")
                            break

                # ---- measure ---------------------------------------------
                reads = bus.read_all(ids)
                # All-silent is the bus going away; one silent servo is a frame
                # to skip. Distinguished here rather than per servo because a
                # single DOF that stops answering while the rest do is a wiring
                # fault worth watching, not a reason to drop the session.
                silent = silent + 1 if all(r is None for r in reads) else 0
                if silent >= MAX_SILENT_STEPS:
                    raise RuntimeError(
                        f"no servo answered in {silent} ticks "
                        f"({silent / cfg.control_hz:.1f}s) on {cfg.port}; "
                        f"check power and the serial cable")
                for i, r in enumerate(reads):
                    if r is None:
                        continue          # hold: never fabricate a position
                    counts[i], load[i] = r[0], r[2]

                # One extra bus transaction at 4 Hz, round-robin: each servo's
                # temperature refreshes about every 1.75 s without seven reads
                # delaying one 20 ms control tick. No catch-up burst after lag.
                now = time.monotonic()
                if now >= next_temperature_at:
                    value = bus.get_temperature(ids[temperature_dof])
                    if value is not None:
                        temperatures[temperature_dof] = int(value)
                    temperature_dof = (temperature_dof + 1) % cfg.n_dof
                    next_temperature_at = now + 0.25

                mm = cfg.counts_to_mm(counts, zero)
                data.qpos[qpos_of] = mm.numpy()[dof_of] / MM_PER_M
                # forward, not step: show the pose the hand reported instead of
                # re-simulating it, and let a pose outside the model's declared
                # travel render as it is rather than clamped into range.
                mujoco.mj_forward(model, data)
                view.sync()
                if web:
                    # 0.60 ms measured with a browser attached, so it runs
                    # inline at the control rate rather than on a thread of its
                    # own -- 3% of a 20 ms tick does not need decimating.
                    web.push(data)
                    web.report(mm, load, want, temperatures)
                if report:
                    report(mm, load, data.ctrl)
                time.sleep(max(0.0, period - (time.time() - tick)))
    finally:
        if web:
            web.close()
        bus.close()


# The joint groups a task actually addresses, as `config`'s own names. A task
# writes `set(BASE_FINGERS, ...)`, never `set([1, 2], ...)`, because the pairing
# is what carries the meaning -- so the composer offers the same vocabulary
# rather than seven checkboxes that would let a user build `[1, 4]`, a base
# finger and an aux jaw, which is a group no procedure on this hand wants.
DOF_GROUPS: dict[str, tuple[int, ...]] = {
    "base fingers": tuple(BASE_FINGERS),
    "aux fingers":  tuple(AUX_FINGERS),
    "all fingers":  tuple(BASE_FINGERS + AUX_FINGERS),
    "both jaws":    (BASE_JAW, AUX_JAW),
    "base jaw":     (BASE_JAW,),
    "aux jaw":      (AUX_JAW,),
    "z stage":      (Z,),
}
WHEN_LABELS = {
    "always": None,
    "source stop succeeded": "stopped_ok",
    "source stop failed": "stopped_failed",
    "source stop >= threshold": "stopped_at_ge",
    "source stop <= threshold": "stopped_at_le",
}
DOF_LABELS = tuple(f"{i} {label}" for i, label in enumerate(LABELS))
WHEN_KINDS = {kind: text for text, kind in WHEN_LABELS.items()}
NO_ROWS = "-"       # the row dropdown's only option while the timeline is empty


class Composer:
    """The panel that edits tasks, and the only part of the studio that writes one.

        Composer(server, cfg)      adds two folders to an existing viser page

    Two folders because there are two things a person does, and they need
    different controls:

    **tune** -- pick an existing task, drag its numbers, save the result as a
    variant. The sliders are built from `tasks.tunables`, so the range a slider
    offers is declared on the field beside the value it bounds; a panel with its
    own bounds table would be a second place to edit, and the stale one is
    whichever nobody reads.

    **compose** -- a history timeline, in the sense a CAD package means it. The
    rows are a list you scrub: select one and its parameters load into the
    editors, change them and **Apply**, or move it, delete it, insert after it.
    **Run to row** executes rows 0..cursor on the hand and leaves it there, which
    is the rollback marker -- you author the next row from the pose the previous
    ones actually produced, rather than from a number you predicted.

    **Load task** fills the timeline from the task the tune dropdown names, so
    an existing procedure can be opened and not only written. It reads the built
    program rather than the source (`compose.rows_from`), which is what lets it
    open a hand-written task whose goals are expressions rather than literals.
    The two halves of the panel divide by what they can reach: tune moves the
    numbers a task declared and keeps its structure, load reaches everything --
    a row's joints, stop rule, frame and predicate -- and keeps nothing, since
    what comes back is a flat program saved under a new name.

    Each row is one `motions.Step.set`. Runtime `When` predicates let each
    environment consume a source joint's previous stop outcome/position without
    changing program length or returning to Python. `stop="external"` accepts an
    explicit backend signal; no producer means timeout, never fabricated contact.

    **Run to row runs the file, not a preview of it.** It writes rows 0..cursor to
    `tasks/_preview.py` and submits that by name, so what the timeline executes
    is the same artifact **Save task file** produces -- one code path, and no
    class of bug that exists only in the thing you did not keep. Cost is that a
    partial run leaves `_preview` in the task list; it is rewritten every time and
    means nothing between runs.

    **Nothing here is in the execution path.** The panel writes `tasks/*.py` and
    then has no further part in it -- the file runs under `sim.run` at N=4096 and
    under `live` on the bus with this module never imported. That is why the
    generation lives in `compose.py`, which has no viser import: the dependency
    runs studio -> compose -> tasks and never back.

    A row is appended by a click on viser's server thread while the control loop
    reads nothing here, so no lock: `self.rows` is touched only by callbacks,
    which viser serializes, and the control loop's only shared slot is
    `WebStudio.pending`.
    """

    def __init__(self, server: "viser.ViserServer", cfg: HandConfig,
                 run: Callable[[str], None] | None = None,
                 pose: Callable[[], Sequence[float]] | None = None) -> None:
        """`run` submits a task name to the control loop; None disables Run to row.

        A callback rather than a reference to the page, so this panel keeps the
        one direction that matters: it can ask for a task to be started and
        cannot reach into the loop's state to find out what is running. The loop
        drops a submission that lands mid-task, which is the same rule the task
        buttons already live under.

        `pose` is where the hand is, in mm, for building a task that is about to
        be loaded into the timeline -- a callback for the same reason, and
        defaulting to zeros so this panel still works with no hand behind it.
        """
        self.server, self.cfg, self.run = server, cfg, run
        self.pose = pose or (lambda: [0.0] * cfg.n_dof)
        self.rows: list[compose.Row] = []
        self._knobs: dict[str, Any] = {}

        # The folder is kept, not just entered: viser tracks the container per
        # *thread*, so `_rebuild` -- which runs from a dropdown callback on
        # viser's server thread -- would otherwise drop its rebuilt "numbers"
        # folder at the top of the page instead of back inside this one.
        self._tune = server.gui.add_folder("tune a task", expand_by_default=False)
        with self._tune:
            self._task = server.gui.add_dropdown("task", tuple(tasks.names()))
            self._task.on_update(lambda _event: self._rebuild())
            self._knob_folder = server.gui.add_folder("numbers")
            # Every rebuild re-declares this folder, and viser orders by when a
            # component was declared, so without pinning the order the numbers
            # walk to the bottom of the folder the first time the task changes.
            self._knob_order = self._knob_folder.order
            server.gui.add_button("Run tuned").on_click(self._run_tuned)
            self._variant_name = server.gui.add_text("save as", "")
            server.gui.add_button("Save variant").on_click(self._save_variant)

        with server.gui.add_folder("TASK TIMELINE", expand_by_default=True):
            self._table = server.gui.add_markdown("")
            self._cursor = server.gui.add_dropdown("row", (NO_ROWS,))
            self._cursor.on_update(lambda _event: self._select())
            server.gui.add_button("Load task").on_click(self._load_task)
            server.gui.add_button("Run to row").on_click(self._run_to_row)
            self._group = server.gui.add_dropdown("joints", tuple(DOF_GROUPS))
            self._goal = server.gui.add_number("goal / distance mm", initial_value=0.0,
                                               step=0.5)
            self._torque = server.gui.add_number("torque", initial_value=50.0,
                                                 min=0, max=1000, step=10)
            self._stop = server.gui.add_dropdown("finish when", motions.STOPS)
            self._timeout = server.gui.add_number("timeout s", initial_value=6.0,
                                                  min=0.1, max=60.0, step=0.5)
            self._frame = server.gui.add_dropdown("goal frame", motions.FRAMES)
            self._when = server.gui.add_dropdown("run when", tuple(WHEN_LABELS))
            self._source = server.gui.add_dropdown("source joint", DOF_LABELS)
            self._threshold = server.gui.add_number(
                "condition threshold mm", initial_value=0.0, step=0.5)
            server.gui.add_button("Insert after row").on_click(self._insert_row)
            server.gui.add_button("Apply to row").on_click(self._apply_row)
            server.gui.add_button("Delete row").on_click(self._delete_row)
            server.gui.add_button("Move up").on_click(self._move_up)
            server.gui.add_button("Move down").on_click(self._move_down)
            self._program_name = server.gui.add_text("task file name", "")
            server.gui.add_button("Save task file").on_click(self._save_program)

        self._status = server.gui.add_markdown("")
        self._rebuild()
        self._show_rows()

    # ── tune ──────────────────────────────────────────────────────────────────

    def _rebuild(self) -> None:
        """Replace the number widgets with the selected task's own tunables.

        Removed and rebuilt rather than hidden: the set of fields differs per
        task (`zero` and `cap` expose different sets) and viser has no way to
        re-label a slider, so a pool of reused widgets would need its own mapping
        from slot to field -- one more place for the panel and task to disagree.

        Rebuilt *inside* `self._tune`: this runs on viser's server thread when
        the dropdown changes, and viser's container is per thread, so without
        re-entering the folder the numbers reappear at the top of the page.
        """
        self._knob_folder.remove()
        self._knobs = {}
        with self._tune, self.server.gui.add_folder(
                "numbers", order=self._knob_order) as folder:
            for field, (value, lo, hi) in tasks.tunables(self._task.value).items():
                self._knobs[field] = self.server.gui.add_slider(
                    field, lo, hi, (hi - lo) / 100.0, value)
            if not self._knobs:
                self.server.gui.add_markdown("_this task declares no tunables_")
        self._knob_folder = folder

    def _changed(self) -> dict[str, float]:
        """Only the knobs actually moved off their default.

        A variant naming every field would freeze the ones nobody touched: the
        parent's defaults are bench numbers that get retuned, and a copy that
        pinned them would silently stop tracking the hand it was measured on.
        """
        shipped = tasks.tunables(self._task.value)
        return {f: round(float(h.value), 4) for f, h in self._knobs.items()
                if abs(h.value - shipped[f][0]) > 1e-9}

    def _run_tuned(self, _event: object) -> None:
        """Run the selected task with the sliders where they are. The tune
        panel's verify step.

        Dragging a number and then having to name and keep a file before the
        hand would move it made every trial a permanent artifact: the point of
        a knob is the one that did *not* work, and `tasks/` filled with them.

        Runs the file, not the values -- same rule as **Run to row**. The
        changed numbers go to `tasks/_preview.py` as a variant and that name is
        submitted, so a tuned trial and the variant **Save variant** keeps are
        the same kind of module, built by the same writer. It shares `_preview`
        with the timeline because both are scratch: whichever ran last is what
        that file means, which is already what its docstring promises.

        Untouched sliders submit the task itself. `write_variant` refuses a
        variant with no changes -- correctly, it would be a copy under another
        name -- and the thing to run in that case is the parent.
        """
        if self.run is None:
            self._status.content = "**nothing to run**"
            return
        name = self._task.value
        if not (changed := self._changed()):
            self.run(name)
            self._status.content = f"running `{name}` with its shipped numbers"
            return
        path = self._write(lambda: compose.write_variant(
            compose.PREVIEW, name, changed,
            doc=f"{name}, tuned in the studio. Rewritten on every tuned run; "
                f"**Save variant** under its own name to keep it.",
            overwrite=True))
        if path is not None:
            self.run(compose.PREVIEW)
            self._status.content = (
                f"running `{name}` with {', '.join(sorted(changed))} changed "
                f"(`{path.name}`)")

    def _save_variant(self, _event: object) -> None:
        self._saved(self._write(lambda: compose.write_variant(
            self._variant_name.value.strip(), self._task.value, self._changed(),
            doc=f"{self._task.value}, retuned in the studio.")))

    def select(self, name: str) -> None:
        """Point both halves of the panel at `name`. For the task buttons.

        Clicking a task button and then finding the tune dropdown still on
        whatever it opened with is how a knob gets dragged against one task and
        run against another -- the two controls are a metre apart on the page
        and nothing tied them together. The button is the statement of intent,
        so it wins.

        `_rebuild` is called rather than left to the dropdown's own callback:
        viser does not promise one for a server-side write, and rebuilding
        twice costs one folder swap and is otherwise invisible.
        """
        if name in self._task.options:
            self._task.value = name
            self._rebuild()

    # ── compose ───────────────────────────────────────────────────────────────

    def _at(self) -> int | None:
        """The selected row's index, or None while the timeline is empty."""
        return int(self._cursor.value) if self.rows else None

    def _edited(self) -> compose.Row:
        """The row the editors currently describe."""
        kind = WHEN_LABELS[self._when.value]
        source = int(self._source.value.split()[0])
        return compose.Row(
            DOF_GROUPS[self._group.value], float(self._goal.value),
            float(self._torque.value), self._stop.value,
            float(self._timeout.value), self._frame.value,
            None if kind is None else
            motions.When(kind, source, float(self._threshold.value)))

    def _insert_row(self, _event: object) -> None:
        """Add the edited row after the cursor, and select it.

        After rather than at, so repeated inserts build the procedure downwards
        in the order they were clicked. Selecting the new row is what makes the
        next click continue from it instead of re-inserting at the old place.
        """
        at = self._at()
        self.rows.insert(0 if at is None else at + 1, self._edited())
        self._show_rows(0 if at is None else at + 1)

    def _apply_row(self, _event: object) -> None:
        """Overwrite the selected row with the editors. The edit half of the panel."""
        if (at := self._at()) is not None:
            self.rows[at] = self._edited()
            self._show_rows(at)

    def _delete_row(self, _event: object) -> None:
        if (at := self._at()) is not None:
            self.rows.pop(at)
            self._show_rows(at - 1)

    def _move_up(self, _event: object) -> None:
        self._move(-1)

    def _move_down(self, _event: object) -> None:
        self._move(+1)

    def _move(self, delta: int) -> None:
        """Swap the selected row with its neighbour and follow it.

        A no-op at the ends rather than a wrap: a row dragged off the top of a
        CAD timeline does not reappear at the bottom, and reordering the first
        and last steps of a procedure by one click is how a jaw ends up closing
        before the fingers have cleared it.
        """
        at = self._at()
        if at is None or not 0 <= at + delta < len(self.rows):
            return
        self.rows[at], self.rows[at + delta] = self.rows[at + delta], self.rows[at]
        self._show_rows(at + delta)

    def _show_rows(self, select: int | None = None) -> None:
        """Redraw the timeline and re-point the cursor. The only writer of both.

        `select` clamps rather than validates: every caller has just changed the
        list length under it (a delete at the end, an insert into an empty
        timeline), and clamping here is one rule instead of four callers each
        getting their own boundary right.
        """
        self._cursor.options = (tuple(str(i) for i in range(len(self.rows)))
                                or (NO_ROWS,))
        if self.rows:
            at = min(max(select if select is not None else int(self._cursor.value),
                         0), len(self.rows) - 1)
            self._cursor.value = str(at)
        else:
            self._table.content = "_no rows yet — set the editors below and **Insert after row**_"
            return

        lines = []
        for i, row in enumerate(self.rows):
            condition = "always" if row.when is None else (
                f"{row.when.kind}({DOF_LABELS[row.when.dof]}"
                + (f", {row.when.threshold_mm:g} mm"
                   if row.when.kind.startswith("stopped_at_") else "") + ")")
            lines.append(
                f"{'▶' if i == at else '　'} {i}. **{self._name(row.dofs)}** -> "
                f"{row.goal:g} mm ({row.frame}) at {row.torque:g}; finish: "
                f"**{row.stop}** or timeout {row.timeout_s:g}s; "
                f"run: **{condition}**")
        self._table.content = "\n".join(lines)
        self._select()

    def _select(self) -> None:
        """Load the selected row back into the editors.

        The half that makes the list editable rather than append-only: without
        it, changing one number of one row means deleting every row after it.
        Sets widget values only, so it cannot re-enter through `_cursor`.
        """
        if (at := self._at()) is None:
            return
        row = self.rows[at]
        if (group := next((n for n, g in DOF_GROUPS.items() if g == row.dofs),
                          None)) is not None:
            self._group.value = group
        self._goal.value, self._torque.value = row.goal, row.torque
        self._stop.value, self._timeout.value = row.stop, row.timeout_s
        self._frame.value = row.frame
        self._when.value = WHEN_KINDS[None if row.when is None else row.when.kind]
        if row.when is not None:
            self._source.value = DOF_LABELS[row.when.dof]
            self._threshold.value = row.when.threshold_mm

    def _load_task(self, _event: object) -> None:
        """Load the task selected above into the timeline, as editable rows.

        The half of "edit a task" that was missing: the timeline could author a
        procedure and could not open one, so every existing task was readable
        only as source and tunable only through the numbers it had thought to
        declare. Loading is what lets a step's joints, stop rule, frame or
        predicate be changed -- none of which is a `Config` field.

        Reads the *built* program, not the file (`compose.rows_from`), so what
        lands in the timeline is what would actually run. That is also why the
        task is built at the hand's current pose: a `frame="abs"` goal a task
        computes from where it started is only meaningful against that pose.

        The tune dropdown chooses the task for both halves rather than this
        folder carrying a second copy of it -- two dropdowns naming a task is
        two things to keep in step, and the stale one is whichever is scrolled
        off screen.

        Two things cannot load, and both are reported rather than half-done:
        a direct `Policy` has no steps to show, and only the *first* program of
        a multi-program task exists before the task has run -- the rest are
        built from measurements it has not taken yet.
        """
        name = self._task.value
        try:
            controller = tasks.make(
                name, self.cfg,
                torch.tensor([self.pose()], dtype=torch.float32))
        except (KeyError, ValueError, RuntimeError) as e:
            self._status.content = f"**{type(e).__name__}**: {e}"
            return
        if isinstance(controller, Policy):
            self._status.content = (
                f"`{name}` is a direct policy, not a step timeline -- there are "
                f"no rows to load. Tune its numbers in **tune a task**.")
            return
        try:
            self.rows = compose.rows_from(next(controller))
        finally:
            controller.close()
        self._show_rows(0)
        self._status.content = (
            f"loaded {len(self.rows)} row(s) from `{name}` (its first program, "
            f"built at the pose the hand is in now). **Save task file** under a "
            f"new name -- this is a copy, not `{name}` itself.")

    def _run_to_row(self, _event: object) -> None:
        """Execute rows 0..cursor on the hand. The timeline's rollback marker.

        Writes them as `tasks/_preview.py` and submits that by name, so the
        thing that runs is the file -- see the class docstring. Clobbers without
        asking because that file exists only to be clobbered.
        """
        if (at := self._at()) is None or self.run is None:
            self._status.content = "**nothing to run**"
            return
        path = self._write(lambda: compose.write_program(
            compose.PREVIEW, self.rows[:at + 1],
            doc=f"Studio timeline, rows 0..{at}. Rewritten on every run; save "
                f"the timeline under its own name to keep it.",
            overwrite=True))
        if path is not None:
            self.run(compose.PREVIEW)
            self._status.content = f"running rows 0..{at} (`{path.name}`)"

    def _save_program(self, _event: object) -> None:
        self._saved(self._write(lambda: compose.write_program(
            self._program_name.value.strip(), self.rows,
            doc=f"Composed in the studio: {len(self.rows)} rows.")))

    # ── both ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _name(dofs: tuple[int, ...]) -> str:
        return next((n for n, g in DOF_GROUPS.items() if g == dofs),
                    "+".join(LABELS[d] for d in dofs))

    def _write(self, make: Callable[[], Path]) -> Path | None:
        """Run a writer, reporting bad input instead of raising into viser.

        Catches only what the writers raise for bad input -- an empty name, a
        name that exists, a variant with nothing changed. A viser callback that
        raises loses the traceback into the server thread and the page just
        stops responding, which is indistinguishable from a hung control loop.
        """
        try:
            return make()
        except (ValueError, FileExistsError, OSError) as e:
            self._status.content = f"**{type(e).__name__}**: {e}"
            return None

    def _saved(self, path: Path | None) -> None:
        """Report where a kept file went, and offer it to the tune dropdown."""
        if path is None:
            return
        self._status.content = (
            f"wrote `{path.name}` -- run it with `--task {path.stem}`, here or "
            f"under `python -m cartesian_hand.sim`")
        self._task.options = tuple(tasks.names())


class WebStudio:
    """The model and its millimetre sliders in one browser page, via viser.

        web = WebStudio(model, cfg, start_mm)
        live(..., goal_mm=web.goal)
        web.push(data)                # per tick, after mj_forward
        web.report(mm, load, want)    # per tick; refreshes at READOUT_HZ

    Both halves of the tool in one place because viser makes them one page: ten
    meshes, seven millimetre sliders, the readout table that turns the 3D gap
    into a number, and the gains that let a torque be found without a restart.
    Served on
    `port`, no GL context in this process and no thread started here. Serves
    whether or not anything is connected, so a page can be opened, closed and
    reopened mid-session.

    Two windows, not one panel. viser's own panel on the right holds the arm
    switch, the readout, the goal sliders and tuning; everything task-shaped --
    the buttons and the whole `Composer` -- goes in a folder that
    `TASK_MENU_CSS` floats to the left of the page, so authoring a timeline and
    watching the hand are side by side rather than one behind the other.
    Watch-only (`sliders=False`) builds the left window and nothing else:
    authoring a task file never needed permission to move the hand.

    Geometry is read **straight off the compiled `MjModel`** -- there is no
    importer and no second scene representation. Each visual geom contributes
    its `mesh_vert`/`mesh_face` once, and `push` moves it with `geom_xpos` and
    `geom_xmat`. Two things make that sound wrong and are not:

    * **`mesh_pos`/`mesh_quat` are not applied here, and must not be.** They
      read up to 88 mm on this model, which looks like a missing transform. It
      is a record of what mujoco already baked into the stored vertices. Checked
      rather than argued: `mjv_updateScene` places every visual geom at
      `data.geom_xpos` to within 1.3e-9 m, which is `mjvGeom.pos` being float32.
    * **Colour comes from `mat_rgba[geom_matid]`, never `geom_rgba`.** All ten
      visual geoms have `mesh_texcoordnum 0`, so they are on the flat-material
      path; reading `geom_rgba` yields default grey for every body, which
      renders plausibly and destroys the left/right measurement the asset's ten
      deliberately distinct materials exist to answer.

    The rack-pair follower needs no special handling on this path, unlike every
    renderer before it: `push` reads whatever `qpos` the loop wrote, and the
    loop already writes both sides of a pair.
    """

    def __init__(self, model: mujoco.MjModel, cfg: HandConfig,
                 start_mm: Sequence[float], port: int = VISER_PORT,
                 host: str = VISER_HOST, sliders: bool = True) -> None:
        self.cfg = cfg
        # Set unconditionally, not inside the `sliders` branch: the loop reads
        # it every tick and a missing attribute would be an AttributeError in
        # the control path rather than a page with no buttons on it.
        self.pending: str | None = None
        # A list of floats, written by viser's server thread from a slider
        # callback and read by the control loop. One float into one slot is the
        # entire write, so there is nothing here a lock would make safer.
        #
        # Clamped to per-DOF travel before anything viser sees: viser's
        # `add_slider` asserts `min <= value <= max` and crashes the loop on a
        # start_mm that landed outside -- which happens the moment saved
        # offsets don't match the hand that's plugged in. The model still
        # shows the true pose from `counts`/`zero`; only the slider's
        # displayed value sits at the bound it has to sit at.
        lower_init, upper_init = cfg.lower().tolist(), cfg.upper().tolist()
        self._want = [min(max(float(v), lower_init[d]), upper_init[d])
                      for d, v in enumerate(start_mm)]
        self.server = viser.ViserServer(host=host, port=port, verbose=False)

        # Per client, not once on the server: viser has no global initial camera
        # and each tab gets its own, so a page opened later would otherwise open
        # at the default. Reopening a tab re-frames, which is what you want -- a
        # lost view is one refresh rather than a restart of the run.
        self.server.scene.set_up_direction("+z")

        @self.server.on_client_connect
        def _(client):
            client.camera.position = CAMERA_POS
            client.camera.look_at = CAMERA_LOOK_AT

        self._handles = []
        for geom in range(model.ngeom):
            if model.geom_group[geom] != VISUAL_GROUP:
                continue
            mesh = model.geom_dataid[geom]
            vert = model.mesh_vertadr[mesh]
            face = model.mesh_faceadr[mesh]
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom)
            self._handles.append((geom, self.server.scene.add_mesh_simple(
                f"/{name or geom}",
                model.mesh_vert[vert:vert + model.mesh_vertnum[mesh]].astype(np.float32),
                model.mesh_face[face:face + model.mesh_facenum[mesh]].astype(np.uint32),
                color=tuple(model.mat_rgba[model.geom_matid[geom]][:3]))))

        self._tick = 0
        self._every = max(1, round(cfg.control_hz / READOUT_HZ))

        # Panel layout, and viser lays out in call order, so this listing *is*
        # the layout -- for the right-hand panel. The task folder at the end of
        # this method is floated to the left of the window instead.
        #
        # First row on the page: the arm switch is the control you reach for
        # while something is going wrong, and hunting for it under a fold is the
        # time you do not have. Unchecked drops torque and stops commanding --
        # see `live`, which owns the transition.
        #
        # One markdown block for the readout, not one per DOF: seven separate
        # components carry seven components' worth of padding for seven lines of
        # text, and only the text was ever the measurement. Aligned in a code
        # block they also read as columns -- see `report`.
        self._armed = self.server.gui.add_checkbox("torque armed", True)
        self._readout = self.server.gui.add_markdown("")

        lower, upper = cfg.lower().tolist(), cfg.upper().tolist()
        self._names = [f"{dof} {label}" for dof, label in enumerate(LABELS)]
        self._speeds, self._torques, self._goal = [], [], []

        if sliders:
            for dof in range(cfg.n_dof):
                # Millimetres, and labelled with the DOF's job rather than an
                # actuator index, because reading `m_down_pair 0.030` off
                # mujoco's unlabelled metre sliders is how the left/right
                # question stayed open this long. Bounds are `config`'s travel,
                # not the model's -- see `narrow_ctrlrange`. It opens on
                # `start_mm`, the pose the hand is already in, because a slider
                # sitting at 0 while the hand is at 30mm commands a full-stroke
                # move the moment the loop starts.
                goal = self.server.gui.add_slider(
                    f"{dof} goal", lower[dof], upper[dof], SLIDER_STEP_MM,
                    self._want[dof])
                goal.on_update(self._setter(dof))
                self._goal.append(goal)
            self._add_tuning(cfg)

        # Everything task-shaped in one folder, which `TASK_MENU_CSS` then lifts
        # out of the right-hand panel into its own window on the left -- see the
        # constant for how, and for what breaks it.
        #
        # Declared last and it does not matter: a fixed-position element is out
        # of flow, so this folder's place in viser's call order never reaches the
        # page. What does matter is that the marker is the folder's *first*
        # child, which is what the selector keys on.
        #
        # The composer is independent of the command sliders. Watch-only mode
        # still composes files; editing source never needs permission to move
        # the connected hand -- which is why this window exists in both modes
        # and the goal sliders do not.
        with self.server.gui.add_folder("tasks"):
            self.server.gui.add_html(TASK_MENU_CSS)
            if sliders:
                self._add_task_buttons()
            self.composer = Composer(self.server, cfg, run=self._submit,
                                     pose=lambda: self._want)
        self._quat = np.empty(4)

    def _add_tuning(self, cfg: HandConfig) -> None:
        """Speed, acc and the seven per-DOF torques, in one collapsed folder.

        Collapsed because these are found once and then left: nine rows that do
        not change during a session were most of the panel's height, and the
        sliders you actually drag were below them. Still one click away, and
        still read every tick through `gains`, so a drag reaches the next write.

        Speed and torque are both per DOF; acc is the one shared number.

        **Speed is reachability on the z stage, not lag.** That is the opposite
        of what this said until the bench disproved it (2026-09-02): z would not
        rise off its stop, and torque was not the lever because `torque` is a
        *cap*, not a command -- the effort a servo develops follows position
        error, and a slow profile keeps the setpoint close enough to the joint
        that the error never grows enough to break static friction. Nudging the
        stage by hand started it every time. So z needs a speed of its own for
        the same reason it needs a torque of its own, and one number for the bus
        cannot express that.

        Torque is per DOF because it is per DOF in the protocol and because one
        number cannot fit this hand: the z stage carries the aux gripper against
        gravity while the other six run level, bisected in
        `config.HandConfig.torque_min_to_move`. Finding a DOF's number is
        otherwise an edit to that table and a restart, which puts the hand back
        at its startup pose to test a number that only means anything mid-move.

        Paired per DOF -- `{dof} speed` directly above `{dof} torque` -- because
        they are bisected together against one joint, and viser lays out in call
        order, so this listing is the layout.
        """
        with self.server.gui.add_folder("tuning", expand_by_default=False):
            self._acc = self.server.gui.add_number(
                "acc", initial_value=int(cfg.gain_vector("acc")[0]),
                min=0, max=255, step=1)
            speeds = cfg.gain_vector("speed").tolist()
            torques = cfg.gain_vector("torque_min_to_move").tolist()
            for dof in range(cfg.n_dof):
                self._speeds.append(self.server.gui.add_slider(
                    f"{dof} speed", 0, 4000, 10, int(speeds[dof])))
                self._torques.append(self.server.gui.add_slider(
                    f"{dof} torque", 0, 1000, 10, int(torques[dof])))

    def _add_task_buttons(self) -> None:
        """One button per task. A click submits; the control loop executes.

        The button writes a name into `pending` and returns immediately. It
        starts no thread, touches no bus, and cannot block viser's server
        thread, because a task returns a controller rather than driving hardware
        itself -- see `tasks/`.

        The thread this replaced is worth naming, because the shape of it looks
        reasonable and is not. Running a task off-loop meant two writers on one
        bus, so the loop had to stop commanding while a task ran; that gate is
        a second control path with its own bugs, it makes "task" mean something
        different from "policy", and none of it can exist in sim. Submitting
        instead means the loop is the only writer at every instant, and the
        same task file runs unchanged under `sim.run`.

        Called inside the `tasks` folder and with no folder of its own: that
        folder is already the grouping, and it is the window `TASK_MENU_CSS`
        floats to the left of the page.
        """
        for name, label in tasks.buttons():
            self.server.gui.add_button(label).on_click(self._submitter(name))

    def _submitter(self, name: str) -> Callable:
        """One callback per button, closing over its own task name.

        The button also points the composer at the task it ran, so the tune
        panel's numbers are the ones belonging to what just moved. Here and not
        in `_submit`, which the composer itself calls: a **Run to row** or a
        tuned run submits `_preview`, and pointing the dropdown at that would
        replace the task being tuned with the scratch copy of it.
        """
        def click(_event: object) -> None:
            self._submit(name)
            self.composer.select(name)
        return click

    def _submit(self, name: str) -> None:
        """Ask the control loop to run task `name`. Also the composer's `run`.

        A submission that lands while a task is running is dropped by the loop,
        not here: this thread cannot see whether one is running without sharing
        state with it, and `pending` is a single slot, so the last write before
        the loop reads it wins.
        """
        self.pending = name

    @property
    def armed(self) -> bool:
        """Whether the page wants torque on. Read by the control loop each tick."""
        return bool(self._armed.value)

    def gains(self) -> tuple[list[int], list[int], list[int]]:
        """`(speed, acc, torque)` for the next `set_positions`, live off the page.

        **All three per-DOF lists, never a mix.** `FtServo.set_positions` is
        nanobind-overloaded on three sequences or three scalars, and a mix
        matches neither -- `(int, int, list)` raised `TypeError` on the first
        hardware run while passing every offline test, because the mock used to
        broadcast each gain on its own. Broadcasting here rather than at the seam
        costs nothing: a sync-write is one packet in which each servo reads its
        own slice, so seven identical speeds are the same time on the wire as one.
        """
        n = len(self._torques)
        return ([h.value for h in self._speeds], [self._acc.value] * n,
                [h.value for h in self._torques])

    def _setter(self, dof: int) -> Callable:
        """One callback per slider, closing over its own DOF. That closure is
        the whole mapping between a slider and a servo; get it wrong and the
        page drives the wrong joint while looking perfectly fine."""
        def on_update(event):
            self._want[dof] = event.target.value
        return on_update

    def goal(self, measured_mm: torch.Tensor,
             load: torch.Tensor) -> list[float]:
        """The `goal_mm` seam: latest slider values, in mm. Never blocks.

        `measured_mm` and `load` are ignored -- a human at a slider is already
        looking at the hand. They are in the signature because that is the seam
        a closed-loop policy will arrive through.
        """
        return self._want

    def rebaseline(self, mm: Sequence[float]) -> None:
        """Move every goal slider to `mm`. Used after a task installs a new zero.

        Without this a slider sitting where the old zero made it sit would
        command the next move from the new absolute pose as if it were a
        delta -- a full-stroke move the moment the loop resumes.
        """
        for dof, v in enumerate(mm):
            v = float(v)
            self._want[dof] = v
            if hasattr(self, "_goal") and dof < len(self._goal):
                self._goal[dof].value = v

    def report(self, mm: torch.Tensor, load: torch.Tensor,
               want: torch.Tensor | None = None,
               temperatures: Sequence[int | None] | None = None) -> None:
        """Live mm, error, load and temperature per DOF: one table, one block.

        Numbers, not bars, and one line per DOF: the two read-only bar rows the
        panel used to spend per DOF were most of its height, and a bar buys
        nothing here -- 0.01 mm of tracking error is a real number and zero
        pixels, the pose itself is in the 3D view, and a number never clamps.
        That last part is why nothing here has a range: `mj_forward` deliberately
        does not clamp to `jnt_range` and the four travel tables disagree by up
        to 76%, so a reading past `config`'s travel is the one that settles
        which number is right, and it must print as it is.

        A fenced code block, so the seven lines are monospace and the columns
        line up under one header instead of each line repeating `mm`/`err`/
        `load` inline. Scanning a column for the DOF that is lagging is the
        thing this readout is for, and proportional text cannot be scanned.

        `want` is what the loop actually commanded, not `self._want`, so `err`
        stays true under `--panel False` where something else is the goal source.
        `None` is teach, which commands nothing -- and an error against a goal
        that was never sent would read as a goal being ignored, so it is dropped.

        `err` is signed: which way the hand is lagging is the whole question on a
        rack that binds in one direction. Load prints signed and inverted: it is
        drive effort, highest during free motion and near zero at rest, so a big
        number is motion, not contact.
        """
        self._tick += 1
        if self._tick % self._every:
            return
        got, loads = mm.tolist(), load.tolist()
        w = want.tolist() if want is not None else None
        # 22 is the longest `self._names` entry ("3 vertical translation"); a
        # narrower field would not truncate it, it would shift that one row's
        # columns and break the scan the block exists for.
        temp = temperatures or [None] * len(got)
        rows = [f"{'':<22}{'mm':>6}{'err':>7}{'load':>6}{'°C':>5}"]
        for dof, name in enumerate(self._names):
            shown_temp = "--" if temp[dof] is None else str(temp[dof])
            rows.append(
                f"{name:<22}{got[dof]:6.2f}"
                + (f"{w[dof] - got[dof]:+7.2f}" if w is not None else f"{'--':>7}")
                + f"{int(loads[dof]):6d}{shown_temp:>5}")
        self._readout.content = "```\n" + "\n".join(rows) + "\n```"

    def push(self, data: mujoco.MjData) -> None:
        """Send the current pose. Call after `mj_forward`; 0.60 ms measured."""
        for geom, handle in self._handles:
            mujoco.mju_mat2Quat(self._quat, data.geom_xmat[geom])
            handle.wxyz = self._quat.copy()
            handle.position = data.geom_xpos[geom]

    def close(self) -> None:
        self.server.stop()
class _NoViewer:
    """Headless stand-in for the passive viewer: same three calls, draws nothing.

    Exists because the hardware path cannot otherwise be exercised on a machine
    with no display, and that is where the hand is plugged in. Runs forever;
    `--seconds` is what stops it.
    """

    def __enter__(self) -> "_NoViewer":
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def is_running(self) -> bool:
        return True

    def sync(self) -> None:
        pass


def _printer(rate: int = 5) -> Callable:
    """Measured mm and the tracking error, every `rate` ticks, in place."""
    state = {"n": 0}

    def report(mm: torch.Tensor, load: torch.Tensor, ctrl: np.ndarray) -> None:
        state["n"] += 1
        if state["n"] % rate:
            return
        pos = " ".join(f"{v:6.1f}" for v in mm.tolist())
        err = max(abs(c * MM_PER_M - m) for c, m in zip(ctrl, mm.tolist()))
        print(f"\rmm [{pos} ]  err {err:5.2f}  load {int(load.abs().max()):4d}",
              end="", flush=True)

    return report


if __name__ == "__main__":
    tyro.cli(live, prog="python -m cartesian_hand.studio")

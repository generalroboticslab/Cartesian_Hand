# TODO: move pipette back in hand before putting down
# TODO: switch hand to the OG nylon hand
# TODO: figure out where to eject pipette tip into trash bin


"""Franka arm motion sequence: spray bottle, then pipette, then cap opening.

Arm motions are Cartesian offsets from home, not chained relative moves like
ee_prompt.py: `main` captures home's pose once (`robot.current_pose`, right
after homing) and every SEQUENCE row is commanded as `home_pose * Affine(offset)`
via `ReferenceType.Absolute`, so each move targets a fixed point in the robot's
base frame regardless of where the arm actually is when it runs. This is what
makes `START_FROM` a plain list-slice (see below) instead of needing to replay
or sum the skipped rows' motion to find out where the arm should be. Hand
tasks are driven directly from cartesian_hand/tasks/ -- no dependency on
cartesian_hand.studio.live(), which also opens a mujoco model, a viewer and a
web page. This script owns the bus and the tick loop itself; it only asks the
tasks package for each task's `Policy` (`tasks.make`) and steps it with
`policy.PolicyRunner`, the same runner class studio.py's own loop uses.

Each `SEQUENCE` row is `(step, dx_mm, dy_mm, dz_mm, description, task,
action)`. `dx_mm`/`dy_mm`/`dz_mm` are `None` for a row with no arm move at
all; otherwise they are that step's running total offset from home (not a
delta from the previous row -- see above), so a row's arm move never depends
on any row before it having actually run. `action` is what to do with the
hand, and is one of, in the order a task normally moves through them:

  None               -- nothing asked of the hand; a plain arm move.
  "grasp"            -- after this arm move, `start_hand_task` ticks `task`'s
                        policy up through its grip `Probe` (the squeeze) and
                        returns the still-running `PolicyRunner` paused there,
                        without going any further into the task. Stopping
                        here, instead of running the whole task in one shot,
                        is what lets the NEXT row actually lift the object
                        while it is held -- running the whole task first
                        would lift nothing, since the grip is the task's own
                        first phase.
  "continue_partial" -- no arm move; `pause_hand_task` resumes that same
                        paused runner and ticks it up through TASK_PAUSE[task]
                        (a (label, occurrence) pair), then pauses again. For
                        a task with an intermediate point the arm must reach
                        before the hand may act further -- e.g. pipette must
                        be moved over the trash bin before its tip ejects.
  "finish"           -- after this arm move, `finish_hand_task` resumes the
                        paused runner and ticks it to the task's actual
                        completion (`runner.finished()`).
  "continue"         -- the same as "finish", but with no arm move first --
                        for a task with nothing left for the arm to do before
                        the hand finishes (e.g. cap, spray: work happens in
                        place).
  "release"          -- purely a timing hint to `main`, not a hand command:
                        this row's arm move is the one that sets the object
                        down, so sleep `RELEASE_SETTLE_S` after it instead of
                        the usual `PAUSE_S`, giving the object a moment to
                        settle before anything else happens. The hand is
                        NOT opened here -- it is still gripping.
  "reset_pose"       -- before this row's arm move, `set_transit_pose` is
                        what actually opens the hand (both jaws to travel
                        max, everything else folded to 0) and blocks until it
                        arrives, so the arm never starts clearing the object
                        while the hand is still closed on it. This is the
                        real release point, however far it is from whichever
                        earlier row was tagged "release".

Every arm move -- including grasp/lift, not just the ones above -- runs
through `move_from_home_holding`, which keeps re-sending the hand's last
command (goal position, torque, and a fresh `enable_torques`) for as long as
the arm is moving, instead of sending it once and going silent. Two real bugs
lived in getting this right: `robot.poll_motion()` reads "not moving" on the
very first check after starting an async move (a race with the trajectory's
own start, not a real "arrived"), so this loops on `join_motion(timeout=...)`
instead, which actually blocks and reports true completion; and a servo's
own overload/stall protection can silently clear TORQUE_ENABLE under a
sustained stall (which is what a held squeeze is, by design), which
`set_positions` alone does not undo, hence the repeated `enable_torques`.

`enable_hand` also sets TORQUE_LIMIT (reg 48/49) to a fixed 800 for every
servo, once, at startup. This is a different register from `set_positions`'s
own `torque` argument (GOAL_TORQUE, reg 44/45): TORQUE_LIMIT is the actual
ceiling on how hard the position loop is allowed to push, and nothing else
in `cartesian_hand` ever writes it, so left alone it sits at whatever each
servo happens to have booted with. Blindly maxing it to 1000 was tried and
broke a gear -- it removed the ceiling that was protecting the gear train
from the squeeze mechanic's own sustained stall. 800 is deliberately a fixed
number with real margin under the value that broke something, not a blind
max and not (currently) tracked per task.

Before any of that, at startup, `run_zero` runs `tasks/zero.py` to find every
DOF's hard stop and saves it as this hand's calibration -- every absolute
millimetre goal downstream (a task's own "height", "open", grip, ...) is
measured from that zero, so nothing else here is trustworthy without it.

franka control here
from home command the following: relative motion commands in cartesian coords.
(each number below is relative to the step before it, as originally dictated;
SEQUENCE further down stores the running total of these instead -- see the
top of this docstring)

0. -20 0 0 back off from home before traversing

1. 90 -145 420 gets behind spray

2. 120 0 0 now can grasp spray bottle using hand

3. 0 0 -100 lift spray bottle clear of the table

4. perform spray task using hand

5. 0 0 100 set spray bottle back down

6. -120 0 0 reset arm to clear spray bottle

7. 0 -225 -167 move to behind pipette

8. 50 0 0 grasp pipette here using hand

9. 0 0 -150 lift pipette

10. perform pipette knob rotation and the two plunges, then wait (no ejecting yet)

10.1 -85 50 0 move pipette over the trash bin

10.2 0 0 97 lower into the bin, then eject the tip

10.3 75 -50 -97 move pipette back to original position

11. 0 0 150 lower pipette, and release hand to ready

11.1 -55 -10 0 open the hand fully, then back off clear of the pipette

12. -30 0 0 reset arm to clear pipette

13. 0 -180 130 gets behind the bottle

14. 80 0 0 grasp bottle here using hand

15. 0 0 -100 lift bottle

16. perform cap opening task on hand

17. 0 0 100 lower bottle, and release hand to ready

18. -150 0 -200 reset arm to clear bottle, then home

19. home
"""

import os
import time

import torch
from franky import (Robot, RealtimeConfig, CartesianMotion, Affine,
                    ReferenceType, JointMotion, ControlException)

from cartesian_hand import motions, tasks
from cartesian_hand.config import (AUX_JAW, BASE_JAW, HANDS, get_hand,
                                    identify, load_offsets, save_offsets)
from cartesian_hand.policy import PolicyRunner
from cartesian_hand.primitives import Probe
from cartesian_hand.servo import open_driver
from cartesian_hand.tasks import cap as cap_task
from cartesian_hand.tasks import pipette as pipette_task
from cartesian_hand.tasks import triggers as triggers_task

PAUSE_S = 0.0  # pause after each arm move
SETTLE_S = 0.0  # settle time after a grip is detected, before lifting
RELEASE_SETTLE_S = 0.0  # extra settle after lowering, so the object sits on the table

# False reuses the saved calibration instead of re-zeroing at startup -- same
# opt-out `bite_probe.py`/`jaw_compare.py`/`jaw_pause_probe.py` already use.
ZERO_AT_START = False

# Set to "spray", "pipette", or "cap" to skip objects already done successfully
# and start partway through. Every SEQUENCE move is an absolute offset from
# home (see `move_from_home`), so this is just a list-slice -- the first move
# of the chosen group already targets the right place, wherever it is.
# Leave as None to run the whole sequence from step 1.
# START_FROM = "pipette"
# START_FROM = "cap"
START_FROM = None
GROUP_START_STEP = {"spray": 1, "pipette": 7, "cap": 13}

# Per-task tuning: `start_hand_task` passes `cfg=` through to `tasks.make`, so
# any field a task's own Config dataclass declares (cartesian_hand/tasks/<name>.py)
# can be overridden here without touching that file. Every field below is one
# of that task's own `tune`-tagged fields (see cartesian_hand/tasks/<name>.py
# for the full docstring on each), listed at its default except where noted --
# edit any value here to tune it.
# TORQUE CEILING -- measured on hand_3's base jaw, stalled against a real object
# with `stall_probe.py` (goal 0.0mm, so position error never shrinks and the
# servo saturates and stays saturated, which is how every `Hold` grip works):
#
#     250 -> load -204,  current -255mA, holds 8s
#     400 -> load -428,  current -385mA, holds 8s
#     550 -> load -676,  current -534mA, then 0 after ~3s   <- current cutout
#     750 -> load -800,  current -653mA, then 0 after ~2s
#
# Past the cutout the servo clears its own TORQUE_ENABLE and goes limp, and it
# LATCHES: re-arming does nothing (`refresh_hand` and a per-tick re-arm were
# both tried and neither recovered it). Only toggling torque off and back on
# brings it back, which is useless mid-grip -- it drops whatever is held.
#
# 400 may not be a fix, only a longer fuse -- UNRESOLVED, see below. Holding
# 400 for 60s: current pinned at -385mA, temp 46 -> 61C at about +15C/min, and
# load crept -469 -> -547 at constant current (winding resistance rising with
# heat -- the creep IS the thermometer, the heating is real and in the motor).
# What is NOT established is whether that ramp reaches a cutout: 60s is short
# next to a motor's thermal time constant, so it may plateau. Reg 13 has since
# been read off all seven servos (`jaw_compare.py`) and the limit is 80C, not
# the 70 assumed here earlier -- about 75s past where the 60s hold ended, so
# reachable on paper. Settle it by holding 400 until it trips or flattens, not
# by extrapolating.
#
# The low cap is NOT fixable, and `jaw_compare.py` ruled out all three ways it
# might have been. Not configuration: registers 13-25 read identically on
# every servo in the hand, MAX_TORQUE (16/17) is 1000, unloading bitmask (19)
# is 13 = voltage|overheat|overload. Not a degraded DOF 0: the aux jaw runs
# the same ladder to the same numbers (-538mA, tripping at t+3s at cap 550,
# against -382mA holding at 400). Not the supply: 12.2 -> 12.1V under a
# stalled motor. What is left is the firmware overload integrator on a motor
# whose continuous rating sits in (385, 538] mA -- a family property with no
# knob on it, so the only fix is not to stall.
#
# If it does trip, a torque cap only buys time and the fix is not to stall at
# all: give the Probe a `measure` and make the Hold's goal `contact - bite_mm`,
# so the position error goes to zero and the grip is a bounded preload instead
# of permanent current saturation. Until that is settled, keep stalled squeeze
# torque at 400 or below.
TASK_CONFIG = {
    "triggers": triggers_task.Config(
        trigger_offset=30.0,    # Z height (mm) the housing squeeze holds -- raised from 0
        squeeze_torque=400.0,   # base-jaw hold torque on the housing -- was 900, which
                                # trips the servo's overload cutout (see TORQUE CEILING above)
        open_mm=45.0,           # aux-jaw "ready" position between pulls
        pull_seconds=3.0,       # how long the pull tries before giving up
        travel_torque=50.0,     # torque for unloaded free moves
        approach_torque=150.0,  # torque while closing the base jaw to find the housing
        approach_speed=800.0,   # servo speed while seeking contact
        timeout_margin=1.5,     # deadline margin on every row
        cycles=3.0,             # spray 3 times, not the default 1
    ),
    "pipette": pipette_task.Config(
        knob_z=28.0,            # Z height of the twist-lock knob
        top_z=50.0,             # Z the fist returns to between presses
        plunger_z=25.0,         # Z at the bottom of a plunger press
        eject_z=4.0,            # Z at the bottom of the tip-eject press
        knob_revs=0.5,          # turns to work the twist knob
        knob_clearance=18.0,    # aux-jaw backoff from the knob between strokes
        base_grip_x=42.0,       # base finger position while pipette rests in the stand -- was 30, fingers were in the way of the post-eject rise
        final_pause_s=1.0,      # pause at the end, holding rise height
        approach_torque=150.0,  # torque closing a jaw onto pipette/knob
        approach_speed=300.0,   # servo speed while finding contact
        travel_speed=1050.0,    # servo speed for every free move
        squeeze_torque=450.0,   # holding torque, both jaws (body + knob) -- was 750, which
                                # trips the servo's overload cutout (see TORQUE CEILING above)
        base_grip_torque=600.0, # torque driving base fingers to base_grip_x
        push_torque=500.0,      # Z torque pressing down (plunge/eject)
        lift_torque=800.0,      # Z torque for every ascent
        travel_torque=100.0,     # torque for unloaded horizontal travel
        finger_stroke=40.0,     # aux-finger sweep during one knob twist
        timeout_margin=1.5,     # deadline margin
        press_seconds=1.0,      # how long a plunger/eject press holds
    ),
    "cap": cap_task.Config(
        cap_offset=25.0,             # height of cap's top face above z zero
        num_revs_up=1.5,             # revolutions opening (unscrewing)
        num_revs_down=0.6,           # revolutions closing (screwing down) -- was 2.3, over-turned and stalled/failed on the real cap
        squeeze_torque=400.0,        # aux-jaw grip for initial bottle grip + opening twist -- was 80, way too weak (max 500)
        close_squeeze_torque=80.0,   # aux-jaw grip for the closing twist only
        close_torque=150.0,          # finger torque while closing the cap
        approach_torque=150.0,       # torque closing a jaw to find contact
        travel_speed=1500.0,         # servo speed for free moves
        approach_speed=800.0,        # servo speed while seeking contact
        travel_torque=50.0,          # torque for unloaded travel
        release_clearance=5.0,       # aux-jaw opening past measured radius between strokes
        timeout_margin=1.5,          # deadline margin
        finger_stroke=45.0,          # aux-finger sweep during a twist
        lift_mm=50.0,                # absolute z holding the removed cap clear
        lift_torque=500.0,           # Z torque lifting the gripper with the cap
        clear_wait_s=1.0,            # max wait with cap held clear
        cycles=1.0,                  # open/close cycles (0 = run until stopped)
        reinsert_z_torque=50.0,      # downward z torque seating/threading the cap
        down_stroke_z=5.0,           # absolute z goal while tightening
        child_safe_press_mm=15.0,    # depth below cap_offset for child-safe press (child_safe only)
        # bulb_offset, bulb_lift_z, bulb_release_mm, bulb_open_mm,
        # bulb_squeeze_torque also exist but only matter under dropper_bottle=True,
        # which this Config leaves at its default (False).
    ),
}

# Where a "continue_partial" action pauses a task: (label, occurrence), fed to
# `row_after_nth_label`. Pipette's tip-eject must wait for the arm to be over
# the trash bin, so it pauses right after the second plunge's "rise".
TASK_PAUSE = {
    "pipette": ("plunger", 2),
}

ROBOT_IP = "172.16.0.2"  # set to your Franka's IP

# Standard Franka "ready" pose: [0, -pi/4, 0, -3pi/4, 0, pi/2, pi/4]
NOMINAL_JOINTS = [0.0, -0.785398163, 0.0, -2.35619449, 0.0, 1.57079632679, 0.785398163397]

# (step, dx_mm, dy_mm, dz_mm, description, task, action) for the arm.
# dx_mm/dy_mm/dz_mm are each step's running total offset from home (not from
# the previous step), i.e. the cumulative sum of the incremental deltas in
# the docstring's numbered plan above; None means no arm move this row.
# `action` semantics (None / "grasp" / "continue_partial" / "finish" /
# "continue" / "release" / "reset_pose") are documented in full at the top of
# this module's docstring.
SEQUENCE = [
    (0, -20, 0, 0, "back off before traversing", None, None),
    (1, 70, -145, 420, "gets behind spray", None, None),
    (2, 190, -145, 420, "now can grasp spray bottle using hand", "triggers", "grasp"),
    (3, 190, -145, 320, "lift spray bottle clear of the table", None, None),
    (4, None, None, None, "perform spray task using hand", "triggers", "continue"),
    (5, 190, -145, 420, "set spray bottle back down", None, None),
    (6, 70, -145, 420, "reset arm to clear spray bottle", None, "reset_pose"),
    (7, 70, -370, 278, "move to behind pipette", None, None),
    (8, 117, -370, 278, "grasp pipette here using hand", "pipette", "grasp"),
    (9, 115, -370, 103, "lift pipette", None, None),
    (10, None, None, None, "perform pipette knob rotation and plunging", "pipette", "continue_partial"),
    ("10.1", 30, -320, 103, "move pipette over the trash bin", None, None),
    ("10.2", 30, -320, 200, "lower into the bin, then eject the tip", "pipette", "finish"),
    ("10.3", 105, -370, 103, "move pipette back to original position", None, None),
    (11, 115, -370, 255, "lower pipette, and release hand to ready", None, "release"),
    ("11.1", 60, -380, 255, "open the hand fully, then back off clear of the pipette", None, "reset_pose"),
    (12, 60, -370, 253, "reset arm to clear pipette", None, None),
    (13, 60, -550, 413, "gets behind the bottle", None, None),
    (14, 160, -550, 413, "grasp bottle here using hand", "cap", "grasp"),
    (15, 160, -550, 313, "lift bottle", None, None),
    (16, None, None, None, "perform cap opening task on hand", "cap", "continue"),
    (17, 160, -550, 413, "lower bottle, and release hand to ready", None, "release"),
    (18, 10, -550, 213, "reset arm to clear bottle, then home", None, "reset_pose"),
]


def move_from_home(robot, home_pose, dx_mm, dy_mm, dz_mm):
    """Move to `home_pose` offset by (dx, dy, dz) mm, as an absolute target
    in the robot's base frame -- not relative to wherever the arm currently
    is, so this lands in the same place whether run in full sequence or
    resumed partway through (`START_FROM`)."""
    dx, dy, dz = dx_mm / 1000, dy_mm / 1000, dz_mm / 1000
    target = home_pose * Affine([dx, dy, dz])
    robot.move(CartesianMotion(target, ReferenceType.Absolute))


def _move_with_recovery(robot, target, asynchronous=False):
    """`robot.move` to `target`, recovering and retrying once on a reflex.

    libfranka aborts a motion and LATCHES an error state when a safety reflex
    fires, and until `recover_from_errors` clears it every later move fails
    too -- so one reflex ends the whole run, not just its own step. Recovering
    and re-issuing costs nothing when nothing tripped and saves the run when
    something did.

    The one this exists for is
    `cartesian_motion_generator_joint_velocity_discontinuity`: the commanded
    joint velocity stepped harder between two control cycles than the arm
    allows. `control_command_success_rate: 1` each time says the network and
    the realtime loop were clean, so it is the trajectory, not the link.

    A retry helps because the second attempt starts from a stationary,
    freshly-recovered arm. It does NOT help if the path itself is the problem
    -- an ill-conditioned stretch where the Cartesian-to-joint map turns a
    modest tool velocity into a large joint one -- and then the same step
    fails twice, which is exactly the signal to route around it. The joint
    angles printed below are for telling those apart.
    """
    for attempt in (1, 2):
        try:
            robot.move(CartesianMotion(target, ReferenceType.Absolute),
                       asynchronous=asynchronous)
            return
        except ControlException as exc:
            print(f"  ** arm reflex on attempt {attempt}: {exc}")
            print(f"     joints at abort: {list(robot.current_joint_positions)}")
            if attempt == 2:
                raise
            robot.recover_from_errors()
            time.sleep(PAUSE_S)


def move_from_home_holding(robot, home_pose, dx_mm, dy_mm, dz_mm,
                           bus, last_command, period):
    """Like `move_from_home`, but re-sends `last_command` throughout the
    move instead of leaving the bus silent while the arm travels -- see
    `refresh_hand` for why. `last_command` is None before anything has ever
    been sent to the hand (there is nothing to hold yet), in which case
    this is a plain blocking move, same as `move_from_home`.

    Loops on `join_motion(timeout=period)`, not `poll_motion()`: measured on
    the real arm, `poll_motion()` read "not moving" on the very first check
    after every single async move, including a 172 mm lift, which means it
    was racing the trajectory's own start rather than reporting whether the
    arm had actually arrived -- `refresh_hand` never ran once, though the
    move itself still completed correctly (`join_motion` at the end still
    blocked for it). `join_motion(timeout=...)` doesn't have that race: it
    blocks for up to `period` and returns whether the motion is actually
    done, so a "not done yet" here is trustworthy the way `poll_motion()`
    reporting "moving" was not.
    """
    dx, dy, dz = dx_mm / 1000, dy_mm / 1000, dz_mm / 1000
    target = home_pose * Affine([dx, dy, dz])
    if last_command is None:
        _move_with_recovery(robot, target)
        return
    _move_with_recovery(robot, target, asynchronous=True)
    refreshes = 0
    while not robot.join_motion(timeout=period):
        refresh_hand(bus, last_command)
        refreshes += 1
    print(f"  (held grip refreshed {refreshes}x during this move, torque={last_command[4]})")


# ── Hand: bus, zero, and the per-tick loop ───────────────────────────────────
# Deliberately not cartesian_hand.studio.open_hand: that module imports mujoco
# and viser at load time for its viewer/page, neither of which this script
# uses. This is the same ~15-line bus-discovery logic without that dependency.

def open_hand(name: str | None = None, port: str | None = None,
              mock: bool = False):
    if name:
        cfg = get_hand(name)
        return (cfg.variant(port=port) if port else cfg), open_driver(
            port or cfg.port, mock=mock)
    if mock:
        cfg = get_hand()
        return cfg, open_driver(cfg.port, mock=True)
    for p in [port] if port else sorted({h.port for h in HANDS.values()}):
        if not os.path.exists(p):
            continue
        bus = open_driver(p, mock=False)
        try:
            cfg = identify(bus, p)
        except RuntimeError:
            bus.close()
            continue
        return cfg, bus
    raise RuntimeError("no hand found -- pass a port or check it is plugged in")


def read_mm(bus, cfg, zero):
    ids = cfg.servo_ids
    reads = bus.read_all(ids)
    if any(r is None for r in reads):
        raise RuntimeError(f"servo(s) did not answer: "
                           f"{[i for i, r in zip(ids, reads) if r is None]}")
    counts = torch.tensor([r[0] for r in reads], dtype=torch.float32)
    return cfg.counts_to_mm(counts, zero)


def enable_hand(bus, cfg):
    """Write the present raw pose, then energize -- so nothing snaps on power-up.

    Raw counts, not millimetres: there is no zero yet at this point in
    startup -- `run_zero` runs right after this, on an energized bus.

    Also sets TORQUE_LIMIT (reg 48/49) to 800 for every servo, once. This is
    a SEPARATE register from `set_positions`'s own `torque` argument (which
    writes GOAL_TORQUE, 44/45) -- TORQUE_LIMIT scales the ceiling the
    position loop's output is allowed to reach at all, and nothing else
    writes it, so it otherwise sits at whatever each servo booted with (SRAM,
    not saved across power cycles -- see `ft_servo_driver.hpp`). A version
    that tracked this to match GOAL_TORQUE exactly per command was tried and
    made every grip weaker than a fixed ceiling, for reasons not yet
    understood; a blind 1000 was tried before that and broke a gear. 800 is
    the current compromise: fixed and blind, like the value that broke a
    gear, but with real margin under it instead of none.
    """
    ids = cfg.servo_ids
    counts = torch.tensor([r[0] for r in bus.read_all(ids)], dtype=torch.float32)
    speed = cfg.gain_vector("speed").tolist()
    acc = cfg.gain_vector("acc").tolist()
    torque = cfg.gain_vector("torque_min_to_move").tolist()
    bus.set_positions(ids, counts.int().tolist(), speed, acc, torque)
    bus.enable_torques(ids, True)
    bus.set_torque_limits(ids, [800] * len(ids))


def run_zero(bus, cfg):
    """Drive `tasks/zero.py` to find every DOF's hard stop, save it, and
    return the new zero (raw counts) -- mirrors `studio.finish`'s zeroing
    branch, the only other place this hand's calibration is written.

    Zeroing is uncalibrated by design (`frame="here"`, see zero.py), so any
    reference works as the frame it starts from; the current raw counts serve
    that role and are discarded once the real zero comes back.
    """
    ids = cfg.servo_ids
    counts = torch.tensor([r[0] for r in bus.read_all(ids)], dtype=torch.float32)
    start_zero = counts.clone()
    start_mm = cfg.counts_to_mm(counts, start_zero)[None, :]
    controller = tasks.module("zero").build(cfg, start_mm)
    runner = motions.TaskRunner(
        controller,
        hold_torque=cfg.gain_vector("torque_min_to_move").to(torch.float32)[None, :])
    period = 1.0 / cfg.control_hz
    speed_gain = cfg.gain_vector("speed").tolist()
    acc_gain = cfg.gain_vector("acc").tolist()
    while True:
        t0 = time.time()
        mm = read_mm(bus, cfg, start_zero)
        step = runner.tick(mm[None, :])
        if step is None:
            break
        goal, torque, speed = step
        # A task's goal is NOT clamped to travel -- zero deliberately asks for
        # more than the rail so the hard stop, not the number, ends the seek.
        speed_now = [round(float(s) * cfg.counts_per_mm) if s > 0 else g
                    for s, g in zip(speed[0].tolist(), speed_gain)]
        torque_now = torque[0].round().int().tolist()
        bus.set_positions(ids, cfg.mm_to_counts(goal[0], start_zero).tolist(),
                          speed_now, acc_gain, torque_now)
        time.sleep(max(0.0, period - (time.time() - t0)))
    value, ok, why = runner.result
    if not bool(ok.all()):
        raise RuntimeError(f"zero: {why}")
    offsets = cfg.mm_to_counts(value, start_zero)[0]
    save_offsets(cfg.name, offsets.tolist())
    print(f"[{cfg.name}] zeroed: {offsets.tolist()}")
    return offsets.to(torch.float32)


JAW_OPEN_TOL_MM = 3.0  # how far short of travel max a jaw may stop and still
                       # count as open. Loose on purpose: the point is to catch
                       # a jaw that never left the object, not to police the
                       # last millimetre of a free move.


def transit_pose_mm(cfg):
    """Both jaws open to travel max, everything else folded to 0."""
    pose = torch.zeros(cfg.n_dof)
    pose[BASE_JAW] = cfg.travel_mm[BASE_JAW]
    pose[AUX_JAW] = cfg.travel_mm[AUX_JAW]
    return pose


def wait_until_reached(bus, cfg, zero, goal_mm, tolerance_mm=1.0,
                       stuck_mm_s=0.3, confirm_ticks=10, timeout_s=None):
    """Block until every DOF is within `tolerance_mm` of `goal_mm`, or has
    stopped moving there (a free move against friction near a fold-in point
    may never hit the goal exactly).

    `timeout_s` defaults to the slowest DOF's full-rail budget -- the same
    margin `Sequence` rows use -- so a genuinely stuck joint raises instead of
    this call returning early and letting the arm move while the hand is
    still committed to reaching `goal_mm`.
    """
    if timeout_s is None:
        timeout_s = cfg.travel_budget(range(cfg.n_dof), margin=1.5)
    period = 1.0 / cfg.control_hz
    quiet = torch.zeros(cfg.n_dof)
    last = read_mm(bus, cfg, zero)
    deadline = time.time() + timeout_s
    while True:
        time.sleep(period)
        mm = read_mm(bus, cfg, zero)
        reached = (mm - goal_mm).abs() <= tolerance_mm
        stopped = (mm - last).abs() / period < stuck_mm_s
        quiet = torch.where(stopped, quiet + 1, torch.zeros_like(quiet))
        last = mm
        if bool((reached | (quiet >= confirm_ticks)).all()):
            return
        if time.time() >= deadline:
            raise RuntimeError(f"hand did not settle within {timeout_s:.1f}s: "
                              f"measured {mm.tolist()}, goal {goal_mm.tolist()}")


def set_transit_pose(bus, cfg, zero):
    """Drive DOFs 0 and 4 open and the rest to 0, and wait until they arrive.

    Blocking, not fire-and-forget: this runs at every "reset" step, and the
    arm move right after it must not start clearing the object until the hand
    has actually let go of it. Returns the raw command sent, for `refresh_hand`.
    """
    ids = cfg.servo_ids
    goal = transit_pose_mm(cfg)
    speed = cfg.gain_vector("speed").tolist()
    acc = cfg.gain_vector("acc").tolist()
    torque = cfg.gain_vector("torque_min_to_move").tolist()
    counts = cfg.mm_to_counts(goal, zero).tolist()
    bus.set_positions(ids, counts, speed, acc, torque)
    wait_until_reached(bus, cfg, zero, goal)
    # `wait_until_reached` accepts "stopped" as well as "arrived", because the
    # fingers fold against a hard stop and never reach 0 exactly. The jaws have
    # no such excuse -- their goal is open air -- so a jaw that stopped short
    # did not open: the servo tripped, or it is still squeezing too hard to
    # move at `torque_min_to_move`. Silently that returns an unreleased hand
    # and the caller's arm move drags the object off the bench, so raise.
    mm = read_mm(bus, cfg, zero)
    for jaw in (BASE_JAW, AUX_JAW):
        if mm[jaw] < goal[jaw] - JAW_OPEN_TOL_MM:
            raise RuntimeError(
                f"hand did not open: dof {jaw} stopped at {mm[jaw]:.1f}mm, "
                f"goal {goal[jaw]:.1f}mm -- refusing to move the arm on a "
                f"closed hand")
    return (ids, counts, speed, acc, torque)


def tick_hand(bus, cfg, zero, runner):
    """One control-tick: measure, step the policy, write its action.

    Returns the raw `(ids, counts, speed, acc, torque)` just sent, so a
    caller that stops ticking (e.g. `_tick_until` pausing at a grip) can
    keep re-sending that exact packet with `refresh_hand` -- see its
    docstring for why a single packet is not enough.

    Deliberately does NOT re-arm torque. A version of this that called
    `enable_torques(ids, True)` every tick was tried against the "grip goes
    loose partway through a task" bug, on the theory that a firmware
    overload trip clears TORQUE_ENABLE and nothing was setting it back. The
    first half is true; the second does not follow. `stall_probe.py` showed
    the trip LATCHES -- after it fires, re-arming a tripped servo leaves it
    limp (load stayed 0 through four seconds of re-arm-and-command), and
    only a torque off/on toggle revives it. So the re-arm bought nothing and
    cost seven extra unicast writes a tick, TORQUE_ENABLE not being in the
    sync-write block. The real defence is upstream: keep squeeze torque
    under the cutout, per TORQUE CEILING above `TASK_CONFIG`.
    """
    ids = cfg.servo_ids
    mm = read_mm(bus, cfg, zero)
    action = runner.tick(mm[None, :])
    goal = cfg.clamp(action.goal_mm[0])
    speed = (action.max_speed_mm_s[0] * cfg.counts_per_mm).round().clamp_min(1).to(torch.int32).tolist()
    torque = (action.effort_limit[0].clamp(0.0, 1.0) * 1000).round().to(torch.int32).tolist()
    acc = cfg.gain_vector("acc").tolist()
    counts = cfg.mm_to_counts(goal, zero).tolist()
    bus.set_positions(ids, counts, speed, acc, torque)
    return (ids, counts, speed, acc, torque)


def refresh_hand(bus, last_command):
    """Re-arm torque and re-send the last raw command, unchanged.

    These servo registers are meant to hold their last value with no
    refresh needed, and that is true for ordinary moves. But `studio.py`'s
    own control loop (see its `bus.set_positions(ids, ..., speed, acc,
    torque)` inside `while view.is_running()`) re-sends the live command on
    every tick regardless, for as long as the page is armed -- it never
    relies on a single packet holding by itself. `sequence.py`'s grip used
    to send exactly one packet (the `Hold` row's single tick) and then go
    completely silent on the bus for the whole arm move that follows,
    which is the one concrete difference between "torque armed in studio,
    hard to move" and "gripped in sequence.py, easy to move" that this
    file and studio.py's own practice disagree on. This closes that gap by
    re-sending the same packet during any arm move that must keep a grip.

    Also re-calls `enable_torques`, not just `set_positions`: this class of
    servo commonly has a firmware-level overload/stall protection that can
    silently clear TORQUE_ENABLE (reg 40) on its own after sustained high
    current against a stall -- exactly the condition the squeeze mechanic
    creates on purpose -- with no signal back to the host beyond the joint
    going limp. `set_positions` alone does not touch that bit, so if this is
    what happened, re-sending the same goal/torque would not have re-armed
    it; only `enable_torques(ids, True)` does. Cheap and harmless to call
    every refresh even when torque was never dropped.
    """
    ids = last_command[0]
    bus.enable_torques(ids, True)
    bus.set_positions(*last_command)


def rearm(bus, ids):
    """Re-enable torque on `ids` alone, with no position resend -- for right
    after a `set_positions` that hasn't been latched into a `last_command`
    tuple yet, so there is nothing for `refresh_hand` to re-send. See
    `refresh_hand` for why a sustained stall can silently clear TORQUE_ENABLE
    even though nothing else changed."""
    bus.enable_torques(ids, True)


def close_dof_hard(bus, cfg, zero, last_command, dof, goal_mm, torque_value):
    """Command a single `dof` to `goal_mm` at `torque_value`, leaving every
    other DOF's held goal/speed/acc/torque in `last_command` untouched.
    Re-enables torque first, same as `refresh_hand`, since the DOFs still
    being held may have tripped their own stall protection while idle.
    Returns the updated `last_command` tuple.
    """
    ids, counts, speed, acc, torque = last_command
    bus.enable_torques(ids, True)
    mm = cfg.counts_to_mm(torch.tensor(counts, dtype=torch.float32), zero)
    mm[dof] = goal_mm
    counts = cfg.mm_to_counts(mm, zero).tolist()
    torque = list(torque)
    torque[dof] = torque_value
    bus.set_positions(ids, counts, speed, acc, torque)
    return (ids, counts, speed, acc, torque)


def row_after_nth_label(rows, label, n):
    """Index of the row right after the `n`th row named `label`.

    Used to pause a task partway through -- e.g. pipette's two plunges are
    each a `Hold(label="plunger")` followed by a `Move(label="rise")`, so
    `row_after_nth_label(rows, "plunger", 2)` is that second `rise`, the point
    to stop at once the plunger is back up, before ejecting the tip.
    """
    seen = 0
    for i, row in enumerate(rows):
        if getattr(row, "label", None) == label:
            seen += 1
            if seen == n:
                return i + 1
    raise ValueError(f"no row labeled {label!r}, occurrence {n}")


def _tick_until(bus, cfg, zero, runner, task_name, stop_row=None):
    """Tick `runner` until it finishes, fails, or passes `stop_row`.

    `stop_row=None` runs to true completion. Raises on failure -- the caller
    must not act (lift, move the arm) on a task that did not do what it said.

    Returns the last raw command `tick_hand` sent, so the caller can keep
    re-sending it with `refresh_hand` once this stops ticking.
    """
    period = 1.0 / cfg.control_hz
    last_command = None
    while True:
        t0 = time.time()
        last_command = tick_hand(bus, cfg, zero, runner)
        if runner.failed():
            raise RuntimeError(f"{task_name}: failed in phase(s) "
                              f"{runner.state.phase.tolist()}")
        if runner.finished() or (stop_row is not None
                                 and runner.state.phase.item() > stop_row):
            return last_command
        time.sleep(max(0.0, period - (time.time() - t0)))


def start_hand_task(bus, cfg, zero, task_name):
    """Build `task_name`'s policy and tick it up through its grip `Probe`.

    Returns the paused `PolicyRunner`, still commanding the grip it found.
    Raises if the grip fails -- the caller must not lift on a failed grasp.

    A `Probe`'s own `grip` field (e.g. triggers.py's "squeeze housing") latches
    its higher holding effort on the very tick contact is confirmed -- see
    `Sequence._run`'s `if row.grip is not None: action = hold(...)`, applied
    before that tick's action is returned. Stopping at `grip_row` there is
    correct.

    But cap.py's default "probe" leaves `grip` unset and defers the higher
    effort to a separate `Hold(label="grip", ...)` row right after it -- and
    `PolicyRunner.tick()` returns the action computed for the phase a row was
    in *before* that tick, so the tick where phase first advances past a
    plain `Probe` still carries its own low contact-seeking effort, not that
    following `Hold`'s. Stopping at `grip_row` there writes the low effort to
    the bus and never ticks again until `finish_hand_task` -- so the object
    rides through the entire lift held at contact-seeking effort, and no
    `TASK_CONFIG` squeeze torque above it does anything until then. Hence the
    `+ 1` below when the Probe itself has no `grip`: one row later, the
    `Hold`'s own tick has actually run and its effort is what's last written
    when this returns.

    A grasp may also need more than one `Probe` -- pipette.py's body and knob
    jaws each latch their own `grip` independently (see its own comment on
    why they are not one combined `Probe(group=JAWS)`) -- so this advances
    through every consecutive `Probe` from the first one found, and only then
    applies the `grip is not None` check above, to the last of them: waiting
    for just the first would return with the second jaw still only at its
    contact-seeking effort.
    """
    start_mm = read_mm(bus, cfg, zero)[None, :]
    kwargs = {"cfg": TASK_CONFIG[task_name]} if task_name in TASK_CONFIG else {}
    policy = tasks.make(task_name, cfg, start_mm, **kwargs)
    grip_row = next(i for i, row in enumerate(policy.rows)
                    if isinstance(row, Probe))
    while (grip_row + 1 < len(policy.rows)
          and isinstance(policy.rows[grip_row + 1], Probe)):
        grip_row += 1
    stop_row = grip_row if policy.rows[grip_row].grip is not None else grip_row + 1
    runner = PolicyRunner(policy, cfg.control_hz)
    last_command = _tick_until(bus, cfg, zero, runner, task_name, stop_row)
    torque = (runner.state.held_effort[0].clamp(0.0, 1.0)
              * 1000).round().int().tolist()
    print(f"{task_name}: gripped, holding torque {torque}")
    return runner, last_command


def pause_hand_task(bus, cfg, zero, runner, task_name, label, occurrence):
    """Resume a runner and tick it up through the row after `label`'s
    `occurrence`th appearance -- see `row_after_nth_label`."""
    stop_row = row_after_nth_label(runner.policy.rows, label, occurrence)
    return _tick_until(bus, cfg, zero, runner, task_name, stop_row)


def finish_hand_task(bus, cfg, zero, runner, task_name):
    """Resume a runner and tick it all the way to completion."""
    return _tick_until(bus, cfg, zero, runner, task_name)


def main():
    robot = Robot(ROBOT_IP, realtime_config=RealtimeConfig.Ignore)
    robot.relative_dynamics_factor = 0.1  # cap speed/accel to 10%

    hand_cfg, bus = open_hand()
    enable_hand(bus, hand_cfg)

    try:
        robot.move(JointMotion(NOMINAL_JOINTS))  # home first
        time.sleep(PAUSE_S)
        home_pose = robot.current_pose  # every SEQUENCE offset is measured from here

        if ZERO_AT_START:
            zero = run_zero(bus, hand_cfg)  # then zero the hand, at home
        else:
            offsets = load_offsets(hand_cfg.name, hand_cfg.n_dof)
            if offsets is None:
                raise RuntimeError(
                    f"ZERO_AT_START=False but no saved calibration for {hand_cfg.name!r}")
            zero = torch.tensor(offsets, dtype=torch.float32)
            print(f"[{hand_cfg.name}] reusing saved zero: {offsets}")
        # then the hand's own init pose -- and the first thing worth re-sending
        # while an arm move runs, so `move_from_home_holding` has something to
        # hold from the very first move onward, not just after the first grasp.
        last_hand_command = set_transit_pose(bus, hand_cfg, zero)
        hand_period = 1.0 / hand_cfg.control_hz

        if START_FROM is not None:
            if START_FROM not in GROUP_START_STEP:
                raise ValueError(f"START_FROM must be one of {list(GROUP_START_STEP)}, "
                                 f"got {START_FROM!r}")
            cutoff = GROUP_START_STEP[START_FROM]
            sequence = [row for row in SEQUENCE if float(row[0]) >= cutoff]
        else:
            sequence = SEQUENCE

        runner = None
        for step, dx_mm, dy_mm, dz_mm, description, task, action in sequence:
            if action == "reset_pose":
                last_hand_command = set_transit_pose(bus, hand_cfg, zero)
            if dx_mm is not None:
                print(f"{step}. {dx_mm} {dy_mm} {dz_mm} -- {description}")
                move_from_home_holding(robot, home_pose, dx_mm, dy_mm, dz_mm,
                                       bus, last_hand_command, hand_period)
                time.sleep(RELEASE_SETTLE_S if action == "release" else PAUSE_S)
            if action == "grasp":
                print(f"{step}. [hand task '{task}'] squeezing")
                runner, last_hand_command = start_hand_task(bus, hand_cfg, zero, task)
                time.sleep(SETTLE_S)
            elif action == "continue_partial":
                print(f"{step}. [hand task '{task}'] {description}")
                last_hand_command = pause_hand_task(
                    bus, hand_cfg, zero, runner, task, *TASK_PAUSE[task])
            elif action in ("continue", "finish"):
                print(f"{step}. [hand task '{task}'] {description}")
                last_hand_command = finish_hand_task(bus, hand_cfg, zero, runner, task)
                runner = None

        robot.move(JointMotion(NOMINAL_JOINTS))  # step 19: home
        time.sleep(PAUSE_S)
    finally:
        # Torque off on completion or on any error -- never leave the servos
        # energized because the sequence stopped partway through.
        try:
            bus.enable_torques(hand_cfg.servo_ids, False)
        finally:
            bus.close()


if __name__ == "__main__":
    main()

"""Squeeze both jaws on the tool, then cycle the top jaw between two heights.

    from ready: pull all four fingers inward
    close z (DOF 3) to its closed position (0)
    squeeze both jaws (DOF 0 and DOF 4); hold `squeeze_hold_s`
    open the top jaw (DOF 4) only -- bottom jaw (DOF 0) never releases
    raise z (DOF 3) by `up_offset_mm`
    squeeze the top jaw again; hold `squeeze_hold_s`
    open the top jaw again
    lower z back to the height the first squeeze happened at

Bottom gripper (DOF 0) grips exactly once, at the very first squeeze, and
never releases again -- every row after that first squeeze names only the
top gripper (DOF 4) and z. Fingers (DOF 1, 2, 5, 6) are pulled in once,
before anything else, and no row after that ever names them again.

`step_seconds` (default 1 s) times every plain reposition/release `Move`
that has no duration of its own. The two `Probe` rows are exempt: contact
search needs however long that actually takes to reach contact, a function
of distance and speed, not a fixed budget -- though every `Probe` already
accepts a confirmed stall as contact by default (`Sequence.stall_fallback`).

**"raise"/"lower" get their own, longer deadline (`z_travel_seconds`), and
"raise" accepts a stall.** A first bench run at the default 30 mm offset
timed out on "raise" 1 s in, only ~5 mm short of goal: z's real climb speed
under `lift_effort` load measured around 25 mm/s on that run, well under the
no-load top speed `Sequence`'s own auto-derived margin assumes for a row with
no explicit `seconds` -- so neither `step_seconds` nor that auto margin is
long enough. Separately, `up_offset_mm` can legitimately put `up_z` past z's
50 mm travel limit depending on the start height (it did on that run), so
"raise" treats a confirmed stall against that hard stop as arrival
(`accept_stall=True`) rather than a fault -- whatever height it actually
reaches is fine, per the operator who watched that run.

No hardware results for this task yet beyond that one run: the sequence is
still not fully bench validated.
"""

from dataclasses import dataclass, field

import torch

from ..config import (AUX_JAW, AUX_LEFT, AUX_RIGHT, BASE_JAW, BASE_LEFT,
                      BASE_RIGHT, HandConfig, Z)
from ..primitives import Hold, Move, Probe, Sequence, lift_effort

FINGERS = (BASE_LEFT, BASE_RIGHT, AUX_LEFT, AUX_RIGHT)


@dataclass
class Config:
    """Bench units. Every `tune` field becomes a slider on the studio page."""

    label: str = "Run electric screwdriver"
    sets_datum: bool = False

    grip_torque: float = field(default=100.0, metadata={"tune": (30.0, 500.0)})
    """Torque both jaws close onto the tool with."""
    up_offset_mm: float = field(default=40.0, metadata={"tune": (10.0, 80.0)})
    """How far z rises, from the task's start height, before the second
    squeeze."""
    squeeze_hold_s: float = field(default=3.0, metadata={"tune": (1.0, 10.0)})
    """Seconds to hold each squeezed position before moving on."""
    release_clearance_mm: float = field(default=10.0, metadata={"tune": (2.0, 20.0)})
    """How far past the top gripper's own measured contact point it opens
    to release, relative to that grip's contact position."""
    step_seconds: float = field(default=1.0, metadata={"tune": (0.2, 5.0)})
    """Deadline for every plain reposition/release move that has no duration
    of its own -- see the module docstring for why `Probe` rows are exempt."""
    z_travel_seconds: float = field(default=10.0, metadata={"tune": (2.0, 20.0)})
    """Deadline for "raise" and "lower" -- longer than `step_seconds` because
    z's measured climb speed under load runs well under the no-load top speed
    the rest of this task's deadlines assume, and by a margin: one bench run
    measured ~25 mm/s, but that was one sample at one grip load, not a floor.
    Set generously (full 50 mm rail at 25 mm/s is ~2 s) rather than tuned
    tight to that sample, since "raise" hitting z's hard stop mid-deadline
    should end in a confirmed stall (see `accept_stall` below), not a timeout
    -- see the module docstring."""
    travel_torque: float = field(default=50.0, metadata={"tune": (30.0, 200.0)})
    """Torque for unloaded travel."""
    approach_torque: float = field(default=100.0, metadata={"tune": (50.0, 300.0)})
    """Torque while a jaw is still seeking contact, before `grip` takes over."""
    approach_speed: float = field(default=300.0, metadata={"tune": (25.0, 800.0)})
    """Servo speed register while closing a jaw to find contact, counts/s."""
    travel_speed: float = field(default=1500.0, metadata={"tune": (200.0, 1500.0)})
    """Unused: every free move now runs at the servo's rated no-load top
    speed regardless of this value."""
    timeout_margin: float = field(default=1.5, metadata={"tune": (1.0, 3.0)})


def build(hand: HandConfig, start_mm: torch.Tensor,
          cfg: Config | None = None) -> Sequence:
    """Squeeze both jaws, hold, cycle the top jaw up and back at a new height."""
    cfg = cfg or Config()
    jaw_floor = max(float(hand.gain_vector("torque_min_to_move")[dof])
                    for dof in (BASE_JAW, AUX_JAW))
    grip_effort = max(cfg.grip_torque, jaw_floor) / 1000.0
    seconds = cfg.step_seconds

    # Ready leaves z at mid travel, not this task's closed position -- close
    # it explicitly so the first squeeze, "raise", and "lower" all measure
    # from 0 instead of wherever ready happened to leave it.
    entry_z = 0.0
    up_z = entry_z + cfg.up_offset_mm

    return Sequence([
        # One-time setup: fingers pinch to a narrow point and are never
        # named again, so they hold there for the rest of the task.
        Move(label="fingers in", goal={FINGERS: 0.0}, seconds=seconds),
        # Nothing is gripped yet, so this is a free move, same as "raise"
        # below; z's own hard stop at 0 is the goal itself, so a confirmed
        # stall counts as arrival rather than a fault.
        Move(label="close z", goal={Z: entry_z}, accept_stall=True,
             seconds=cfg.z_travel_seconds),
        # Both jaws close together. Contact position recorded for the top
        # gripper's release below; the bottom gripper never releases, so
        # its own contact point is never needed.
        Probe(label="squeeze both", group=(BASE_JAW, AUX_JAW), creep=True,
              grip=grip_effort, measure={"aux_contact": AUX_JAW}),
        Hold(label="hold squeeze", seconds=cfg.squeeze_hold_s),
        # Top gripper opens; bottom gripper keeps its grip untouched --
        # this row never names it.
        Move(label="open top",
             goal={AUX_JAW: lambda m: m.aux_contact + cfg.release_clearance_mm},
             effort=grip_effort, loaded=True, seconds=seconds),
        # Free move -- top gripper is open, carrying nothing. Longer deadline
        # and `accept_stall`: real climb speed under load and z's own 50 mm
        # hard stop can both leave this short of `up_z` -- see module
        # docstring.
        Move(label="raise", goal={Z: up_z}, effort=lift_effort(hand),
             accept_stall=True, seconds=cfg.z_travel_seconds),
        # Squeeze the top gripper again at the new height.
        Probe(label="squeeze top", group=AUX_JAW, creep=True,
              grip=grip_effort, measure={"aux_contact": AUX_JAW}),
        Hold(label="hold squeeze again", seconds=cfg.squeeze_hold_s),
        # Top gripper opens again.
        Move(label="open top again",
             goal={AUX_JAW: lambda m: m.aux_contact + cfg.release_clearance_mm},
             effort=grip_effort, loaded=True, seconds=seconds),
        # Back down to the height the first squeeze happened at -- same
        # travel distance as "raise", so the same longer deadline applies.
        # Always inside z's travel range (unlike "raise"), so no
        # `accept_stall` here.
        Move(label="lower", goal={Z: entry_z}, seconds=cfg.z_travel_seconds),
    ], hand=hand,
       start_mm=start_mm,
       travel_torque=cfg.travel_torque,
       approach_torque=cfg.approach_torque,
       approach_speed=cfg.approach_speed,
       timeout_margin=cfg.timeout_margin,
       travel_speed=cfg.travel_speed)


# =============================================================================
# ARCHIVED, 2026-09-08 -- previous design: grip with one jaw at a time,
# alternating which one anchors the tool as z carries it down to the hard
# stop and back up to the start height (position for the first squeeze,
# squeeze the top gripper, creep z down to the hard stop carried by that
# grip, hold, hand off to the bottom gripper, a free reposition move up,
# hand back to the top gripper, slow climb back to the start height). Kept
# for reference; not wired to `build` above. To restore: rename this
# `Config`/`build` pair back to the live names (and the ones above to
# something else or delete them), and add `Loop` back to the `primitives`
# import if the commented-out loop body inside the old `build` is restored
# too.
# =============================================================================
#
# @dataclass
# class _ArchivedConfig:
#     """Bench units. Every `tune` field becomes a slider on the studio page."""
#
#     label: str = "Run electric screwdriver"
#     sets_datum: bool = False
#
#     grip_torque: float = field(default=100.0, metadata={"tune": (30.0, 500.0)})
#     """Torque either jaw closes onto the driver's body with."""
#     button_torque: float = field(default=300.0, metadata={"tune": (100.0, 1000.0)})
#     """Torque the top gripper closes onto the side button with -- needs to
#     be enough that it cannot slip off the button while carrying the tool."""
#     up_button_offset: float = field(default=15.0, metadata={"tune": (5.0, 40.0)})
#     """How far the top gripper (z) rises, from the task's start height,
#     before the first squeeze -- clearing the body grip out of the way and
#     lining the top gripper up on the tool."""
#     pre_squeeze_offset_mm: float = field(default=20.0, metadata={"tune": (5.0, 40.0)})
#     """How far z lowers, from `up_button_offset`, right before the first
#     squeeze -- 2 cm by default."""
#     screw_down_offset: float = field(default=15.0, metadata={"tune": (5.0, 40.0)})
#     """How far z feeds down, from `up_button_offset`, while the motor
#     drills."""
#     screw_hold_s: float = field(default=3.0, metadata={"tune": (1.0, 10.0)})
#     """Seconds to hold the fed-down depth once `screw_down_offset` is
#     reached, motor still running."""
#     bottom_hold_s: float = field(default=2.0, metadata={"tune": (1.0, 10.0)})
#     """Seconds to hold z at the fully closed position after the slow
#     descent, before re-gripping."""
#     z_slow_speed_mm_s: float = field(default=5.0, metadata={"tune": (2.0, 30.0)})
#     """z's commanded speed for the slow descent to bottom and the slow climb
#     back to the start height -- an explicit mm/s, not the shared creep
#     register (which every jaw's contact-seeking `Probe` also uses)."""
#     reposition_up_mm: float = field(default=50.0, metadata={"tune": (10.0, 60.0)})
#     """How far z rises, from wherever the descent stopped, once the top
#     gripper alone is carrying the tool -- 5 cm by default."""
#     release_clearance_mm: float = field(default=10.0, metadata={"tune": (2.0, 20.0)})
#     """How far past a jaw's own measured contact point it opens to release,
#     relative to that grip's contact position."""
#     step_seconds: float = field(default=1.0, metadata={"tune": (0.2, 5.0)})
#     """Deadline for every plain reposition/release move that has no duration
#     of its own -- see the module docstring for why `Probe` rows are exempt."""
#     cycles: float = field(default=0.0, metadata={"tune": (0.0, 20.0)})
#     """Complete loops, rounded to an integer. Zero repeats until the task is
#     stopped (disarm torque in Studio, or Ctrl-C headless)."""
#     travel_torque: float = field(default=50.0, metadata={"tune": (30.0, 200.0)})
#     """Torque for unloaded travel."""
#     approach_torque: float = field(default=100.0, metadata={"tune": (50.0, 300.0)})
#     """Torque while a jaw is still seeking contact, before `grip` takes over."""
#     approach_speed: float = field(default=300.0, metadata={"tune": (25.0, 800.0)})
#     """Servo speed register while closing a jaw to find contact, counts/s."""
#     travel_speed: float = field(default=1500.0, metadata={"tune": (200.0, 1500.0)})
#     """Unused: every free move now runs at the servo's rated no-load top
#     speed regardless of this value."""
#     timeout_margin: float = field(default=1.5, metadata={"tune": (1.0, 3.0)})
#
#
# def _archived_build(hand: HandConfig, start_mm: torch.Tensor,
#           cfg: "_ArchivedConfig | None" = None) -> Sequence:
#     """Grip with one jaw at a time, alternating which one anchors the tool
#     as z carries it down to the hard stop and back up to the start height.
#     """
#     cfg = cfg or _ArchivedConfig()
#     jaw_floor = max(float(hand.gain_vector("torque_min_to_move")[dof])
#                     for dof in (BASE_JAW, AUX_JAW))
#     grip_effort = max(cfg.grip_torque, jaw_floor) / 1000.0
#     seconds = cfg.step_seconds
#     # `creep_speed` is the shared register every jaw's contact-seeking
#     # `Probe` also runs at; z's slow travel wants its own explicit mm/s
#     # instead, so this scales that same register just for the rows below.
#     z_slow_scale = cfg.z_slow_speed_mm_s * hand.counts_per_mm / cfg.approach_speed
#
#     entry_z = start_mm[:, Z]
#     up_z = entry_z + cfg.up_button_offset
#     pre_squeeze_z = up_z - cfg.pre_squeeze_offset_mm
#
#     return Sequence([
#         # One-time setup: fingers pinch to a narrow point and are never
#         # named again, so they hold there for the rest of the task.
#         Move(label="fingers in", goal={FINGERS: 0.0}, seconds=seconds),
#         # Clear the top gripper up before squeezing -- also where it
#         # returns to become the button height later (up_z + reposition_up_mm).
#         Move(label="raise top gripper", goal={Z: up_z},
#              effort=lift_effort(hand), seconds=seconds),
#         # Back down a little from the button height, right before squeezing
#         # -- descent, so no lift floor needed; gravity assists.
#         Move(label="lower top gripper", goal={Z: pre_squeeze_z},
#              seconds=seconds),
#         # Top gripper only -- bottom gripper (DOF0) stays open. Top gripper
#         # rides on z, so gripping here is what lets it carry the tool down
#         # as z descends, rather than z sliding empty past a tool anchored
#         # from below.
#         Probe(label="squeeze top", group=AUX_JAW, creep=True,
#               grip=grip_effort, measure={"aux_contact": AUX_JAW}),
#         # Slow descent (z_slow_speed_mm_s), carried by the top gripper's
#         # grip, all the way to the hard stop. `loaded=True`: this is not a
#         # free jaw sweep, so effort is floored and a confirmed stall counts
#         # as arrival -- the screw's own resistance stopping it short of 0 is
#         # arrival, not a fault. No fixed deadline: at this speed, z's full
#         # travel needs more than `step_seconds`, so this one keeps the
#         # auto-derived margin budget instead. Measured, since the rows below
#         # need wherever it actually stopped, not literal 0.
#         Move(label="descend to bottom", goal={Z: 0.0}, creep=True,
#              speed_scale=z_slow_scale, loaded=True, measure={"z_bottom": Z}),
#         # Settle at the bottom for a couple seconds before switching grips.
#         Hold(label="hold at bottom", group=Z, goal=lambda m: m.z_bottom,
#              seconds=cfg.bottom_hold_s),
#         # Bottom gripper takes over at the bottom -- squeeze it before
#         # opening the top gripper, so the tool is never held by neither.
#         Probe(label="squeeze bottom", group=BASE_JAW, creep=True,
#               grip=grip_effort, measure={"base_contact": BASE_JAW}),
#         # Top gripper lets go; bottom gripper alone now holds the tool at
#         # the bottom while z (with the now-empty top gripper) is free to
#         # move on its own.
#         Move(label="unsqueeze top",
#              goal={AUX_JAW: lambda m: m.aux_contact + cfg.release_clearance_mm},
#              effort=grip_effort, loaded=True, seconds=seconds),
#         # Free move -- top gripper is open, carrying nothing. Back up from
#         # wherever the descent actually stopped, not from a literal 0, in
#         # case the confirmed stall ended it short.
#         Move(label="raise top gripper again",
#              goal={Z: lambda m: m.z_bottom + cfg.reposition_up_mm},
#              effort=lift_effort(hand), seconds=seconds),
#         # Top gripper grips again at the new height, before the bottom
#         # gripper lets go.
#         Probe(label="squeeze top again", group=AUX_JAW, creep=True,
#               grip=grip_effort),
#         # Bottom gripper lets go; top gripper alone carries the tool the
#         # rest of the way.
#         Move(label="unsqueeze bottom",
#              goal={BASE_JAW: lambda m: m.base_contact + cfg.release_clearance_mm},
#              effort=grip_effort, loaded=True, seconds=seconds),
#         # Slow climb (z_slow_speed_mm_s) straight back to the task's start
#         # height, carried by the top gripper. No fixed deadline, same reason
#         # as the descent above.
#         Move(label="raise to start", goal={Z: entry_z}, creep=True,
#              speed_scale=z_slow_scale, loaded=True),
#         # --- rest of the loop this design was heading toward, never wired
#         # in -- see the archived note above for how far it got. ---
#         # button_effort = max(cfg.button_torque, jaw_floor) / 1000.0
#         # down_z = up_z - cfg.screw_down_offset
#         # reposition_z = down_z + cfg.reposition_up_mm
#         # cycles = float("inf") if cfg.cycles <= 0 else max(1, round(cfg.cycles))
#         # Loop(count=cycles, rows=[
#         #     # Feed down while the motor drills. `accept_stall`: the screw's
#         #     # own resistance stopping the feed short of depth is arrival,
#         #     # not a fault. Full travel speed, not creep: the motor supplies
#         #     # the actual drive, z only needs to keep pace, and creep speed
#         #     # cannot cover `screw_down_offset` inside `step_seconds`.
#         #     Move(label="feed", goal={Z: down_z}, accept_stall=True,
#         #          seconds=seconds),
#         #     Hold(label="hold depth", group=Z, goal=down_z,
#         #          seconds=cfg.screw_hold_s),
#         #     # Bottom gripper closes again -- both jaws holding once more.
#         #     # Re-measured: this is the contact position the later release
#         #     # (after the button press) computes from.
#         #     Probe(label="close bottom", group=BASE_JAW, creep=True,
#         #           grip=grip_effort, measure={"base_contact": BASE_JAW}),
#         #     # Top gripper lets go; bottom gripper alone now holds the body.
#         #     Move(label="release top",
#         #          goal={AUX_JAW: lambda m: m.aux_contact + cfg.release_clearance_mm},
#         #          effort=grip_effort, loaded=True, seconds=seconds),
#         #     Move(label="reposition top gripper", goal={Z: reposition_z},
#         #          effort=lift_effort(hand), seconds=seconds),
#         #     # Same close as `squeeze both`, but at a height where what is
#         #     # physically there is the side button, not the body -- see the
#         #     # module docstring.
#         #     Probe(label="press button", group=AUX_JAW, creep=True,
#         #           grip=button_effort, measure={"button_contact": AUX_JAW}),
#         #     Move(label="release bottom again",
#         #          goal={BASE_JAW: lambda m: m.base_contact + cfg.release_clearance_mm},
#         #          effort=grip_effort, loaded=True, seconds=seconds),
#         #     # Straight back to the task's start height, top gripper still on
#         #     # the button -- this carries the tool up with it. `loaded=True`:
#         #     # the ascent is done through that grip, not a free jaw sweep.
#         #     Move(label="lift to start", goal={Z: entry_z},
#         #          effort=lift_effort(hand), loaded=True, seconds=seconds),
#         #     # Bottom gripper closes again, ready to be the sole hold for the
#         #     # next loop.
#         #     Probe(label="close bottom again", group=BASE_JAW, creep=True,
#         #           grip=grip_effort),
#         #     # Top gripper releases the button -- loop ends exactly where the
#         #     # next pass's first row assumes: bottom gripped, top open.
#         #     Move(label="release button",
#         #          goal={AUX_JAW: lambda m: m.button_contact + cfg.release_clearance_mm},
#         #          effort=button_effort, loaded=True, seconds=seconds),
#         # ]),
#     ], hand=hand,
#        start_mm=start_mm,
#        travel_torque=cfg.travel_torque,
#        approach_torque=cfg.approach_torque,
#        approach_speed=cfg.approach_speed,
#        timeout_margin=cfg.timeout_margin,
#        travel_speed=cfg.travel_speed)

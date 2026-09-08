"""Run an electric screwdriver's own motor to feed a screw in.

    entry (z parked, both jaws half open, fingers untouched)
      -> grip driver body (taut, base jaw)
      -> press trigger (aux jaw)
      -> feed z down while the motor drives the screw, up to a fixed timer
      -> release trigger a little
      -> reposition z up (actively driven)
      -> relax z (drop its holding effort)
      -> press trigger again -- z is left slack, so the screw's own
         self-feeding pulls it down rather than z's held position fighting it

Single direction throughout: there is no separate cw/ccw button pair here,
only one trigger the aux jaw presses twice. Fingers (DOF 1, 2, 5, 6) are
never named by any row and stay wherever they started -- inactive for this
task.

**The feed is a `Hold`, not a `Move`, and that is deliberate.** A `Move`
that never reaches its goal and never registers a confirmed stall (see
`primitives.STUCK_SPEED_MM_S`/`CONFIRM_TICKS`) times out as a FAILURE, which
retires the whole `Sequence` for that environment -- `Sequence.step` folds a
failed row straight into `done`, so nothing after it runs. That is the
opposite of what was asked: the feed should push for up to
`screw_feed_timeout_s` and then move on regardless of whether the screw
finished seating. `Hold` never fails on its own timeout -- it just keeps
commanding the push for its `seconds` and then advances -- so it is the
row that actually gives that guarantee, not a `Move` with `accept_stall`
(which still fails if the joint keeps almost-moving without ever settling
into either arrival or a confirmed stall).

**The feed's torque is explicit and deliberately not floored.** `_effort`
in `primitives.Sequence` only raises a row to `torque_min_to_move` when the
row gives no effort of its own, or is `loaded=True`; an explicit number
below the floor is understood there as a row naming its own number on
purpose. Every other DOF here (the two jaw grips, the reposition lift) still
goes through the usual floor/`lift_effort` treatment -- only this one push
is meant to stay as light as `screw_feed_torque` says, same contract the
previous version of this task used.

**Relaxing z before the second trigger press is the point of the whole
second half.** After the feed `Hold`, z is still actively holding whatever
position it last commanded -- if that standing command were left alone, it
would keep opposing any motion once the trigger drives the screw again. The
explicit low-effort `Hold` in between removes that standing resistance so
the screw's own thread engagement, not z's servo, decides where z ends up
next.

No hardware results for this task yet: the sequence is written, not bench
run.
"""

from dataclasses import dataclass, field

import torch

from ..config import AUX_JAW, BASE_JAW, HandConfig, Z
from ..primitives import Hold, Move, Probe, Sequence, lift_effort

JAWS = (BASE_JAW, AUX_JAW)


@dataclass
class Config:
    """Bench units. Every `tune` field becomes a slider on the studio page."""

    label: str = "Run electric screwdriver"
    sets_datum: bool = False

    entry_z: float = field(default=40.0, metadata={"tune": (10.0, 90.0)})
    """Absolute z height, mm above z zero, both jaws close onto the driver
    at -- the initialization height."""
    base_grip_torque: float = field(default=30.0, metadata={"tune": (10.0, 100.0)})
    """Torque the base jaw closes onto the driver body with, and holds for
    the whole task -- taut enough to steady the tool, light enough that z
    can still slide the body through the grip as it feeds the bit."""
    trigger_torque: float = field(default=300.0, metadata={"tune": (100.0, 1000.0)})
    """Torque the aux jaw closes onto the trigger with, both times it
    presses it -- needs to be enough that the jaw cannot slip off the
    trigger under z's reaction force while feeding."""
    screw_feed_mm: float = field(default=15.0, metadata={"tune": (5.0, 40.0)})
    """How far z feeds down, from `entry_z`, once the trigger is pressed."""
    screw_feed_torque: float = field(default=50.0, metadata={"tune": (10.0, 200.0)})
    """Constant z torque for the feed -- light and explicit, not floored to
    `torque_min_to_move`; see the module docstring."""
    screw_feed_timeout_s: float = field(default=12.0, metadata={"tune": (2.0, 30.0)})
    """Maximum time the feed keeps pushing before moving on regardless of
    whether the screw finished seating."""
    release_clearance_mm: float = field(default=10.0, metadata={"tune": (2.0, 20.0)})
    """How far past the trigger's measured contact point the aux jaw opens
    to release it, stopping the motor before repositioning."""
    reposition_mm: float = field(default=15.0, metadata={"tune": (5.0, 40.0)})
    """How far z is actively raised, from where the feed targeted, after
    releasing the trigger and before it is relaxed."""
    relax_torque: float = field(default=0.0, metadata={"tune": (0.0, 50.0)})
    """z's effort once repositioned and the trigger is pressed the second
    time -- deliberately low so the screw's own feed pulls z down rather
    than z's held position fighting it."""
    z_speed_scale: float = field(default=3.0, metadata={"tune": (1.0, 6.0)})
    """Multiply z's creep speed by this for the feed."""
    travel_torque: float = field(default=50.0, metadata={"tune": (30.0, 200.0)})
    """Torque for unloaded travel."""
    approach_torque: float = field(default=100.0, metadata={"tune": (50.0, 300.0)})
    """Torque while a jaw is still seeking contact, before `grip` takes over."""
    approach_speed: float = field(default=300.0, metadata={"tune": (25.0, 800.0)})
    """Servo speed register while closing a jaw to find contact, and while
    creeping z, counts/s."""
    travel_speed: float = field(default=1500.0, metadata={"tune": (200.0, 1500.0)})
    """Unused: every free move now runs at the servo's rated no-load top
    speed regardless of this value."""
    timeout_margin: float = field(default=1.5, metadata={"tune": (1.0, 3.0)})


def build(hand: HandConfig, start_mm: torch.Tensor,
          cfg: Config | None = None) -> Sequence:
    """Park z, half-open both jaws, then grip the driver and feed the screw."""
    cfg = cfg or Config()
    mm = hand.clamped_mm
    jaw_floor = max(float(hand.gain_vector("torque_min_to_move")[dof])
                    for dof in JAWS)
    base_grip_effort = max(cfg.base_grip_torque, jaw_floor) / 1000.0
    trigger_effort = max(cfg.trigger_torque, jaw_floor) / 1000.0
    feed_effort = cfg.screw_feed_torque / 1000.0
    relax_effort = cfg.relax_torque / 1000.0

    entry_z = mm(Z, cfg.entry_z)
    base_half = mm(BASE_JAW, hand.travel_mm[BASE_JAW] / 2.0)
    aux_half = mm(AUX_JAW, hand.travel_mm[AUX_JAW] / 2.0)
    down_z = mm(Z, cfg.entry_z - cfg.screw_feed_mm)
    up_z = mm(Z, cfg.entry_z - cfg.screw_feed_mm + cfg.reposition_mm)

    return Sequence([
        # z on its own value, jaws half open: fingers (1, 2, 5, 6) are never
        # named here or below, so they stay wherever they started.
        Move(label="entry",
             goal={Z: entry_z, BASE_JAW: base_half, AUX_JAW: aux_half}),
        # Taut hold on the driver body -- steadies it without pinning it, so
        # z can still slide the body through the grip as it feeds.
        Probe(label="grip body", group=BASE_JAW, creep=True,
              grip=base_grip_effort),
        # Close onto the trigger and hold it -- the jaw that keeps the motor
        # running. Contact position is recorded, not assumed to be 0, so the
        # release below knows where it actually stopped.
        Probe(label="press trigger", group=AUX_JAW, creep=True,
              grip=trigger_effort, measure={"trigger": AUX_JAW}),
        # Keep pushing z down for up to `screw_feed_timeout_s`, then move on
        # regardless -- a `Hold`, not a `Move`, is what makes that guarantee
        # (see module docstring). The aux jaw keeps holding the trigger
        # throughout: this row never names it.
        Hold(label="feed", group=Z, goal=down_z, effort=feed_effort,
             creep=True, speed_scale=cfg.z_speed_scale,
             seconds=cfg.screw_feed_timeout_s),
        # Let go of the trigger to stop the motor before repositioning.
        # `loaded=True` floors the effort at travel torque and accepts a
        # confirmed stall -- retracting against the grip that just closed
        # needs at least as much torque as made it.
        Move(label="release trigger",
             goal={AUX_JAW: lambda m: m.trigger + cfg.release_clearance_mm},
             effort=trigger_effort, loaded=True, creep=True),
        # Actively raise z back up, floored at the lift effort so it
        # actually clears against gravity.
        Move(label="reposition", goal={Z: up_z}, effort=lift_effort(hand)),
        # Drop z's holding effort so it stops opposing whatever the screw's
        # own feed does next -- see the module docstring.
        Hold(label="relax z", group=Z, goal=up_z, effort=relax_effort),
        # Press the trigger again; z is left slack, so the screw drives
        # itself from here.
        Probe(label="press trigger again", group=AUX_JAW, creep=True,
              grip=trigger_effort),
    ], hand=hand,
       start_mm=start_mm,
       travel_torque=cfg.travel_torque,
       approach_torque=cfg.approach_torque,
       approach_speed=cfg.approach_speed,
       timeout_margin=cfg.timeout_margin,
       travel_speed=cfg.travel_speed)

"""Find each DOF's hard stop and record it as the zero offset.

Runs in raw counts because mm are undefined until this completes. Phases are
ordered so the fingers retract before the jaws close and the jaws clear before
the z stage drops, which keeps the mechanism from binding on itself.
"""

from dataclasses import dataclass
from typing import Optional

from .primitives import wait_for_stall_counts, wait_until_counts

DESCRIPTION = "Zero all DOFs against their hard stops"


@dataclass
class Config:
    dof: Optional[int] = None
    """Zero a single DOF instead of running the full sequence. Debug aid."""
    zero_torque: int = 50
    """Torque used while creeping into the hard stop. Keep it low."""
    creep_speed: int = 50
    """Speed used while seeking the stop."""
    save: bool = True
    """Write the resulting offsets to the calibration file."""

# Phase order matters mechanically: x fingers, then y jaws, then z.
PHASES = [("x fingers", [1, 2, 5, 6]),
          ("y jaws", [0, 4]),
          ("z stage", [3])]

OVERTRAVEL_COUNTS = 10000     # far enough past the stop that the servo keeps pressing


def zero_all(hand, stall_threshold: int = 5, confirm_count: int = 2,
             zero_torque: int = 50, creep_speed: int = 50,
             transit_speed: int = 100, transit_acc: int = 20,
             transit_torque: int = 400, transit_tolerance: int = 80,
             transit_timeout: float = 5.0, stall_timeout: float = 30.0,
             save: bool = True) -> bool:
    """Drive every DOF to its hard stop, phase by phase, then park at mid travel.

    Returns True if all DOFs zeroed. Torque is left enabled so the caller can
    move immediately; use hand.release() when done.
    """
    cfg = hand.config

    # The control loop would fight the raw creep commands below, overwriting
    # them with its own target vector mid-approach.
    hand.stop_loop()

    # Phases write offsets as they succeed. If a later phase fails we must put
    # back what was there, or the hand is left with some axes on new offsets and
    # some on stale ones, still flagged as zeroed and still accepting motion.
    saved_offsets = hand.zero_offset.copy()
    saved_is_zeroed = hand.is_zeroed
    hand.is_zeroed = False

    def restore(reason):
        hand.zero_offset[:] = saved_offsets
        hand.is_zeroed = saved_is_zeroed
        print(f"  {reason} Offsets rolled back to "
              f"{'the previous calibration' if saved_is_zeroed else 'unzeroed'}.")
        return False

    hand.servo.enable_torques(cfg.servo_ids, True)

    for name, dof_ids in PHASES:
        print(f"[{hand.name}] {name}: seeking hard stop")

        for d in dof_ids:
            sid = cfg[d].servo_id
            here = hand.servo.read_position(sid)
            if here is None:
                return restore(f"DOF {d} (servo {sid}) not responding, aborting.")
            # Command far past the stop so the servo stays loaded against it.
            hand.servo.set_position(sid, here - cfg[d].orientation * OVERTRAVEL_COUNTS,
                                    creep_speed, transit_acc, zero_torque)

        stops = wait_for_stall_counts(hand, dof_ids, stall_threshold=stall_threshold,
                                      confirm_count=confirm_count, timeout=stall_timeout)
        missing = [d for d, c in stops.items() if c is None]
        if missing:
            # Recording a zero for a DOF that never stalled would put the origin
            # mid-travel and silently corrupt every later mm command.
            return restore(f"DOFs {missing} never reached a hard stop, aborting.")
        for d, counts in stops.items():
            hand.zero_offset[d] = counts

        print(f"[{hand.name}] {name}: moving to mid travel")
        # Safe to use mm here: these DOFs now have valid offsets.
        counts = [cfg.mm_to_counts(d, cfg[d].max_mm / 2, hand.zero_offset[d])
                  for d in dof_ids]
        hand.servo.set_positions([cfg[d].servo_id for d in dof_ids], counts,
                                 transit_speed, transit_acc, transit_torque)
        if not wait_until_counts(hand, dof_ids, counts, transit_tolerance, transit_timeout):
            print(f"  warning: DOFs {dof_ids} did not reach mid travel")

    hand.is_zeroed = True
    print(f"[{hand.name}] zeroed: {hand.zero_offset.tolist()}")
    if save:
        print(f"[{hand.name}] saved to {hand.save_calibration()}")
    return True


def zero_single(hand, dof_id: int, stall_threshold: int = 5, confirm_count: int = 5,
                zero_torque: int = 50, creep_speed: int = 50,
                stall_timeout: float = 30.0):
    """Zero one DOF. Debug aid: does not mark the hand as zeroed or save."""
    cfg = hand.config
    sid = cfg[dof_id].servo_id
    hand.servo.enable_torque(sid, True)

    here = hand.servo.read_position(sid)
    if here is None:
        raise RuntimeError(f"DOF {dof_id} (servo {sid}) not responding")

    print(f"[{hand.name}] zeroing DOF {dof_id} from {here}")
    hand.servo.set_position(sid, here - cfg[dof_id].orientation * OVERTRAVEL_COUNTS,
                            creep_speed, 20, zero_torque)
    stops = wait_for_stall_counts(hand, [dof_id], stall_threshold=stall_threshold,
                                  confirm_count=confirm_count, timeout=stall_timeout)
    counts = stops[dof_id]
    if counts is None:
        raise RuntimeError(f"DOF {dof_id} never stalled")
    hand.zero_offset[dof_id] = counts
    print(f"DOF {dof_id} zeroed at {counts} (not saved; run without --dof to save)")
    return counts


def run(hand, cfg: Config):
    try:
        if cfg.dof is not None:
            zero_single(hand, cfg.dof, zero_torque=cfg.zero_torque,
                        creep_speed=cfg.creep_speed)
        else:
            zero_all(hand, zero_torque=cfg.zero_torque,
                     creep_speed=cfg.creep_speed, save=cfg.save)
    finally:
        hand.release()

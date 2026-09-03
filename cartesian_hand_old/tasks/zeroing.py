"""Find each DOF's hard stop and record it as the zero offset.

Runs in raw counts because mm are undefined until this completes. Phases are
ordered so fingers retract before jaws close and jaws clear before z drops.
"""
from dataclasses import dataclass

from ..hand import ZEROING_TORQUE
from .primitives import wait_for_stall_counts, wait_until_counts

DESCRIPTION = "Zero all DOFs against their hard stops"


@dataclass
class Config:
    save: bool = True


# Each phase drives its DOFs past the hard stop by ~120mm and records the
# stall count as that DOF's zero offset.
PHASES = [
    ([1, 2, 5, 6], "fingers"),
    ([0, 4],       "jaws"),
    ([3],          "z"),
]


def zero_all(hand, save: bool = True, stall_timeout: float = 30.0) -> bool:
    """Drive every DOF to its hard stop. Returns True on success.

    Hardware implementation: a Python loop over PHASES, driving raw counts.
    Sim uses `zero_program()` in `sim_zeroing.py` for the batched path; the
    two share PHASES and OVERTRAVEL but the driver is different because
    pre-zero mm is undefined on hardware.
    """
    cfg = hand.config
    hand.stop_loop()
    saved, was_zeroed = hand.zero_offset.copy(), hand.is_zeroed
    hand.is_zeroed = False
    hand.servo.enable_torques(cfg.servo_ids, True)

    for dof_ids, name in PHASES:
        sids = [cfg[d].servo_id for d in dof_ids]
        # One sync-read packet for the whole phase, not one unicast per DOF.
        here_all = hand.servo.read_all(sids)
        if any(r is None or r[0] is None for r in here_all):
            hand.zero_offset[:], hand.is_zeroed = saved, was_zeroed
            print(f"  {name}: servos silent. Rolled back.")
            return False
        here = [r[0] for r in here_all]
        targets = [int(here[i] - cfg[d].orientation * 10000) for i, d in enumerate(dof_ids)]
        # Creep speed 60 (was 120): high-speed creep at torque=50 builds
        # momentum that the stall window can't catch in time, and the joint
        # over-shoots the stop and climbs a gear tooth.
        hand.servo.set_positions(sids, targets,
                                 [60] * len(dof_ids), [200] * len(dof_ids),
                                 [ZEROING_TORQUE[d] for d in dof_ids])
        stops = wait_for_stall_counts(hand, dof_ids,
                                       stall_speed=2.0, confirm_s=1.0,
                                       timeout=stall_timeout)
        if None in stops.values():
            hand.zero_offset[:], hand.is_zeroed = saved, was_zeroed
            print(f"  {name}: never stalled. Rolled back.")
            return False
        for d, counts in stops.items():
            hand.zero_offset[d] = counts
        # Park at mid travel, then drop torque. Without dropping torque the
        # servos sit at torque=400 against the stop until the next operation,
        # pressing the mechanism.
        mid = [cfg.mm_to_counts(d, cfg[d].max_mm / 2, hand.zero_offset[d]) for d in dof_ids]
        hand.servo.set_positions(sids, mid, [500] * len(dof_ids), [200] * len(dof_ids), [400] * len(dof_ids))
        wait_until_counts(hand, dof_ids, mid)
        hand.servo.enable_torques(sids, False)
        print(f"  {name}: {[int(c) for c in stops.values()]}")

    hand.is_zeroed = True
    print(f"[{hand.name}] zeroed: {hand.zero_offset.tolist()}")
    if save:
        print(f"[{hand.name}] saved to {hand.save_calibration()}")
    return True


def run(hand, cfg: Config):
    try:
        return zero_all(hand, save=cfg.save)
    finally:
        hand.release()

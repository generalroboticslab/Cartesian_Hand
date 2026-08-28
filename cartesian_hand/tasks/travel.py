"""Measure each DOF's full travel by driving it to the stop opposite its zero.

Zeroing finds one hard stop per DOF and calls it the origin. That fixes where
travel starts but says nothing about how far it runs, so `max_mm` has always
been an assumption. This drives the other way until each DOF stalls again and
reports the span between the two stops.

The span is reported in counts, which is exact. Converting it to millimetres
needs `counts_per_mm`, which is the very thing in doubt, so the mm column here
is only what the current configured value implies. Measure the real distance
with calipers and divide: counts_per_mm = span_counts / measured_mm.
"""

from dataclasses import dataclass
from typing import Optional

from .primitives import wait_for_stall_counts
from .zeroing import OVERTRAVEL_COUNTS, PHASES

DESCRIPTION = "Measure full travel by seeking the stop opposite zero"


@dataclass
class Config:
    dof: Optional[int] = None
    """Measure a single DOF instead of all of them."""
    creep_speed: int = 50
    """Speed used while seeking the far stop. Keep it low."""
    stall_timeout: float = 45.0
    """Per-phase budget. Longer than zeroing's: this crosses the full travel."""
    restore: bool = True
    """Return each DOF to mid travel when done, instead of leaving it at the stop."""


def measure(hand, dof_ids, creep_speed: int = 50, stall_timeout: float = 45.0) -> dict:
    """Drive dof_ids away from zero until they stall. Returns {dof_id: span_counts}.

    None for any DOF that never stalled, so a timeout is never mistaken for a
    measurement.
    """
    cfg = hand.config
    for d in dof_ids:
        # Away from zero is the +mm direction, which is +orientation in counts.
        # Overshoot the expected stop so the servo stays loaded against it.
        target = hand.zero_offset[d] + cfg[d].orientation * OVERTRAVEL_COUNTS
        # Use the hand's own per-DOF torque: the z stage cannot climb on the
        # value the horizontal DOFs use, and this is the direction that lifts.
        hand.servo.set_position(cfg[d].servo_id, target, creep_speed, 20,
                                hand.gains(d)["torque"])

    stops = wait_for_stall_counts(hand, dof_ids, timeout=stall_timeout)
    return {d: (None if c is None else abs(c - int(hand.zero_offset[d])))
            for d, c in stops.items()}


# A DOF that stops because it arrived at the commanded position looks exactly
# like one that stopped against a hard stop: both simply cease moving. The
# command is placed OVERTRAVEL_COUNTS past zero, so a span at that value means
# the travel is at least that long and was never actually bounded.
def _reached_command(span: int, slack: int = 20) -> bool:
    return span is not None and span >= OVERTRAVEL_COUNTS - slack


def run(hand, cfg: Config):
    hand.require_zeroed()
    # Raw count commands below; the control loop would overwrite them.
    hand.stop_loop()
    hand.servo.enable_torques(hand.config.servo_ids, True)

    # Reverse of the zeroing order. Zeroing retracts fingers, closes jaws, then
    # drops z; unwinding in the same order would drive z up into a mechanism
    # that is still folded.
    phases = ([("dof %d" % cfg.dof, [cfg.dof])] if cfg.dof is not None
              else list(reversed(PHASES)))

    spans = {}
    try:
        for name, dof_ids in phases:
            print(f"[{hand.name}] {name}: seeking far stop")
            spans.update(measure(hand, dof_ids, cfg.creep_speed, cfg.stall_timeout))

        cpm = hand.config.geometry.counts_per_mm
        print(f"\n[{hand.name}] travel, counts_per_mm={cpm:.2f}\n")
        print(f"{'DOF':<5}{'label':<26}{'span counts':>12}{'implied mm':>12}{'max_mm':>9}")
        print("-" * 64)
        for d in range(hand.n_dof):
            span = spans.get(d)
            label = hand.config[d].label[:25]
            if span is None:
                print(f"{d:<5}{label:<26}{'no stall':>12}{'-':>12}"
                      f"{hand.config[d].max_mm:>9.1f}")
                continue
            flag = "  unbounded" if _reached_command(span) else ""
            print(f"{d:<5}{label:<26}{span:>12}{span / cpm:>12.1f}"
                  f"{hand.config[d].max_mm:>9.1f}{flag}")

        unbounded = [d for d, s in spans.items() if _reached_command(s)]
        if unbounded:
            print(f"\nDOFs {unbounded} stopped at the commanded position rather "
                  f"than a hard stop, so their span is a lower bound, not a "
                  f"measurement. Re-run with a larger OVERTRAVEL_COUNTS.")

        measured = [s for d, s in spans.items()
                    if s is not None and not _reached_command(s)]
        if measured:
            print(f"\nMeasure one DOF's travel with calipers, then set "
                  f"counts_per_mm = span_counts / measured_mm in hands.py.")
            print("Nothing is saved: this is a measurement, not a calibration.")

        if cfg.restore:
            print(f"\n[{hand.name}] returning to mid travel")
            hand.set_pos([hand.config[d].max_mm / 2 for d in range(hand.n_dof)],
                         timeout=10.0)
    finally:
        hand.release()

    return spans

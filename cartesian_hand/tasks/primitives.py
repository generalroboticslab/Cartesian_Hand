"""Motion primitives shared by tasks.

Stall detection was previously reimplemented in zeroing, caps_contact_based and
primitives, with three different sets of thresholds and three different bugs.
There is one implementation here, in mm, plus a counts-level variant for use
before zero offsets exist and mm is meaningless.
"""

import time

import numpy as np


def wait_for_stall(hand, dof_ids, stall_threshold: float = 0.5,
                   confirm_count: int = 3, timeout: float = 10.0,
                   poll: float = 0.05, label: str = "stall") -> dict:
    """Block until every DOF stops moving. Returns {dof_id: position_mm}.

    A DOF counts as stalled once it moves less than stall_threshold mm for
    confirm_count consecutive polls. DOFs still moving at timeout are reported
    at their last position, so callers always get an entry for every DOF and
    never have to handle a partial dict.
    """
    dof_ids = list(dof_ids)
    prev = {d: hand.positions[d] for d in dof_ids}
    consecutive = {d: 0 for d in dof_ids}
    stalled = {}
    deadline = time.time() + timeout

    while len(stalled) < len(dof_ids):
        if time.time() > deadline:
            pos = hand.positions
            pending = [d for d in dof_ids if d not in stalled]
            print(f"  {label}: timeout on DOFs {pending}")
            for d in pending:
                stalled[d] = pos[d]
            break

        time.sleep(poll)
        current = hand.positions
        for d in dof_ids:
            if d in stalled:
                continue
            if abs(current[d] - prev[d]) < stall_threshold:
                consecutive[d] += 1
                if consecutive[d] >= confirm_count:
                    stalled[d] = current[d]
                    print(f"  DOF {d} {label} at {current[d]:.1f}mm")
            else:
                consecutive[d] = 0
            prev[d] = current[d]

    return {d: stalled[d] for d in dof_ids}


def approach(hand, dof_ids, torque: int = 150, speed: int = 50,
             target_mm: float = 0.0, **kwargs) -> dict:
    """Drive DOFs toward target_mm at reduced torque until they stall on contact.

    Returns {dof_id: contact_mm}. For a jaw closing on an object, the contact
    position is the object's radius.
    """
    hand.require_zeroed()
    hand.enable()
    hand.set_gains(dof_ids, speed=speed, torque=torque)
    hand.hold(dof_ids, torque=torque, position_mm=target_mm)
    return wait_for_stall(hand, dof_ids, label="contact", **kwargs)


def squeeze(hand, dof_ids, torque: int, position_mm: float = 0.0):
    """Hold DOFs against an object at a given torque."""
    hand.hold(dof_ids, torque=torque, position_mm=position_mm)


def squeeze_until_stall(hand, dof_ids, torque: int, **kwargs) -> dict:
    squeeze(hand, dof_ids, torque)
    return wait_for_stall(hand, dof_ids, label="squeeze", **kwargs)


# ── Pre-calibration variant ───────────────────────────────────────────────────

def wait_for_stall_counts(hand, dof_ids, stall_threshold: int = 5,
                          confirm_count: int = 2, timeout: float = 30.0,
                          poll: float = 0.1, verbose: bool = True) -> dict:
    """Stall detection in raw servo counts, for use before zeroing.

    Returns {dof_id: counts}, with None for any DOF that never stalled or whose
    servo never answered. None must stay None: reporting a timed-out DOF at its
    last position would record a zero offset in the middle of travel, and every
    subsequent mm command on that axis would be wrong.
    """
    dof_ids = list(dof_ids)
    read = lambda d: hand.servo.read_position(hand.config[d].servo_id)

    prev = {d: read(d) for d in dof_ids}
    consecutive = {d: 0 for d in dof_ids}
    stalled = {}
    deadline = time.time() + timeout

    while len(stalled) < len(dof_ids):
        if time.time() > deadline:
            pending = [d for d in dof_ids if d not in stalled]
            print(f"  timeout: DOFs {pending} never reached a hard stop")
            for d in pending:
                stalled[d] = None
            break

        time.sleep(poll)
        for d in dof_ids:
            if d in stalled:
                continue
            actual = read(d)
            if actual is None or prev[d] is None:
                # Dropped frame: no movement estimate this round, keep waiting.
                consecutive[d] = 0
                prev[d] = actual
                continue
            movement = abs(actual - prev[d])
            prev[d] = actual
            if verbose:
                print(f"  DOF {d} | counts: {actual} | movement: {movement}")
            if movement < stall_threshold:
                consecutive[d] += 1
                if consecutive[d] >= confirm_count:
                    stalled[d] = actual
                    print(f"  DOF {d} hard stop at {actual}")
            else:
                consecutive[d] = 0

    return {d: stalled.get(d) for d in dof_ids}


def wait_until_counts(hand, dof_ids, counts, tolerance: int = 80,
                      timeout: float = 5.0, poll: float = 0.02) -> bool:
    """Wait for DOFs to reach raw count targets. Returns False on timeout."""
    deadline = time.time() + timeout
    targets = dict(zip(dof_ids, counts))
    while time.time() < deadline:
        # One read per DOF per round. The previous version read twice in the
        # same expression and null-checked a different read than it compared.
        actual = {d: hand.servo.read_position(hand.config[d].servo_id) for d in dof_ids}
        if all(a is not None and abs(a - targets[d]) <= tolerance
               for d, a in actual.items()):
            return True
        time.sleep(poll)
    return False

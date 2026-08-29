"""Motion primitives shared by tasks.

Stall detection was previously reimplemented in zeroing, caps_contact_based and
primitives, with three different sets of thresholds and three different bugs.
There is one implementation here, in mm, plus a counts-level variant for use
before zero offsets exist and mm is meaningless.
"""

import time

import numpy as np


def wait_for_stall(hand, dof_ids, stall_speed: float = 0.3,
                   confirm_count: int = 3, timeout: float = 10.0,
                   poll: float = 0.1, label: str = "stall") -> dict:
    """Block until every DOF stops moving. Returns {dof_id: position_mm}.

    A DOF counts as stalled once it moves slower than stall_speed mm/s for
    confirm_count consecutive polls. DOFs still moving at timeout are reported
    at their last position, so callers always get an entry for every DOF and
    never have to handle a partial dict.

    The threshold is a rate, not a per-poll distance, because a per-poll
    distance silently depends on the poll interval. It was 0.5mm per 0.05s
    poll, which is 10mm/s -- an order of magnitude faster than anything this
    hand commands, so every DOF read as stalled on the third poll and approach()
    returned its starting position as the contact point. The gap is now wide in
    both directions: creeping at speed=50 is 0.61mm/s, well above the threshold,
    while one count of encoder noise over a 0.1s poll is 0.12mm/s, well below.
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

        t0 = time.time()
        time.sleep(poll)
        # Measured, not assumed: a slow bus read stretches the interval, and
        # dividing by the nominal poll would read that as extra speed.
        elapsed = max(time.time() - t0, 1e-6)
        current = hand.positions
        for d in dof_ids:
            if d in stalled:
                continue
            if abs(current[d] - prev[d]) / elapsed < stall_speed:
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
    hand.set_pos({d: target_mm for d in dof_ids},
                 speed=speed, torque=torque, wait=False)
    return wait_for_stall(hand, dof_ids, label="contact", **kwargs)


def squeeze(hand, dof_ids, torque: int, position_mm: float = 0.0):
    """Drive DOFs toward a position at reduced torque so they stall on contact
    and keep pressing. This is how the hand grips."""
    hand.set_pos({d: position_mm for d in dof_ids}, torque=torque, wait=False)


def squeeze_until_stall(hand, dof_ids, torque: int, **kwargs) -> dict:
    squeeze(hand, dof_ids, torque)
    return wait_for_stall(hand, dof_ids, label="squeeze", **kwargs)


# ── Pre-calibration variant ───────────────────────────────────────────────────

def wait_for_stall_counts(hand, dof_ids, stall_speed: float = 25.0,
                          confirm_count: int = 2, timeout: float = 30.0,
                          poll: float = 0.1, verbose: bool = True) -> dict:
    """Stall detection in raw servo counts, for use before zeroing.

    Returns {dof_id: counts}, with None for any DOF that never stalled or whose
    servo never answered. None must stay None: reporting a timed-out DOF at its
    last position would record a zero offset in the middle of travel, and every
    subsequent mm command on that axis would be wrong.

    stall_speed is counts/sec, for the same reason wait_for_stall's is mm/sec: a
    per-poll distance silently depends on the poll interval. This one used to
    ask for 5 counts per 0.1s poll, which is 50 counts/sec -- exactly the creep
    speed zeroing commands, so a DOF travelling at precisely the commanded rate
    sat on the boundary and either reading could win. It happened to work. The
    default is now half the creep speed, so travelling and stalled are an
    unambiguous factor of two apart in each direction.
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

        t0 = time.time()
        time.sleep(poll)
        # Measured, not assumed: one read per DOF per round means the real
        # interval grows with len(dof_ids), and a phase of four fingers polls
        # noticeably slower than a phase of one z stage.
        elapsed = max(time.time() - t0, 1e-6)
        for d in dof_ids:
            if d in stalled:
                continue
            actual = read(d)
            if actual is None or prev[d] is None:
                # Dropped frame: no movement estimate this round, keep waiting.
                consecutive[d] = 0
                prev[d] = actual
                continue
            speed = abs(actual - prev[d]) / elapsed
            prev[d] = actual
            if verbose:
                print(f"  DOF {d} | counts: {actual} | speed: {speed:.0f}/s")
            if speed < stall_speed:
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

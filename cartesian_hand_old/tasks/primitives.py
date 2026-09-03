"""The two things a task cannot write for itself.

There used to be a vocabulary here -- `travel_to`, `probe`, `grip`, `hold`,
`slide_finger` -- and every one of them was `Move(goal, torque, <a literal>,
timeout)` under a different name. `probe` and `grip` had byte-identical bodies
and differed only in a default torque that no call site ever used. A primitive
that adds no information is a lookup, so they are gone: tasks write
`Move(x, tq, "goal", t)`, which says the same thing in the same space.

What survives is what carries information:

`twist` binds joints. It returns the `{joint: Move}` dict that `Program.step`
consumes, so the caller cannot pair the wrong Move with the wrong finger. The
old version took `left_joint`/`right_joint`, ignored both, and returned a
positional 2-list the caller unpacked by convention -- which is exactly how the
aux fingers ended up reversed once already.

The raw-count waits drive the bus directly. Zeroing runs before mm exist, so the
motion engine cannot drive it; these two exist for that one caller.

Travel clamping used to live in `caps_motions` as three private helpers. It is
`HandConfig.clamped_mm` now -- every task needs it, and the second copy is the
one that gets it wrong.
"""
from ..motions import Move


def twist(left_joint: int, right_joint: int, span: float,
          torque: float = 80.0, timeout_s: float = 6.0,
          when=None) -> dict:
    """The aux fingers' coordinated half-twist, as one step's worth of moves.

    Left goes span -> 0, right goes 0 -> span. Returns `{joint: Move}` so the
    caller spreads it straight into a step: `p.step(twist(L, R, span))`. The
    joint arguments are the point -- they are what makes this a primitive rather
    than two `Move`s the caller has to label correctly.

    A full turn is two of these back-to-back with the cap re-gripped between,
    which is what `caps_motions.stroke_program` builds.
    """
    return {left_joint:  Move(0.0,  torque, "goal", timeout_s, when),
            right_joint: Move(span, torque, "goal", timeout_s, when)}


# ── Raw-count waits (zeroing only) ─────────────────────────────────────────────
#
# Zeroing runs before mm exist, so the motion engine cannot drive it. These are
# the only primitives that touch the bus, and they have one caller:
# `tasks/zeroing.py`.

def wait_for_stall_counts(hand, dof_ids, stall_speed: float = 5.0,
                          confirm_s: float = 1.0, timeout: float = 30.0,
                          poll: float = 0.1, verbose: bool = True) -> dict:
    """Stall detection in raw servo counts, for use before zeroing.

    Returns {dof_id: counts}, with None for any DOF that never stalled or whose
    servo never answered. None must stay None: reporting a timed-out DOF at its
    last position would record a zero offset in the middle of travel, and every
    subsequent mm command on that axis would be wrong.
    """
    import time
    dof_ids = list(dof_ids)
    sids = [hand.config[d].servo_id for d in dof_ids]

    # One sync-read per poll, not one unicast per DOF. Reads of contiguous
    # registers (position, speed, load) cost the same bus packet; we use just
    # position here.
    read = lambda: hand.servo.read_all(sids)

    anchor = {d: (None, time.time()) for d in dof_ids}
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
        now = time.time()
        snapshot = read()
        for d, (sid, reading) in zip(dof_ids, zip(sids, snapshot)):
            if d in stalled:
                continue
            if reading is None:
                continue
            actual = reading[0]                        # position from (pos, speed, load)
            counts0, started = anchor[d]
            if counts0 is None:
                anchor[d] = (actual, now)
                continue
            window = now - started
            if window < confirm_s:
                continue
            speed = abs(actual - counts0) / window
            if verbose:
                print(f"  DOF {d} | counts: {actual} | speed: {speed:.1f}/s")
            if speed < stall_speed:
                stalled[d] = actual
                print(f"  DOF {d} hard stop at {actual}")
            else:
                anchor[d] = (actual, now)

    return {d: stalled.get(d) for d in dof_ids}


def wait_until_counts(hand, dof_ids, counts, tolerance: int = 80,
                      timeout: float = 5.0, poll: float = 0.02) -> bool:
    """Wait for DOFs to reach raw count targets. Returns False on timeout."""
    import time
    deadline = time.time() + timeout
    targets = dict(zip(dof_ids, counts))
    while time.time() < deadline:
        actual = {d: hand.servo.read_position(hand.config[d].servo_id) for d in dof_ids}
        if all(a is not None and abs(a - targets[d]) <= tolerance
               for d, a in actual.items()):
            return True
        time.sleep(poll)
    return False

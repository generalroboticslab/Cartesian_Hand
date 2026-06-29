"""
Contact-based bottle cap open task.

Standalone usage:
    python tasks/caps_contact_based.py --cap-offset 25 --num-revs 3
"""

import argparse
import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cartesian_hand import DEFAULT_TORQUE, DEFAULT_SPEED, DEFAULT_ACC


def approach(controller, dof_ids, approach_torque=150, approach_speed=50,
             stall_threshold=0.5, confirm_count=3, timeout=10.0):
    """Drive y DOFs toward 0mm until stall. Returns {dof_id: contact_mm (= radius)}."""
    if not controller.running:
        controller.enable()
    with controller.lock:
        for d in dof_ids:
            controller._speed[d]  = approach_speed
            controller._torque[d] = approach_torque
            controller.target[d]  = 0.0
        prev = {d: controller.actual[d] for d in dof_ids}
    consecutive = {d: 0     for d in dof_ids}
    stalled     = {d: False for d in dof_ids}
    contact     = {}
    start = time.time()
    while not all(stalled.values()):
        if time.time() - start > timeout:
            print(f"approach: timeout for DOFs {[d for d in dof_ids if not stalled[d]]}")
            with controller.lock:
                for d in dof_ids:
                    if not stalled[d]:
                        contact[d] = controller.actual[d]
            break
        time.sleep(0.05)
        with controller.lock:
            current = {d: controller.actual[d] for d in dof_ids}
        for d in dof_ids:
            if stalled[d]:
                continue
            if abs(current[d] - prev[d]) < stall_threshold:
                consecutive[d] += 1
                if consecutive[d] >= confirm_count:
                    stalled[d] = True
                    contact[d] = current[d]
                    print(f"  DOF {d} contact at {current[d]:.1f}mm")
            else:
                consecutive[d] = 0
            prev[d] = current[d]
    return contact


def squeeze(controller, dof_ids, squeeze_torque):
    """Lower torque and keep target=0 — servo stalls at contact with reduced force."""
    with controller.lock:
        for d in dof_ids:
            controller._torque[d] = squeeze_torque
            controller.target[d]  = 0.0


def caps_open(controller, cap_offset, num_revs, squeeze_torque,
              approach_torque=150, approach_speed=50,
              speed=DEFAULT_SPEED, acc=DEFAULT_ACC, torque=DEFAULT_TORQUE):
    """
    Open a bottle cap via contact-based stall detection.

    cap_offset:     z height of cap top face from zero (mm)
    num_revs:       full revolutions to remove cap
    squeeze_torque: holding torque during twist strokes
    """
    if not controller.is_zeroed:
        raise RuntimeError("No zero offsets found — run zeroing.py first.")

    cfg   = controller.dof_config
    X_MAX = cfg[1]["max_mm"]
    Y_MAX = cfg[0]["max_mm"]
    Z_MAX = cfg[3]["max_mm"]
    cap_z = min(cap_offset, Z_MAX)

    sp       = dict(speed=speed, acc=acc, torque=torque)
    sp_slide = dict(speed=speed, acc=acc)  # no torque= — preserves per-DOF squeeze during x moves
    kw_ap    = dict(approach_torque=approach_torque, approach_speed=approach_speed)

    print("--- Caps: Open ---")

    # Entry: z to cap height, all x to halfway, all y to max
    controller.set_pos([Y_MAX, X_MAX/2, X_MAX/2, cap_z, Y_MAX, X_MAX/2, X_MAX/2], **sp)
    time.sleep(1)

    # Probe: close DOF 0 and DOF 4 simultaneously until stall
    print("Probing...")
    contact       = approach(controller, [0, 4], **kw_ap)
    bottle_radius = contact.get(0, 0.0)
    cap_radius    = contact.get(4, 0.0)
    if cap_radius < 1.0:
        raise RuntimeError(f"Cap probe failed — contact at {cap_radius:.1f}mm")
    print(f"  Bottle radius {bottle_radius:.1f}mm  Cap radius {cap_radius:.1f}mm")
    squeeze(controller, [0, 4], squeeze_torque)

    # Twist strokes: circumference = 2π × cap_radius, each stroke covers X_MAX mm
    total_strokes = max(1, int(np.ceil(num_revs * 2 * np.pi * cap_radius / X_MAX)))
    print(f"  {total_strokes} strokes for {num_revs} revs")

    for stroke in range(total_strokes):
        print(f"  Stroke {stroke + 1}/{total_strokes}")

        # Release cap jaw 1mm past contact
        controller.set_pos([None, None, None, None, cap_radius + 1.0, None, None], **sp)
        squeeze(controller, [0], squeeze_torque)  # restore bottle after torque broadcast

        # Reset x: DOF 5 → max, DOF 6 → 0
        controller.set_pos([None, None, None, None, None, X_MAX, 0.0], **sp_slide)

        # Grip cap: DOF 4 closes to cap_radius, then squeeze
        controller.set_pos([None, None, None, None, cap_radius, None, None], **sp_slide)
        squeeze(controller, [4], squeeze_torque)

        # Twist: DOF 5 → 0, DOF 6 → max (only after grip confirmed above)
        controller.set_pos([None, None, None, None, None, 0.0, X_MAX], **sp_slide)

    # Extract: release cap jaw, center x, re-grip (stall), lift z, clear base jaws
    print("Extracting...")
    controller.set_pos([None, None, None, None, cap_radius + 1.0, None, None], **sp)
    squeeze(controller, [0], squeeze_torque)
    controller.set_pos([None, None, None, None, None, X_MAX/2, X_MAX/2], **sp_slide)
    approach(controller, [4], **kw_ap)
    squeeze(controller, [4], squeeze_torque)
    controller.set_pos([None, None, None, Z_MAX, None, None, None], **sp_slide)
    controller.set_pos([None, X_MAX, X_MAX, None, None, 0.0, 0.0], **sp)
    print("Cap removed.")


if __name__ == "__main__":
    from cartesian_hand import CartesianHand, PORT_1, PORT_2, config_1, config_2

    parser = argparse.ArgumentParser(description="Contact-based bottle cap open")
    parser.add_argument("--config",          choices=["1", "2"], default="2")
    parser.add_argument("--cap-offset",      type=float, default=20.0)
    parser.add_argument("--num-revs",        type=float, default=3.0)
    parser.add_argument("--squeeze-torque",  type=int,   default=80)
    parser.add_argument("--approach-torque", type=int,   default=150)
    parser.add_argument("--approach-speed",  type=int,   default=50)
    parser.add_argument("--torque",          type=int,   default=DEFAULT_TORQUE)
    parser.add_argument("--speed",           type=int,   default=DEFAULT_SPEED)
    parser.add_argument("--acc",             type=int,   default=DEFAULT_ACC)
    args = parser.parse_args()

    port   = PORT_1   if args.config == "1" else PORT_2
    config = config_1 if args.config == "1" else config_2

    with CartesianHand(port, config) as hand:
        if not hand.is_zeroed:
            raise RuntimeError("No zero offsets found — run zeroing.py first.")
        time.sleep(0.5)
        caps_open(hand,
                  cap_offset=args.cap_offset,
                  num_revs=args.num_revs,
                  squeeze_torque=args.squeeze_torque,
                  approach_torque=args.approach_torque,
                  approach_speed=args.approach_speed,
                  torque=args.torque,
                  speed=args.speed,
                  acc=args.acc)
        hand.release()

"""
Bottle cap manipulation task — open and/or close a threaded cap.

Standalone usage:
    python tasks/caps.py --open
    python tasks/caps.py --close
    python tasks/caps.py --open --close --bottle-diameter 42.2 --cap-diameter 36.7 --num-revs 2.5 --cap-offset 15
"""

import argparse
import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cartesian_hand import DEFAULT_TORQUE, DEFAULT_SPEED, DEFAULT_ACC


def caps(controller,
         torque: int = DEFAULT_TORQUE, speed: int = DEFAULT_SPEED, acc: int = DEFAULT_ACC,
         bottle_diameter: float = 60.0, cap_diameter: float = 30.0,
         num_revs: float = 3.0, cap_offset: float = 20.0,
         do_open: bool = True, do_close: bool = True):
    """
    Open or close a bottle cap using the gripper.

    bottle_diameter: diameter of bottle body (mm)
    cap_diameter:    diameter of cap (mm)
    num_revs:        number of full revolutions to remove/replace cap
    cap_offset:      height of cap top face from z=0 (mm)
    open:            run opening sequence
    close:           run closing sequence
    """
    if not controller.is_zeroed:
        print("Must zero controller first.")
        return

    # ── Geometry ──────────────────────────────────────────────────────────────
    X_MAX = controller.dof_config[1]["max_mm"]
    Y_MAX = controller.dof_config[0]["max_mm"]
    Z_MAX = controller.dof_config[3]["max_mm"]

    grip_clearance = 10.0
    tight_grip     = 1
    tighter_grip   = 2
    cap_release    = 5
    extract_lift   = 30

    cap_circumference = np.pi * cap_diameter
    stroke_length     = 2 * X_MAX
    strokes_per_rev   = cap_circumference / stroke_length
    total_strokes     = int(np.ceil(num_revs * strokes_per_rev))

    print(f"Cap circumference: {cap_circumference:.1f}mm")
    print(f"Stroke length: {stroke_length:.1f}mm")
    print(f"Strokes per revolution: {strokes_per_rev:.2f}")
    print(f"Total strokes needed: {total_strokes}")

    approach_mm    = min((max(bottle_diameter, cap_diameter) / 2) + grip_clearance, Y_MAX)
    cap_z          = min(cap_offset, Z_MAX)
    bottle_grip_mm = max(0, min((bottle_diameter / 2) - tighter_grip, Y_MAX))
    cap_grip_mm    = max(0, min((cap_diameter / 2) - tight_grip, Y_MAX))
    extract_z      = min(cap_offset + extract_lift, Z_MAX)

    sp = dict(speed=speed, acc=acc, torque=torque)

    if do_open:
        print("--- Caps: Opening sequence ---")

        # Step 1: open both y jaws
        print(f"Step 1: opening y jaws to {approach_mm:.1f}mm")
        controller.set_pos([approach_mm, None, None, None, approach_mm, None, None], **sp)

        # Step 2: move z to cap offset
        print(f"Step 2: moving z to cap offset {cap_z:.1f}mm")
        controller.set_pos([None, None, None, cap_z, None, None, None], **sp)

        time.sleep(3)

        # Step 3: grip bottle, base slides to halfway
        print(f"Step 3: gripping bottle at {bottle_grip_mm:.1f}mm, slides to halfway")
        controller.set_pos([bottle_grip_mm, X_MAX/2, X_MAX/2, None, None, None, None], **sp)

        # Step 4: twist sequence
        print(f"Step 4: {total_strokes} twist strokes to open cap")
        for stroke in range(total_strokes):
            print(f"  Stroke {stroke + 1}/{total_strokes}")
            controller.set_pos([None, None, None, None, None, 0.0, X_MAX], **sp)
            controller.set_pos([None, None, None, None, cap_grip_mm, None, None], **sp)
            controller.set_pos([None, None, None, None, None, X_MAX, 0.0], **sp)
            controller.set_pos([None, None, None, None, cap_grip_mm + cap_release, None, None], **sp)

        # Step 5: extract cap
        print("Step 5: extracting cap")
        controller.set_pos([None, None, None, None, None, X_MAX/2, X_MAX/2], **sp)
        controller.set_pos([None, None, None, None, cap_grip_mm, None, None], **sp)
        controller.set_pos([None, None, None, extract_z, None, None, None], **sp)
        controller.set_pos([None, X_MAX, X_MAX, None, None, 0.0, 0.0], **sp)

        print("Cap removed. Bottle presented.")

    if do_close:
        print("--- Caps: Closing sequence ---")

        # Step 1: reposition cap above bottle neck
        print("Step 1: repositioning cap above bottle neck")
        controller.set_pos([None, X_MAX/2, X_MAX/2, None, None, X_MAX/2, X_MAX/2], **sp)
        controller.set_pos([None, None, None, cap_z, None, None, None], **sp)

        # Step 2: reverse twist strokes
        print(f"Step 2: {total_strokes} reverse twist strokes to close cap")
        for stroke in range(total_strokes):
            print(f"  Stroke {stroke + 1}/{total_strokes}")

            # press dof3 down at 50 torque while sliding; wait only on slide DOFs
            with controller.lock:
                controller._torque[3] = 50
            controller.set_pos([None, None, None, 0.0, None, X_MAX, 0.0],
                               wait_dofs=[5, 6], **sp)

            # release grip — set_pos(**sp) broadcasts torque back to all DOFs including dof3
            controller.set_pos([None, None, None, None, cap_grip_mm + cap_release, None, None], **sp)
            controller.set_pos([None, None, None, cap_z, None, None, None], **sp)

            # reset slides for next stroke, re-grip
            controller.set_pos([None, None, None, None, None, 0.0, X_MAX], **sp)
            controller.set_pos([None, None, None, None, cap_grip_mm, None, None], **sp)

        # Step 3: release and clear
        print("Step 3: releasing and clearing")
        controller.set_pos([None, None, None, cap_z, None, None, None], **sp)
        controller.set_pos([approach_mm, None, None, None, approach_mm, None, None], **sp)
        print("Cap closed. Gripper cleared.")


if __name__ == "__main__":
    from cartesian_hand import CartesianHand, PORT_1, PORT_2, config_1, config_2

    parser = argparse.ArgumentParser(description="Bottle cap open/close task")
    parser.add_argument("--config",  choices=["1", "2"], default="2")
    parser.add_argument("--open",    action="store_true", default=False)
    parser.add_argument("--close",   action="store_true", default=False)
    parser.add_argument("--bottle-diameter", type=float, default=60.0)
    parser.add_argument("--cap-diameter",    type=float, default=30.0)
    parser.add_argument("--num-revs",        type=float, default=3.0)
    parser.add_argument("--cap-offset",      type=float, default=20.0)
    parser.add_argument("--torque", type=int, default=DEFAULT_TORQUE)
    parser.add_argument("--speed",  type=int, default=DEFAULT_SPEED)
    parser.add_argument("--acc",    type=int, default=DEFAULT_ACC)
    args = parser.parse_args()

    if not args.open and not args.close:
        parser.error("Specify at least one of --open or --close")

    port   = PORT_1   if args.config == "1" else PORT_2
    config = config_1 if args.config == "1" else config_2

    with CartesianHand(port, config) as hand:
        if not hand.is_zeroed:
            raise RuntimeError("No zero offsets found — run zeroing.py first.")
        time.sleep(0.5)
        caps(hand,
             torque=args.torque, speed=args.speed, acc=args.acc,
             bottle_diameter=args.bottle_diameter,
             cap_diameter=args.cap_diameter,
             num_revs=args.num_revs,
             cap_offset=args.cap_offset,
             do_open=args.open,
             do_close=args.close)
        hand.release()

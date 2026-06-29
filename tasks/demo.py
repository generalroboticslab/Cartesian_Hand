"""
Range-of-motion demo task — sweeps all DOFs to max then to zero.

Standalone usage:
    python tasks/demo.py
    python tasks/demo.py --side left
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def demo(controller):
    """Sweep all DOFs to max then zero. Requires zeroed controller."""
    if not controller.is_zeroed:
        print("Must zero controller first.")
        return
    MAX  = [controller.dof_config[d]["max_mm"] for d in range(7)]
    ZERO = [0, 0, 0, 0, 0, 0, 0]
    controller.set_pos(MAX)
    time.sleep(0.5)
    controller.set_pos(ZERO)
    print("Demo complete.")


if __name__ == "__main__":
    import argparse
    from cartesian_hand import CartesianHand, PORT_1, PORT_2, config_1, config_2

    parser = argparse.ArgumentParser(description="Range-of-motion demo")
    parser.add_argument("--config", choices=["1", "2"], default="2")
    args = parser.parse_args()

    port   = PORT_1   if args.config == "1" else PORT_2
    config = config_1 if args.config == "1" else config_2

    with CartesianHand(port, config) as hand:
        if not hand.is_zeroed:
            raise RuntimeError("No zero offsets found — run zeroing.py first.")
        time.sleep(0.5)
        demo(hand)
        hand.release()

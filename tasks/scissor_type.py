"""
Scissor-type manipulation task — stub.

Intended for tasks where the gripper operates in a scissor/shear motion
rather than parallel jaw or rotational modes.

Standalone usage:
    python tasks/scissor_type.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run(controller):
    """Scissor-type task. Not yet implemented."""
    raise NotImplementedError("scissor_type task not yet implemented")


if __name__ == "__main__":
    import argparse
    import time
    from cartesian_hand import CartesianHand, PORT_1, PORT_2, config_1, config_2

    parser = argparse.ArgumentParser(description="Scissor-type task")
    parser.add_argument("--config", choices=["1", "2"], default="2")
    args = parser.parse_args()

    port   = PORT_1   if args.config == "1" else PORT_2
    config = config_1 if args.config == "1" else config_2

    with CartesianHand(port, config) as hand:
        if not hand.is_zeroed:
            raise RuntimeError("No zero offsets found — run zeroing.py first.")
        time.sleep(0.5)
        run(hand)
        hand.release()

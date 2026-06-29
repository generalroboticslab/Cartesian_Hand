"""
Rubik's cube manipulation task — stub.

Intended for tasks where the gripper grips and rotates a Rubik's cube face,
using the x-slides to apply lateral friction and the y-jaw for squeeze force.

Standalone usage:
    python tasks/rubiks_cube.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run(controller, face: str = "U", turns: int = 1):
    """
    Rubik's cube face rotation. Not yet implemented.

    face:  cube face identifier (U/D/F/B/L/R)
    turns: number of 90-degree clockwise turns (negative = counter-clockwise)
    """
    raise NotImplementedError("rubiks_cube task not yet implemented")


if __name__ == "__main__":
    import argparse
    import time
    from cartesian_hand import CartesianHand, PORT_1, PORT_2, config_1, config_2

    parser = argparse.ArgumentParser(description="Rubik's cube face rotation")
    parser.add_argument("--config", choices=["1", "2"], default="2")
    parser.add_argument("--face",   default="U", choices=["U", "D", "F", "B", "L", "R"])
    parser.add_argument("--turns",  type=int, default=1)
    args = parser.parse_args()

    port   = PORT_1   if args.config == "1" else PORT_2
    config = config_1 if args.config == "1" else config_2

    with CartesianHand(port, config) as hand:
        if not hand.is_zeroed:
            raise RuntimeError("No zero offsets found — run zeroing.py first.")
        time.sleep(0.5)
        run(hand, face=args.face, turns=args.turns)
        hand.release()

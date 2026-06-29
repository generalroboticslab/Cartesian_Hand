"""
demo.py — Franka arm + CartesianHand combined example.

Run from the project root:
    micromamba run -n cartesian_hand python3 franky/demo.py

The arm and hand run sequentially here. For true parallel control, wrap
arm.move_cartesian() in a thread so the hand can move concurrently.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # expose arm.py

import numpy as np
from franky import Affine
from arm import FrankaArm
from cartesian_hand import CartesianHand, PORT_1, config_2

ARM_IP = "172.16.0.2"

# Panda ready pose (joints, radians)
HOME_JOINTS = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]

# Example target pose — edit to match your workspace.
# Columns: [x_axis | y_axis | z_axis | translation] in robot base frame (metres).
TARGET_POSE = np.array([
    [ 1,  0,  0,  0.50],
    [ 0, -1,  0,  0.00],
    [ 0,  0, -1,  0.35],
    [ 0,  0,  0,  1.00],
], dtype=float)

# Hand positions (mm) for open/closed — tune to your gripper geometry.
HAND_OPEN   = [0,  0,  0, 0,  0,  0,  0]
HAND_CLOSED = [20, 15, 15, 0, 20, 15, 15]


def main():
    with FrankaArm(ARM_IP, speed=0.15) as arm, \
         CartesianHand(PORT_1, config_2) as hand:

        if not hand.is_zeroed:
            print("Hand not zeroed — run tasks/zeroing.py first")
            return

        print("Moving arm to home joints...")
        arm.move_joints(HOME_JOINTS)

        print("Opening hand...")
        hand.set_pos(HAND_OPEN)

        print(f"Current arm pose:\n{arm.pose.matrix}")

        print("Moving arm to target pose...")
        arm.move_cartesian(Affine(TARGET_POSE))

        print("Closing hand...")
        hand.set_pos(HAND_CLOSED)

        print("Returning to home...")
        arm.move_joints(HOME_JOINTS)
        hand.set_pos(HAND_OPEN)

        print("Done.")


if __name__ == "__main__":
    main()

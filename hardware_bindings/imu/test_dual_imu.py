#!/usr/bin/env python3
"""Test script for running 2 IMUs simultaneously with viser visualization."""

import time
import numpy as np
import viser
from .py_imu import IMU


def xyzw_to_wxyz(q):
    """Convert quaternion from xyzw to wxyz format for viser."""
    return np.array([q[3], q[0], q[1], q[2]])


def main():
    # Specify port names for each IMU
    # Adjust these based on your setup (check with: ls /dev/ttyACM*)
    port1 = "/dev/ttyACM0"
    port2 = "/dev/ttyACM1"

    # IMU 1 has 180 degree rotation about Z axis relative to IMU 2
    # Rotation matrix for 180° about Z: [[−1, 0, 0], [0, −1, 0], [0, 0, 1]]
    rotation_offset = np.array([
        [-1,  0,  0],
        [ 0, -1,  0],
        [ 0,  0,  1]
    ], dtype=np.float32)

    imu1 = IMU(port_name=port1)

    # imu2 = IMU(port_name=port2)
    imu2 = IMU(port_name=port2, rotation_offset=rotation_offset)


    # Setup viser visualization
    server = viser.ViserServer(host="0.0.0.0", port=8080)
    print("Viser server running at http://localhost:8080")

    # World reference frame
    server.scene.add_frame("/world", axes_length=0.5, axes_radius=0.01)

    # IMU1 frame
    imu1_frame = server.scene.add_frame(
        "/imu1", axes_length=0.3, axes_radius=0.008)
    server.scene.add_label("/imu1/label", "IMU1", position=(0.0, 0.35, 0))


    # IMU2 frame as child of rotation_offset
    imu2_frame = server.scene.add_frame(
        "/imu2", axes_length=0.3, axes_radius=0.008)
    server.scene.add_label("/imu2/label", "IMU2", position=(0.0, 0.35, 0))

    print("Press Ctrl+C to stop.\n")

    counter1 = 0
    counter2 = 0

    try:
        while True:
            # Wait for new data from either IMU
            if counter1 == imu1.counter and counter2 == imu2.counter:
                time.sleep(1e-4)
                continue

            # Update counters
            counter1 = imu1.counter
            counter2 = imu2.counter

            # Update viser frames with quaternion data
            q1 = imu1.quat_xyzw

            q2 = imu2.transformed_quat_xyzw
            if np.linalg.norm(q1) > 1e-6:
                imu1_frame.wxyz = xyzw_to_wxyz(q1)
            if np.linalg.norm(q2) > 1e-6:
                imu2_frame.wxyz = xyzw_to_wxyz(q2)

            # Clear screen and print header
            print("\033[2J\033[H", end="")  # Clear screen
            print("=" * 80)
            print("DUAL IMU TEST - Viser: http://localhost:8080")
            print("=" * 80)

            with np.printoptions(precision=3, suppress=True,
                                formatter={"float": lambda x: f"{x:+7.3f}"},
                                linewidth=140):
                e1, e2 = imu1.euler, imu2.transformed_euler
                g1, g2 = imu1.gravity_vec, imu2.transformed_gravity_vec
                a1, a2 = imu1.raw_acc, imu2.transformed_raw_acc
                w1, w2 = imu1.ang_vel, imu2.transformed_ang_vel

                print(f"\n{'':15} {'IMU1 (reference)':>35} {'IMU2 (transformed)':>35} {'Difference':>25}")
                print(f"{'Euler (R,P,Y)':15} {str(e1):>35} {str(e2):>35} {str(e1-e2):>25}")
                print(f"{'Quaternion':15} {str(q1):>35} {str(q2):>35} {str(q1-q2):>25}")
                print(f"{'Gravity':15} {str(g1):>35} {str(g2):>35} {str(g1-g2):>25}")
                print(f"{'Lin Accel':15} {str(a1):>35} {str(a2):>35} {str(a1-a2):>25}")
                print(f"{'Ang Vel':15} {str(w1):>35} {str(w2):>35} {str(w1-w2):>25}")

            print("\n" + "=" * 80)
            time.sleep(0.05)  # ~20 Hz display update

    except KeyboardInterrupt:
        print("\n\nShutting down...")
    finally:
        imu1.shutdown()
        imu2.shutdown()
        print("Both IMUs closed.")

if __name__ == "__main__":
    main()

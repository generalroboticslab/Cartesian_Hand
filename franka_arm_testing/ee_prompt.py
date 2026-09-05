"""Interactive command prompt for moving the Franka end-effector with franky.

Type 'dx dy dz [dtheta_z]' (millimeters, and optionally a rotation about the
end-effector's own z axis in degrees) to move the end-effector relatively
along its current orientation's axes, 'home' to return to the nominal joint
pose, or 'q' to quit.
"""

import math

from franky import Robot, RealtimeConfig, CartesianMotion, Affine, ReferenceType, JointMotion

ROBOT_IP = "172.16.0.2"  # set to your Franka's IP

# Standard Franka "ready" pose: [0, -pi/4, 0, -3pi/4, 0, pi/2, pi/4]
NOMINAL_JOINTS = [0.0, -0.785398163, 0.0, -2.35619449, 0.0, 1.57079632679, 0.785398163397]


def main():
    robot = Robot(ROBOT_IP, realtime_config=RealtimeConfig.Ignore)
    robot.relative_dynamics_factor = 0.1  # cap speed/accel to 10%

    print("Commands: 'dx dy dz [dtheta_z]' (mm, deg; relative move), 'home', 'q' to quit")
    while True:
        raw = input("ee> ").strip()
        if not raw:
            continue
        if raw in ("q", "quit", "exit"):
            break
        if raw == "home":
            robot.move(JointMotion(NOMINAL_JOINTS))
            continue
        parts = raw.split()
        try:
            values = [float(v) for v in parts]
            if len(values) not in (3, 4):
                raise ValueError
        except ValueError:
            print("Expected 'dx dy dz [dtheta_z]' (mm, deg), 'home', or 'q'")
            continue
        dx_mm, dy_mm, dz_mm = values[:3]
        dx, dy, dz = dx_mm / 1000, dy_mm / 1000, dz_mm / 1000
        if len(values) == 4:
            half_angle = math.radians(values[3]) / 2
            quaternion = [0.0, 0.0, math.sin(half_angle), math.cos(half_angle)]
            motion = Affine([dx, dy, dz], quaternion)
        else:
            motion = Affine([dx, dy, dz])
        robot.move(CartesianMotion(motion, ReferenceType.Relative))


if __name__ == "__main__":
    main()

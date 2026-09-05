"""Move the Franka arm to its standard nominal (ready) joint pose using franky.

Requires the `franky` package (https://github.com/TimSchneider42/franky) and
the arm to be in FCI mode with joints unlocked via Desk.
"""

from franky import Robot, JointMotion, RealtimeConfig

ROBOT_IP = "172.16.0.2"  # set to your Franka's IP

# Standard Franka "ready" pose used by libfranka examples and franka_ros:
# [0, -pi/4, 0, -3pi/4, 0, pi/2, pi/4]
NOMINAL_JOINTS = [0.0, -0.785398163, 0.0, -2.35619449, 0.0, 1.57079632679, 0.785398163397]

if __name__ == "__main__":
    robot = Robot(ROBOT_IP, realtime_config=RealtimeConfig.Ignore)  # skip RT kernel requirement
    robot.relative_dynamics_factor = 0.1  # cap speed/accel to 10% for a first move

    robot.move(JointMotion(NOMINAL_JOINTS))

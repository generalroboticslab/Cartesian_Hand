"""
arm.py — minimal FrankaArm wrapper around franky-control.

Usage:
    from arm import FrankaArm
    from franky import Affine
    import numpy as np

    with FrankaArm("172.16.0.2") as arm:
        arm.move_joints([0, -0.785, 0, -2.356, 0, 1.571, 0.785])
        arm.move_cartesian(Affine(some_4x4_matrix))
"""

import numpy as np
from franky import (
    Robot,
    CartesianMotion,
    JointWaypointMotion,
    JointWaypoint,
    RobotPose,
    Affine,
    ReferenceType,
    RelativeDynamicsFactor,
)

ROBOT_IP    = "172.16.0.2"
DEFAULT_SPEED = 0.05   # fraction of robot max (0–1)


class FrankaArm:
    """Thin wrapper around franky Robot with speed-controlled motions."""

    def __init__(self, ip: str = ROBOT_IP, speed: float = DEFAULT_SPEED):
        self.robot = Robot(ip)
        self.default_speed = speed

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def pose(self) -> Affine:
        """Current end-effector pose as Affine (4×4 homogeneous transform)."""
        return self.robot.current_cartesian_state.pose.end_effector_pose

    @property
    def joints(self) -> np.ndarray:
        """Current joint positions in radians, shape (7,)."""
        return np.array(self.robot.state.q)

    @property
    def has_errors(self) -> bool:
        return self.robot.has_errors

    # ── Motion ────────────────────────────────────────────────────────────────

    def move_cartesian(self, target: Affine, speed: float = None,
                       relative: bool = False):
        """
        Move end-effector to target Affine pose.

        target:   Affine — construct from a 4×4 numpy matrix or translation vector.
        speed:    0–1 fraction of robot max; defaults to self.default_speed.
        relative: if True, target is interpreted relative to current pose.
        """
        rdf = self._rdf(speed)
        ref = ReferenceType.Relative if relative else ReferenceType.Absolute
        self.robot.move(CartesianMotion(RobotPose(target), ref, rdf))

    def move_joints(self, q, speed: float = None):
        """
        Move to joint configuration q (list or ndarray of 7 angles in radians).
        """
        rdf = self._rdf(speed)
        self.robot.move(JointWaypointMotion([JointWaypoint(list(q))], rdf))

    # ── Control ───────────────────────────────────────────────────────────────

    def recover(self):
        self.robot.recover_from_errors()

    def stop(self):
        self.robot.stop()

    # ── Context manager ───────────────────────────────────────────────────────

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.robot.has_errors:
            try:
                self.robot.recover_from_errors()
            except Exception:
                pass
        return False

    # ── Internal ──────────────────────────────────────────────────────────────

    def _rdf(self, speed: float = None) -> RelativeDynamicsFactor:
        s = speed if speed is not None else self.default_speed
        return RelativeDynamicsFactor(s, s, s)

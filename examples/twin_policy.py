"""Worked example of the twin-to-real path.

    PYTHONPATH=. python examples/twin_policy.py     # writes /tmp/twin_rollout.npz
    python -m cartesian_hand policy /tmp/twin_rollout.npz --mock
    python -m cartesian_hand policy /tmp/twin_rollout.npz --hand hand_2

Or skip the artifact and run the policy object directly:

    PYTHONPATH=.:examples python -m cartesian_hand policy twin_policy:PinchPolicy --mock

(PYTHONPATH is only needed when running from a source checkout without
`pip install -e .`.)

The point is that nothing below knows the hand's dimensions, serial port, servo
IDs or count directions. It works in normalized [-1, 1] joint space against the
contract exported by `python -m cartesian_hand contract`, which is the same
contract the twin should be built from.
"""

import numpy as np

from cartesian_hand.hands import get_hand
from cartesian_hand.policy import Obs, Policy, Rollout


class PinchPolicy(Policy):
    """Close the two jaws while holding everything else centred.

    A policy written against a twin looks exactly like this: a function of the
    observation returning normalized targets. Attach the fingerprint of the
    geometry it was tuned against, and the runner refuses to drive a hand whose
    travel limits or DOF ordering differ.
    """

    def __init__(self, config, period_steps: int = 100):
        self.n_dof = config.n_dof
        self.jaws = [i for i, d in enumerate(config.dofs) if d.axis == "y"]
        self.period = period_steps
        self.fingerprint = config.fingerprint()

    def act(self, obs: Obs) -> np.ndarray:
        action = np.zeros(self.n_dof)          # 0 == mid travel
        # Squeeze and release on a slow cycle, -1 being fully closed.
        phase = np.sin(2 * np.pi * obs.step / self.period)
        action[self.jaws] = phase
        return action

    def done(self, obs: Obs) -> bool:
        return obs.step >= 2 * self.period


def main():
    config = get_hand("hand_1")
    policy = PinchPolicy(config)

    print(f"contract fingerprint: {config.fingerprint()}")

    # In a real workflow these actions come out of the twin. Rolling the policy
    # forward open-loop here is enough to produce a replayable artifact.
    actions = []
    for step in range(2 * policy.period):
        obs = Obs(step=step, t=step / config.motion.control_hz,
                  position=np.zeros(config.n_dof),
                  position_mm=np.zeros(config.n_dof),
                  last_action=np.zeros(config.n_dof))
        actions.append(policy.act(obs))

    path = Rollout(fingerprint=config.fingerprint(),
                   hz=config.motion.control_hz,
                   actions=np.array(actions),
                   meta={"source": "examples/twin_policy.py"}).save("/tmp/twin_rollout.npz")
    print(f"wrote {path} ({len(actions)} steps)")
    print("replay it:  python -m cartesian_hand policy /tmp/twin_rollout.npz --mock")


if __name__ == "__main__":
    main()

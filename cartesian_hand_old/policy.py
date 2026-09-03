"""The bridge between a policy and the hardware.

A policy optimized in a digital twin and a policy written by hand are the same
object here: something that maps an observation to an action. This module fixes
what those two words mean, so a policy developed against the twin runs on the
real hand without edits.

The contract:

  * Actions and observed positions are normalized to [-1, 1] per DOF. The policy
    never sees millimetres, serial ports, servo IDs or count directions.
  * An action is an absolute position target, not a delta or a velocity.
  * DOF ordering is the ordering in HandConfig.dofs.
  * Steps run at a fixed rate, config.motion.control_hz unless overridden.

`HandConfig.fingerprint()` hashes exactly the fields above. A rollout recorded in
the twin carries that fingerprint, and `run_policy` refuses to drive real
hardware with a policy built against different geometry.

What the runner adds on top of the twin, because hardware is not the twin:

  * Slew limiting. A twin can teleport a joint between steps; a servo cannot.
  * NaN rejection. A diverged network must not reach the bus.
  * Position clipping to real travel limits.
"""

import json
import time
from dataclasses import dataclass, field

import numpy as np


class ContractMismatch(RuntimeError):
    """Policy was built against different hand geometry than the one connected."""


@dataclass
class Obs:
    """What a policy sees. Normalized unless the field name says otherwise."""
    step: int
    t: float                          # seconds since episode start
    position: np.ndarray              # measured, [-1, 1]
    position_mm: np.ndarray           # measured, raw mm
    last_action: np.ndarray           # previous commanded target, [-1, 1]

    def as_vector(self) -> np.ndarray:
        """Flat observation for a network that wants one array."""
        return np.concatenate([self.position, self.last_action])


class Policy:
    """Base policy. Subclass and implement `act`.

    `fingerprint` is the HandConfig fingerprint this policy was built against.
    Leave it None for a policy that is geometry-agnostic by construction.

    Set `requires_contract = True` if the policy is meaningless without a known
    geometry. A recorded trajectory is the clear case: it is a fixed sequence of
    positions, so an unlabelled one must not be allowed to drive an arbitrary
    hand just because it forgot to say which hand it came from.
    """

    fingerprint = None
    requires_contract = False

    def reset(self, obs: Obs) -> None:
        """Called once before the first step."""

    def act(self, obs: Obs) -> np.ndarray:
        raise NotImplementedError

    def done(self, obs: Obs) -> bool:
        """Return True to end the episode early."""
        return False


@dataclass
class Rollout:
    """A recorded episode. Round-trips between twin and hardware."""
    fingerprint: str
    hz: float
    actions: np.ndarray = None            # (T, n_dof) normalized commands
    positions: np.ndarray = None          # (T, n_dof) normalized measurements
    times: np.ndarray = None              # (T,) seconds since start
    meta: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return 0 if self.actions is None else len(self.actions)

    def save(self, path: str) -> str:
        """Save as .npz. The fingerprint travels with the data, which is the
        whole point: a trajectory is meaningless without the geometry it assumes.

        Absent fields are written as empty float arrays. `np.asarray(None)`
        produces an object array, which will not load under allow_pickle=False,
        and loading untrusted pickles to replay a trajectory is not a trade
        worth making.
        """
        actions = np.asarray(self.actions, dtype=float)
        n_dof = actions.shape[1] if actions.ndim == 2 else 0
        empty2d = np.empty((0, n_dof), dtype=float)

        np.savez(
            path,
            actions=actions,
            positions=empty2d if self.positions is None
                      else np.asarray(self.positions, dtype=float),
            times=np.empty(0, dtype=float) if self.times is None
                  else np.asarray(self.times, dtype=float),
            meta=json.dumps({"fingerprint": self.fingerprint, "hz": self.hz,
                             **self.meta}),
        )
        return path

    @classmethod
    def load(cls, path: str) -> "Rollout":
        d = np.load(path, allow_pickle=False)
        meta = json.loads(str(d["meta"]))
        return cls(
            fingerprint=meta.pop("fingerprint", None),
            hz=meta.pop("hz", None),
            actions=d["actions"],
            positions=d["positions"] if "positions" in d else None,
            times=d["times"] if "times" in d else None,
            meta=meta,
        )


# ── Stock policies ────────────────────────────────────────────────────────────

class ReplayPolicy(Policy):
    """Play back a trajectory recorded in the twin.

    The most common artifact an optimizer produces is a sequence of joint
    targets, not a network. Replaying it is the shortest path from twin to
    hardware, and it is the first thing to try when checking whether a transfer
    gap is in the policy or in the dynamics.
    """

    requires_contract = True

    def __init__(self, rollout, loop: bool = False):
        if isinstance(rollout, str):
            rollout = Rollout.load(rollout)
        self.rollout = rollout
        self.actions = np.asarray(rollout.actions, dtype=float)
        self.fingerprint = rollout.fingerprint
        self.loop = loop
        if self.actions.ndim != 2:
            raise ValueError(f"expected (T, n_dof) actions, got {self.actions.shape}")

    def act(self, obs: Obs) -> np.ndarray:
        i = obs.step % len(self.actions) if self.loop else min(obs.step, len(self.actions) - 1)
        return self.actions[i]

    def done(self, obs: Obs) -> bool:
        return not self.loop and obs.step >= len(self.actions)


class FunctionPolicy(Policy):
    """Wrap a plain function as a policy. For quick tests and LLM-written policies
    that are one expression long."""

    def __init__(self, fn, fingerprint: str = None, horizon: int = None):
        self.fn = fn
        self.fingerprint = fingerprint
        self.horizon = horizon

    def act(self, obs: Obs) -> np.ndarray:
        return np.asarray(self.fn(obs), dtype=float)

    def done(self, obs: Obs) -> bool:
        return self.horizon is not None and obs.step >= self.horizon


class HoldPolicy(Policy):
    """Hold a fixed normalized pose. Useful as a transfer-test baseline."""

    def __init__(self, action, horizon: int = None):
        self.action = np.asarray(action, dtype=float)
        self.horizon = horizon

    def act(self, obs: Obs) -> np.ndarray:
        return self.action

    def done(self, obs: Obs) -> bool:
        return self.horizon is not None and obs.step >= self.horizon


# ── Runner ────────────────────────────────────────────────────────────────────

def _fail(msg, strict):
    if strict:
        raise ContractMismatch(msg)
    print(f"WARNING: {msg}")


def check_contract(hand, policy, strict: bool = True, hz: float = None):
    """Compare the policy's geometry assumptions against the connected hand."""
    expected = getattr(policy, "fingerprint", None)
    actual = hand.config.fingerprint()

    if expected is None:
        if getattr(policy, "requires_contract", False):
            _fail(f"{type(policy).__name__} carries no fingerprint, so there is "
                  f"nothing to check it against. A recorded trajectory is a fixed "
                  f"sequence of positions and is only valid for the geometry it "
                  f"was recorded on. Re-record it, or stamp it with "
                  f"{actual!r} if you are certain it belongs to {hand.name}.",
                  strict)
        return

    if expected != actual:
        _fail(f"Policy was built for hand geometry {expected}, but {hand.name} is "
              f"{actual}. DOF count, ordering, travel limits or control rate "
              f"differ.\nConnected hand contract:\n"
              f"{json.dumps(hand.config.contract(), indent=2)}", strict)
        return

    # control_hz is part of the fingerprint, so running at a different rate
    # silently invalidates the check that just passed: a policy tuned for 50Hz
    # moves four times as fast per unit time at 200Hz.
    if hz is not None and abs(hz - hand.config.motion.control_hz) > 1e-9:
        _fail(f"Policy was built for {hand.config.motion.control_hz}Hz but the "
              f"run requests {hz}Hz. The rate is part of the contract; a "
              f"trajectory replayed faster commands proportionally faster motion.",
              strict)


def run_policy(hand, policy, steps: int = None, duration: float = None,
               hz: float = None, max_delta: float = 0.05, strict: bool = True,
               record: bool = True, on_step=None, settle: float = 0.5,
               speed: int = None, acc: int = None, torque: int = None) -> Rollout:
    """Step a policy against a hand at a fixed rate.

    steps / duration: stop conditions. Without either, runs until policy.done().
    hz:        control rate. Defaults to the hand's configured rate.
    max_delta: slew limit in normalized units per step. With the default 0.05
               and 60mm of travel at 50Hz, a DOF moves at most 3mm per step, so
               a policy that teleports in the twin ramps on hardware instead of
               commanding a step the servo answers with maximum current. Set
               None to disable, but measure the current draw before you do.
    strict:    raise on a policy/hand geometry mismatch rather than warn.
    settle:    seconds to wait after enabling, before the first observation.
    """
    hand.require_zeroed()
    check_contract(hand, policy, strict=strict, hz=hz)

    hz = hz or hand.control_hz
    period = 1.0 / hz
    cfg = hand.config

    hand.enable()
    if speed is not None or acc is not None or torque is not None:
        hand.set_gains(speed=speed, acc=acc, torque=torque)
    if settle:
        time.sleep(settle)

    # Seed the slew limiter at the current pose so the first action ramps from
    # where the hand actually is, not from an arbitrary origin. Clip it: a hand
    # parked outside its configured travel normalizes past [-1, 1], and since
    # the slew window is applied around this value it would otherwise drag the
    # first commanded actions out of range with it.
    last_action = np.clip(hand.normalized_positions(), -1.0, 1.0)

    actions, positions, times = [], [], []
    t0 = time.time()
    step = 0

    def observe(i):
        mm = hand.positions
        return Obs(step=i, t=time.time() - t0, position=cfg.normalize(mm),
                   position_mm=mm, last_action=last_action.copy())

    policy.reset(observe(0))

    try:
        while True:
            loop_start = time.time()
            obs = observe(step)

            if policy.done(obs):
                break
            if steps is not None and step >= steps:
                break
            if duration is not None and obs.t >= duration:
                break

            action = np.asarray(policy.act(obs), dtype=float).reshape(-1)
            if action.shape[0] != hand.n_dof:
                raise ValueError(
                    f"policy returned {action.shape[0]} values, hand has {hand.n_dof} DOFs")
            if not np.all(np.isfinite(action)):
                raise ValueError(f"policy returned non-finite action at step {step}: {action}")

            # Slew limit first, then bound to the action space. Doing it the
            # other way round lets the slew window override the [-1, 1] clip and
            # emit an out-of-range action.
            if max_delta is not None:
                action = np.clip(action, last_action - max_delta, last_action + max_delta)
            action = np.clip(action, -1.0, 1.0)

            hand.set_pos(cfg.denormalize(action), wait=False)
            last_action = action

            if record:
                actions.append(action.copy())
                positions.append(obs.position.copy())
                times.append(obs.t)
            if on_step is not None:
                on_step(obs, action)

            step += 1
            time.sleep(max(0.0, period - (time.time() - loop_start)))
    except KeyboardInterrupt:
        print(f"\ninterrupted at step {step}")

    return Rollout(
        fingerprint=cfg.fingerprint(),
        hz=hz,
        actions=np.array(actions) if actions else np.empty((0, hand.n_dof)),
        positions=np.array(positions) if positions else np.empty((0, hand.n_dof)),
        times=np.array(times) if times else np.empty(0),
        meta={"hand": hand.name, "steps": step, "max_delta": max_delta},
    )

"""Backend-neutral tensor policy contract.

Policies see millimetres and normalized effort, whether the executor is a real
servo bus or a simulator.  Every tensor has a leading environment dimension;
hardware is the ordinary ``N=1`` case rather than a separate API.
"""
from collections import deque
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar, runtime_checkable

import torch


@dataclass(frozen=True, slots=True)
class Observation:
    """One control-tick observation in backend-neutral units."""

    position_mm: torch.Tensor       # [N, J]
    velocity_mm_s: torch.Tensor     # [N, J]
    contact: torch.Tensor           # [N, J] bool
    elapsed_ticks: torch.Tensor     # [N] integer


@dataclass(frozen=True, slots=True)
class Action:
    """One control-tick command shared by hardware and simulation."""

    goal_mm: torch.Tensor           # [N, J]
    max_speed_mm_s: torch.Tensor    # [N, J]
    effort_limit: torch.Tensor      # [N, J], normalized to [0, 1]


class PolicyState(Protocol):
    @property
    def done(self) -> torch.Tensor: ...

    @property
    def failed(self) -> torch.Tensor: ...


StateT = TypeVar("StateT", bound=PolicyState)


@runtime_checkable
class Policy(Protocol[StateT]):
    """Structural policy type; implementations need no base class."""

    def initial_state(self, observation: Observation) -> StateT: ...

    def step(self, observation: Observation,
             state: StateT) -> tuple[Action, StateT]: ...


VELOCITY_WINDOW_TICKS = 10
"""Ticks of position history the reported velocity is measured over.

**A one-tick difference cannot resolve a stopped joint.** The encoder reports
whole counts, so at 81.5 counts/mm and 50 Hz one count of change per tick is
0.61 mm/s -- twice `primitives.STUCK_SPEED_MM_S`, the threshold everything on
this path calls "not moving". A servo dithering a single count against an
object therefore reads as *moving*, forever: the contact never registers, the
row runs out its whole deadline, and on the bench that is a hand sitting still
for twenty seconds and then reporting a fault.

Ten ticks resolves 0.3 mm/s to 4.9 counts, so +/-1 count of dither uses a fifth
of the budget instead of twice it. Same number and the same reasoning as
`motions.STUCK_WINDOW_STEPS`, whose comment records this costing a set of
hard-stop measurements on hand_1 -- the policy path was written with a
single-tick difference and did not inherit it.

Invisible to both backends: `MockServo` integrates a float from wall-clock time
and mujoco reports a float `qpos`, so neither quantizes and neither can produce
the dither. Do not expect a simulated run to fail if this is reverted.
"""


class PolicyRunner(Generic[StateT]):
    """Build canonical observations while an executor owns I/O and pacing."""

    def __init__(self, policy: Policy[StateT], control_hz: float):
        self.policy = policy
        self.control_hz = control_hz
        self.state: StateT | None = None
        self._history: deque[torch.Tensor] = deque(
            maxlen=VELOCITY_WINDOW_TICKS)
        self._elapsed_ticks: torch.Tensor | None = None

    def tick(self, position_mm: torch.Tensor,
             contact: torch.Tensor | None = None) -> Action:
        if position_mm.ndim != 2:
            raise ValueError(f"position_mm must be [N, J], got {position_mm.shape}")
        if contact is None:
            contact = torch.zeros_like(position_mm, dtype=torch.bool)
        if contact.shape != position_mm.shape or contact.dtype != torch.bool:
            raise ValueError("contact must be bool [N, J] matching position_mm")

        if not self._history:
            velocity = torch.zeros_like(position_mm)
            self._elapsed_ticks = torch.zeros(
                position_mm.shape[0], dtype=torch.int64,
                device=position_mm.device)
        else:
            # Over however much history exists, so the window fills in rather
            # than reporting zero for its first ten ticks -- a joint that
            # started moving immediately must not read as stopped.
            span = len(self._history)
            velocity = ((position_mm - self._history[0])
                        * self.control_hz / span)
            self._elapsed_ticks = self._elapsed_ticks + 1

        observation = Observation(position_mm, velocity, contact,
                                  self._elapsed_ticks)
        if self.state is None:
            self.state = self.policy.initial_state(observation)
        action, self.state = self.policy.step(observation, self.state)
        _check_action(action, position_mm)
        self._history.append(position_mm.clone())
        return action

    def finished(self) -> bool:
        """Executor-only completion check; policy ticks never sync to Python."""
        return self.state is not None and bool(self.state.done.all())

    def failed(self) -> bool:
        return self.state is not None and bool(self.state.failed.any())


def _check_action(action: Action, reference: torch.Tensor) -> None:
    shape = reference.shape
    for name, value in (
            ("goal_mm", action.goal_mm),
            ("max_speed_mm_s", action.max_speed_mm_s),
            ("effort_limit", action.effort_limit)):
        if value.shape != shape:
            raise ValueError(f"{name} must be {tuple(shape)}, got {tuple(value.shape)}")
        if value.device != reference.device:
            raise ValueError(f"{name} must be on {reference.device}")

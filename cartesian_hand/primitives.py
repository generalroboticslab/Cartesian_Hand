"""Everything a task is built from: one move, and an ordering of moves.

Three sections, in dependency order.

1. **`Step` helpers** -- `twist`, `tilt`, `rotate_in_place`. They fill one
   `motions.Step`, keeping the legacy `Motions` tasks (`zero`, `tilt`) readable:

       twist(p.step(), AUX_LEFT, AUX_RIGHT, span, squeeze)

2. **Closed-loop primitives** -- `move_to`, `close_until_contact`, `hold`, and
   the composite `twist_stroke`. Unlike aliases for `Step.set`, these own
   closed-loop state and publish typed measurements a later primitive can
   consume. One of them answers exactly one question per tick: did these DOFs
   arrive, or touch something, or give up?

3. **Phase tables** -- `Move`/`Probe`/`Hold`/`Twist`/`Loop` rows and the
   `Sequence` that walks them. A task declares its rows; `Sequence` calls the
   primitive each row names and decides what runs next. This is the general
   form of combining section 2, and `twist_stroke` is a special case of it that
   was hand-written before it existed.

Both the `Motions` and policy paths live here because fixed timelines and
feedback-driven manipulation have different smallest useful representations,
and a task reaching for either should not have to know which file it is in.

Goals are millimetres, positive-is-extend on every DOF -- `counts_to_mm` already
applied orientation, so a task writes `-80` for "retract 80 mm" on any DOF and
never an `orientation` term. The raw-count version needs one and gets it wrong
about half the time it is re-derived.

The paired goals below are `[1, n]` rather than a bare length-n list because
`Step.set` refuses an ambiguous 1-D value -- see `Step._spread`.
"""
import math
from collections.abc import Callable, Sequence as Rows
from dataclasses import dataclass, replace

import torch

from .config import (AUX_LEFT, AUX_RIGHT, BASE_LEFT, BASE_RIGHT, LABELS, Z,
                     HandConfig)
from .motions import Step
from .policy import Action, Observation


def twist(step: Step, left_joint: int, right_joint: int, span: float,
          torque: float, timeout_s: float = 6.0,
          when: torch.Tensor | None = None) -> Step:
    """The paired fingers' coordinated half-twist: left to 0, right to `span`.

    Ids and goals are built together so a caller cannot line the wrong number up
    against the wrong finger -- which is how the aux fingers ended up reversed
    once already. Swapping the two id arguments is what runs the twist backwards,
    so a stroke and its reset are the same call with the ids exchanged.

    A full turn is two of these with the object re-gripped between, which is what
    the stroke phase of `tasks.cap.build` emits.
    """
    return step.set([left_joint, right_joint],
                    torch.tensor([[0.0, float(span)]], device=step.device),
                    torque, "goal", timeout_s, when)


def tilt(step: Step, stage_a: int, stage_b: int, other_a: int, other_b: int,
         span: float, torque: float, timeout_s: float = 6.0,
         when: torch.Tensor | None = None) -> Step:
    """Pitch one stage's grip relative to the other: `stage_*` move, `other_*` hold.

    `*_a` names the same column (both left, or both right) on each stage.
    Crossing the columns pitches the grip sideways of what the task meant, which
    a symmetric gripper cannot report, so the same-id trap is refused here rather
    than built silently. Hinged-lid opens are two of these with the lid re-seated
    between.
    """
    if stage_a == stage_b or other_a == other_b:
        raise ValueError(
            f"tilt: each column must name two distinct joints, got "
            f"({stage_a}, {stage_b}) and ({other_a}, {other_b})")
    return step.set([stage_a, stage_b, other_a, other_b],
                    torch.tensor([[float(span), float(span), 0.0, 0.0]],
                                 device=step.device),
                    torque, "goal", timeout_s, when)


def rotate_in_place(step: Step, span: float, torque: float,
                    timeout_s: float = 6.0,
                    when: torch.Tensor | None = None) -> Step:
    """Yaw the object about the vertical axis without translating it.

    Two opposing twists, one per stage, so the object's centre holds still on
    average. Built here rather than as two `twist` calls in a task for the same
    reason `twist` exists: the four-way pairing is the primitive, it is
    positional in LAYOUT order, and the runtime has no other clue which joint
    opposes which.
    """
    return step.set([BASE_LEFT, BASE_RIGHT, AUX_LEFT, AUX_RIGHT],
                    torch.tensor([[float(span), 0.0, 0.0, float(span)]],
                                 device=step.device),
                    torque, "goal", timeout_s, when)


# ── Direct tensor policy primitives ──────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class PrimitiveState:
    """Flat per-environment state shared by the first direct primitives."""

    elapsed_ticks: torch.Tensor       # [N] integer
    quiet_ticks: torch.Tensor         # [N, J] integer
    retired: torch.Tensor             # [N, J] bool
    succeeded_joints: torch.Tensor    # [N, J] bool
    stopped_at_mm: torch.Tensor       # [N, J]
    timed_out: torch.Tensor           # [N] bool
    reached_goal_without_contact: torch.Tensor  # [N] bool
    done: torch.Tensor                # [N] bool
    failed: torch.Tensor              # [N] bool


@dataclass(frozen=True, slots=True)
class PrimitiveResult:
    """Facts a task may name, store, and pass to its next primitive."""

    done: torch.Tensor                # [N] bool
    succeeded: torch.Tensor           # [N] bool
    timed_out: torch.Tensor           # [N] bool
    reached_goal_without_contact: torch.Tensor  # [N] bool
    stopped_at_mm: torch.Tensor       # [N, J]
    stopped_valid: torch.Tensor       # [N, J] bool


@dataclass(frozen=True, slots=True)
class TwistStrokeParameters:
    """Batched values varied by a task or optimizer for one twist stroke."""

    release_clearance_mm: torch.Tensor  # [N]
    finger_span_mm: torch.Tensor        # [N]
    travel_speed_mm_s: torch.Tensor     # [N, J]
    grip_speed_mm_s: torch.Tensor       # [N, J]
    travel_effort: torch.Tensor         # [N, J]
    contact_effort: torch.Tensor        # [N]
    grip_effort: torch.Tensor           # [N]
    travel_timeout_ticks: torch.Tensor  # [N]
    contact_timeout_ticks: torch.Tensor  # [N]
    turn_effort: torch.Tensor | None = None  # [N], explicit finger-turn cap


@dataclass(frozen=True, slots=True)
class TwistStrokeState:
    """Internal release/reset/re-grip/turn state, with no backend objects."""

    phase: torch.Tensor                 # [N] integer
    phase_elapsed_ticks: torch.Tensor   # [N] integer
    phase_quiet_ticks: torch.Tensor     # [N, J] integer
    phase_retired: torch.Tensor         # [N, J] bool
    phase_succeeded_joints: torch.Tensor  # [N, J] bool
    phase_stopped_at_mm: torch.Tensor   # [N, J]
    gripped_at_mm: torch.Tensor         # [N, J]
    grip_valid: torch.Tensor            # [N, J] bool
    timed_out: torch.Tensor             # [N] bool
    reached_goal_without_contact: torch.Tensor  # [N] bool
    done: torch.Tensor                  # [N] bool
    failed: torch.Tensor                # [N] bool
    turn_stalled: torch.Tensor          # [N] bool


@dataclass(frozen=True, slots=True)
class TwistPress:
    """An optional axis press held through the turn, for threading tasks.

    A bulb going back into its socket and a screw being driven in engage by
    depth as well as rotation, so the stroke presses `dof` to `goal_mm` after
    the re-grip, keeps it there through the turn, and backs off to
    `return_mm` before the next stroke releases. `active` is per environment,
    so one batch may thread and unthread at once.

    `return_effort` is separate from `effort` because backing off raises the
    stage against gravity, and z torque is directional: the validated
    implementation reused one number for both and the ascent is the half that
    silently does not move. A task supplies an ascent effort floored at that
    DOF's measured `torque_min_to_move`; see `cap`'s lift.
    """

    dof: int
    active: torch.Tensor           # [N] bool
    goal_mm: torch.Tensor          # [N]
    return_mm: torch.Tensor        # [N]
    effort: torch.Tensor           # [N]
    return_effort: torch.Tensor    # [N]


(RELEASE, RESET_FINGERS, SETTLE_FINGERS, REGRIP,
 PRESS, TURN, RETRACT, STROKE_DONE) = range(8)

STUCK_SPEED_MM_S = 0.3
"""Below this a joint counts as not moving. One servo step at the slowest
commanded rate is well above it, so ordinary travel never trips it."""
CONFIRM_TICKS = 10
"""Consecutive quiet ticks before a stop is believed -- 0.2 s at 50 Hz. One
quiet tick is a servo between steps; ten under load is the joint having
arrived at whatever the object allows."""
START_GRACE_TICKS = 25
"""Ticks a `move_to` row runs before a stop can count against it -- 0.5 s.

**A joint starting from rest is quiet, and that is not evidence.** The torque
register is a force CAP and the effort actually developed follows position
error, so a profile ramping out of a standstill needs time before it pushes
hard enough to break stiction. Without this, `CONFIRM_TICKS` elapses first and
the row rejects a joint that was merely still accelerating: `cap` failed phase
11 (`cap align`) at exactly 0.2 s while reversing two loaded fingers, which then
travelled the full 22.5 mm under the standing command after the task had
already retired. MEMORY records this as the predicted failure of the stiction
finding (2026-09-02) -- see the `Torque is a CAP` entry.

**This costs no motion time.** It applies only where a stall is a FAULT -- a
free `move_to`. A row that reaches its goal finishes exactly when it always
did; the only thing that gets slower is *declaring a fault*, from 0.2 s to
grace + `CONFIRM_TICKS` = 0.7 s, still far inside the multi-second deadlines a
rail implies.

Deliberately NOT applied where a stall is *arrival*: `stall_fallback` rows and
`close_until_contact`. Those end by stalling on nearly every stroke -- a twist's
PRESS, TURN and RETRACT all do -- so a grace there is 0.5 s of dead time per
phase per stroke, buying nothing. For a probe it is also 1.8 mm of extra creep
at the 3.68 mm/s approach speed, past the 1 mm tolerance those work to.

The mirror risk on that side (a joint reading *arrival* in its first ten ticks
because it has not started moving yet) is real, is the same root cause with the
opposite sign, and is NOT addressed here. It predates this change and has not
been observed on the bench."""


def initial_primitive_state(observation: Observation) -> PrimitiveState:
    """Allocate one primitive's state beside its observation tensors."""
    n, joints = observation.position_mm.shape
    device = observation.position_mm.device
    return PrimitiveState(
        elapsed_ticks=torch.zeros(n, dtype=torch.int64, device=device),
        quiet_ticks=torch.zeros((n, joints), dtype=torch.int64, device=device),
        retired=torch.zeros((n, joints), dtype=torch.bool, device=device),
        succeeded_joints=torch.zeros(
            (n, joints), dtype=torch.bool, device=device),
        stopped_at_mm=torch.zeros_like(observation.position_mm),
        timed_out=torch.zeros(n, dtype=torch.bool, device=device),
        reached_goal_without_contact=torch.zeros(
            n, dtype=torch.bool, device=device),
        done=torch.zeros(n, dtype=torch.bool, device=device),
        failed=torch.zeros(n, dtype=torch.bool, device=device),
    )


def initial_twist_stroke_state(observation: Observation) -> TwistStrokeState:
    """Allocate one reusable twist stroke beside its observation tensors."""
    primitive = initial_primitive_state(observation)
    n = observation.position_mm.shape[0]
    return TwistStrokeState(
        phase=torch.full((n,), RELEASE, dtype=torch.int64,
                         device=observation.position_mm.device),
        phase_elapsed_ticks=primitive.elapsed_ticks,
        phase_quiet_ticks=primitive.quiet_ticks,
        phase_retired=primitive.retired,
        phase_succeeded_joints=primitive.succeeded_joints,
        phase_stopped_at_mm=primitive.stopped_at_mm,
        gripped_at_mm=torch.zeros_like(observation.position_mm),
        grip_valid=torch.zeros_like(observation.position_mm, dtype=torch.bool),
        timed_out=primitive.timed_out,
        reached_goal_without_contact=primitive.reached_goal_without_contact,
        done=primitive.done,
        failed=primitive.failed,
        turn_stalled=torch.zeros(n, dtype=torch.bool,
                                 device=observation.position_mm.device),
    )


def hold(action: Action, active: torch.Tensor, dofs: torch.Tensor,
         goal_mm: torch.Tensor, max_speed_mm_s: torch.Tensor,
         effort_limit: torch.Tensor) -> Action:
    """Update named DOFs and preserve every other standing command."""
    _check_direct_inputs(action, active, dofs, goal_mm, max_speed_mm_s,
                         effort_limit)
    selected = active[:, None] & dofs[None, :]
    return Action(
        goal_mm=torch.where(selected, goal_mm, action.goal_mm),
        max_speed_mm_s=torch.where(
            selected, max_speed_mm_s, action.max_speed_mm_s),
        effort_limit=torch.where(selected, effort_limit, action.effort_limit),
    )


def move_to(observation: Observation, action: Action, state: PrimitiveState,
            active: torch.Tensor, dofs: torch.Tensor,
            goal_mm: torch.Tensor, max_speed_mm_s: torch.Tensor,
            effort_limit: torch.Tensor, timeout_ticks: torch.Tensor,
            tolerance_mm: float = 1.0, stall_fallback: bool = False,
            stuck_speed_mm_s: float = STUCK_SPEED_MM_S,
            confirm_ticks: int = CONFIRM_TICKS,
            grace_ticks: int = START_GRACE_TICKS,
            ) -> tuple[Action, PrimitiveState, PrimitiveResult]:
    """Move selected DOFs until each reaches its goal or its env times out.

    ``stall_fallback`` also accepts *stopped moving* as arrival, and a **loaded**
    move needs it. Effort is a force cap, not a force command, so a row pushing
    an object -- a twist turning a cap, a jaw pulling free of one, z pressing
    into a thread -- legitimately parks short of the coordinate it was given.
    Without this it never reports success, waits out the whole deadline, and
    then fails in a way that reads like a mechanical fault rather than a torque
    setting. Off by default: for a *free* move, stopping short is the fault.

    A free move reports that fault **on the tick it is detectable**, not when
    the clock runs out. Both cases need the same stall measurement; only the
    verdict differs. Waiting out the deadline for a joint already known to have
    stopped is the whole "task hangs after the first row" symptom: a blocked
    approach sat quiet for the fifteen-second budget its rail implies before
    saying anything, so the one obstruction that a person could have cleared in
    two seconds read as a freeze. Reported through
    ``reached_goal_without_contact``, which is `_closed_loop`'s "failed for a
    reason that is not the clock" channel -- misnamed for this use because
    `close_until_contact` got there first; renaming it touches five tasks not
    yet on this runner.

    The *fault* verdict is not drawn from the row's first ``grace_ticks``, when
    a joint still accelerating out of a standstill is quiet for reasons that are
    not about the goal. ``stall_fallback`` reads the same measurement as arrival
    and gets no grace, so it costs no time -- see ``START_GRACE_TICKS``.
    """
    # Grace only where a stall is a FAULT. Where it is arrival, waiting it out
    # is dead time on every stroke and buys nothing.
    state, stalled = _stalled(observation, state, active, dofs, goal_mm,
                              stuck_speed_mm_s, confirm_ticks,
                              0 if stall_fallback else grace_ticks)
    reached = (observation.position_mm - goal_mm).abs() <= tolerance_mm
    if stall_fallback:
        return _closed_loop(observation, action, state, active, dofs, goal_mm,
                            max_speed_mm_s, effort_limit, timeout_ticks,
                            reached | stalled)
    return _closed_loop(observation, action, state, active, dofs, goal_mm,
                        max_speed_mm_s, effort_limit, timeout_ticks, reached,
                        stalled)


def _stalled(observation: Observation, state: PrimitiveState,
             active: torch.Tensor, dofs: torch.Tensor, goal_mm: torch.Tensor,
             stuck_speed_mm_s: float, confirm_ticks: int,
             grace_ticks: int = 0) -> tuple[PrimitiveState, torch.Tensor]:
    """Per-joint "stopped away from its goal", confirmed over several ticks.

    One quiet tick is a servo between steps; `confirm_ticks` of them under load
    is the joint having arrived at whatever the object allows. The counter lives
    in `quiet_ticks` and resets whenever the joint moves again.

    `grace_ticks` suppresses counting for the row's opening ticks, when a joint
    accelerating out of a standstill is quiet for reasons that say nothing about
    the goal -- see `START_GRACE_TICKS`. Zero by default, so a caller that reads
    a stall as success is unchanged.

    `stuck_speed_mm_s` is well below one encoder count per tick, so this is only
    meaningful because `PolicyRunner` measures velocity over a window rather
    than a single difference -- see `policy.VELOCITY_WINDOW_TICKS`. Anything
    else building an `Observation` by hand owes the same, or a parked joint
    reads as moving and every contact here waits out its deadline.
    """
    running = (active[:, None] & ~state.done[:, None] & dofs[None, :]
               & ~state.retired
               & (state.elapsed_ticks >= grace_ticks)[:, None])
    away = ((observation.velocity_mm_s.abs() < stuck_speed_mm_s)
            & ((observation.position_mm - goal_mm).abs() > 1.0))
    quiet = torch.where(running, torch.where(away, state.quiet_ticks + 1, 0),
                        state.quiet_ticks)
    return replace(state, quiet_ticks=quiet), quiet >= confirm_ticks


def close_until_contact(
        observation: Observation, action: Action, state: PrimitiveState,
        active: torch.Tensor, dofs: torch.Tensor, goal_mm: torch.Tensor,
        max_speed_mm_s: torch.Tensor, effort_limit: torch.Tensor,
        timeout_ticks: torch.Tensor, stall_fallback: bool = False,
        stuck_speed_mm_s: float = STUCK_SPEED_MM_S,
        confirm_ticks: int = CONFIRM_TICKS,
        ) -> tuple[Action, PrimitiveState, PrimitiveResult]:
    """Close selected DOFs; only canonical contact counts as success.

    Reaching ``goal_mm`` without contact fails immediately. It is neither
    contact nor timeout, which prevents an empty jaw at its closed limit from
    becoming an object measurement or a long unexplained pause.
    """
    state, stalled = _stalled(observation, state, active, dofs, goal_mm,
                              stuck_speed_mm_s, confirm_ticks)
    satisfied = (observation.contact
                 | (stalled if stall_fallback else torch.zeros_like(stalled)))
    reached_without_contact = (
        (observation.position_mm - goal_mm).abs() <= 1.0) & ~satisfied
    return _closed_loop(
        observation, action, state, active, dofs, goal_mm, max_speed_mm_s,
        effort_limit, timeout_ticks, satisfied, reached_without_contact)


def twist_stroke(
        observation: Observation, action: Action, state: TwistStrokeState,
        active: torch.Tensor, radius_mm: torch.Tensor,
        jaw_dof: int, left_finger_dof: int, right_finger_dof: int,
        parameters: TwistStrokeParameters, stall_fallback: bool = False,
        reverse: torch.Tensor | None = None,
        press: TwistPress | None = None,
        ) -> tuple[Action, TwistStrokeState, PrimitiveResult]:
    """Release, reset the fingers, re-grip, and perform one half-twist.

    Environments advance independently. The measured re-grip position is kept
    in ``gripped_at_mm`` and published in the result, so a caller can score it
    or feed it into another primitive without reaching into this state machine.

    ``reverse`` is ``[N] bool`` and swaps which finger leads the reset and
    which finishes the turn, so a stroke and its mirror are one call with a
    tensor rather than two state machines -- `bulb` unscrews and re-threads in
    a single task. ``press`` optionally drives a third axis to depth after the
    re-grip and holds it through the turn; see `TwistPress`.

    Environments that do not press skip both press phases on the transition
    itself, not by passing through them, so a stroke without a press costs
    exactly the ticks it did before this parameter existed.
    """
    n, joints = observation.position_mm.shape
    _check_twist_inputs(state, active, radius_mm, parameters, n, joints)
    ids = (jaw_dof, left_finger_dof, right_finger_dof)
    if min(ids) < 0 or max(ids) >= joints or len(set(ids)) != len(ids):
        raise ValueError(f"twist stroke needs three distinct DOFs, got {ids}")
    device = observation.position_mm.device
    joint_ids = torch.arange(joints, device=device)
    jaw = joint_ids == jaw_dof
    left = joint_ids == left_finger_dof
    right = joint_ids == right_finger_dof
    fingers = left | right
    shape = observation.position_mm.shape

    if reverse is None:
        reverse = torch.zeros(n, dtype=torch.bool, device=device)
    elif reverse.shape != (n,) or reverse.dtype != torch.bool:
        raise ValueError("reverse must be bool [N]")
    _check_press(press, n, joints, device)

    travel_speed = parameters.travel_speed_mm_s
    grip_speed = parameters.grip_speed_mm_s
    travel_effort = parameters.travel_effort
    loaded_effort = torch.maximum(
        parameters.grip_effort[:, None].expand(shape), travel_effort)
    turn_effort = (loaded_effort if parameters.turn_effort is None else
                   parameters.turn_effort[:, None].expand(shape))
    contact_effort = parameters.contact_effort[:, None].expand(shape)
    grip_effort = parameters.grip_effort[:, None].expand(shape)
    release_goal = (radius_mm + parameters.release_clearance_mm)[:, None].expand(shape)
    # Which finger opens the gap the turn then closes. Exchanging the two is
    # what runs the stroke the other way round, which is why direction is a
    # tensor here and not a second call site.
    lead = torch.where(reverse[:, None], right[None, :], left[None, :])
    trail = torch.where(reverse[:, None], left[None, :], right[None, :])
    reset_goal = parameters.finger_span_mm[:, None] * lead
    grip_goal = torch.zeros_like(observation.position_mm)
    turn_goal = parameters.finger_span_mm[:, None] * trail

    presses = (press.active if press is not None
               else torch.zeros(n, dtype=torch.bool, device=device))
    press_dofs = (joint_ids == press.dof if press is not None
                  else torch.zeros(joints, dtype=torch.bool, device=device))
    press_goal = (press.goal_mm[:, None].expand(shape)
                  if press is not None else grip_goal)
    press_effort = (press.effort[:, None].expand(shape)
                    if press is not None else travel_effort)
    return_goal = (press.return_mm[:, None].expand(shape)
                   if press is not None else grip_goal)
    return_effort = (press.return_effort[:, None].expand(shape)
                     if press is not None else travel_effort)

    primitive = PrimitiveState(
        state.phase_elapsed_ticks, state.phase_quiet_ticks, state.phase_retired,
        state.phase_succeeded_joints, state.phase_stopped_at_mm,
        state.timed_out, state.reached_goal_without_contact,
        torch.zeros_like(state.done), torch.zeros_like(state.failed))
    phase = state.phase

    # The jaw must reach its release clearance before the fingers reset;
    # accepting a short stall here can drag the object backwards. Do not reject
    # one either: a loaded servo may need longer than the 0.2 s contact window
    # to start moving, so give it the normal travel deadline. PRESS, TURN and
    # RETRACT push against the object, so a stop is an arrival for them.
    action, primitive, release = move_to(
        observation, action, primitive,
        active & (phase == RELEASE) & ~state.done,
        jaw, release_goal, travel_speed, loaded_effort,
        parameters.travel_timeout_ticks, stuck_speed_mm_s=0.0)
    action, primitive, reset = move_to(
        observation, action, primitive,
        active & (phase == RESET_FINGERS) & ~state.done,
        fingers, reset_goal, travel_speed, travel_effort,
        parameters.travel_timeout_ticks)
    action, primitive, grip = close_until_contact(
        observation, action, primitive,
        active & (phase == REGRIP) & ~state.done,
        jaw, grip_goal, grip_speed, contact_effort,
        parameters.contact_timeout_ticks, stall_fallback=stall_fallback)
    action, primitive, pressed = move_to(
        observation, action, primitive,
        active & (phase == PRESS) & presses & ~state.done,
        press_dofs, press_goal, travel_speed, press_effort,
        parameters.travel_timeout_ticks, stall_fallback=stall_fallback)
    action, primitive, turn = move_to(
        observation, action, primitive,
        active & (phase == TURN) & ~state.done,
        fingers, turn_goal, travel_speed, turn_effort,
        parameters.travel_timeout_ticks, stall_fallback=stall_fallback)
    action, primitive, retracted = move_to(
        observation, action, primitive,
        active & (phase == RETRACT) & presses & ~state.done,
        press_dofs, return_goal, travel_speed, return_effort,
        parameters.travel_timeout_ticks, stall_fallback=stall_fallback)

    release_ok = active & (phase == RELEASE) & release.succeeded
    reset_ok = active & (phase == RESET_FINGERS) & reset.succeeded
    # Position tolerance ends RESET_FINGERS, but the servo may still be moving
    # through its last fraction of a millimetre. Keep the jaw open until both
    # measured finger velocities have settled, then re-grip on the next tick.
    settling = active & (phase == SETTLE_FINGERS) & ~state.done
    settle_elapsed = (primitive.elapsed_ticks
                      + settling.to(primitive.elapsed_ticks.dtype))
    fingers_still = ((observation.velocity_mm_s.abs() < STUCK_SPEED_MM_S)
                     | ~fingers[None, :]).all(dim=1)
    settle_ok = settling & fingers_still
    settle_timed_out = (settling & ~settle_ok
                        & (settle_elapsed >= parameters.travel_timeout_ticks))
    primitive = replace(primitive, elapsed_ticks=settle_elapsed)
    grip_ok = active & (phase == REGRIP) & grip.succeeded
    press_ok = active & (phase == PRESS) & pressed.succeeded
    turn_ok = active & (phase == TURN) & turn.succeeded
    turn_reached = (((observation.position_mm - turn_goal).abs() <= 1.0)
                    | ~fingers[None, :]).all(dim=1)
    turn_stalled = state.turn_stalled | (turn_ok & ~turn_reached)
    retract_ok = active & (phase == RETRACT) & retracted.succeeded
    action = hold(action, grip_ok, jaw, grip_goal, grip_speed, grip_effort)
    transitioned = (release_ok | reset_ok | settle_ok | grip_ok | press_ok
                    | turn_ok | retract_ok)
    failed_now = ((active & (phase == RELEASE) & release.timed_out)
                  | (active & (phase == RESET_FINGERS) & reset.timed_out)
                  | settle_timed_out
                  | (active & (phase == REGRIP) & grip.timed_out)
                  | (active & (phase == PRESS) & presses & pressed.timed_out)
                  | (active & (phase == TURN) & turn.timed_out)
                  | (active & (phase == RETRACT) & presses
                     & retracted.timed_out))
    rejected_now = (
        (active & (phase == RELEASE) & release.reached_goal_without_contact)
        | (active & (phase == RESET_FINGERS)
           & reset.reached_goal_without_contact)
        | (active & (phase == REGRIP)
           & grip.reached_goal_without_contact)
        | (active & (phase == TURN) & turn.reached_goal_without_contact))

    # An environment with no press jumps both z phases on the transition
    # itself rather than spending a tick idling in each, so a stroke without a
    # press issues exactly the actions it did before `press` existed.
    next_phase = torch.where(release_ok, RESET_FINGERS, phase)
    next_phase = torch.where(reset_ok, SETTLE_FINGERS, next_phase)
    next_phase = torch.where(settle_ok, REGRIP, next_phase)
    next_phase = torch.where(
        grip_ok, torch.where(presses, PRESS, TURN), next_phase)
    next_phase = torch.where(press_ok, TURN, next_phase)
    next_phase = torch.where(
        turn_ok, torch.where(presses, RETRACT, STROKE_DONE), next_phase)
    next_phase = torch.where(retract_ok, STROKE_DONE, next_phase)
    finished = retract_ok | (turn_ok & ~presses)
    gripped_at = torch.where(
        grip_ok[:, None] & grip.stopped_valid,
        grip.stopped_at_mm, state.gripped_at_mm)
    grip_valid = state.grip_valid | (grip_ok[:, None] & grip.stopped_valid)

    primitive = reset_primitive_state(primitive, transitioned)
    done = state.done | finished | failed_now | rejected_now
    failed = state.failed | failed_now | rejected_now
    next_state = TwistStrokeState(
        next_phase, primitive.elapsed_ticks, primitive.quiet_ticks, primitive.retired,
        primitive.succeeded_joints, primitive.stopped_at_mm,
        gripped_at, grip_valid,
        state.timed_out | failed_now,
        state.reached_goal_without_contact | rejected_now,
        done, failed, turn_stalled)
    result = PrimitiveResult(
        done=done,
        succeeded=done & ~failed,
        timed_out=next_state.timed_out,
        reached_goal_without_contact=next_state.reached_goal_without_contact,
        stopped_at_mm=gripped_at,
        stopped_valid=grip_valid,
    )
    return action, next_state, result


def _closed_loop(
        observation: Observation, action: Action, state: PrimitiveState,
        active: torch.Tensor, dofs: torch.Tensor, goal_mm: torch.Tensor,
        max_speed_mm_s: torch.Tensor, effort_limit: torch.Tensor,
        timeout_ticks: torch.Tensor, satisfied: torch.Tensor,
        rejected: torch.Tensor | None = None,
        ) -> tuple[Action, PrimitiveState, PrimitiveResult]:
    _check_direct_inputs(action, active, dofs, goal_mm, max_speed_mm_s,
                         effort_limit)
    shape = observation.position_mm.shape
    if satisfied.shape != shape or satisfied.dtype != torch.bool:
        raise ValueError("satisfied must be bool [N, J]")
    if timeout_ticks.shape != shape[:1]:
        raise ValueError("timeout_ticks must be [N]")

    running_env = active & ~state.done
    running_joint = running_env[:, None] & dofs[None, :] & ~state.retired
    stopped_now = running_joint & satisfied
    rejected_now = (torch.zeros_like(running_joint) if rejected is None else
                    running_joint & rejected & ~satisfied)
    rejected_env = rejected_now.any(dim=1)
    retired = (state.retired | stopped_now
               | (rejected_env[:, None] & dofs[None, :]))
    succeeded_joints = state.succeeded_joints | stopped_now
    stopped_at = torch.where(
        stopped_now, observation.position_mm, state.stopped_at_mm)
    elapsed = (state.elapsed_ticks
               + running_env.to(state.elapsed_ticks.dtype))

    selected_done = (retired | ~dofs[None, :]).all(dim=1)
    timed_out = (running_env & ~rejected_env & ~selected_done
                 & (elapsed >= timeout_ticks))
    retired = retired | (timed_out[:, None] & dofs[None, :])
    done = state.done | (running_env & selected_done) | timed_out | rejected_env
    timed_out_total = state.timed_out | timed_out
    rejected_total = state.reached_goal_without_contact | rejected_env
    failed = state.failed | timed_out | rejected_env

    next_state = PrimitiveState(elapsed, state.quiet_ticks, retired, succeeded_joints,
                                stopped_at, timed_out_total, rejected_total,
                                done, failed)
    result = PrimitiveResult(
        done=done,
        succeeded=(done & ~failed
                   & (succeeded_joints | ~dofs[None, :]).all(dim=1)),
        timed_out=timed_out_total,
        reached_goal_without_contact=rejected_total,
        stopped_at_mm=stopped_at,
        stopped_valid=succeeded_joints,
    )
    next_action = hold(action, running_env, dofs, goal_mm, max_speed_mm_s,
                       effort_limit)
    return next_action, next_state, result


def _check_direct_inputs(action: Action, active: torch.Tensor,
                         dofs: torch.Tensor, goal_mm: torch.Tensor,
                         max_speed_mm_s: torch.Tensor,
                         effort_limit: torch.Tensor) -> None:
    shape = action.goal_mm.shape
    if len(shape) != 2:
        raise ValueError("action fields must be [N, J]")
    if active.shape != shape[:1] or active.dtype != torch.bool:
        raise ValueError("active must be bool [N]")
    if dofs.shape != shape[1:] or dofs.dtype != torch.bool:
        raise ValueError("dofs must be bool [J]")
    for name, value in (
            ("action.max_speed_mm_s", action.max_speed_mm_s),
            ("action.effort_limit", action.effort_limit),
            ("goal_mm", goal_mm),
            ("max_speed_mm_s", max_speed_mm_s),
            ("effort_limit", effort_limit)):
        if value.shape != shape:
            raise ValueError(f"{name} must be {tuple(shape)}")


def reset_primitive_state(state: PrimitiveState,
                          rows: torch.Tensor) -> PrimitiveState:
    """Fresh primitive state only for selected environment rows."""
    selected = rows[:, None]
    return PrimitiveState(
        elapsed_ticks=torch.where(rows, 0, state.elapsed_ticks),
        quiet_ticks=torch.where(selected, 0, state.quiet_ticks),
        retired=torch.where(selected, False, state.retired),
        succeeded_joints=torch.where(
            selected, False, state.succeeded_joints),
        stopped_at_mm=torch.where(selected, 0.0, state.stopped_at_mm),
        timed_out=torch.where(rows, False, state.timed_out),
        reached_goal_without_contact=torch.where(
            rows, False, state.reached_goal_without_contact),
        done=torch.where(rows, False, state.done),
        failed=torch.where(rows, False, state.failed),
    )


def reset_twist_stroke_state(state: TwistStrokeState,
                             rows: torch.Tensor) -> TwistStrokeState:
    """Restart the snippet for selected rows while other envs keep running."""
    selected = rows[:, None]
    return TwistStrokeState(
        phase=torch.where(rows, RELEASE, state.phase),
        phase_elapsed_ticks=torch.where(rows, 0, state.phase_elapsed_ticks),
        phase_quiet_ticks=torch.where(selected, 0, state.phase_quiet_ticks),
        phase_retired=torch.where(selected, False, state.phase_retired),
        phase_succeeded_joints=torch.where(
            selected, False, state.phase_succeeded_joints),
        phase_stopped_at_mm=torch.where(
            selected, 0.0, state.phase_stopped_at_mm),
        gripped_at_mm=torch.where(selected, 0.0, state.gripped_at_mm),
        grip_valid=torch.where(selected, False, state.grip_valid),
        timed_out=torch.where(rows, False, state.timed_out),
        reached_goal_without_contact=torch.where(
            rows, False, state.reached_goal_without_contact),
        done=torch.where(rows, False, state.done),
        failed=torch.where(rows, False, state.failed),
        turn_stalled=torch.where(rows, False, state.turn_stalled),
    )


def _check_twist_inputs(state: TwistStrokeState, active: torch.Tensor,
                        radius_mm: torch.Tensor,
                        parameters: TwistStrokeParameters,
                        n: int, joints: int) -> None:
    if active.shape != (n,) or active.dtype != torch.bool:
        raise ValueError("active must be bool [N]")
    if radius_mm.shape != (n,):
        raise ValueError("radius_mm must be [N]")
    for name, value in (
            ("release_clearance_mm", parameters.release_clearance_mm),
            ("finger_span_mm", parameters.finger_span_mm),
            ("contact_effort", parameters.contact_effort),
            ("grip_effort", parameters.grip_effort),
            ("travel_timeout_ticks", parameters.travel_timeout_ticks),
            ("contact_timeout_ticks", parameters.contact_timeout_ticks)):
        if value.shape != (n,):
            raise ValueError(f"{name} must be [N]")
    if (parameters.turn_effort is not None
            and parameters.turn_effort.shape != (n,)):
        raise ValueError("turn_effort must be [N]")
    for name, value in (
            ("travel_speed_mm_s", parameters.travel_speed_mm_s),
            ("grip_speed_mm_s", parameters.grip_speed_mm_s),
            ("travel_effort", parameters.travel_effort)):
        if value.shape != (n, joints):
            raise ValueError(f"{name} must be [N, J]")
    if state.phase.shape != (n,) or state.phase_retired.shape != (n, joints):
        raise ValueError("twist stroke state does not match observation [N, J]")


def _check_press(press: TwistPress | None, n: int, joints: int,
                 device: torch.device) -> None:
    if press is None:
        return
    if not 0 <= press.dof < joints:
        raise ValueError(f"press dof {press.dof} outside [0, {joints})")
    if press.active.shape != (n,) or press.active.dtype != torch.bool:
        raise ValueError("press.active must be bool [N]")
    for name, value in (("goal_mm", press.goal_mm),
                        ("return_mm", press.return_mm),
                        ("effort", press.effort),
                        ("return_effort", press.return_effort)):
        if value.shape != (n,):
            raise ValueError(f"press.{name} must be [N], got {tuple(value.shape)}")
        if value.device != device:
            raise ValueError(f"press.{name} must be on {device}")


def joint_mask(joint_ids: torch.Tensor, selected: Rows[int]) -> torch.Tensor:
    """`[J]` bool selecting a named mechanism group.

    The Python loop is over `LAYOUT`'s static roles, never over anything a
    sensor said, so it costs one build per tick and no synchronisation.
    """
    mask = torch.zeros_like(joint_ids, dtype=torch.bool)
    for dof in selected:
        mask |= joint_ids == dof
    return mask


def strokes_for_revolutions(revolutions: torch.Tensor, radius_mm: torch.Tensor,
                            finger_span_mm: torch.Tensor | float) -> torch.Tensor:
    """How many half-twists turn a probed object `revolutions` times. `[N]` int.

    One stroke rolls the object by at most one finger travel or half a turn.
    The count therefore follows from the *measured* radius but never drops
    below two strokes per revolution.

    `clamp_min(1)` because a stroke count of zero is a task that grips an
    object and then reports success without turning it; the floor makes an
    under-measured radius a short run rather than a silent no-op.

    `finger_span_mm` is a constant of the task and may be supplied as a
    Python float -- the math divides a tensor by it, broadcasting to `[N]`,
    and the divide-by-zero guard it needs (`clamp_min(1e-6)`) only matters
    against a tensor.
    """
    span = (torch.as_tensor(finger_span_mm, dtype=radius_mm.dtype,
                            device=radius_mm.device)
            if not isinstance(finger_span_mm, torch.Tensor) else finger_span_mm)
    by_span = revolutions * 2 * math.pi * radius_mm / span.clamp_min(1e-6)
    return torch.ceil(torch.maximum(by_span, 2 * revolutions)).to(
        torch.int64).clamp_min(1)


# ── phase tables ─────────────────────────────────────────────────────────────

Value = float | torch.Tensor | Callable[["Measures"], "float | torch.Tensor"]
"""A number, an `[N]` tensor, or a callable of the measurements taken so far."""
Dofs = int | tuple[int, ...]


def lift_effort(hand: HandConfig, register: float = 0.0) -> float:
    """Effort for a row that RAISES a joint against gravity.

    Floored at z's measured `torque_min_to_move`: under it the row runs,
    expires and reports nothing wrong while the stage never moved, and neither
    backend can see that. Every z ascent in every task uses this.
    """
    return max(register, float(hand.gain_vector("torque_min_to_move")[Z])) / 1000.0


# ── rows ─────────────────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True, kw_only=True)
class Row:
    """What every phase carries. Subclasses add only what they alone need.

    **Every field is keyword-only, here and in every subclass.** A row is a
    handful of bench numbers that are interchangeable by type -- an effort, a
    deadline, a goal, three bools -- so a positional call is a silent swap
    waiting to happen, which is the same failure the pairing helpers at the top
    of this file exist to refuse. It also stops `label` from sitting in the
    first slot looking like something the engine reads.
    """

    label: str = ""
    """Free text for the bench trace, and nothing else.

    Optional because it is a comment that happens to be a string: no code
    branches on it, nothing validates it, nothing requires it to be unique, and
    the hand does the identical thing if it is left out. `studio.py` records the
    matching rule for task names -- selection is `Config.sets_datum`, never
    `name == "zero"` -- and rows never earned an identity either.

    Omitted, `_trace` falls back to the row's type and position (`move 0`,
    `probe 1`), which is all the trace really has to do: name which of nine rows
    the hand is sitting on.
    """
    effort: Value | None = None
    """Normalized [0, 1]. None takes the task's travel effort, or a `Probe`'s
    approach effort."""
    creep: bool = False
    """Command the slow contact speed instead of the hand's travel speed."""
    loaded: bool = False
    """This row pushes an object -- see the module docstring."""
    seconds: float | None = None
    """Deadline override. None derives one from the DOFs the row moves, at the
    speed it commands."""


@dataclass(frozen=True, slots=True, kw_only=True)
class Move(Row):
    """Drive DOFs to a goal; done when every one is inside tolerance."""

    goal: dict[Dofs, Value] | None = None
    tolerance_mm: float = 1.0
    accept_stall: bool = False
    """Also accept a confirmed stop short of the goal, without changing effort."""


@dataclass(frozen=True, slots=True, kw_only=True)
class Probe(Row):
    """Close DOFs until they touch. Reaching the goal instead is a failure.

    An empty jaw at its closed limit is not a measurement, so it is neither
    contact nor timeout: it fails at once rather than becoming a radius of zero
    that every later row silently inherits.
    """

    group: Dofs = ()
    goal: Value = 0.0
    """How far the close is allowed to travel if it never touches anything.

    Zero is the usual case -- a jaw shutting on nothing ends at its own stop.
    A probe names a floor when closing past one would hit structure before the
    object: `syringe`'s aux jaw stops at the plunger's flange guide, and its
    final seat stops above the barrel's flanges. Reaching this without contact
    is still a failure, which is the whole point of a `Probe`.
    """
    measure: dict[str, int] | None = None
    """`{"radius": AUX_JAW}`. A probe closing two jaws may name both."""
    grip: Value | None = None
    """Latch this effort on the tick it touches, holding what it found."""


@dataclass(frozen=True, slots=True, kw_only=True)
class Hold(Row):
    """Change a standing command, or keep one, for `seconds`.

    Contact is already established when this runs after a `Probe`, so it does
    not probe the same object twice; it raises the effort cap the probe left
    behind and continues on the next tick.

    `seconds` makes it a settle: an empty `group` commands nothing and every
    DOF simply keeps what an earlier row left on it. That is what a settle
    *is*, which is why there is no separate `Wait` row -- one row type, and it
    reuses `Row.seconds` because for a row that only holds, its deadline is its
    duration.

    Seconds, never ticks, and the conversion happens in `Sequence` where
    `control_hz` lives. A task computing `round(s * hand.control_hz)` itself is
    doing the arithmetic in the one place that cannot check it: the result is
    wrong by exactly the ratio between the configured `control_hz` and the rate
    the loop really achieved, silently. `studio.report_task_rate` prints that
    ratio at the end of every task.
    """

    group: Dofs = ()
    goal: Value = 0.0
    interruptible: bool = False
    """End a timed hold early when any external/contact signal is true."""


@dataclass(frozen=True, slots=True, kw_only=True)
class Twist(Row):
    """One release / reset / re-grip / turn stroke -- see `twist_stroke`."""

    jaw: int = 0
    left: int = 0
    right: int = 0
    radius: Value = 0.0
    span: Value = 0.0
    """Full sweep of one finger during the turn, mm."""
    clearance: Value = 0.0
    """How far past the measured radius the jaw opens between strokes, mm."""
    grip: Value = 0.0
    """Holding effort while turning. A stroke is loaded by definition."""
    turn: Value | None = None
    """Optional finger-turn effort. None uses `grip`; setting it separately
    leaves jaw release and re-grip strength unchanged."""
    reverse: torch.Tensor | None = None
    press: TwistPress | None = None
    measure: dict[str, int] | None = None
    """The re-grip position, which tracks an object as it unscrews upward."""
    count: Value | None = None
    """`[N]` repetitions. None runs once; a number, tensor, or `lambda m:`
    matches the same callable seam `radius` already uses, so the count can be
    a function of the measured radius without the caller building a `Loop`.
    """
    stop_on_stall: bool = False
    """Finish a counted Twist when its turn stalls short of the finger goal."""


@dataclass(frozen=True, slots=True, kw_only=True)
class Loop:
    """Repeat `rows` `count` times.

    `count` is `[N]`, so one environment may run three strokes and another six
    off the same declaration.
    """

    count: Value
    rows: Rows[Row]


class Measures:
    """`m.radius` -> `[N]`. What an earlier row named, for a later row's goal."""

    __slots__ = ("columns", "values")

    def __init__(self, columns: dict[str, int], values: torch.Tensor):
        object.__setattr__(self, "columns", columns)
        object.__setattr__(self, "values", values)

    def __getattr__(self, name: str) -> torch.Tensor:
        columns = object.__getattribute__(self, "columns")
        if name not in columns:
            raise AttributeError(
                f"no measurement {name!r}: a row must name it before another "
                f"reads it. Named so far: {sorted(columns) or 'nothing'}")
        return object.__getattribute__(self, "values")[:, columns[name]]


@dataclass(frozen=True, slots=True)
class SequenceState:
    """One state type for every task."""

    phase: torch.Tensor          # [N] index into the flattened row list
    phase_ticks: torch.Tensor    # [N] ticks spent in it, for a timed Hold
    loops: torch.Tensor          # [N, L] completed repetitions per Loop
    twist_counts: torch.Tensor   # [N, T] completed repetitions per Twist that carries its own count
    measured: torch.Tensor       # [N, M] one column per named measurement
    held_goal_mm: torch.Tensor   # [N, J] standing command, carried between rows
    held_speed_mm_s: torch.Tensor        # so a DOF no row names keeps the grip
    held_effort: torch.Tensor            # an earlier row took
    motion: PrimitiveState
    twist: TwistStrokeState
    done: torch.Tensor           # [N] bool
    failed: torch.Tensor         # [N] bool


class Sequence:
    """Walk a declared phase list. The same policy object for every task."""

    def __init__(self, rows: Rows[Row | Loop], *, hand: HandConfig,
                 start_mm: torch.Tensor, travel_torque: float,
                 approach_torque: float, approach_speed: float,
                 timeout_margin: float = 1.5, travel_speed: float | None = None,
                 stall_fallback: bool = True):
        """`travel_speed` is counts/s for every free move; None takes the hand's
        own `speed` gains.

        **Keyword-only past `rows`.** Five of the eight are bare floats in bench
        units, and three of those are torques that differ only in which move
        they cap. Positionally that is `..., 50.0, 150.0, 300.0, 1.5, 1100.0`,
        where transposing any adjacent pair type-checks, runs, and shows up as a
        hand that creeps or grips wrong -- the same silent-swap failure the
        pairing helpers at the top of this file exist to refuse.

        A task owns this because `config.STANDARD_SPEED` is one number for the
        whole session and is set by what `zero` needs: hand_2's fingers sit at
        200 counts/s (2.45 mm/s), so a 40 mm twist sweep takes 16 s. The seek
        wants slow -- a fast creep overshoots a hard stop and climbs a gear
        tooth -- and manipulation wants fast, and those are different moves.
        `Motions` could not express the difference (it carries goal and torque
        but no speed channel, which MEMORY records as unfixable there); a direct
        policy commands `Action.max_speed_mm_s` every tick, so here it is free.

        Deadlines are derived from this same number, so raising it shortens the
        budgets with it rather than leaving them stale and over-generous.
        """
        self.rows: list[Row] = []
        self.next_index: list[int] = []
        self.loops: list[tuple[int, Value]] = []   # (first row, count)
        self.closes: dict[int, int] = {}           # last row -> loop id
        # Per-twist repetition: a `Twist` may carry its own `count` and unroll
        # the `Loop` for the case the caller almost always wants. Maps row
        # index -> column index in `state.twist_counts`.
        self.twist_counts: list[tuple[int, Value]] = []
        self.twist_count_col: dict[int, int] = {}
        for item in rows:
            body = item.rows if isinstance(item, Loop) else [item]
            start = len(self.rows)
            self.rows.extend(body)
            self.next_index.extend(range(start + 1, start + len(body) + 1))
            if isinstance(item, Loop):
                self.closes[len(self.rows) - 1] = len(self.loops)
                self.loops.append((start, item.count))
            for offset, row in enumerate(body):
                if isinstance(row, Twist) and row.count is not None:
                    index = start + offset
                    if index in self.closes:
                        raise ValueError(
                            "a counted Twist cannot be the last row of a Loop")
                    self.twist_count_col[index] = len(self.twist_counts)
                    self.twist_counts.append((index, row.count))
        self.columns: dict[str, int] = {}
        for row in self.rows:
            for name in getattr(row, "measure", None) or ():
                self.columns.setdefault(name, len(self.columns))
        self.done_index = len(self.rows)
        self.stall_fallback = stall_fallback

        self.hand = hand
        self.n, self.n_dof = start_mm.shape
        self.device = start_mm.device
        self.margin = timeout_margin
        self.creep_counts = approach_speed
        self.travel_counts = (
            hand.gain_vector("speed", self.device).to(torch.float32)
            if travel_speed is None else
            torch.full((self.n_dof,), float(travel_speed), device=self.device))
        self.travel_speed = (self.travel_counts / hand.counts_per_mm)[
            None, :].expand(self.n, self.n_dof).clone()
        self.creep_speed = torch.full_like(
            self.travel_speed, approach_speed / hand.counts_per_mm)
        floor = (hand.gain_vector("torque_min_to_move", self.device)
                 .to(torch.float32) / 1000.0)
        flat = torch.full_like(self.travel_speed, travel_torque / 1000.0)
        # Every DOF, z included. z was exempt on the theory that its floor is
        # gravity and a descent wants less; that is wrong twice. Effort is a
        # force CAP and a free descent's error never approaches it, so the
        # exemption bought no gentleness -- and 50 against z's measured floor of
        # 800 does not make the stage gentle, it makes it immobile in both
        # directions. `cap`'s approach commands z and so could never complete:
        # it burned its 6.1 s deadline, and one failed row retires the whole
        # `Sequence`, so every row after it silently never ran.
        #
        # A row that genuinely presses z into something names its own effort,
        # which is NOT floored (`_effort` only raises a `loaded` row). That is
        # the contract `lift_effort` and `TwistPress.effort` already are.
        self.travel_effort = torch.maximum(flat, floor[None, :])
        approach = torch.full_like(
            self.travel_effort, approach_torque / 1000.0)
        # Per DOF: using `floor.max()` made every jaw inherit z's 800/1000
        # gravity floor.
        self.approach_effort = torch.maximum(approach, floor[None, :])

    # ── the tick ─────────────────────────────────────────────────────────────

    def initial_state(self, observation: Observation) -> SequenceState:
        n, joints = observation.position_mm.shape
        if joints != self.n_dof or n != self.n:
            raise ValueError(f"task built for [{self.n}, {self.n_dof}], "
                             f"got [{n}, {joints}]")
        zeros = torch.zeros(n, dtype=torch.int64, device=self.device)
        return SequenceState(
            phase=zeros.clone(),
            phase_ticks=zeros.clone(),
            loops=torch.zeros((n, max(1, len(self.loops))),
                              dtype=torch.int64, device=self.device),
            twist_counts=torch.zeros((n, max(1, len(self.twist_counts))),
                                     dtype=torch.int64, device=self.device),
            measured=torch.zeros((n, max(1, len(self.columns))),
                                 device=self.device),
            held_goal_mm=observation.position_mm.clone(),
            held_speed_mm_s=self.travel_speed,
            held_effort=self.travel_effort,
            motion=initial_primitive_state(observation),
            twist=initial_twist_stroke_state(observation),
            done=torch.zeros(n, dtype=torch.bool, device=self.device),
            failed=torch.zeros(n, dtype=torch.bool, device=self.device),
        )

    def step(self, observation: Observation,
             state: SequenceState) -> tuple[Action, SequenceState]:
        """One pure tensor tick. No environment chooses Python control flow."""
        active = ~state.done
        phase = state.phase
        measures = Measures(self.columns, state.measured)
        action = Action(state.held_goal_mm, state.held_speed_mm_s,
                        state.held_effort)
        motion, twist = state.motion, state.twist
        measured, loops, twist_counts = state.measured, state.loops, state.twist_counts
        # Counted here, for every row, rather than only inside the one row type
        # that reads it: a timed `Hold` needs it and so does the trace, and two
        # places incrementing the same counter is how it ends up double-counted.
        ticks = state.phase_ticks + active.to(state.phase_ticks.dtype)
        advanced = torch.zeros_like(active)
        failed = torch.zeros_like(active)
        target = phase.clone()

        for index, row in enumerate(self.rows):
            here = active & (phase == index)
            action, motion, twist, measured, ok, bad = self._run(
                row, observation, action, motion, twist, measured, ticks,
                here, measures)
            loop = self.closes.get(index)
            tc_col = self.twist_count_col.get(index)
            if loop is None and tc_col is None:
                nxt = torch.full_like(phase, self.next_index[index])
            elif loop is not None:
                start, count = self.loops[loop]
                counted = loops[:, loop] + ok.to(loops.dtype)
                again = ok & (counted < self._value(count, measures))
                loops = _write(loops, loop, ok, torch.where(again, counted, 0))
                nxt = torch.where(again, start, self.next_index[index])
            else:
                # A `Twist` carrying its own `count`: the inline form of a
                # single-row `Loop`.
                start, count = self.twist_counts[tc_col]
                counted = twist_counts[:, tc_col] + ok.to(twist_counts.dtype)
                stop = (twist.turn_stalled if row.stop_on_stall else
                        torch.zeros_like(ok))
                again = ok & ~stop & (counted < self._value(count, measures))
                twist_counts = _write(
                    twist_counts, tc_col, ok,
                    torch.where(again, counted, 0))
                nxt = torch.where(again, start, self.next_index[index])
            # A later Twist (including one in the opposite direction) must not
            # inherit STROKE_DONE from this one. Reset on every successful
            # stroke; the standing action issued on this tick is unchanged.
            if isinstance(row, Twist):
                twist = reset_twist_stroke_state(twist, ok)
            target = torch.where(ok, nxt, target)
            advanced |= ok
            failed |= bad

        if self.n == 1:
            self._trace(int(phase[0]), int(ticks[0]), observation, action,
                        bool(advanced[0] | failed[0]), bool(failed[0]),
                        bool(active[0]))

        return action, SequenceState(
            phase=target,
            phase_ticks=torch.where(advanced, 0, ticks),
            loops=loops,
            twist_counts=twist_counts,
            measured=measured,
            held_goal_mm=action.goal_mm,
            held_speed_mm_s=action.max_speed_mm_s,
            held_effort=action.effort_limit,
            motion=reset_primitive_state(motion, advanced),
            twist=twist,
            done=state.done | (active & (target >= self.done_index)) | failed,
            failed=state.failed | failed,
        )

    def _run(self, row, observation, action, motion, twist, measured, ticks,
             here, measures):
        """One row, for the environments sitting on it."""
        none = torch.zeros_like(here)

        if isinstance(row, Hold):
            # An empty group masks nothing, so `hold` is a no-op and the row is
            # purely a settle. No branch needed for the two uses.
            action = hold(action, here, self._mask(row.group),
                          self._spread(row.goal, measures),
                          self._speed(row), self._effort(row, measures))
            elapsed = (here if row.seconds is None else
                       here & ((ticks >= self._ticks(row.seconds))
                               | (row.interruptible
                                  & observation.contact.any(dim=1))))
            return action, motion, twist, measured, elapsed, none

        if isinstance(row, Twist):
            action, twist, result = twist_stroke(
                observation, action, twist, here,
                self._value(row.radius, measures), row.jaw, row.left, row.right,
                self._twist_parameters(row, measures),
                stall_fallback=self.stall_fallback,
                reverse=row.reverse, press=row.press)
            ok = here & result.succeeded
            measured = self._record(measured, row.measure,
                                    ok & result.stopped_valid.any(dim=1),
                                    result.stopped_at_mm)
            return (action, motion, twist, measured, ok,
                    here & (result.timed_out
                            | result.reached_goal_without_contact))

        if isinstance(row, Probe):
            dofs = self._mask(row.group)
            goal = self._spread(row.goal, measures)
            action, motion, result = close_until_contact(
                observation, action, motion, here, dofs, goal,
                self._speed(row), self._effort(row, measures, approach=True),
                self._deadline(row, dofs), stall_fallback=self.stall_fallback)
            ok = here & result.succeeded
            if row.grip is not None:
                action = hold(action, ok, dofs, goal, self._speed(row),
                              self._effort(row, measures, override=row.grip))
        else:
            dofs, goal = self._goals(row.goal, measures)
            action, motion, result = move_to(
                observation, action, motion, here, dofs, goal,
                self._speed(row), self._effort(row, measures),
                self._deadline(row, dofs), tolerance_mm=row.tolerance_mm,
                stall_fallback=row.loaded or row.accept_stall)
            ok = here & result.succeeded

        measured = self._record(measured, getattr(row, "measure", None), ok,
                                result.stopped_at_mm)
        return (action, motion, twist, measured, ok,
                here & (result.timed_out | result.reached_goal_without_contact))

    # ── what the bench sees ──────────────────────────────────────────────────

    def _trace(self, index: int, ticks: int, observation: Observation,
               action: Action, ended: bool, failed: bool,
               active: bool) -> None:
        """Narrate rows, once a second, on a single-environment run.

        A waiting row and a hung program are the same thing from outside the
        loop: the tick rate holds, the readout holds, and nothing names which
        of nine rows is in front or how much of its budget is left. Every bench
        report of "stuck after the Nth step" has so far cost a round trip to
        establish only that much. Printed rather than returned because the
        thing that needs it is a person watching a hand move.

        Guarded on `n == 1`, which is hardware and the bench; a batch of 4096
        would print 4096 rows' worth of one environment's opinion.
        """
        if not active:
            return
        row = self.rows[index]
        dofs = self._row_dofs(row).nonzero().flatten().tolist()
        at = "  ".join(f"{LABELS[d]} {observation.position_mm[0, d]:.1f}"
                       f"->{action.goal_mm[0, d]:.1f}" for d in dofs)
        elapsed = ticks / self.hand.control_hz
        name = row.label or f"{type(row).__name__.lower()} {index}"
        if ended:
            print(f"  {name:9s} "
                  f"{'FAILED' if failed else 'ok':6s} {elapsed:5.1f}s  {at}")
        elif ticks % round(self.hand.control_hz) == 0:
            budget = float(self._deadline(row, self._row_dofs(row))[0])
            print(f"  {name:9s} ...    {elapsed:5.1f}s of "
                  f"{budget / self.hand.control_hz:.1f}s  {at}")

    def _row_dofs(self, row: Row) -> torch.Tensor:
        """Which DOFs a row drives, for its deadline and its trace.

        Keys only -- a `Move`'s goal *values* may be callables that need
        measurements this row has not taken yet, and the mask never does.
        """
        if isinstance(row, Move):
            mask = torch.zeros(self.n_dof, dtype=torch.bool, device=self.device)
            for key in row.goal or {}:
                mask |= self._mask(key)
            return mask
        if isinstance(row, Twist):
            return self._mask((row.jaw, row.left, row.right))
        return self._mask(row.group)

    # ── bench units to tensors ───────────────────────────────────────────────

    def _mask(self, dofs: Dofs) -> torch.Tensor:
        index = torch.arange(self.n_dof, device=self.device)
        ids = (dofs,) if isinstance(dofs, int) else tuple(dofs)
        mask = torch.zeros(self.n_dof, dtype=torch.bool, device=self.device)
        for dof in ids:
            mask |= index == dof
        return mask

    def _value(self, value: Value, measures: Measures) -> torch.Tensor:
        """A row's `[N]` number, whether typed in or measured."""
        if callable(value):
            value = value(measures)
        if isinstance(value, torch.Tensor):
            return value if value.ndim else value.expand(self.n)
        return torch.full((self.n,), float(value), device=self.device)

    def _spread(self, value: Value, measures: Measures) -> torch.Tensor:
        return self._value(value, measures)[:, None].expand(self.n, self.n_dof)

    def _goals(self, goal: dict, measures: Measures):
        """`{dofs: value}` -> the `[J]` mask it names and its `[N, J]` goal."""
        dofs = torch.zeros(self.n_dof, dtype=torch.bool, device=self.device)
        out = torch.zeros((self.n, self.n_dof), device=self.device)
        for key, value in goal.items():
            here = self._mask(key)
            dofs |= here
            out = torch.where(here[None, :],
                              self._value(value, measures)[:, None], out)
        return dofs, out

    def _speed(self, row: Row) -> torch.Tensor:
        return self.creep_speed if row.creep else self.travel_speed

    def _effort(self, row: Row, measures: Measures, approach: bool = False,
                override: Value | None = None) -> torch.Tensor:
        """The row's `[N, J]` effort cap.

        A `loaded` row never drops below the effort that took the grip: every
        old task carried that comment, and dropping to travel torque silently
        fails to pull free of what is being held.
        """
        value = override if override is not None else row.effort
        effort = (self.approach_effort if value is None
                  and (approach or row.loaded) else
                  self.travel_effort if value is None else
                  self._spread(value, measures))
        return (torch.maximum(effort, self.travel_effort) if row.loaded
                else effort)

    def _ticks(self, seconds: float) -> torch.Tensor:
        """`[N]` control ticks. The only place seconds become ticks."""
        return torch.full((self.n,),
                          max(1, round(float(seconds) * self.hand.control_hz)),
                          dtype=torch.int64, device=self.device)

    def _deadline(self, row: Row, dofs: torch.Tensor) -> torch.Tensor:
        """`[N]` ticks for this row's own travel at the speed it commands."""
        if row.seconds is not None:
            return self._ticks(row.seconds)
        ids = dofs.nonzero().flatten().tolist()
        counts_per_s = (self.creep_counts if row.creep else
                        float(self.travel_counts[ids].min()))
        rail = max(self.hand.travel_mm[dof] for dof in ids)
        return self._ticks(
            self.margin * rail * self.hand.counts_per_mm / counts_per_s)

    def _twist_parameters(self, row: Twist,
                          measures: Measures) -> TwistStrokeParameters:
        """The stroke's own knobs: its re-grip creeps, its turn does not, so
        the two get separate deadlines."""
        jaw = self._mask(row.jaw)
        return TwistStrokeParameters(
            release_clearance_mm=self._value(row.clearance, measures),
            finger_span_mm=self._value(row.span, measures),
            travel_speed_mm_s=self.travel_speed,
            grip_speed_mm_s=self.creep_speed,
            travel_effort=self.travel_effort,
            contact_effort=self.approach_effort[:, row.jaw],
            grip_effort=self._value(row.grip, measures),
            travel_timeout_ticks=self._deadline(row, ~jaw),
            contact_timeout_ticks=self._deadline(replace(row, creep=True), jaw),
            turn_effort=(None if row.turn is None
                         else self._value(row.turn, measures)),
        )

    def _record(self, measured: torch.Tensor, measure: dict | None,
                rows: torch.Tensor,
                stopped_at_mm: torch.Tensor) -> torch.Tensor:
        for name, dof in (measure or {}).items():
            measured = _write(measured, self.columns[name], rows,
                              stopped_at_mm[:, dof])
        return measured


def _write(table: torch.Tensor, column: int, rows: torch.Tensor,
           values: torch.Tensor) -> torch.Tensor:
    """Set one column of an `[N, K]` table for the rows that changed it."""
    index = torch.arange(table.shape[1], device=table.device) == column
    return torch.where(rows[:, None] & index, values[:, None], table)

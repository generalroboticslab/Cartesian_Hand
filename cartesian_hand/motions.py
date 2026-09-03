"""Motion engine — a robot protocol expressed as tensors, advanced one tick at a time.

A task is normally a Python program, and a Python program cannot be searched by
an optimizer or run 4096 times at once: its behaviour lives in loop bounds and
`if` statements, and a program counter is not a tensor. This module is that
program turned into data.

Shapes
------
Program  [N, J, K]  written once by `Program.build`, read-only during a run
Runtime  [N, J]     plus one step counter per env, [N]

    N  envs in sim, or physical hands on real
    J  joints
    K  steps in the program

A **motion** is one cell: one joint, one goal, one torque, one stop rule.
Motions on different joints at the same step run concurrently; motions on the
same joint at different steps run in order. That ordering is what the cap task
needs — the aux jaw must release before the fingers slide, and be back on the
cap before they turn.

Cells are the data, not the notation. A task authors a step by indexing joints
and assigning a broadcast value — `Step.set` — the same way the tick reads them,
so there is no per-joint Python object anywhere between a task and the servo.

Standing orders
---------------
`step_once` returns the last commanded goal and torque for every joint, not
only for joints still running. A joint that has finished its motion keeps
commanding what it last asked for, exactly like `hand.target` keeps whatever
`set_pos` last wrote. That is what makes a grip work: `squeeze` commands the
jaw *past* the object (goal 0.0, reduced torque) and lets the object stop it,
so the standing order is what keeps the pressure on. Zeroing a retired joint's
goal to its current position would release every grip the moment it was made,
and dropping its torque would do the same.

Design decisions
----------------
Torch, not numpy, because `device="cuda"` is meant to be the only difference
between the hardware and sim backends, and numpy has no GPU. Nothing here needs
autograd.

Deadlines are tick counts, not wall-clock seconds: sim has no wall clock and
does not run at real time, so a tick is the only unit meaning the same thing on
both backends.

Every cell is a number and every update is a `torch.where` or an in-place
masked fill — no Python callables in the arrays, no boolean-mask indexing, no
branch on motion type. One Python `if` here would make this a loop over N.

The step barrier is global per env rather than one counter per joint. Per-joint
counters would let a joint run ahead of the others, and the ordering above is
exactly what must not be lost. Cost is no overlap between steps, which the cap
task does not need.

Contact is sensed as stall (velocity), not as load. The servos do report a load
byte, but it reads zero at rest on every servo on both hands, so a decode that
works and a decode returning padding are indistinguishable today. A stop
condition that cannot be verified is worse than one that does not exist, so
there is none until the bytes are proven.
"""
from collections.abc import Generator, Sequence
from dataclasses import dataclass
from typing import NamedTuple

import torch

# What a goal or a torque may be written as: a Python scalar, a per-env or
# per-joint sequence, or an already-built tensor. `Step._spread` normalizes them.
Broadcast = float | Sequence[float] | torch.Tensor
Device = str | torch.device

RUNNING, GOAL, STUCK, TIMEOUT, EXTERNAL = 0, 1, 2, 3, 4

# Arm-time predicates over the last row a joint executed. Names live in one
# table for the same reason stop rules do: GUI, generated task and engine share
# one vocabulary, and integer tensors remain batchable.
ALWAYS, STOPPED_OK, STOPPED_FAILED, STOPPED_AT_GE, STOPPED_AT_LE = range(5)
PREDICATES = {
    "always": ALWAYS,
    "stopped_ok": STOPPED_OK,
    "stopped_failed": STOPPED_FAILED,
    "stopped_at_ge": STOPPED_AT_GE,
    "stopped_at_le": STOPPED_AT_LE,
}


@dataclass(frozen=True)
class When:
    """Arm a row from one joint's most recent executed-row result.

    Evaluated once when the row arms, never while it runs. `stopped_at_*`
    predicates require that source row to have succeeded; a timeout retains the
    previous valid position but cannot make a new row use it as fresh contact.
    """
    kind: str
    dof: int
    threshold_mm: float = 0.0

    def __post_init__(self) -> None:
        if self.kind not in PREDICATES or self.kind == "always":
            raise ValueError(
                f"unknown runtime predicate {self.kind!r}, expected one of "
                f"{tuple(k for k in PREDICATES if k != 'always')}")
        if self.dof < 0:
            raise ValueError(f"predicate source DOF must be nonnegative, got {self.dof}")

# What a task may ask for, and the outcome code each ask is satisfied by. One
# table, so a stop rule's name, the criterion it arms and the outcome that means
# it worked cannot drift apart -- `succeeded` is literally `outcome == wants`.
#
# `hold` maps to TIMEOUT rather than to a code of its own: a hold arms no
# condition and spends its (1-tick) budget on purpose, which is how a standing
# order gets placed. Naming that TIMEOUT is not a fudge, it is what it wants.
#
# `wait` is the same trick with the budget left alone: arm nothing, spend the
# seconds, succeed. A dwell -- hold the pipette plunger down while liquid moves,
# sit at each end of a sweep -- has always been expressible and never had a
# name, so tasks that need one reach for `time.sleep`, which no sim can run.
WANTS = {"goal": GOAL, "stuck": STUCK, "hold": TIMEOUT, "wait": TIMEOUT,
         "external": EXTERNAL}

# For humans reading a bench log. Derived from the codes so it cannot drift.
OUTCOME_NAMES = {GOAL: "goal", STUCK: "stuck", TIMEOUT: "timeout",
                 EXTERNAL: "external", 0: "never ran"}

POSITION_TOLERANCE_MM = 1.0
STUCK_SPEED_MM_S      = 0.3    # matches tasks/primitives.py wait_for_stall
# 1.0 s at 50 Hz, matching wait_for_stall_counts' confirm_s. Not shorter: the
# encoder quantizes to whole counts, so a window of W ticks resolves speed only
# in steps of one count per W. At 81.5 counts/mm a 5-tick window makes the 0.3
# mm/s threshold 2.4 counts wide, and a joint dithering +/-1 count against a
# stop eats half of it -- the bug that recorded hard stops mid-rail on hand_1.
STUCK_WINDOW_STEPS    = 10

STOPS = tuple(WANTS)

# What a goal is measured from. "here" is resolved to the joint's own position
# at the moment its step arms (see `Motions._arm`), so a task writes the
# *distance* it wants to travel and never the origin it starts from.
#
# That is what zeroing needs: it runs before millimetres are calibrated, so
# every goal it commands must be `here ± something` for the frame's origin to
# cancel. Without this a task has to yield, read the measurement back, and
# build a second program just to add a constant to it.
FRAMES = ("abs", "here")


class Motions:
    """Executes an [N, J, K] motion program. No servo, no bus, no threads, no clock."""

    def __init__(self, n_envs: int, n_joints: int, n_steps: int, hz: float,
                 device: Device = "cpu",
                 stuck_speed_mm_s: float = STUCK_SPEED_MM_S,
                 position_tolerance_mm: float = POSITION_TOLERANCE_MM) -> None:
        N, J, K = n_envs, n_joints, n_steps
        prog  = lambda dt: torch.zeros((N, J, K), dtype=dt, device=device)
        joint = lambda dt: torch.zeros((N, J), dtype=dt, device=device)
        self.hz, self.K, self.device = hz, K, device
        self.stuck_speed_mm_s = stuck_speed_mm_s
        self.position_tolerance_mm = position_tolerance_mm

        # ── program ───────────────────────────────────────────────────────────
        self.acts            = prog(torch.bool)     # joint acts at this step
        self.goal_mm         = prog(torch.float32)
        self.torque          = prog(torch.float32)
        self.timeout_steps   = prog(torch.int32)
        # Which stop condition this cell arms, as the outcome code that means it
        # worked -- see `WANTS`. TIMEOUT, not zeros, so a cell nobody wrote
        # wants what an unarmed row wants.
        self.wants           = torch.full((N, J, K), TIMEOUT, dtype=torch.int8,
                                          device=device)
        # Whether this cell's goal is a distance from where the joint is when
        # the step arms, rather than an absolute millimetre. False, so a cell
        # nobody wrote is absolute -- see `FRAMES`.
        self.relative        = prog(torch.bool)
        self.outcome         = prog(torch.int8)
        # Arm-time condition over one source joint's latest result. ALWAYS for
        # ordinary/build-time-masked rows.
        self.predicate_kind      = prog(torch.int8)
        self.predicate_source    = prog(torch.int64)
        self.predicate_threshold = prog(torch.float32)
        # Which authored cells really armed after resolving runtime predicates.
        # `acts` is source code; `executed` is what search/result scoring grades.
        self.executed         = prog(torch.bool)
        # What the joint was doing at the instant its row ended: the windowed
        # speed the stall test read, and where it was. Diagnostic only -- nothing
        # branches on these -- but a row that wanted STUCK and timed out is
        # otherwise indistinguishable between "it never touched anything", "it
        # touched something and the detector could not see it", and "it closed
        # all the way, which STUCK deliberately refuses to call contact". Those
        # have three different fixes and the log could not tell them apart.
        self.stop_speed_mm_s  = prog(torch.float32)
        self.stop_position_mm = prog(torch.float32)

        # ── runtime ───────────────────────────────────────────────────────────
        self.step               = torch.zeros(N, dtype=torch.int64, device=device)
        self.retired            = joint(torch.bool)
        self.active_step        = joint(torch.bool)
        self.steps_remaining    = joint(torch.int32)
        self.anchor_mm          = joint(torch.float32)   # tumbling stall window
        self.steps_since_anchor = joint(torch.int32)
        self.held_goal          = joint(torch.float32)   # the standing orders
        self.held_torque        = joint(torch.float32)
        # What a relative row's goal is added to: the position captured when
        # this step armed. Zero for absolute rows, so one add serves both and
        # the tick needs no branch on the frame.
        self.goal_base          = joint(torch.float32)
        # Latest executed row for each joint. Failure invalidates `stopped_ok`
        # but never overwrites the last trustworthy position with a timeout.
        self.stopped_at         = joint(torch.float32)
        self.stopped_ok         = joint(torch.bool)
        self.stopped_valid      = joint(torch.bool)

    # ── program access ────────────────────────────────────────────────────────

    def _index(self, n_joints: int) -> torch.Tensor:
        """[N, J, 1] gather index for each env's current step."""
        return self.step.clamp(max=self.K - 1)[:, None, None].expand(-1, n_joints, 1)

    def _at(self, program: torch.Tensor,
            idx: torch.Tensor | None = None) -> torch.Tensor:
        """[N, J] slice of an [N, J, K] program at each env's current step.

        `idx` is `_index`, hoisted by a caller that reads several programs at
        the same step -- `step_once` reads six, and rebuilding the index for
        each is six expands per tick per env. Pass it only when `self.step` has
        not moved since it was built.
        """
        if idx is None:
            idx = self._index(program.shape[1])
        return program.gather(2, idx).squeeze(-1)

    def _arm(self, where: torch.Tensor, position_mm: torch.Tensor) -> None:
        """Arm the row at the current step, for envs selected by `where` [N].

        Runs at start and again on every advance, and the two must stay
        identical: a row armed without resetting the stall anchor inherits the
        previous row's, so `steps_since_anchor` is already past the window and
        the joint reports STUCK on its first tick -- a contact that never
        happened, reported as a measurement.

        A joint that acts at this step is RE-armed: it may have retired in a
        previous step (reached its goal) and now has a new role (probe the
        object, take a grip, turn). A joint that idles here keeps its retired flag
        if it has one; retirement only happens in `step_once` via stop rules
        or timeouts. Idling is handled by the standing orders.

        State updates only apply to joints that act at this step. A joint with
        no `Move` here has `timeout_steps=0` and would expire instantly if we
        wrote that into `steps_remaining`.

        Arming is also where a relative goal becomes a number: `goal_base` takes
        the joint's position now, and `step_once` adds it. Resolving here rather
        than in the tick is what makes "here" mean *where the step started*
        instead of a goal that runs away from a joint chasing it.
        """
        idx = self._index(position_mm.shape[1])
        authored = self._at(self.acts, idx)
        kind = self._at(self.predicate_kind, idx)
        source = self._at(self.predicate_source, idx).clamp(max=position_mm.shape[1] - 1)
        source_ok = self.stopped_ok.gather(1, source)
        source_valid = self.stopped_valid.gather(1, source)
        source_at = self.stopped_at.gather(1, source)
        threshold = self._at(self.predicate_threshold, idx)
        enabled = torch.where(kind == ALWAYS, torch.ones_like(authored),
                  torch.where(kind == STOPPED_OK, source_valid & source_ok,
                  torch.where(kind == STOPPED_FAILED, source_valid & ~source_ok,
                  torch.where(kind == STOPPED_AT_GE, source_ok & (source_at >= threshold),
                              source_ok & (source_at <= threshold)))))
        active = where[:, None] & authored & enabled          # [N, J]
        self.active_step = torch.where(where[:, None], active, self.active_step)
        # Record what really armed. Search and generated-task results must not
        # grade a condition-disabled cell as a failed command.
        was_executed = self.executed.gather(2, idx).squeeze(-1)
        self.executed.scatter_(
            2, idx, torch.where(where[:, None], active,
                                was_executed)[..., None])
        # Every joint in an advancing env starts retired, then active joints are
        # opened. A predicate-disabled authored cell must not block the barrier.
        self.retired.masked_fill_(where[:, None], True)
        self.retired.masked_fill_(active, False)
        self.goal_base          = torch.where(
            active, torch.where(self._at(self.relative, idx), position_mm,
                                torch.zeros_like(position_mm)),
            self.goal_base)
        self.steps_remaining    = torch.where(active, self._at(self.timeout_steps, idx),
                                              self.steps_remaining)
        self.anchor_mm          = torch.where(active, position_mm, self.anchor_mm)
        self.steps_since_anchor.masked_fill_(active, 0)

    def start(self, position_mm: torch.Tensor, torque: Broadcast) -> None:
        """Arm step 0.

        `position_mm` [N, J] is where the joints are right now, and seeds the
        standing orders: a joint the program never touches must be told to stay
        where it is, not to go to zero. `torque` is scalar or [N, J] and seeds
        the same registers, so an untouched joint keeps the hand's current gain
        rather than going limp.
        """
        self.held_goal   = position_mm.to(torch.float32).clone()
        self.held_torque = torch.as_tensor(
            torque, dtype=torch.float32,
            device=self.device).expand_as(self.held_goal).clone()
        self._arm(torch.ones_like(self.step, dtype=torch.bool), position_mm)

    def done(self) -> bool:
        return bool((self.step >= self.K).all())

    def succeeded(self) -> torch.Tensor:
        """[N, J, K] — rows that ended the way their stop rule implied they should.

        One comparison, because `wants` already stores the criterion as the
        outcome code that satisfies it. This used to be two boolean planes
        (`stop_at_goal`, `stop_when_stuck`) resolved by a nested `where` — four
        representable states for three meanings, with both-true silently
        resolved in favour of STUCK. A single int cannot spell that.

        Index with `acts` to ignore cells where a joint deliberately idled.
        """
        return self.outcome == self.wants

    def all_reached(self, dofs: int | Sequence[int], step: int = 0) -> torch.Tensor:
        """[N] — did every joint in `dofs` end `step` the way its rule wanted?

        Reduces over the joints and keeps N: envs are independent, so one env's
        finger missing its stop says nothing about the other 4095. `step`
        defaults to 0 because the caller is usually a one-step program.
        """
        ids = [dofs] if isinstance(dofs, int) else list(dofs)
        return self.succeeded()[:, ids, step].all(dim=1)

    # ── the tick ──────────────────────────────────────────────────────────────

    def step_once(self, position_mm: torch.Tensor,
                  external: torch.Tensor | None = None
                  ) -> tuple[torch.Tensor, torch.Tensor]:
        """Advance every env by one tick. Returns (goal_mm, torque), both [N, J].

        `position_mm` is [N, J] and is treated as READ-ONLY. `external`, when
        supplied, is exactly [N,J] bool: an explicit backend-owned signal for
        rows with `stop="external"`. None means all false -- never infer contact
        from load, a limit, or a missing producer.
        """
        if external is None:
            external = torch.zeros_like(position_mm, dtype=torch.bool)
        else:
            external = torch.as_tensor(external, dtype=torch.bool, device=self.device)
            if external.shape != position_mm.shape:
                raise ValueError(
                    f"external has shape {tuple(external.shape)}, expected "
                    f"{tuple(position_mm.shape)} [N, J]")
        pos    = position_mm
        # Built once: six of the reads below are the same step of six different
        # programs, and `self.step` does not move until the barrier at the end.
        idx    = self._index(pos.shape[1])
        # `goal_base` is the arm-time position on relative rows and 0 on
        # absolute ones, so this one add covers both frames with no branch.
        goal   = self._at(self.goal_mm, idx) + self.goal_base
        torque = self._at(self.torque, idx)
        live   = ~self.retired & self.active_step & (self.step < self.K)[:, None]
        live_i = live.to(torch.int32)                # torch refuses bool arithmetic

        # Age the clocks BEFORE testing them, or a 1-tick budget survives 2 ticks.
        self.steps_remaining    -= live_i
        self.steps_since_anchor += live_i

        # * hz because the division yields mm-per-tick, not mm/s.
        mm_s    = ((pos - self.anchor_mm).abs()
                   / self.steps_since_anchor.clamp(min=1) * self.hz)
        looked  = self.steps_since_anchor >= STUCK_WINDOW_STEPS
        at_goal = (pos - goal).abs() <= self.position_tolerance_mm
        # ~at_goal: a joint that ARRIVED also has zero speed. Without this term
        # an empty gripper closes fully, reads no motion, and reports contact --
        # and the caller takes that position as an object's size.
        stuck   = looked & ~at_goal & (mm_s < self.stuck_speed_mm_s)
        expired = self.steps_remaining <= 0          # timeout is always armed

        wants   = self._at(self.wants, idx)
        w_goal    = (wants == GOAL)     & at_goal
        w_stuck   = (wants == STUCK)    & stuck
        w_external = (wants == EXTERNAL) & external
        stop = live & (w_goal | w_stuck | w_external | expired)

        # Requested conditions beat TIMEOUT on the final tick. Timeout last is
        # load-bearing: a signal that arrives on its deadline is success.
        keep = self.outcome.gather(2, idx).squeeze(-1)
        code = torch.where(stop & w_goal,     GOAL,
               torch.where(stop & w_stuck,    STUCK,
               torch.where(stop & w_external, EXTERNAL,
               torch.where(stop,              TIMEOUT, keep))))
        self.outcome.scatter_(2, idx, code[..., None])
        # Same scatter shape as `outcome`, and only on the tick a row ends.
        self.stop_speed_mm_s.scatter_(2, idx, torch.where(
            stop, mm_s, self.stop_speed_mm_s.gather(2, idx).squeeze(-1))[..., None])
        self.stop_position_mm.scatter_(2, idx, torch.where(
            stop, pos, self.stop_position_mm.gather(2, idx).squeeze(-1))[..., None])

        wanted = stop & (code == wants)
        self.stopped_valid |= stop
        self.stopped_ok = torch.where(stop, wanted, self.stopped_ok)
        self.stopped_at = torch.where(wanted, pos, self.stopped_at)
        self.retired = self.retired | stop

        # Tumbling window: re-anchor only on real travel, so mm_s stays a current
        # speed. A lifetime average never registers a joint that moved then stopped.
        moving = live & looked & ~stuck
        self.anchor_mm = torch.where(moving, pos, self.anchor_mm)
        self.steps_since_anchor.masked_fill_(moving, 0)

        # The standing orders. Only an ACTIVE joint rewrites them; an
        # inactive one keeps commanding what it last asked for (or where it
        # started, if it has never acted). See the module docstring.
        # `live` includes joints that are merely waiting; only joints with a
        # Move at this step should update their standing orders.
        active_step = self.active_step                                # [N, J]
        self.held_goal   = torch.where(active_step, goal, self.held_goal)
        self.held_torque = torch.where(active_step, torque, self.held_torque)

        # Barrier: advance when every joint has finished its row. `& step < K`
        # stops finished envs from running the counter past the program.
        advance   = self.retired.all(dim=1) & (self.step < self.K)
        self.step = self.step + advance.to(torch.int64)
        self._arm(advance, pos)
        return self.held_goal, self.held_torque


class Step:
    """One step of a program, as `[N, J]` tensors filled by joint index.

        p.step().set(FINGERS, mid, travel_torque, "goal", 6.0)
                .set(AUX_JAW, 0.0, squeeze, "stuck", 10.0, when=on)

    The authoring counterpart of the tick: a task indexes joints and assigns a
    broadcast value, the same shape as everything below it, instead of building
    one object per joint for `build` to unpack in a Python loop. Joints nobody
    writes idle at this step and keep their standing orders.

    **The tensors are not the interface — `set` is.** They are ordinary
    attributes and nothing stops a caller writing them, but `acts` is what says
    a joint acts here, and a `goal_mm` written without it does nothing while an
    `acts` written without a `goal_mm` commands 0.0, which on this hand is a
    full-stroke move into a hard stop. `set` writes them together or not at all.
    """

    def __init__(self, n_envs: int, n_joints: int, hz: float,
                 device: Device) -> None:
        self.N, self.J, self.hz, self.device = n_envs, n_joints, hz, device
        joint = lambda dt: torch.zeros((n_envs, n_joints), dtype=dt, device=device)
        self.acts            = joint(torch.bool)
        self.goal_mm         = joint(torch.float32)
        self.torque          = joint(torch.float32)
        self.timeout_steps   = joint(torch.int32)
        self.wants           = torch.full((n_envs, n_joints), TIMEOUT,
                                          dtype=torch.int8, device=device)
        self.relative        = joint(torch.bool)
        self.predicate_kind      = joint(torch.int8)
        self.predicate_source    = joint(torch.int64)
        self.predicate_threshold = joint(torch.float32)

    def set(self, dofs: int | Sequence[int], goal: Broadcast,
            torque: Broadcast, stop: str = "goal", timeout_s: float = 6.0,
            when: torch.Tensor | When | None = None, frame: str = "abs") -> "Step":
        """Give one order to a group of joints. Returns self, so sets chain.

        `dofs` is a joint index or a sequence of them. `goal` and `torque`
        broadcast across the group -- scalar, `[N]`, `[len(dofs)]` or
        `[N, len(dofs)]` -- so a torque that varies per joint *and* per env is
        one write rather than one object per joint.

        `stop` is one of:

            "goal"   retire on arrival      (a travel move)
            "stuck"  retire on contact      (a probe, or taking a grip)
            "hold"   retire immediately     (place a standing order and move on)
            "wait"   retire on the budget   (a dwell)

        A "hold" never blocks the barrier and needs no budget: its whole effect
        is the standing order it leaves behind. A "wait" is the same row with
        its budget honoured -- it arms no condition, so `timeout_s` is what it
        spends and spending it is success.

        `when` is `[N]` bool, None for every env, or `When(...)` resolved once
        when the row arms from a prior row's stop position/outcome. Envs where
        it is false idle without changing their standing command.

        `frame` is `"abs"` (default) or `"here"`, where `goal` is a DISTANCE
        from wherever the joint sits when this step arms. `"here"` is what a
        task uses when it has no calibrated origin to command against, or when
        it means "back off half a rail from whatever you just hit" -- both are
        zeroing. See `FRAMES`.

        The stop rule and the frame are checked here rather than in `build` so a
        typo raises where it was written, not one call later with no clue which
        step it was.
        """
        if stop not in STOPS:
            raise ValueError(f"unknown stop {stop!r}, expected one of {STOPS}")
        if frame not in FRAMES:
            raise ValueError(f"unknown frame {frame!r}, expected one of {FRAMES}")
        ids = [dofs] if isinstance(dofs, int) else list(dofs)
        n = len(ids)
        self.goal_mm[:, ids] = self._spread(goal, n, "goal")
        self.torque[:, ids] = self._spread(torque, n, "torque")
        self.wants[:, ids] = WANTS[stop]
        self.relative[:, ids] = (frame == "here")
        self.timeout_steps[:, ids] = (
            1 if stop == "hold" else max(1, round(timeout_s * self.hz)))
        if isinstance(when, When):
            if when.dof >= self.J:
                raise ValueError(
                    f"predicate source DOF {when.dof} outside 0..{self.J - 1}")
            self.acts[:, ids] = True
            self.predicate_kind[:, ids] = PREDICATES[when.kind]
            self.predicate_source[:, ids] = when.dof
            self.predicate_threshold[:, ids] = when.threshold_mm
        else:
            self.acts[:, ids] = (
                True if when is None else
                torch.as_tensor(when, dtype=torch.bool,
                                device=self.device)[:, None])
        return self

    def _spread(self, value: Broadcast, n_sel: int, field: str) -> torch.Tensor:
        """One field's value, broadcast to `[N, n_sel]`.

        A 1-D value is read by its length: `N` is per env, `n_sel` is per joint.
        Unambiguous except when the batch is exactly as wide as the selection,
        where it is **refused rather than guessed** — torch would silently align
        on the trailing axis, so a per-env goal would be written across joints
        and the program would still run, commanding plausible wrong numbers.
        """
        v = torch.as_tensor(value, dtype=torch.float32, device=self.device)
        if v.dim() == 1:
            if self.N == n_sel and self.N > 1:
                raise ValueError(
                    f"{field}: a 1-D value is ambiguous when the batch (N="
                    f"{self.N}) is as wide as the {n_sel} joints being set. "
                    f"Pass [N, {n_sel}] to say which axis it varies over.")
            if v.shape[0] == self.N:
                v = v[:, None]
            elif v.shape[0] == n_sel:
                v = v[None, :]
            else:
                raise ValueError(
                    f"{field}: length {v.shape[0]} is neither N={self.N} nor "
                    f"the {n_sel} joints being set")
        return v.expand(self.N, n_sel)


class Program:
    """Collects steps, emits a Motions.

    The Python loop that builds N strokes runs HERE, once, at build time. At run
    time there is no loop bound and no branch: envs needing fewer strokes carry
    `when=False` on the extra rows and idle through them. That is the whole
    answer to how `for i in range(ceil(...))` survives being batched — the bound
    becomes the max over envs and the mask does the rest.

    `build` stacks the steps' `[N, J]` tensors into the `[N, J, K]` program along
    a new K axis, so the only per-step Python is one `torch.stack` per field.
    """

    def __init__(self, n_envs: int, n_joints: int, hz: float,
                 device: Device = "cpu",
                 stuck_speed_mm_s: float = STUCK_SPEED_MM_S,
                 position_tolerance_mm: float = POSITION_TOLERANCE_MM) -> None:
        self.N, self.J, self.hz, self.device = n_envs, n_joints, hz, device
        self.stuck_speed_mm_s = stuck_speed_mm_s
        self.position_tolerance_mm = position_tolerance_mm
        self.steps: list[Step] = []

    def step(self) -> Step:
        """Append an empty step and hand it back to be filled by `set`."""
        self.steps.append(Step(self.N, self.J, self.hz, self.device))
        return self.steps[-1]

    def build(self) -> Motions:
        m = Motions(self.N, self.J, len(self.steps), self.hz, self.device,
                    stuck_speed_mm_s=self.stuck_speed_mm_s,
                    position_tolerance_mm=self.position_tolerance_mm)
        if not self.steps:
            return m                  # K=0: already the right empty shape
        stack = lambda pick: torch.stack([pick(s) for s in self.steps], dim=2)
        m.acts            = stack(lambda s: s.acts)
        m.goal_mm         = stack(lambda s: s.goal_mm)
        m.torque          = stack(lambda s: s.torque)
        m.timeout_steps   = stack(lambda s: s.timeout_steps)
        m.wants              = stack(lambda s: s.wants)
        m.relative           = stack(lambda s: s.relative)
        m.predicate_kind      = stack(lambda s: s.predicate_kind)
        m.predicate_source    = stack(lambda s: s.predicate_source)
        m.predicate_threshold = stack(lambda s: s.predicate_threshold)
        return m


class Result(NamedTuple):
    """What a task returns: its measurement, and which envs that measurement is
    valid for.

    `ok` is [N] bool, one flag per env. Envs fail independently -- one env whose
    finger never found its stop says nothing about the other 4095 -- so failure
    is reported per env rather than raised. A raise would also unwind the
    generator, throwing away every env that did succeed along with the one that
    did not.

    The task decides what failed. The *caller* decides what to do about it:
    `studio.finish` refuses to save a calibration when any env failed, `sim.run`
    raises, a batched trainer masks the bad rows and carries on. `why` is the
    one-line diagnostic those consumers print, and is the reason this is not a
    bare `(value, ok)` pair -- the message a `RuntimeError` used to carry is what
    an operator reads off the console.
    """
    value: torch.Tensor   # [N, J] stops, or [N] radii -- whatever the task measures
    ok: torch.Tensor      # [N] bool
    why: str = ""


# What every task under `tasks/` is: yields programs, is sent the millimetres
# measured when each finished, returns its measurement. Named here rather than
# spelled out at each `build`, since the three-parameter form of `Generator` is
# what a reader gets wrong when re-deriving it.
Task = Generator[Motions, torch.Tensor, Result]


class TaskRunner:
    """Drives a task to completion, one program at a time.

    A task is not one program. Zeroing seeks a hard stop and only *then* knows
    where mid travel is; the cap task probes the cap and only then knows how
    many strokes to emit. Both are "a program, some Python, another program",
    and the Python has to run between programs rather than inside a tick --
    otherwise the shape of a program would depend on a measurement, which is
    the one thing that stops it being batched.

    A generator is exactly that shape, so a task is one:

        def zero(cfg, n_envs=1):
            for dof_ids, name in PHASES:
                stalled_mm = yield seek(cfg, dof_ids, ...)
                yield park(cfg, dof_ids, stalled_mm, ...)
            return offsets

    `yield` hands out a program and evaluates to the measurement taken when
    that program finished, which is the plan's "measurement bus" with no extra
    mechanism. The task's return value arrives as `StopIteration.value` and is
    left in `result`, untouched -- the tasks under `tasks/` return a `Result`,
    but nothing here requires that, and a caller driving a one-off generator can
    return whatever it likes.

    The runner owns no bus, no clock, no device and no config. Both backends
    drive it identically -- give it the measured millimetres each tick, write
    back the goal and torque it returns -- which is the whole of what "the same
    task deploys on sim and real" means. The hand executes; the task submits.
    """

    def __init__(self, task: Task, hold_torque: Broadcast) -> None:
        """`hold_torque` seeds joints the current program never mentions.

        It cannot default to zero: `Motions.start` copies this into
        `held_torque` for every joint, and a joint the program does not touch
        would be commanded limp rather than told to stay where it is. The
        hand's own configured torque is the right seed, and only the caller
        knows it.
        """
        self.task = task
        self.hold_torque = hold_torque
        self.program: Motions | None = None
        self.result: Result | None = None
        self.finished = False
        # Every program the task issued, not just the one being driven. A task is
        # several (`zero` is six, `cap` is three) and the last one alone cannot
        # explain a run: `zero`'s `Result.ok` reports on its seeks and says
        # nothing about its parks. Kept so `slow_rows` can name the row that
        # spent the time after the fact -- the engine has no other trace, which
        # is how a probe that never detected contact hid behind a budget for a
        # whole bench session.
        self.issued: list[Motions] = []

    def tick(self, position_mm: torch.Tensor,
             external: torch.Tensor | None = None
             ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """One tick. `position_mm` is [N, J]; returns (goal_mm, torque) or None.

        `None` means the task is over. It keeps meaning that -- a finished
        runner never resumes -- and the caller decides whether to hold the last
        goal or hand the joints back to its own goal source.

        The `while` covers a task that yields an already-finished program (an
        empty one, or one every env skipped) without spending a tick on it.
        """
        while not self.finished and (self.program is None or self.program.done()):
            self._advance(position_mm)
        return (None if self.finished else
                self.program.step_once(position_mm, external))

    def slow_rows(self, labels: Sequence[str], env: int = 0) -> list[str]:
        """One line per commanded row that did NOT get the outcome it asked for.

        The diagnostic the engine was missing. `caps_contact_based.py` printed a
        `mark(label)` per phase for exactly this reason -- *"so a run log shows
        which phase actually costs the time instead of guessing"* -- and the
        rewrite dropped it, which is how `cap`'s probe could fail to detect
        contact on every stroke and show up only as the hand standing still.

        A row that misses its condition still ends tidily, on its deadline. So
        the wall-clock cost of a failure is exactly its budget, and a budget is
        the one number here that is chosen rather than measured: if the reported
        seconds equal the budget, the row did not stop -- it was stopped.

        `env` because this is for a person reading a bench log, where N is 1.
        """
        out = []
        for p, program in enumerate(self.issued):
            for k in range(program.K):
                for j in range(len(labels)):
                    if not program.executed[env, j, k]:
                        continue
                    if bool(program.succeeded()[env, j, k]):
                        continue
                    want = OUTCOME_NAMES.get(int(program.wants[env, j, k]), "?")
                    got = OUTCOME_NAMES.get(int(program.outcome[env, j, k]), "?")
                    at = float(program.stop_position_mm[env, j, k])
                    speed = float(program.stop_speed_mm_s[env, j, k])
                    goal = float(program.goal_mm[env, j, k])
                    # The three ways a STUCK row times out, named. Which one it
                    # is decides the fix, and the seconds alone never said.
                    why = ""
                    if int(program.wants[env, j, k]) == STUCK:
                        if abs(at - goal) <= program.position_tolerance_mm:
                            why = ("  <- closed onto its goal; STUCK cannot fire "
                                   "at_goal, so nothing could ever end this row")
                        elif speed >= program.stuck_speed_mm_s:
                            why = (f"  <- still reading {speed:.2f} mm/s of motion "
                                   f"at the {program.stuck_speed_mm_s} threshold: "
                                   f"contact was made but never looked stopped")
                    out.append(
                        f"program {p} step {k} {labels[j]}: wanted {want}, got "
                        f"{got} after {int(program.timeout_steps[env, j, k]) / program.hz:.1f}s"
                        f" at torque {float(program.torque[env, j, k]):.0f}"
                        f"; ended {at:.2f}mm (goal {goal:.2f}) moving "
                        f"{speed:.2f}mm/s{why}")
        return out

    def _advance(self, position_mm: torch.Tensor) -> None:
        """Pull the next program out of the task, or record that it is done.

        The measurement sent back is the position at this instant, which is
        where the program that just finished left the joints -- for a
        `stop="stuck"` row that is the contact, i.e. the measurement the task
        asked for.
        """
        try:
            program = (next(self.task) if self.program is None
                       else self.task.send(position_mm))
        except StopIteration as end:
            self.result, self.finished = end.value, True
            return
        program.start(position_mm, self.hold_torque)
        self.program = program
        self.issued.append(program)

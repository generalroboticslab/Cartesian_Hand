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
from dataclasses import dataclass

import torch

RUNNING, GOAL, STUCK, TIMEOUT = 0, 1, 2, 3

POSITION_TOLERANCE_MM = 1.0
STUCK_SPEED_MM_S      = 0.3    # matches tasks/primitives.py wait_for_stall
# 1.0 s at 50 Hz, matching wait_for_stall_counts' confirm_s. Not shorter: the
# encoder quantizes to whole counts, so a window of W ticks resolves speed only
# in steps of one count per W. At 81.5 counts/mm a 5-tick window makes the 0.3
# mm/s threshold 2.4 counts wide, and a joint dithering +/-1 count against a
# stop eats half of it -- the bug that recorded hard stops mid-rail on hand_1.
STUCK_WINDOW_STEPS    = 50

STOPS = ("goal", "stuck", "hold")


class Motions:
    """Executes an [N, J, K] motion program. No servo, no bus, no threads, no clock."""

    def __init__(self, n_envs, n_joints, n_steps, hz, device="cpu",
                 stuck_speed_mm_s: float = STUCK_SPEED_MM_S,
                 position_tolerance_mm: float = POSITION_TOLERANCE_MM):
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
        self.stop_at_goal    = prog(torch.bool)
        self.stop_when_stuck = prog(torch.bool)
        self.outcome         = prog(torch.int8)

        # ── runtime ───────────────────────────────────────────────────────────
        self.step               = torch.zeros(N, dtype=torch.int64, device=device)
        self.retired            = joint(torch.bool)
        self.steps_remaining    = joint(torch.int32)
        self.anchor_mm          = joint(torch.float32)   # tumbling stall window
        self.steps_since_anchor = joint(torch.int32)
        self.held_goal          = joint(torch.float32)   # the standing orders
        self.held_torque        = joint(torch.float32)

    # ── program access ────────────────────────────────────────────────────────

    def _at(self, program):
        """[N, J] slice of an [N, J, K] program at each env's current step."""
        idx = self.step.clamp(max=self.K - 1)[:, None, None]
        return program.gather(2, idx.expand(-1, program.shape[1], 1)).squeeze(-1)

    def _arm(self, where, position_mm):
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
        """
        active = where[:, None] & self._at(self.acts)        # [N, J]
        # Re-arm active joints: clear retired, reset timeout/anchor/window.
        self.retired            = torch.where(active, torch.zeros_like(self.retired),
                                              self.retired)
        self.steps_remaining    = torch.where(active, self._at(self.timeout_steps),
                                              self.steps_remaining)
        self.anchor_mm          = torch.where(active, position_mm, self.anchor_mm)
        self.steps_since_anchor = torch.where(active, torch.zeros_like(self.steps_since_anchor),
                                              self.steps_since_anchor)

    def start(self, position_mm, torque):
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

    def done(self):
        return bool((self.step >= self.K).all())

    def succeeded(self):
        """[N, J, K] — rows that ended the way their stop rule implied they should.

        Derived rather than stored: a row arms exactly one non-timeout stop
        condition, so the condition it armed *is* the outcome it wanted. A
        `hold` row arms none and wants TIMEOUT — spending its whole budget is
        how a standing order gets placed, not a failure.

        Index with `acts` to ignore cells where a joint deliberately idled.
        """
        i8 = lambda v: torch.tensor(v, dtype=torch.int8, device=self.device)
        want = torch.where(self.stop_when_stuck, i8(STUCK),
               torch.where(self.stop_at_goal,    i8(GOAL), i8(TIMEOUT)))
        return self.outcome == want

    # ── the tick ──────────────────────────────────────────────────────────────

    def step_once(self, position_mm):
        """Advance every env by one tick. Returns (goal_mm, torque), both [N, J].

        `position_mm` is [N, J] and is treated as READ-ONLY.
        """
        pos    = position_mm
        goal   = self._at(self.goal_mm)
        torque = self._at(self.torque)
        live   = ~self.retired & (self.step < self.K)[:, None]
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

        w_goal  = self._at(self.stop_at_goal)    & at_goal
        w_stuck = self._at(self.stop_when_stuck) & stuck
        stop = live & (w_goal | w_stuck | expired)

        # Priority GOAL > STUCK > TIMEOUT. Timeout last is the load-bearing
        # part: a row that reaches its goal on its final tick is a success, and
        # the reverse order would score it as a failure. The chain is anchored
        # on the existing outcome so it stays int8 -- and so a row that does not
        # stop keeps whatever it already had.
        idx  = self.step.clamp(max=self.K - 1)[:, None, None].expand(-1, pos.shape[1], 1)
        keep = self.outcome.gather(2, idx).squeeze(-1)
        code = torch.where(stop & w_goal,  GOAL,
               torch.where(stop & w_stuck, STUCK,
               torch.where(stop,           TIMEOUT, keep)))
        self.outcome.scatter_(2, idx, code[..., None])

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
        active_step = self._at(self.acts)                              # [N, J]
        self.held_goal   = torch.where(active_step, goal, self.held_goal)
        self.held_torque = torch.where(active_step, torque, self.held_torque)

        # Barrier: advance when every joint has finished its row. `& step < K`
        # stops finished envs from running the counter past the program.
        advance   = self.retired.all(dim=1) & (self.step < self.K)
        self.step = self.step + advance.to(torch.int64)
        self._arm(advance, pos)
        return self.held_goal, self.held_torque


@dataclass
class Move:
    """One joint's order for one step.

    `goal` is millimetres, scalar or [N] — per-env goals are the point, since a
    measured cap radius differs per env. `stop` is one of:

        "goal"   retire on arrival      (a travel move)
        "stuck"  retire on contact      (a probe, or taking a grip)
        "hold"   retire immediately     (place a standing order and move on)

    A "hold" never blocks the barrier and never needs a timeout: its whole
    effect is the standing order it leaves behind.
    """
    goal: float = 0.0
    torque: float = 50.0
    stop: str = "goal"
    timeout_s: float = 6.0
    when: object = None          # [N] bool; None means every env


class Program:
    """Collects steps, emits a Motions.

    The Python loop that builds N strokes runs HERE, once, at build time. At run
    time there is no loop bound and no branch: envs needing fewer strokes carry
    `when=False` on the extra rows and idle through them. That is the whole
    answer to how `for i in range(ceil(...))` survives being batched — the bound
    becomes the max over envs and the mask does the rest.
    """

    def __init__(self, n_envs, n_joints, hz, device="cpu",
                 stuck_speed_mm_s: float = STUCK_SPEED_MM_S,
                 position_tolerance_mm: float = POSITION_TOLERANCE_MM):
        self.N, self.J, self.hz, self.device = n_envs, n_joints, hz, device
        self.stuck_speed_mm_s = stuck_speed_mm_s
        self.position_tolerance_mm = position_tolerance_mm
        self.steps = []

    def step(self, moves):
        """Append one step. `moves` is {joint_index: Move}; omitted joints idle."""
        self.steps.append(moves)
        return self

    def build(self) -> Motions:
        m = Motions(self.N, self.J, len(self.steps), self.hz, self.device,
                    stuck_speed_mm_s=self.stuck_speed_mm_s,
                    position_tolerance_mm=self.position_tolerance_mm)
        col = lambda v: torch.as_tensor(v, dtype=torch.float32,
                                        device=self.device).expand(self.N)
        for k, moves in enumerate(self.steps):
            for j, mv in moves.items():
                if mv.stop not in STOPS:
                    raise ValueError(
                        f"unknown stop {mv.stop!r}, expected one of {STOPS}")
                m.acts[:, j, k]            = True if mv.when is None else mv.when
                m.goal_mm[:, j, k]         = col(mv.goal)
                m.torque[:, j, k]          = col(mv.torque)
                m.stop_at_goal[:, j, k]    = mv.stop == "goal"
                m.stop_when_stuck[:, j, k] = mv.stop == "stuck"
                m.timeout_steps[:, j, k]   = (
                    1 if mv.stop == "hold" else max(1, round(mv.timeout_s * self.hz)))
        return m

"""Unscrewing a bottle cap, as a motion program instead of as Python.

Two protocols
-------------
The human protocol is five steps and mentions no mechanism -- it would read the
same for a hand with a rotating wrist:

    1. put the gripper over the bottle, at cap height
    2. close both jaws until they touch          -> tells you both radii
    3. hold the bottle, hold the cap
    4. turn the cap three full revolutions
    5. lift the cap clear and present the bottle

The robot protocol is one of those steps broken into what *this* machine can do.
The fingers travel 55 mm and a cap's circumference is more, so step 4 becomes
"repeat: release the cap, slide the fingers back, re-grip, twist". That
breakdown exists only because this gripper cannot rotate continuously, and it is
the only place a hardware limit is allowed to enter the design.

Steps 1-2, 3-4 and 5 are the three programs `build` yields, in that order.

Measurements may set values. They may not set structure
-------------------------------------------------------
They are three programs rather than one because step 2 is a measurement and
everything after it is parameterised by that measurement. A wider cap has a
longer circumference and needs more strokes, so the stroke count is per env --
but it is decided when the stroke program is *built*, and by the time a program
starts ticking its shape is fixed. The Python loop that emits N strokes runs
once, at build time; at run time there is no loop bound and no branch, because
envs needing fewer strokes carry `when=False` on the extra rows and idle through
them.

That is the whole answer to how `for i in range(ceil(measurement))` survives
being batched, and the reason this task can be searched at N=4096 while
`cartesian_hand_old/tasks/caps_contact_based.py`, which is the same task in
straight-line Python, cannot.

Why the base jaw is mentioned once
----------------------------------
It takes its grip in the first step of the stroke program and never appears again.
A retired joint keeps its standing order (see `motions.py`), so the base jaw
goes on pressing the bottle at squeeze torque for the rest of the run without
occupying a row.
"""
import math
from dataclasses import dataclass, field

import torch

from ..config import (AUX_FINGERS, AUX_JAW, AUX_LEFT, AUX_RIGHT, BASE_FINGERS,
                      BASE_JAW, BASE_LEFT, BASE_RIGHT, HandConfig, Z)
from ..motions import Program, Result, Task
from ..primitives import twist

PROBE_STEP = 1          # which step of the probe program closes the jaws on the object


@dataclass
class Config:
    """Everything about this task that is not its procedure.

    Mostly the theta a search varies; `label` is the exception and is here so a
    task has one configuration object rather than a dataclass plus a scatter of
    module constants. Only `num_revs` changes the program's shape, and only by
    changing how many stroke steps get emitted, which is a build-time decision.
    """
    label: str = "Open cap"
    """Button text on the studio page. Empty string means no button.

    Read off `Config()` by `tasks.buttons()` without building a program. Opt-in
    by default value: a variant that does not name its own label inherits none,
    because it reaches this class through `cap.Config` rather than importing the
    name into its own module. See `tasks/__init__.py`.
    """
    cap_offset: float = field(default=20.0, metadata={"tune": (5.0, 40.0)})
    """Height of the cap's top face above z zero, in mm."""
    num_revs: float = field(default=1, metadata={"tune": (0.5, 6.0)})
    """Full turns needed to free the cap."""
    squeeze_torque: float = field(default=80.0, metadata={"tune": (40.0, 200.0)})
    """Holding torque while gripping and twisting."""
    approach_torque: float = field(default=150.0, metadata={"tune": (50.0, 300.0)})
    """Torque used while closing a jaw onto the object."""
    travel_torque: float = field(default=50.0, metadata={"tune": (30.0, 200.0)})
    """Torque for moves through free space."""
    release_clearance: float = field(default=3.0, metadata={"tune": (1.0, 10.0)})
    """How far past the cap radius the aux jaw opens between strokes, in mm."""
    timeout_margin: float = field(default=1.5, metadata={"tune": (1.0, 3.0)})
    """Multiplier on expected travel time, as in `zero.Config.timeout_margin`.

    Replaces the flat `move_timeout_s = 3.0` / `probe_timeout_s = 5.0` this
    shipped with. At 300 counts/s the hand moves 3.68 mm/s, so 3 s buys 11 mm --
    it covered a 3 mm jaw release and nothing else, and in particular not the
    40 mm twist, which is the row that turns the cap. Driven through mujoco (now
    that `sim.profile` models the servo's own ramp) the shipped constants
    expired 15 of 18 commanded rows."""
    jaw_opening: float = field(default=25.0, metadata={"tune": (10.0, 40.0)})
    """How wide to open each jaw before closing on the object, in mm.

    A jaw only has to clear the bottle. Every geometric field here is a
    clearance the task asks for, passed through `HandConfig.clamped_mm` before
    it reaches a Move -- never a DOF's `max_mm`, which is CAD and reads high.
    """
    finger_stroke: float = field(default=40.0, metadata={"tune": (10.0, 55.0)})
    """Full sweep of a finger during one twist, in mm."""
    max_cap_radius: float = 25.0
    """Widest cap this task will turn, in mm. Sets the stroke count, and so the
    program's shape -- a constant, deliberately, so shape never depends on a
    measurement. Matches `jaw_opening`: a cap wider than the jaw opens is one
    the jaw cannot reach around anyway. Raising it lengthens every run; a cap
    that exceeds it is reported not-ok rather than silently left screwed on.

    **Deliberately not tunable.** It is the one field that changes `K`, and two
    candidates with different program lengths are not comparable -- see
    `tasks.tunables`."""
    lift_mm: float = field(default=35.0, metadata={"tune": (5.0, 50.0)})
    """How high the z stage lifts the freed cap, in mm.

    Not affected by `travel_torque`: the lift is the one row that raises the
    loaded axis, so it commands `torque_min_to_move[Z]` and sizes its own budget
    from this height. See the row in `build`."""


def build(hand: HandConfig, start_mm: torch.Tensor, cfg: Config | None = None,
          cap_radius: float | None = None, **kwargs) -> Task:
    """Entry point for `--task cap`. Yields three programs, returns a `Result`.

    `start_mm` is [N, J] and fixes the batch width and device; the cap task does
    not read the starting position, since every goal it commands is absolute in
    the hand's millimetre frame.

    **This task is contact-based and there is no other kind.** `cap_radius` is
    not a second protocol and not a mode -- it is an injection point for the two
    callers that have no cap to touch: `sim.run`, whose model is objectless, and
    `test_motions`, which passes a *tensor* of per-env radii to exercise the
    batched stroke-count predication. On hardware it is always None and the
    probe always runs. Do not add a variant that ships with it set; sizing the
    grip from contact instead of from a number typed by the operator is the
    point of the file.

    `Result.value` is [N] mm radii; `Result.ok` is [N] -- whether the probe ended
    on contact rather than on its timeout. The outcome, not the resulting number,
    is what tells "the cap is 2 mm across" apart from "there was no cap and the
    jaw shut on air": those are indistinguishable by position alone, and taking
    the second for a measurement builds the rest of the run on nothing.

    A failed env is left to run the strokes anyway rather than masked out with
    `when=`. Every goal downstream of the measurement is bounded even when the
    measurement is nonsense -- `release` is `radius + clearance` and the stroke
    count is `clamp(min=1)` -- so a dead env turns once against air. Wasteful,
    not dangerous, and the alternative costs a `when=` on every remaining step.

    A variant is a new file calling this with a different `Config` -- the
    dataclass constructor is the configuration. See `tasks/__init__.py`.
    """
    cfg = cfg or Config()
    n_envs, n_dof = start_mm.shape
    device = start_mm.device
    program = lambda: Program(n_envs, n_dof, hand.control_hz, device)

    jaw = hand.clamped_mm(BASE_JAW, cfg.jaw_opening)
    span = hand.clamped_mm(AUX_LEFT, cfg.finger_stroke)
    mid = span / 2
    sq, travel_tq = cfg.squeeze_torque, cfg.travel_torque
    floor = hand.gain_vector("torque_min_to_move", device).float()
    JAWS = [BASE_JAW, AUX_JAW]
    FINGERS = BASE_FINGERS + AUX_FINGERS
    # Full-rail deadline at this group's own speed -- see `HandConfig.travel_budget`.
    budget = lambda dofs: hand.travel_budget(
        [dofs] if isinstance(dofs, int) else dofs, cfg.timeout_margin)

    # Every free-space move in this task shares a stop rule; only the joints, the
    # goal, the budget and sometimes the torque differ.
    travel = lambda s, dofs, goal, tq=travel_tq, when=None: s.set(
        dofs, goal, tq, "goal", budget(dofs), when)

    # ── Steps 1-2: reach over the bottle at cap height, close both jaws ───────
    if cap_radius is None:
        p = program()
        s = p.step()
        travel(s, JAWS, jaw)
        travel(s, FINGERS, mid)
        travel(s, Z, hand.clamped_mm(Z, cfg.cap_offset))
        # The jaws close past 0 with reduced torque and retire on contact:
        # stop="stuck", so the object is what ends the move.
        p.step().set(JAWS, 0.0, cfg.approach_torque, "stuck", budget(JAWS))

        probe = p.build()
        measured = yield probe
        # Per env, not `.all()`: with a cap present in some envs and not others,
        # squashing them would throw away every good measurement in the batch.
        ok = probe.succeeded()[:, AUX_JAW, PROBE_STEP]            # [N]
        why = "" if bool(ok.all()) else (
            f"cap probe did not end on contact for envs "
            f"{(~ok).nonzero().flatten().tolist()} (outcome "
            f"{probe.outcome[:, AUX_JAW, PROBE_STEP].tolist()}). Is a cap "
            f"present at z={hand.clamped_mm(Z, cfg.cap_offset):.1f}mm?")
        # Where each jaw stopped IS the radius of what it is holding.
        radius = measured[:, AUX_JAW].clone()
        # No env measured anything, so every stroke below would turn against
        # air. At N=1 -- every hardware run -- that is any failure at all, and
        # a hand that unscrews nothing for a minute before saying so is a minute
        # the operator spends watching a run whose result is already void.
        if not bool(ok.any()):
            return Result(radius, ok, why)
    else:
        radius = torch.as_tensor(cap_radius, dtype=torch.float32,
                                 device=device).expand(n_envs).clone()
        ok, why = torch.ones(n_envs, dtype=torch.bool, device=device), ""

    # ── Steps 3-4: hold both, then turn the cap `num_revs` times ──────────────
    # One stroke rotates the cap by one finger sweep along its circumference.
    # Denominated in `span`, the distance the fingers actually travel below --
    # using the table's max here instead would under-count strokes whenever the
    # sweep is bounded, and the cap would come out short of num_revs.
    strokes = torch.ceil(cfg.num_revs * 2 * math.pi * radius / span).clamp(min=1)
    # The loop bound is a CONSTANT, never `int(strokes.max())`. A measured bound
    # is a host sync, and it couples the batch: one wide cap in 4096 envs
    # lengthens the program for all of them, and the other 4095 idle through
    # rows that exist only because of their neighbour. Envs needing fewer
    # strokes still idle the surplus, at 1 tick per row. Under an optimizer this
    # is load-bearing twice over -- theta must not change K, or two candidates
    # are not comparable.
    max_strokes = math.ceil(cfg.num_revs * 2 * math.pi * cfg.max_cap_radius / span)
    # A cap wider than the budget comes out still on the bottle, and a cap still
    # on the bottle is indistinguishable by position from a free one -- the same
    # argument as the probe's outcome check.
    short = strokes > max_strokes
    if bool(short.any()):
        ok = ok & ~short
        why = why or (
            f"cap in envs {short.nonzero().flatten().tolist()} is wider than "
            f"max_cap_radius={cfg.max_cap_radius}mm, so {max_strokes} strokes "
            f"is short of {cfg.num_revs} turns")

    p = program()
    # Both jaws take their grip on the cap: same stop rule as the probe, but at
    # holding torque rather than contact-finding torque.
    p.step().set(JAWS, 0.0, sq, "stuck", budget(JAWS))

    for i in range(max_strokes):
        on = strokes > i                      # [N] bool: envs still stroking

        # Retract against our own grip at squeeze torque: pulling free of a grip
        # needs at least the torque that made it. Travel torque here is the
        # documented bug in caps_contact_based -- the jaw silently fails to open
        # and the fingers reset and twist against a held cap.
        #
        # `frame="here"` because the row above left the jaw ON the cap, so
        # "here + clearance" IS `radius + clearance` with no radius term. The
        # measurement stops feeding a goal and only gates `on`.
        p.step().set(AUX_JAW, cfg.release_clearance, sq, "goal",
                     budget(AUX_JAW), on, frame="here")

        # Fingers reset to the far end of their sweep, ready to turn. The reset
        # is the turn run backwards, so it is the same call with the ids swapped.
        twist(p.step(), AUX_RIGHT, AUX_LEFT, span, travel_tq,
              budget(AUX_FINGERS), on)

        # Back onto the cap. stop="stuck", not "goal": this drives into a
        # physical obstruction, so position convergence can legitimately never
        # fire.
        p.step().set(AUX_JAW, 0.0, sq, "stuck", budget(AUX_JAW), on)

        # The turn itself, at squeeze torque -- it is turning a gripped cap.
        twist(p.step(), AUX_LEFT, AUX_RIGHT, span, sq, budget(AUX_FINGERS), on)
    yield p.build()

    # ── Step 5: lift the freed cap clear and present the bottle ───────────────
    p = program()
    # Retract at squeeze torque: same reason, and the same frame, as the
    # in-loop release -- the jaw is still standing on the cap.
    p.step().set(AUX_JAW, cfg.release_clearance, sq, "goal",
                 budget(AUX_JAW), frame="here")
    travel(p.step(), AUX_FINGERS, mid)
    p.step().set(AUX_JAW, 0.0, sq, "stuck", budget(AUX_JAW))

    # The lift, and the only row in this task that raises z against gravity.
    #
    # **Not `travel`.** `travel_torque` is a horizontal-DOF number. Pressing z
    # DOWN at it is fine and the entry move above relies on that -- so this is
    # an asymmetry in the axis, not a bug in `travel` to be fixed there. Lifting
    # at it moves nothing: hand_2's bisect (lifting 5 mm, travel after 3 s) put
    # the cliff between 150 and 200 unloaded, which is why the working hardware
    # carried its travel torque as a per-DOF table with z alone at 300
    # (`STANDARD_TORQUE`) and why `zero`'s park had to lift z to its own floor
    # before it moved off the stop. At 50 this row runs, expires, and reports
    # nothing wrong while the cap never leaves the bottle.
    #
    # `torque_min_to_move[Z]` rather than a constant: it is the per-hand
    # measured floor and it is already what zeroing lifts z with. It is a floor
    # bisected *unloaded*, and this lift is carrying a cap, so if a cap is ever
    # seen to hang short the number to raise is that one, in `config`, for the
    # hand it was measured on -- not a pad hidden here.
    #
    # The budget is `budget(Z)` like every other row: z's own rail at z's own
    # speed. z is the one DOF with a speed of its own (500, not 300), which is
    # exactly what `budget`'s per-group lookup is for.
    p.step().set(Z, hand.clamped_mm(Z, cfg.lift_mm), float(floor[Z]), "goal",
                 budget(Z))

    # z is not named again, and a retired joint keeps its standing order, so it
    # goes on holding the cap up at lift torque while the rows below open the
    # base jaw underneath it. At travel torque it would sag back onto the bottle.
    s = p.step()
    travel(s, [BASE_LEFT, BASE_RIGHT], span)
    travel(s, AUX_FINGERS, 0.0)
    travel(s, BASE_JAW, jaw)
    yield p.build()
    return Result(radius, ok, why)

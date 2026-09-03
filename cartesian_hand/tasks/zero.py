"""Find every DOF's hard stop and report it as that hand's zero offset.

One sentence: for each group of joints, in mechanical order, creep it into its
hard stop, remember where that was, back off to mid travel.

Runs in a relative frame -- millimetres are undefined before this completes.
Every goal here is `here ± something`, so the origin cancels and the task is
correct in the uncalibrated startup frame, the calibrated frame, and the sim's
where `q = 0` is the rest pose.

Returns `motions.Result`: `[N, J]` stop positions + `[N]` ok flags. Never
raises -- envs fail independently; a bare `.all()` would let one env out of
4096 discard 4095 that succeeded.

Bench, hand_1, 2026-09-03
-------------------------
Re-zeroed against a calibration recorded on an earlier run and compared count
for count. Deltas were `[+6, +1, -2, +1, +2, -1, +3]` counts, i.e. **0.074 mm
worst case** at 81.5 counts/mm, so an independent re-zero reproduces the datum
to well under the 1.0 mm `POSITION_TOLERANCE_MM` the parks retire inside.

That is the number that makes the rest of the file's claims checkable, and it
exercises three of them at once: every DOF parked at its *own* mid travel
(27.5 mm fingers, 25.0 mm jaws and z, from a travel table where those differ),
the z park saturated the load channel at 1000 where the others sat near 280 --
which is `park_torque` being lifted to z's 800 floor and z alone being under
gravity -- and no phase expired, so `timeout_margin` covers the real creep
speed rather than the nominal one.
"""
from dataclasses import dataclass, field

import torch

from ..config import AUX_FINGERS, AUX_JAW, BASE_FINGERS, BASE_JAW, HandConfig, Z
from ..motions import Program, Result, Task

# Mechanical order: fingers retract before jaws close, jaws clear before z drops.
# Tuple, not a `Config` field -- a wrong number in `Config` is a retry; a wrong
# order here closes a jaw on a finger.
PHASES = (
    (tuple(BASE_FINGERS + AUX_FINGERS), "fingers"),
    ((BASE_JAW, AUX_JAW),               "jaws"),
    ((Z,),                              "z"),
)


@dataclass
class Config:
    label: str = "Zero hand"
    """Button text on the studio page. Empty means no button."""
    sets_datum: bool = True
    """Whether this task's result becomes the hand's zero. Read off `Config()`
    rather than compared against the name "zero", so a variant still installs
    its calibration instead of having it printed and dropped."""
    overtravel_mm: float = 80.0
    """How far past the stop to command. Further than any DOF can travel (widest
    rail is 55 mm), so the stop and never the number is what ends the move."""
    timeout_margin: float = field(default=1.5, metadata={"tune": (1.0, 3.0)})
    """Multiplier on expected travel time. At 300 counts/s the hand moves
    3.68 mm/s, so half a finger rail is 7.5 s; a fixed 6 s park used to expire
    mid-move on every run. Tracks `HandConfig.speed` too."""
    park_torque: float = field(default=300.0, metadata={"tune": (100.0, 800.0)})
    """Floor for backing off the stop. Raised per DOF to `torque_min_to_move`
    where that is higher (z on both hands). A flat 300 is below what lifts z,
    so the z park moved nothing. Invisible in both sims -- mujoco drops torque,
    `MockServo` ignores it."""


def build(hand: HandConfig, start_mm: torch.Tensor, cfg: Config | None = None,
          **kwargs) -> Task:
    cfg = cfg or Config()
    n_envs, n_dof = start_mm.shape
    program = lambda: Program(n_envs, n_dof, hand.control_hz, start_mm.device)
    floor = hand.gain_vector("torque_min_to_move", start_mm.device).float()

    here = start_mm
    stops = start_mm.clone()
    alive = torch.ones(n_envs, dtype=torch.bool, device=start_mm.device)
    why = ""

    for dof_ids, name in PHASES:
        ids = list(dof_ids)
        # Per-phase, so a z failure reports in z-time and not in finger-time.
        # The seek crosses a whole rail; the park backs off half of one.
        seek_timeout = hand.travel_budget(ids, cfg.timeout_margin)
        park_timeout = hand.travel_budget(
            ids, cfg.timeout_margin, max(hand.travel_mm[d] for d in ids) / 2)

        # Seek the hard stop. `frame="here"` so the goal is a distance, not a
        # place -- millimetres have no origin yet. Overtravel outruns the rail,
        # so the stop and never the number is what ends the move, which is why
        # the executor must not clamp this goal (see `studio.live`).
        #
        # Torque is `torque_min_to_move`, the LEAST that moves the joint. Force
        # left over at contact deflects the rack instead of stopping the
        # carriage and the stop is recorded that far past where it is -- see
        # `config.torque_min_to_move`, which carries the full argument.
        p = program()
        p.step().set(ids, -cfg.overtravel_mm, floor[None, ids], "stuck",
                     seek_timeout, frame="here")
        seek = p.build()
        here = yield seek

        # Ask the outcome, not the position -- both stall and timeout end
        # stationary, indistinguishable in mm.
        reached = seek.all_reached(ids)
        lost, alive = ~reached & alive, reached & alive
        if not why and bool(lost.any()):
            why = (f"phase {name!r}: envs {lost.nonzero().flatten().tolist()} "
                   f"had a DOF that never reached a hard stop")
        stops[:, ids] = here[:, ids]

        # Back off to mid travel: the same frame, half this DOF's own rail.
        # Dead envs get an offset of 0 and park in place -- their "stop" is
        # wherever the budget ran out, so `+ travel/2` aims past the open end.
        # `.clamp(min=cfg.park_torque)` lifts z above its 350/800 floor so the
        # park actually moves it off the stop.
        offset = hand.upper(here.device)[None, ids] / 2 * alive[:, None]
        p = program()
        p.step().set(ids, offset,
                     floor[None, ids].clamp(min=cfg.park_torque), "goal",
                     park_timeout, frame="here")
        here = yield p.build()

        if not alive.any():
            break
    return Result(stops, alive, why)

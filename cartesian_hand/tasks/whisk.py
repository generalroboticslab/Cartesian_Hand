"""Squeeze, roll, and tilt a non-articulated whisk held in hand.

    init dof1256 to 0 -> squeeze jaws (dof0 & dof4 together)
      -> grip (gentle search to full span) -> squeeze (latch)
      -> repeat cycles( roll out: dof1/5 out, dof2/6 in -> roll in: dof1/5 in, dof2/6 out )
      -> [if Config.tilt] close top fingers (dof1/2/5/6 in)
      -> [if Config.tilt] repeat cycles( tilt out: dof1/2 out -> tilt in: dof1/2 in )
      -> ready (fingers to half span, jaws left closed)

`Config.tilt` (default True) gates the two tilt-only rows above; `ready`
always runs. `tasks/rotate_object.py` is this task with `tilt=False` under
its own "Rotate object" button -- a roll-only rod-turning maneuver with no
tilt.

0 mm is a finger's own retracted hard stop -- outward, away from whatever it
is holding. `finger_span_mm` is the other end of its travel -- inward, closed
onto the object. Every finger goal below is one of those two values.

No jaw twisting and no radius measurement: a whisk handle is a plain rod, not
a threaded cap, so there is nothing for `Twist`'s alternating-finger walk to
size itself against. The three demo motions come straight from which DOFs
move which way, driven directly with `Move`:

- **Squeezing** -- dof0 and dof4, the two jaws' own coarse parallel clamps,
  close on the rod together (one `Probe` over both, each latching contact
  independently) before any finger does anything finer. This row is what
  actually holds the whisk; the finger maneuvers below reposition around an
  object the jaws are already carrying, the same relationship `pump`'s and
  `triggers`' body-then-secondary-mechanism order uses.
- **Rolling** -- dof1 (base-left) and dof5 (aux-left) swing out to 0 while
  dof2 (base-right) and dof6 (aux-right) swing in to `finger_span_mm`, then
  the pair flips: dof1/5 in, dof2/6 out. Each pair spans a different jaw at
  the same corner (left or right), so the two diagonals sliding opposite
  ways rolls the whisk about its own long axis between the four contact
  points -- the same idea `Twist` uses to walk a cap round, just without the
  release/re-grip machinery a plain rod does not need. `cycles` full out/in
  cycles are that many rolls, not half-swings recentred in between -- there
  is nothing to recentre back to.
- **Tilting** -- dof1 and dof2 (the base jaw's own finger pair) swing out to
  0 together, then back in to `finger_span_mm` together, `cycles` times,
  while dof5 and dof6 hold still. Moving only the base jaw's pair rocks the
  whisk back and forth at the aux jaw's fixed hold, instead of rolling it.
  Roll's last row leaves dof1/dof2 on opposite ends (one just rolled in, the
  other just rolled out) and dof5/dof6 the same way, so `close top fingers`
  brings all four in to `finger_span_mm` first -- the first "tilt out" then
  starts from dof1/dof2 matched and dof5/dof6 both confirmed inward and
  matching each other, rather than split with one in and one still out.

Once both loops finish, `ready` brings all four fingers to `half_span_mm` --
neither rolled nor tilted to an extreme -- while the jaws are left exactly
where `squeeze jaws` put them, still closed on the whisk. Chaining another
run of this task from there starts from the same neutral finger pose every
time, without first releasing and re-finding the whisk.

Every row from `squeeze` (fingers) on pushes fingers back toward an object
they are already holding, so each one runs at `squeeze_torque` and accepts a
confirmed stall as arrival (`accept_stall=True`) -- the same "squeeze
whatever it finds" contact philosophy `cap`'s grip rows use. Only `grip`
itself, the one row searching for contact rather than re-squeezing it, runs
at the gentler `approach_torque` -- see the bench note for why it is still a
`Move` and not a `Probe`.

Torque is kept low throughout (`squeeze_torque`/`approach_torque` default
100) because a whisk is light and easily crushed or knocked from the
fingers. Speed is not correspondingly slowed: every free `Move` already runs
at the servo's no-load top speed regardless of any config here (see
`Sequence.__init__`'s `SERVO_NO_LOAD_RPM` note), and `approach_speed` --
the creep for the initial contact search -- defaults fast for `cap`'s own
reason: a slow creep is indistinguishable from a stalled contact.

Every goal is absolute in the hand's millimetre frame, so the hand must be
zeroed first.

**Bench, 2026-09-07: the grip probe's ceiling was wrong, twice, then the
probe itself was the wrong primitive.** First run set it to `half_span`
(20 mm); the fingers stalled on the whisk at ~19.2 mm, inside the fixed 1 mm
"reached goal without contact" band a `Probe` fails on. Raising the ceiling
to the full `finger_span_mm` (40 mm) reproduced the identical failure at
~39.3 mm. Raising it again to the finger's own physical travel limit
(`hand.travel_mm`) -- the same margin-past-the-object shape `cap`'s and
`screwdriver`'s probes use, going to a jaw's full travel (0) rather than any
estimate of where the object is -- did not fix hand_1: it stalled at 0.7-0.9
mm short of that ceiling too. Every ceiling tried lands the real contact
point inside the last ~1 mm, because that is where a whisk this size against
this hand's travel actually is, on every hand tried so far -- not something a
bigger ceiling ever escapes, since `clamped_mm` refuses to command past
`hand.travel_mm` at all (the far end of every rail is open, not a hard stop;
see `config.STANDARD_TRAVEL`).

The fix is not the ceiling but the primitive: `Probe` (`close_until_contact`)
fails immediately on landing this close without a *confirmed* stall, and
`_stalled`'s own `away` condition can never confirm one this close (it needs
*more* than 1 mm from goal to count a quiet tick at all) -- so no ceiling
within roughly 1 mm of the real contact point can ever pass a `Probe`,
regardless of how long the joint sits there loaded. `grip` is a plain `Move`
now, like every row after it: a free `Move` has no such all-or-nothing check,
just `reached` (widened to `contact_tolerance_mm`, catching the near-ceiling
stall directly) or `accept_stall` (catching a stall further out, for a
thicker whisk that stops well short of the ceiling). The following `squeeze`
row raises the effort from the gentle search cap to `squeeze_torque`, the
same latch `Probe`'s `grip=` parameter used to do on contact.

No other hardware results yet: the squeeze, roll, and tilt maneuvers are
authored, not bench-validated.
"""
from dataclasses import dataclass, field

import torch

from ..config import (AUX_JAW, AUX_LEFT, AUX_RIGHT, BASE_JAW, BASE_LEFT,
                      BASE_RIGHT, HandConfig)
from ..primitives import Loop, Move, Probe, Sequence

BASE = (BASE_LEFT, BASE_RIGHT)      # dof1, dof2 -- the base jaw's own pair
AUX = (AUX_LEFT, AUX_RIGHT)         # dof5, dof6 -- the aux jaw's own pair
LEFT = (BASE_LEFT, AUX_LEFT)        # dof1, dof5 -- both jaws' left corner
RIGHT = (BASE_RIGHT, AUX_RIGHT)     # dof2, dof6 -- both jaws' right corner
FINGERS = BASE + AUX                # dof1, dof2, dof5, dof6
JAWS = (BASE_JAW, AUX_JAW)          # dof0, dof4


@dataclass
class Config:
    """Bench units. Every `tune` field becomes a slider on the studio page."""

    label: str = "Roll & tilt whisk"
    sets_datum: bool = False

    finger_stroke: float = field(default=40.0, metadata={"tune": (10.0, 55.0)})
    """Full sweep of a finger pair, in mm. Roll and tilt both drive between
    0 and this span; the grip searches out to the finger's full travel
    instead (see the bench note on why)."""
    squeeze_torque: float = field(default=100.0, metadata={"tune": (50.0, 300.0)})
    """Torque for every row from the jaw squeeze on -- fingers, roll,
    close-top-fingers, tilt, and the final `ready`. Low by default: a whisk
    is light and easily crushed or knocked loose."""
    approach_torque: float = field(default=100.0, metadata={"tune": (50.0, 300.0)})
    """Torque while the initial grip is still searching for contact, before
    `squeeze_torque` latches in."""
    approach_speed: float = field(default=800.0, metadata={"tune": (25.0, 1500.0)})
    """Servo speed register for the initial contact search, counts/s.

    Fast for `cap`'s reason: there is no contact sensor, so contact is read as
    a confirmed stop under 0.3 mm/s, and a slow creep runs close enough to
    that threshold that ordinary servo hesitation reads as contact."""
    travel_torque: float = field(default=50.0, metadata={"tune": (30.0, 200.0)})
    """Torque for the one unloaded row: driving dof1256 to 0 at task start."""
    travel_speed: float = field(default=1500.0, metadata={"tune": (200.0, 1500.0)})
    """Unused: every free move now runs at the servo's rated no-load top
    speed regardless of this value."""
    cycles: float = field(default=2.0, metadata={"tune": (1.0, 10.0)})
    """Repetitions of each maneuver, roll then tilt, rounded to an
    integer."""
    contact_tolerance_mm: float = field(default=2.0, metadata={"tune": (1.0, 5.0)})
    """Arrival tolerance for the grip row's approach, mm.

    Widened past `Move`'s 1 mm default because the whisk stalls within a
    millimetre of the probed ceiling on every hand tried so far -- see the
    bench note. `accept_stall` alone does not cover this: it needs `_stalled`
    to confirm a stop *more than* 1 mm from goal, which a stall this close can
    never do."""
    tilt: bool = True
    """True runs `close top fingers` and the tilt loop after roll; False
    stops after roll and goes straight to `ready`. Not tunable: it is not a
    number, and half the range is a different maneuver -- the same idiom
    `scissors.py`'s `normally_closed` uses. `tasks/rotate_object.py` is this
    task with `tilt=False` under its own button."""
    timeout_margin: float = field(default=1.5, metadata={"tune": (1.0, 3.0)})


def build(hand: HandConfig, start_mm: torch.Tensor,
          cfg: Config | None = None) -> Sequence:
    """Squeeze a whisk in both jaws, then roll it and tilt it with the fingers."""
    cfg = cfg or Config()
    mm = hand.clamped_mm
    finger_floor = max(float(hand.gain_vector("torque_min_to_move")[dof])
                       for dof in FINGERS)
    squeeze_effort = max(cfg.squeeze_torque, finger_floor) / 1000.0
    approach_effort = max(cfg.approach_torque, finger_floor) / 1000.0
    finger_span_mm = mm(BASE_LEFT, cfg.finger_stroke)
    half_span_mm = finger_span_mm / 2
    # The finger's true physical extreme, not a value derived from
    # `finger_stroke` -- see the module's bench note for why a goal merely
    # larger than the expected contact point is not enough, and why this is
    # a `Move` rather than a `Probe`.
    probe_reach_mm = mm(BASE_LEFT, hand.travel_mm[BASE_LEFT])
    cycles = max(1, round(cfg.cycles))

    rows = [
        Move(label="init", goal={FINGERS: 0.0}),
        # The two coarse parallel-jaw clamps, closed on the rod together --
        # one `Probe` over both dof0 and dof4, each searching its own hard
        # stop (0) and latching independently, rather than one jaw closing
        # fully before the other starts.
        Probe(label="squeeze jaws", group=JAWS, creep=True,
              grip=squeeze_effort),
        # Gentle search creep toward the ceiling; `squeeze` below latches the
        # real grip torque once this arrives. `tolerance_mm` (not `Probe`'s
        # hair-trigger 1 mm) covers landing right at the ceiling; `accept_stall`
        # covers stopping well short of it, for a thicker whisk.
        Move(label="grip", goal={FINGERS: probe_reach_mm}, creep=True,
             effort=approach_effort, tolerance_mm=cfg.contact_tolerance_mm,
             accept_stall=True),
        Move(label="squeeze", goal={FINGERS: probe_reach_mm},
             effort=squeeze_effort, accept_stall=True),
        Loop(count=cycles, rows=[
            Move(label="roll out", goal={LEFT: 0.0, RIGHT: finger_span_mm},
                 effort=squeeze_effort, accept_stall=True),
            Move(label="roll in", goal={LEFT: finger_span_mm, RIGHT: 0.0},
                 effort=squeeze_effort, accept_stall=True),
        ]),
    ]
    if cfg.tilt:
        rows += [
            # See the "Tilting" bullet above for why this row exists. dof5
            # and dof6 are pinned in too -- roll's last row leaves them on
            # opposite ends, and tilt wants the aux jaw's hold matched and
            # firm, not one finger in and the other still out.
            Move(label="close top fingers", goal={BASE: finger_span_mm,
                                                  AUX: finger_span_mm},
                 effort=squeeze_effort, accept_stall=True),
            Loop(count=cycles, rows=[
                Move(label="tilt out", goal={BASE: 0.0},
                     effort=squeeze_effort, accept_stall=True),
                Move(label="tilt in", goal={BASE: finger_span_mm},
                     effort=squeeze_effort, accept_stall=True),
            ]),
        ]
    # Ready for the next run: fingers to neutral (half span, neither
    # rolled nor tilted to an extreme), jaws left alone -- still closed
    # on the whisk from `squeeze jaws`, not released.
    rows.append(Move(label="ready", goal={FINGERS: half_span_mm},
                     effort=squeeze_effort, accept_stall=True))

    return Sequence(rows, hand=hand,
       start_mm=start_mm,
       travel_torque=cfg.travel_torque,
       approach_torque=cfg.approach_torque,
       approach_speed=cfg.approach_speed,
       timeout_margin=cfg.timeout_margin,
       travel_speed=cfg.travel_speed)

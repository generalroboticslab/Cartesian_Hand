"""Run an electric screwdriver's own motor in, then back out.

    entry -> height -> grip driver body (taut)
      -> repeat cycle(
           press cw button -> feed z down at constant torque (screw in)
           -> release cw -> raise z to the ccw button's height
           -> press ccw button -> crawl z up (screw out)
           -> release ccw -> return z
         )

Unlike `screwdriver`'s manual variant, the hand does no twisting itself: the
tool spins its own bit. The hand's job is to hold the driver body still,
press its two trigger buttons, and feed its own z axis up and down -- z
doubles as the tool's own vertical travel, since driving the bit down/up is
what feeds the screw in/out once the motor is running.

The cw and ccw buttons sit at different heights on the tool body, so what
looks like the same jaw-close row twice is really two different presses:
`cw_offset` is where entry parks z and where the aux jaw's first close each
cycle reaches the cw button; `ccw_offset` is a second, higher z the aux jaw
is raised to before it closes again, this time onto ccw.

The base jaw's role is the same taut, never-squeezed hold `screwdriver` gives
the tip: fingers pinched at 0, closed onto the driver's shaft/collar by a
`Probe`'s own contact alone, holding whatever effort that contact left behind
for the rest of the task -- one grip for every cycle, never released, so the
same driver stays held between screws exactly as it does between strokes of
one screw. It is only steadying the tool, not gripping anything that needs to
hold load, so it is never squeezed harder than `base_approach_torque`.

**No settle waits.** An earlier draft paused after entry (to place the driver
by hand), after the body grip (for camera setup), and after releasing cw
(before repositioning). None of that is needed: like every other task here,
a `Probe`'s own closed-loop contact detection is what actually gates the next
row, not a fixed delay guessed to be long enough -- see `cap`'s "no pre-probe
sleep" note. Placing the driver and setting up a camera happen before the
task is started, not inside it.

**Screwing in is a single, constant-torque feed.** `screw_torque` is light
(50/1000 by default) and z drives straight to zero at that cap. Gentle and
constant is enough here because the cap itself is the safety margin: at
50/1000 there isn't the sudden weight behind the feed that a stiction "shoot"
would need to matter. But that same low torque can stall well short of the
screw before the bit has actually engaged, and a stall this light is common,
not exceptional, so the feed is two rows: a `Hold` that just keeps commanding
the push for `screw_min_push_s` regardless of an early stall (`Hold` never
fails on one), then a `Move` with `accept_stall` to finish -- turning the
screw's own resistance stopping it short of zero, once genuinely seated, into
arrival rather than a fault. The aux jaw's grip on the cw button (`button_torque`)
has to survive z's reaction force through all of this without slipping off the
button, which is why it is squeezed harder than a probe's approach effort.

**Screwing out crawls.** The pull runs at `extract_torque` (max) to keep the
motor spinning ccw the whole climb, and *that* is exactly the resistance a
servo's onboard position PID cannot see coming: it saturates torque fighting
static friction, then "shoots" once it breaks free, a stick-slip burst the
firmware gives no direct handle on (no P/I/D or punch/min-force register,
only Speed/Acc/Torque). A single `Move` straight to the far target at that
torque lets one burst run the whole remaining distance, so the pull is
`crawl_step_mm` moves instead, one row per step (`Loop.rows` cannot itself
hold a `Loop`, so this repeats a plain `Move` rather than nesting one): every
step's goal is computed from `m.z_pos`, the position the *previous* step
actually stopped at (via `Move.measure` -- see `primitives.Move`), not from a
plan made before either step ran, so an overshoot on one step is accounted
for rather than compounding into the next. `crawl_settle_s` between steps,
with `crawl_step_mm`, sets the average pull rate; default 1mm / 1s = 1mm/s.

`cycles` follows `cap`'s and `triggers`' convention exactly: zero repeats
until the task is stopped (disarm torque in Studio, or Ctrl-C headless); a
positive value runs that many complete screw-in/screw-out cycles. Every cycle
ends with z back at `cw_offset` and the aux jaw open, which is also where the
task starts -- so cycles run back to back with no seam between them, and a
fresh run of this same task picks up exactly as if it were one more cycle.

Only the thick driver body is implemented -- the thin body is a different
enough shape that it needs its own bench pass, and is out of scope here.

No hardware results for this task yet: the sequence is validated, the port
of it is not.
"""

import math
from dataclasses import dataclass, field

import torch

from ..config import AUX_FINGERS, AUX_JAW, BASE_FINGERS, BASE_JAW, HandConfig, Z
from ..primitives import Hold, Loop, Move, Probe, Sequence, lift_effort

JAWS = (BASE_JAW, AUX_JAW)
BASE = tuple(BASE_FINGERS)
AUX = tuple(AUX_FINGERS)


@dataclass
class Config:
    """Bench units. Every `tune` field becomes a slider on the studio page."""

    label: str = "Run electric screwdriver"
    sets_datum: bool = False

    jaw_opening: float = field(default=25.0, metadata={"tune": (10.0, 40.0)})
    """How wide each jaw opens at entry before the base jaw closes on the
    driver body, in mm."""
    cw_offset: float = field(default=25.0, metadata={"tune": (5.0, 40.0)})
    """Absolute z height, mm above z zero, entry parks at and every cycle
    returns to -- and, with no other row moving z first, where the aux jaw's
    first close each cycle reaches the cw button."""
    base_approach_torque: float = field(default=50.0,
                                        metadata={"tune": (20.0, 300.0)})
    """Torque the base jaw closes onto the driver's shaft/collar with, and
    holds for the whole task -- taut, never squeezed harder, since it is only
    steadying the tool."""
    button_torque: float = field(default=300.0, metadata={"tune": (100.0, 1000.0)})
    """Torque the aux jaw closes onto either trigger button with, and holds
    through the feed that follows -- the jaw that keeps the motor running.
    Needs to be enough that the jaw cannot slip off the button under z's
    reaction force during the screw-in push, not just enough to trigger it."""
    screw_torque: float = field(default=50.0, metadata={"tune": (10.0, 200.0)})
    """Constant z torque for the screw-in feed -- light, so `screw_min_push_s`
    below is what gets the bit engaged, not force. A single move to zero, not
    crawled: see the module docstring for why screwing out needs the crawl
    and this doesn't."""
    screw_min_push_s: float = field(default=8.0, metadata={"tune": (1.0, 10.0)})
    """Minimum time the screw-in feed keeps pushing toward zero before a
    stall is allowed to end it. At `screw_torque`'s light cap the feed can
    stall -- a confirmed stop, not a fault -- within `CONFIRM_TICKS` (0.2s) of
    starting, well before the bit has actually engaged the screw; holding the
    push for this long first gives the low torque time to work rather than
    reading that first stall as arrival."""
    crawl_step_mm: float = field(default=1.0, metadata={"tune": (0.5, 3.0)})
    """Distance z advances per crawl step during the extraction pull, instead
    of driving straight to the target in one move -- see the module
    docstring."""
    crawl_settle_s: float = field(default=1.0, metadata={"tune": (0.2, 3.0)})
    """Pause between crawl steps. Combined with `crawl_step_mm`, sets the
    average pull rate -- 1mm / 1s = 1mm/s by default."""
    z_speed_scale: float = field(default=3.0, metadata={"tune": (1.0, 6.0)})
    """Multiply z's creep speed by this for the screw-in feed and the
    extraction crawl. `creep` is `approach_speed`'s register, shared with
    every jaw probe in this task, so raising `approach_speed` itself to speed
    up z would also speed up -- and erode the contact-detection margin of --
    the button and body probes; this scales only z's own rows instead."""
    release_clearance: float = field(default=10.0, metadata={"tune": (2.0, 20.0)})
    """How far past a button's measured contact point the aux jaw opens to
    release it. Relative to that press's own contact position, not an
    absolute mm target -- the button contact point isn't at 0, and an
    absolute target below it would drive the jaw further onto the button
    instead of off it, leaving the motor spinning."""
    ccw_offset: float = field(default=30.0, metadata={"tune": (10.0, 60.0)})
    """Absolute z height, mm above z zero, the aux jaw is raised to before it
    closes onto the ccw button. Set a few mm above the button's measured
    reach, so the jaw doesn't scrape it on the way up."""
    extract_offset: float = field(default=50.0, metadata={"tune": (20.0, 90.0)})
    """Absolute z height the extraction pull crawls to, with the aux jaw held
    on the ccw button the whole way."""
    extract_torque: float = field(default=1000.0, metadata={"tune": (500.0, 1000.0)})
    """Z torque for the extraction pull. Max -- the pull has to keep the
    motor spinning ccw against the screw's resistance for the entire climb,
    unlike the controlled `screw_torque` feed on the way down."""
    travel_torque: float = field(default=50.0, metadata={"tune": (30.0, 200.0)})
    """Torque for unloaded travel."""
    approach_torque: float = field(default=100.0, metadata={"tune": (50.0, 300.0)})
    """Torque while a jaw is still seeking contact, before `grip` takes over."""
    approach_speed: float = field(default=300.0, metadata={"tune": (25.0, 800.0)})
    """Servo speed register while closing a jaw to find contact, and while
    crawling z, counts/s.

    300, for `cap`'s reason: there is no contact sensor, so contact is read
    as a confirmed stop under 0.3 mm/s, and at 50 counts/s the creep itself
    runs at twice that threshold, so any servo hesitation reads as contact."""
    travel_speed: float = field(default=1500.0, metadata={"tune": (200.0, 1500.0)})
    """Unused: every free move now runs at the servo's rated no-load top
    speed regardless of this value."""
    timeout_margin: float = field(default=1.5, metadata={"tune": (1.0, 3.0)})
    cycles: float = field(default=1.0, metadata={"tune": (0.0, 10.0)})
    """Complete screw-in/screw-out cycles, rounded to an integer. Zero repeats
    until the task is stopped (disarm torque in Studio, or Ctrl-C headless)."""


def _crawl_steps(distance_mm: float, step_mm: float) -> int:
    """How many `crawl_step_mm` steps cover `distance_mm`, rounded up.

    A generous, not exact, count: each step aims from where the previous one
    actually stopped (see `Move.measure`), so once that position reaches the
    step's own target the remaining steps compute a goal equal to their
    current position and finish at once -- a few extra steps here cost a
    handful of no-op ticks, not wall-clock time.
    """
    return max(1, math.ceil(distance_mm / step_mm))


def build(hand: HandConfig, start_mm: torch.Tensor,
          cfg: Config | None = None) -> Sequence:
    """Take a taut body grip, then crawl z through the cw and ccw presses."""
    cfg = cfg or Config()
    mm = hand.clamped_mm
    jaw_floor = max(float(hand.gain_vector("torque_min_to_move")[dof])
                    for dof in JAWS)
    body_effort = max(cfg.base_approach_torque, jaw_floor) / 1000.0
    button_effort = max(cfg.button_torque, jaw_floor) / 1000.0
    screw_effort = cfg.screw_torque / 1000.0
    extract_effort = cfg.extract_torque / 1000.0
    jaw_opening_mm = mm(BASE_JAW, cfg.jaw_opening)
    cw_height = mm(Z, cfg.cw_offset)
    ccw_height = mm(Z, cfg.ccw_offset)
    extract_height = mm(Z, cfg.extract_offset)
    feed_target = mm(Z, 0.0)
    step_mm = cfg.crawl_step_mm
    cycles = float("inf") if cfg.cycles <= 0 else max(1, round(cfg.cycles))

    # One crawl step of the pull. Free, not `accept_stall`: the motor has to
    # keep spinning ccw for the entire pull, not just long enough to break the
    # screw free, so stopping short here is a fault, not arrival. Repeated
    # (not looped -- `Loop.rows` cannot itself hold a `Loop`) rather than
    # nested: the row is frozen and stateless, so the same instance
    # re-appearing is exactly as good as that many distinct ones.
    pull_step = Move(
        label="pull step",
        goal={Z: lambda m: torch.clamp(m.z_pos + step_mm, max=extract_height)},
        effort=extract_effort, creep=True, speed_scale=cfg.z_speed_scale,
        measure={"z_pos": Z})
    extract_crawl = [pull_step, Hold(label="pull settle", seconds=cfg.crawl_settle_s)
                     ] * _crawl_steps(extract_height - ccw_height, step_mm)

    return Sequence([
        # Jaws wide, fingers pinched at 0 (not centred): the shaft/collar and
        # the buttons both want a narrow point contact, not a cradle.
        Move(label="entry", goal={JAWS: jaw_opening_mm, BASE: 0.0, AUX: 0.0}),
        # z on its own, after: the entry move may ascend and so carries the
        # lift floor, which is far too much force to also put behind a jaw
        # sweeping through free space.
        Move(label="height", goal={Z: cw_height}, effort=lift_effort(hand)),
        # Taut hold on the shaft/collar, never squeezed harder -- it holds
        # this grip for every cycle while the aux jaw does the pressing.
        Probe(label="grip body", group=BASE_JAW, creep=True, grip=body_effort),
        Loop(count=cycles, rows=[
            # Close onto the cw button and hold it -- the jaw that keeps the
            # motor spinning cw. Contact position is recorded, not assumed to
            # be 0, so the release below knows where it actually stopped.
            Probe(label="cw button", group=AUX_JAW, creep=True,
                  grip=button_effort, measure={"cw": AUX_JAW}),
            # Constant, low torque, straight to zero -- see the module
            # docstring for why this doesn't need the crawl the pull does.
            # A `Hold` first, not a `Move`: at this light a torque the feed
            # can stall well before the bit has actually engaged, and `Hold`
            # never fails on one -- it just keeps commanding the push for
            # `screw_min_push_s` regardless, which is the minimum push this
            # needs. The aux jaw keeps holding the cw button throughout: this
            # row never names it, and a DOF no row names keeps whatever the
            # last one left it at.
            Hold(label="screw in (min push)", group=Z, goal=feed_target,
                 effort=screw_effort, creep=True, speed_scale=cfg.z_speed_scale,
                 seconds=cfg.screw_min_push_s),
            # Past the minimum push, a normal `Move` finishes the feed:
            # `accept_stall` because the screw's own resistance stops it
            # short of the target once fully seated, which is arrival here,
            # not a fault.
            Move(label="screw in", goal={Z: feed_target}, effort=screw_effort,
                 accept_stall=True, creep=True, speed_scale=cfg.z_speed_scale),
            # Let go of the cw button to stop the motor before repositioning.
            # `loaded=True` floors the effort at travel torque and accepts a
            # confirmed stall -- retracting against the grip that just closed
            # needs at least as much torque as made it.
            Move(label="release cw",
                 goal={AUX_JAW: lambda m: m.cw + cfg.release_clearance},
                 effort=button_effort, loaded=True, creep=True),
            # Raise to the ccw button's height, then close the aux jaw again
            # -- same close, different z, different button. `measure` seeds
            # `m.z_pos` for the extraction crawl below.
            Move(label="reposition", goal={Z: ccw_height},
                 effort=lift_effort(hand), measure={"z_pos": Z}),
            Probe(label="ccw button", group=AUX_JAW, creep=True,
                  grip=button_effort, measure={"ccw": AUX_JAW}),
            *extract_crawl,
            # Let go to stop the motor, then return to the cw button's
            # height -- ready for the next cycle, or for the next run of
            # this same task, with the driver body still gripped throughout.
            Move(label="release ccw",
                 goal={AUX_JAW: lambda m: m.ccw + cfg.release_clearance},
                 effort=button_effort, loaded=True, creep=True),
            Move(label="return", goal={Z: cw_height}, effort=lift_effort(hand)),
        ]),
    ], hand=hand,
       start_mm=start_mm,
       travel_torque=cfg.travel_torque,
       approach_torque=cfg.approach_torque,
       approach_speed=cfg.approach_speed,
       timeout_margin=cfg.timeout_margin,
       travel_speed=cfg.travel_speed)

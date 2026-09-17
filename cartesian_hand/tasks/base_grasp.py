"""Contact-probe the base jaw and hold the grip -- `cap.py`'s opening stage,
alone.

    probe base jaw -> hold grip

The same two rows `cap.py` opens with (its "probe" and "grip"), narrowed to
the base jaw only: no aux jaw, no z move, no twist. The grip is a bounded
preload: the probe measures where it touched and the hold commands `bite_mm`
past that, so the position error the servo works against is `bite_mm` and not
the whole remaining travel. See `bite_mm` for why that matters.

Every goal is absolute in the hand's millimetre frame, so the hand must be
zeroed first.
"""
from dataclasses import dataclass, field

import torch

from ..config import BASE_JAW, HandConfig
from ..primitives import Hold, Probe, Sequence


@dataclass
class Config:
    """Bench units. Every `tune` field becomes a slider on the studio page."""

    label: str = "Base grasp"
    sets_datum: bool = False

    squeeze_torque: float = field(default=400.0, metadata={"tune": (40.0, 500.0)})
    """Base jaw grip effort, latched on contact and held for the whole task.

    400 is the hardest cap this hand can hold continuously. Grip force IS
    current, so the hardest grip is the highest current the servo survives
    indefinitely, and a bench comparison put that between 400 and 550:
    stalled at 400 both jaws pull ~385mA and hold, at 550 both pull ~538mA
    and the firmware clears TORQUE_ENABLE within 3s. The trip latches, so
    overshooting is not a soft failure -- it drops the object, and only a
    torque off/on toggle revives the servo.

    This is a ceiling and MUST stay one -- a grip that actually saturates at
    400 is not continuous. Measured, a full stall there holds 385mA and runs
    40 -> 72C in 120 seconds, still climbing +15C/min with the slope barely
    bending; fitted, it settles near 140C, and the firmware cuts at 80. So it
    self-destructs in about two and a half minutes.

    Whether the jaw reaches the cap is `bite_mm`'s job: the cap bounds the
    current, the position error makes it. At the 2.0mm default the jaw draws
    about 90mA, which is why a cap this high is safe to carry."""
    bite_mm: float = field(default=2.0, metadata={"tune": (0.0, 4.0)})
    """How far past contact the hold commands, mm. This is what sets grip
    force, and it is the whole fix for the base jaw going slack mid-task.

    The grip used to be `Hold(goal=0.0)` -- jaw fully shut -- which with an
    object in it commands the servo ~12mm past where it can physically go,
    permanently. A position servo turns that error into current capped at the
    effort limit, and since the error never shrinks it sits at the cap
    forever. FeeTech firmware answers sustained saturation by clearing
    TORQUE_ENABLE, and the trip LATCHES: re-arming does not clear it, only a
    torque off/on toggle does, which mid-grip drops whatever is held. Measured
    on hand_3: stalled at cap 550 the jaw pulled 534mA and tripped in 3s; at
    cap 400 it pulled 385mA, survived, but heated ~15C/min with no sign of
    levelling inside a minute.

    Commanding `contact - bite_mm` makes the error `bite_mm` instead, so
    current follows the preload and the servo is not saturated at all.

    Swept on hand_3 against a real object, squeeze cap 250, steady state
    after 15s:

        bite 0.5mm -> settled 12.48 vs goal 12.46,    0mA   <- no grip at all
        bite 1.0mm -> settled 11.81 vs goal 11.60,  -55mA
        bite 2.0mm -> settled 11.01 vs goal 10.71,  -90mA

    Note what 0.5 does: the jaw REACHES its goal, so the position error is
    zero and so is the force. The object has about 0.5mm of free play -- slop,
    compliance, backlash -- and a bite that lands inside it grips nothing.
    That is the failure mode to watch for, not the stall: too small is silent.
    Hence the default of 2.0 rather than something timid. Only the part of the
    bite past the free play turns into grip, so a softer object wants more.

    That sweep ran at cap 250; `squeeze_torque` is now 400, so the same bite
    is free to pull harder and a deeper one is bounded by the cap rather than
    by the object. Erring deep is cheap and erring shallow is a dropped
    object."""
    approach_torque: float = field(default=150.0, metadata={"tune": (50.0, 300.0)})
    """Torque while the jaw is still searching for contact, before
    `squeeze_torque` latches in."""
    approach_speed: float = field(default=800.0, metadata={"tune": (25.0, 800.0)})
    """Servo speed register while closing the jaw to find contact, counts/s.

    Fast for `cap`'s reason: there is no contact sensor, so contact is read
    as a confirmed stop under 0.3 mm/s, and a slow creep runs close enough to
    that threshold that ordinary servo hesitation reads as contact."""
    travel_torque: float = field(default=50.0, metadata={"tune": (30.0, 200.0)})
    """Unused: this task never issues a free move. Carried only because
    `Sequence` takes it."""
    travel_speed: float = field(default=1500.0, metadata={"tune": (200.0, 1500.0)})
    """Unused: every free move now runs at the servo's rated no-load top
    speed regardless of this value."""
    timeout_margin: float = field(default=1.5, metadata={"tune": (1.0, 3.0)})


def build(hand: HandConfig, start_mm: torch.Tensor,
          cfg: Config | None = None) -> Sequence:
    """Close the base jaw onto contact, then hold the grip."""
    cfg = cfg or Config()
    jaw_floor = float(hand.gain_vector("torque_min_to_move")[BASE_JAW])
    squeeze_effort = max(cfg.squeeze_torque, jaw_floor) / 1000.0
    bite_mm = cfg.bite_mm

    return Sequence([
        Probe(label="probe", group=BASE_JAW, creep=True,
              measure={"contact_mm": BASE_JAW}),
        Hold(label="grip", group=BASE_JAW, effort=squeeze_effort,
             goal=lambda m: m.contact_mm - bite_mm),
    ], hand=hand,
       start_mm=start_mm,
       travel_torque=cfg.travel_torque,
       approach_torque=cfg.approach_torque,
       approach_speed=cfg.approach_speed,
       timeout_margin=cfg.timeout_margin,
       travel_speed=cfg.travel_speed)

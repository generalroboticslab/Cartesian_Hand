"""Child-safe cap: `cap`'s twist, but pressed down through the opening turn
as well as the closing one.

Just the button. The mechanism is `cap.build`'s `child_safe` branch -- see
that module's docstring and the branch itself for the press-hold-turn-retract
cycle it runs on every opening stroke. `from . import cap`, not `from .cap
import Config`: the latter would bind `cap`'s own `Config` name here, and
`tasks.config` would then hand back `cap`'s label and defaults instead of
this file's -- see the "Writing a variant" note in `tasks/__init__.py`.
"""
from . import cap


def Config(**overrides) -> cap.Config:
    return cap.Config(label="Child-safe cap", child_safe=True, **overrides)


def build(hand, start_mm, cfg: cap.Config | None = None, **kwargs) -> cap.Sequence:
    return cap.build(hand, start_mm, cfg=cfg or Config(), **kwargs)

"""Dropper bottle: `cap`'s cap-opening twist, plus a bulb-squeeze insert.

Just the button. The mechanism is `cap.build`'s `dropper_bottle` branch --
see that module's docstring and the branch itself for the row-by-row squeeze
and release. `from . import cap`, not `from .cap import Config`: the latter
would bind `cap`'s own `Config` name here, and `tasks.config` would then hand
back `cap`'s label and defaults instead of this file's -- see the "Writing a
variant" note in `tasks/__init__.py`.
"""
from . import cap


def Config(**overrides) -> cap.Config:
    return cap.Config(label="Dropper bottle", dropper_bottle=True, **overrides)


def build(hand, start_mm, cfg: cap.Config | None = None, **kwargs) -> cap.Sequence:
    return cap.build(hand, start_mm, cfg=cfg or Config(), **kwargs)

"""Task discovery: one file name builds one controller for any executor.

A task never opens a bus, sleeps, or steps a simulator. Feedback-heavy tasks
return a typed `Policy`; fixed timelines and pre-calibration zeroing return a
`motions.Task` generator. Executors accept both controller forms by design.

The whole dispatch, in one sentence
-----------------------------------
`--task zero` imports `tasks/zero.py` and calls its `build()`. That is all of
it. There is no registry, nothing is registered at import time, and the file
stem is the name because it *is* the import.

A task module supplies exactly two names:

    build(hand, start_mm, cfg=None, **kwargs)   required -- returns its controller
    Config                                      optional -- dataclass of defaults

`build` constructs the controller at the requested batch width and device.
Direct policies keep phase and results in typed tensor state. Program generators
return `motions.Result`; they remain the smaller representation for fixed
timelines and for zeroing, whose uncalibrated overtravel cannot be clamped like
a normal direct action.

`Config` is everything else, including what the dispatch here reads:

    label      = "Zero hand"   puts a button on the page; "" means none
    sets_datum = True          the result becomes the hand's new zero

Those are fields rather than module constants so a task has one configuration
object instead of a dataclass plus a scatter of `UPPER_CASE` beside it. They are
read off `Config()` -- building one runs no program, so the page can be laid out
before any task exists.

Writing a variant
-----------------
A new file next to the one it varies, reaching what it reuses through the
module:

    # tasks/cap_gentle.py
    \"\"\"Squeeze 80 -> 40: hand_2 crushed a PET cap on stroke 2 at 80.\"\"\"
    from . import cap

    def build(hand, start_mm, **kwargs):
        return cap.build(hand, start_mm, cfg=cap.Config(squeeze_torque=40.0),
                         **kwargs)

Reachable immediately as `--task cap_gentle`, and every name in it is a real
import an editor can follow. `Config` is a dataclass, so its constructor is the
configuration -- there is no `configure()` hook to learn, and
`dataclasses.replace` composes one variant onto another when that is wanted.

**`from . import cap`, not `from .cap import Config`.** The second binds `Config`
into the variant's own namespace, where `config()` below finds it and hands back
cap's label -- so the variant would inherit a button it never asked for. Reaching
through the module is what keeps opt-in working.

The module docstring is the bench log: a result kept only in working notes is
not versioned beside the task variant it describes.
"""
import dataclasses
import importlib
import os
import pkgutil
from pathlib import Path
from types import ModuleType

import torch

from ..config import CALIB_PATH, HandConfig
from ..motions import Task
from ..policy import Policy

Controller = Task | Policy


def names() -> list[str]:
    """Every task: the file stems under `tasks/`, which are the `--task` values."""
    return sorted(m.name for m in pkgutil.iter_modules([str(Path(__file__).parent)]))


def module(name: str) -> ModuleType:
    """`tasks/<name>.py`. Raises KeyError listing the directory.

    Validated because `name` arrives from a CLI flag or a page click, and a
    bare `ModuleNotFoundError` names the wrong problem.
    """
    if name not in names():
        raise KeyError(f"unknown task {name!r}. Known tasks:\n{describe()}")
    return importlib.import_module(f".{name}", __package__)


def make(name: str, hand: HandConfig, start_mm: torch.Tensor,
         **kwargs) -> Controller:
    """Build task `name` for this hand, batch width, and tensor device.

    `start_mm` is [N, J], where the joints are now. It is an argument rather
    than something the task reads because task code owns no backend. `kwargs`
    is reserved for task-specific construction values.
    """
    return module(name).build(hand, start_mm, **kwargs)


def config(name: str) -> object | None:
    """A task's defaults: its `Config()`, or None if the module defines none.

    Instantiated rather than read off the class, so every default lives in the
    dataclass and nowhere else. Cheap -- a `Config` is plain fields and building
    one runs no program.

    A variant reaches the class it reuses through the module (`from . import
    cap`, then `cap.Config`). Importing the *name* would rebind it here too, and
    the variant would answer with the original's label and datum flag.
    """
    cls = getattr(module(name), "Config", None)
    return cls() if cls is not None else None


def tunables(name: str) -> dict[str, tuple[float, float, float]]:
    """`{field: (value, lo, hi)}` -- every number the studio's tune panel offers.

    Marked on the field itself, `field(default=80.0, metadata={"tune": (30,
    200)})`, so the bound sits beside the value and its docstring. A separate
    table of bounds would be a second place to edit and the one that goes stale
    is the one nobody reads.

    Opt-in, not opt-out, but the bar is low: `label` and `sets_datum` are out
    because they are not numbers.

    **A declared bound is not a claim that the number is measurable.** These are
    ranges a human may drag with the hand in front of them. An automatic tuner
    reading them would also need to know which of them the backend it runs on
    can actually observe: mujoco drops torque on the floor (`sim.profile` takes
    the goal and nothing else), so every torque field here is invisible there,
    and half of a composed task's fields are torques. A tuner that sampled them
    anyway would report noise as a finding -- which is what happened, and why
    the search driver was deleted on 2026-09-03 rather than left to be trusted.
    """
    cfg = config(name)
    return {f.name: (getattr(cfg, f.name), *f.metadata["tune"])
            for f in dataclasses.fields(cfg) if "tune" in f.metadata} if cfg else {}


def sets_datum(name: str) -> bool:
    """Whether this task's result becomes the hand's zero.

    Read by `studio.live` instead of comparing the name against "zero", so a
    zeroing variant in its own file still installs its calibration.
    """
    return bool(getattr(config(name), "sets_datum", False))


def result_path(name: str) -> str | None:
    """Where this task's result should be appended as JSON, or None to only
    print it -- the file-persistence analog of `sets_datum` for a task whose
    result is a bench measurement rather than a calibration.

    Opt-in via `Config.save_json`. A bare filename resolves beside the
    calibration file: bench data is per-machine the same way
    `zero_offsets.json` is, and keeping the two side by side is one less path
    to remember.
    """
    path = getattr(config(name), "save_json", None)
    if path and not os.path.isabs(path):
        path = os.path.join(os.path.dirname(CALIB_PATH), path)
    return path


def buttons() -> list[tuple[str, str]]:
    """`[(name, label), ...]` for tasks whose Config names one.

    Opt-in: a variant is reachable through `--task` by name, and every variant
    sprouting its own button would turn the panel into a wall of them.
    """
    return [(n, label) for n in names()
            if (label := getattr(config(n), "label", ""))]


def describe() -> str:
    """Every task and its module docstring's first line."""
    out = []
    for n in names():
        head = (module(n).__doc__ or "").strip().split("\n")[0]
        out.append(f"  {n}" + (f"\n    {head}" if head else ""))
    return "\n".join(out)

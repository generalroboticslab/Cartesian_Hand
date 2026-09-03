"""Tasks: generators that submit motion programs to whatever hand is executing.

A task never opens a bus, never sleeps, and never steps a simulator. It yields
`Motions` programs and receives back the millimetres measured when each one
finished (see `motions.TaskRunner`). That is the whole interface, and it is why
the same task file runs on hardware under `studio.live` and on mujoco under
`sim.run` with nothing swapped out.

The whole dispatch, in one sentence
-----------------------------------
`--task zero` imports `tasks/zero.py` and calls its `build()`. That is all of
it. There is no registry, nothing is registered at import time, and the file
stem is the name because it *is* the import.

A task module supplies exactly two names:

    build(hand, start_mm, cfg=None, **kwargs)   required -- returns the generator
    Config                                      optional -- dataclass of defaults

`build` is the whole procedure: one generator, no helper split out of it. Three
`_program` builders called once each from a fourth function is four names for
one procedure, and the phase boundaries are already visible as the `yield`s.

It returns a `motions.Result`: the measurement, plus `[N]` flags saying which
envs that measurement is valid for. A task does not raise on a failure it can
attribute to particular envs -- `start_mm` is `[N, J]` and the envs are
independent, so an exception would let one of 4096 discard the rest. Deciding
what a failure *means* is the caller's: `sim.run` raises, `studio.finish`
declines to save, a batched trainer masks.

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

The module docstring is the bench log, and it has to be: `plan/` is not
committed and neither is `MEMORY.md`, so a result that is not in a docstring is
not anywhere.
"""
import dataclasses
import importlib
import pkgutil
from pathlib import Path
from types import ModuleType

import torch

from ..config import HandConfig
from ..motions import Task


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


def make(name: str, hand: HandConfig, start_mm: torch.Tensor, **kwargs) -> Task:
    """The generator for task `name`.

    `start_mm` is [N, J], where the joints are now. It is an argument rather
    than something the task reads because a task has no way to read anything --
    that is what makes it run on both backends. `kwargs` is per-task extra
    (`cap_radius`); a task ignores what it does not use.
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
    because they are not numbers, and `cap.max_cap_radius` is out because it is
    the one field that changes `K` -- two configurations whose programs are
    different lengths are not comparable.

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

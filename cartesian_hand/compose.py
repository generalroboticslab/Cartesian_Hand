"""Write a task out as a task file. No GUI here, and no GUI above it either.

This is the "the result is a file" half of the studio, kept in its own module
with no viser import so that anything unattended -- a tuner, a batch job -- can
write a task the same way a person writes a variant, and so the engine never
acquires a path to a toolkit. `studio.py` calls in here; nothing here calls back.

Two things can be written, matching the two things a person does:

    write_variant   keep the procedure, change the numbers   (tune)
    write_program   arrange rows into a new procedure        (combine)

Both produce a module under `tasks/`, which *is* the registration -- dispatch is
`pkgutil` over file stems, so a written file is reachable as `--task <name>` with
nothing else edited (see `tasks/__init__.py`).

Why generate Python and not a JSON spec
---------------------------------------
Because the file has to be readable and editable *after* the tool writes it. A
variant is a dataclass constructor call, so codegen of it is an f-string over
the changed fields and is lossless by construction -- there is no
`deg(-50) -> -0.8726646259971648` to recover, which is the failure that made
robot_studio's patcher 539 lines. And the module docstring is where a bench
result goes; a JSON spec has nowhere to put "hand_2 crushed a PET cap on stroke
2 at 80".

The generated program file is *not* meant to be write-only. It reads as the
hand-written tasks do, because it is the same three calls (`Program`, `set`,
`build`) a person would have made.
"""
import importlib
import sys
from pathlib import Path
from typing import NamedTuple

from .config import LABELS
from .motions import FRAMES, PREDICATES, STOPS, When

# Where the studio's timeline puts the rows it is about to run. A real task file
# under a reserved name, not a separate preview path, so what the timeline
# executes is byte-for-byte the artifact the same rows would ship as -- a preview
# that ran the rows through a second interpreter would be a second language, and
# the bug you chase would be the one that only exists in the language you did not
# save. Rewritten on every run, so it is the only file `_write` may clobber
# without being asked.
PREVIEW = "_preview"

# How wide a composed row's tuning bounds open around the value the author set.
# Half to double, because the author's number is evidence -- they watched it work
# -- and a slider that may wander an order of magnitude away from it is not
# tuning, it is starting over. Widen by editing the written file; that is what
# makes the output a source file rather than a black box.
TUNE_SPAN = 2.0


class Row(NamedTuple):
    """One step of a composed task: exactly the arguments of `Step.set`.

    Not a richer intermediate form. Anything this holds that `set` does not take
    is a thing the GUI can express and the engine cannot run, and that gap is
    where a visual editor stops being a view of the task and starts being a
    second language.
    """
    dofs: tuple[int, ...]
    goal: float
    torque: float
    stop: str = "goal"
    timeout_s: float = 6.0
    frame: str = "abs"
    when: When | None = None

    def validate(self) -> None:
        """Refuse a row the engine would refuse, but while it can still be named.

        `Step.set` raises on a bad `stop` or `frame` at build time, which for a
        generated file means the traceback arrives on the first run of a task
        nobody has read yet. Checking here points at the row.
        """
        if not self.dofs:
            raise ValueError("a row must name at least one DOF")
        if any(not 0 <= d < len(LABELS) for d in self.dofs):
            raise ValueError(f"row names DOF outside 0..{len(LABELS) - 1}: {self.dofs}")
        if self.stop not in STOPS:
            raise ValueError(f"unknown stop {self.stop!r}, expected one of {STOPS}")
        if self.frame not in FRAMES:
            raise ValueError(f"unknown frame {self.frame!r}, expected one of {FRAMES}")
        if self.when is not None:
            if self.when.kind not in PREDICATES or self.when.kind == "always":
                raise ValueError(f"unknown predicate {self.when.kind!r}")
            if self.when.dof >= len(LABELS):
                raise ValueError(
                    f"predicate source DOF {self.when.dof} outside "
                    f"0..{len(LABELS) - 1}")


def tasks_dir() -> Path:
    return Path(__file__).parent / "tasks"


def _write(name: str, source: str, overwrite: bool) -> Path:
    """Put `source` at `tasks/<name>.py`. Refuses to clobber unless told.

    Overwriting is opt-in because the file being written over may be the only
    copy of a bench result: `tasks/` is committed but a variant written this
    morning is not, and a writer that silently replaced yesterday's would be
    destroying the one artifact this module exists to produce.
    """
    if not name.isidentifier():
        raise ValueError(f"{name!r} is not a usable module name")
    path = tasks_dir() / f"{name}.py"
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"{path} exists. Pass overwrite=True, or pick another name.")
    path.write_text(source)
    # Import machinery caches a directory's contents keyed on its mtime, at
    # whatever resolution the filesystem reports. A file written and imported
    # inside one process -- the studio saving a task and then listing it, or
    # writing `_preview` and then running it -- lands inside that window and
    # raises `ModuleNotFoundError` for a file that is plainly on disk.
    # `tasks.names()` does not hit this (pkgutil re-scans), so the symptom is a
    # task that appears in the registry and then refuses to import.
    importlib.invalidate_caches()
    # An overwrite of something already imported is worse than a cache miss: the
    # file on disk is new and `import_module` hands back the old module object,
    # so the studio's timeline would rewrite `_preview.py` and then run the
    # previous edit -- silently, every time but the first.
    if (stale := sys.modules.get(f"{__package__}.tasks.{name}")) is not None:
        importlib.reload(stale)
    return path


def variant_source(name: str, base: str, changes: dict[str, float],
                   doc: str = "") -> str:
    """Source for a task that is `base` with some numbers changed.

    Reaches its parent as `from . import base`, never `from .base import
    Config`. The second binds `Config` into the variant's own namespace where
    `tasks.config` finds it, so the variant would answer with the parent's
    label and datum flag -- inheriting a button it never asked for. Reaching
    through the module is what keeps that opt-in.
    """
    if not changes:
        raise ValueError("a variant with no changes is the task it copies")
    unknown = set(changes) - set(tasks_tunable_names(base))
    if unknown:
        raise ValueError(
            f"{base} has no tunable field(s) {sorted(unknown)}. Known: "
            f"{sorted(tasks_tunable_names(base))}")
    args = ", ".join(f"{k}={v!r}" for k, v in sorted(changes.items()))
    head = doc or f"{base}, with {', '.join(sorted(changes))} changed."
    # Continuation aligned under the open paren, which moves with the length of
    # the parent's name. A fixed indent reads as a mistake in the one artifact
    # this module exists to make readable.
    return (
        f'"""{head}\n\nWritten by `cartesian_hand.compose`. Edit freely -- it is '
        f'an ordinary\ntask module, and `{base}.Config` supplies every field it '
        f'does not name.\n"""\n'
        f"import dataclasses\n\n"
        f"from . import {base}\n\n\n"
        f"def Config(**kwargs):\n"
        f'    """`{base}.Config` with this variant\'s numbers already in it.\n\n'
        f"    A function rather than a class because the only difference from the\n"
        f"    parent is defaults, and a dataclass cannot inherit one field's\n"
        f"    default without restating every field -- which would freeze the\n"
        f"    parent's other numbers at whatever they were the day this was\n"
        f"    written. `tasks.config` calls this, so the variant appears in the\n"
        f"    studio's tune panel like any other task.\n\n"
        f"    `label` is cleared: a button is opt-in, and inheriting the\n"
        f"    parent's would put a second copy of it on the panel. Name one here\n"
        f'    to give this variant its own.\n    """\n'
        f"    return dataclasses.replace({base}.Config({args}),\n"
        f'                               **{{"label": "", **kwargs}})\n\n\n'
        f"def build(hand, start_mm, cfg=None, **kwargs):\n"
        f"    return {base}.build(hand, start_mm, cfg=cfg or Config(), **kwargs)\n")


def tasks_tunable_names(base: str) -> list[str]:
    """`tasks.tunables(base)`'s keys. Imported late to keep this module leaf-ish.

    `tasks/__init__.py` imports every task module on `names()`, and a task is
    free to import `compose` (a written variant does not, but nothing stops
    one); importing it at module scope here would make that a cycle.
    """
    from . import tasks
    return list(tasks.tunables(base))


def _field(name: str, value: float) -> str:
    """One `Config` line: the value, and the bounds it may be moved in.

    A value of 0.0 gets **no** bounds and is emitted as a plain default. Its
    neighbourhood here is multiplicative, and half of zero is zero -- a
    degenerate range, which as a slider is a control that cannot move and to a
    sampler is a lie about how much it explored.
    `stop="stuck"` rows are the ones that hit this: their goal is 0.0 because
    the object is meant to stop the joint before it arrives, so the number was
    never the thing that mattered. Give it a range by editing the file.
    """
    lo, hi = sorted((value / TUNE_SPAN, value * TUNE_SPAN))
    if lo == hi:
        return f"    {name}: float = {value!r}"
    return (f"    {name}: float = field(default={value!r}, "
            f'metadata={{"tune": ({lo!r}, {hi!r})}})')


def program_source(name: str, rows: list[Row], doc: str = "") -> str:
    """Source for a task that is `rows`, in order, as one motion program.

    One row is one step, so the rows run strictly in sequence. Steps are the
    engine's ordering barrier and a step holds any number of joints, so packing
    independent rows into one step would be faster -- but only by the tick the
    barrier costs, and "the order I wrote is the order it runs" is the property
    that makes a row table legible. Merge by hand in the written file if a
    phase is worth overlapping.

    Every number becomes a tunable `Config` field, which is what lets a composed
    task be retuned (`tasks.tunables`, the studio's tune panel) rather than only
    run. Bounds open `TUNE_SPAN` around the authored value.

    Feedback stays inside the fixed program. A row's `when=When(...)` is resolved
    when it arms from the source joint's latest successful stop position/outcome;
    each environment chooses independently without changing K or returning to
    Python. `frame="here"` then expresses the contact-follow-up value: after a
    successful probe, move a clearance from where contact occurred.
    """
    for row in rows:
        row.validate()
    if not rows:
        raise ValueError("a task with no rows commands nothing")

    fields, calls = [], []
    for i, row in enumerate(rows):
        fields += [
            f'    # {"+".join(LABELS[d] for d in row.dofs)}: '
            f'{row.stop} from {row.frame}',
            _field(f"goal_{i}", row.goal),
            _field(f"torque_{i}", row.torque),
        ]
        when = ("None" if row.when is None else
                f"When({row.when.kind!r}, {row.when.dof}, "
                f"{row.when.threshold_mm!r})")
        calls.append(
            f"    p.step().set({list(row.dofs)}, cfg.goal_{i}, cfg.torque_{i},\n"
            f"                 {row.stop!r}, {row.timeout_s!r}, when={when}, "
            f"frame={row.frame!r})")

    head = doc or f"Composed task: {len(rows)} rows."
    body = "\n".join(fields)
    return (
        f'"""{head}\n\n'
        f"Written by `cartesian_hand.compose`. An ordinary task module: one\n"
        f"program, {len(rows)} steps, run in the order below. Every number is a\n"
        f"`Config` field with declared bounds, so the studio's tune panel can\n"
        f"move it, and `--task {name}` runs it on the hand or in mujoco.\n"
        f'"""\n'
        f"from dataclasses import dataclass, field\n\n"
        f"import torch\n\n"
        f"from ..config import HandConfig\n"
        f"from ..motions import Program, Result, Task, When\n\n\n"
        f"@dataclass\nclass Config:\n"
        f'    label: str = ""\n'
        f"    sets_datum: bool = False\n"
        f"{body}\n\n\n"
        f"def build(hand: HandConfig, start_mm: torch.Tensor,\n"
        f"          cfg: Config | None = None, **kwargs) -> Task:\n"
        f"    cfg = cfg or Config()\n"
        f"    n_envs, n_dof = start_mm.shape\n"
        f"    p = Program(n_envs, n_dof, hand.control_hz, start_mm.device)\n"
        f"{chr(10).join(calls)}\n"
        f"    program = p.build()\n"
        f"    measured = yield program\n"
        f"    ok = (program.succeeded() | ~program.executed).all(dim=(1, 2))\n"
        f"    why = (\"\" if bool(ok.all()) else\n"
        f"           f\"composed task failed in envs {{(~ok).nonzero().flatten().tolist()}}\")\n"
        f"    return Result(program.stopped_at, ok, why)\n")


def write_variant(name: str, base: str, changes: dict[str, float],
                  doc: str = "", overwrite: bool = False) -> Path:
    """`variant_source`, written to `tasks/<name>.py`. Returns the path."""
    return _write(name, variant_source(name, base, changes, doc), overwrite)


def write_program(name: str, rows: list[Row], doc: str = "",
                  overwrite: bool = False) -> Path:
    """`program_source`, written to `tasks/<name>.py`. Returns the path."""
    return _write(name, program_source(name, rows, doc), overwrite)

"""Self-check for what the studio writes: task files, and their tunable numbers.

    python tests/test_compose.py

The claim under test is the goal in one sentence: a task composed by clicking is
the same kind of thing as a task written by hand -- a file under `tasks/` that
runs on the bus, in one simulator, or in four thousand, and whose numbers are
declared where something other than a human can reach them.

So the assertions are all *round trips through the file*. Nothing here checks
that `program_source` emits a particular string; it checks that what it emits
imports, dispatches, and runs, because the string is an implementation detail
and the file being a real task is the whole claim.

Every test writes into the real `tasks/` directory, since that directory *is*
the registration -- there is no registry to fake -- and deletes what it wrote in
a `finally`. A leftover file would make `test_the_task_name_is_the_file_name`
fail in a different suite, which is the confusing kind of red.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import contextlib
import io

import torch

from cartesian_hand import compose, tasks
from cartesian_hand.config import HAND_2
from cartesian_hand.motions import GOAL, STUCK, TIMEOUT

ROWS = [
    compose.Row((1, 2), 35.0, 50.0, "goal", 10.0),
    compose.Row((0, 4), 0.0, 80.0, "stuck", 10.0),
    compose.Row((3,), 15.0, 50.0, "goal", 10.0, "here"),
]


@contextlib.contextmanager
def written(name, make):
    """Write a task, yield its name, delete it. Cleans up on failure too."""
    path = make(name)
    try:
        yield name
    finally:
        if path.exists():
            os.unlink(path)


def test_a_composed_file_is_an_ordinary_task():
    """Written by the tool, dispatched by the same `pkgutil` scan as the rest.

    The four assertions are the four things "it is a task" has to mean: the
    stem appears in the registry, `tasks.make` returns a generator, the program
    that generator yields carries the rows in the order they were added, and the
    numbers reached the cells rather than only the docstring.

    Asserted on the built program rather than on the source text, because a
    generated file that imports and then commands the wrong joint is exactly the
    failure a string comparison cannot see.
    """
    with written("t_composed", lambda n: compose.write_program(n, ROWS)) as name:
        assert name in tasks.names()
        gen = tasks.make(name, HAND_2, torch.zeros(1, HAND_2.n_dof))
        program = next(gen)
        assert program.goal_mm.shape[2] == len(ROWS), "one row is one step"
        for k, row in enumerate(ROWS):
            acting = [d for d in range(HAND_2.n_dof) if program.acts[0, d, k]]
            assert acting == list(row.dofs), (k, acting, row.dofs)
            assert float(program.goal_mm[0, row.dofs[0], k]) == row.goal
            assert float(program.torque[0, row.dofs[0], k]) == row.torque
        # The stop rule survives as the outcome code that satisfies it, which is
        # what `succeeded()` compares against -- a row whose `wants` is wrong
        # runs correctly and is then scored backwards.
        assert [int(program.wants[0, r.dofs[0], k]) for k, r in enumerate(ROWS)] \
            == [GOAL, STUCK, GOAL]
        # `frame="here"` is not a goal, it is a flag on the row; a generator
        # that dropped it would command an absolute 15 mm and still run.
        assert bool(program.relative[0, 3, 2]) and not bool(program.relative[0, 1, 0])
        gen.close()


def test_a_composed_task_runs_in_simulation():
    """The portability claim for a file nobody typed. Executed, not asserted.

    Runs the same `sim.run` the hand-written tasks use. What would fail here and
    nowhere else: a generated file that imports at module scope but whose
    `build` is not a generator, or whose `Result` has the wrong batch width --
    both of which pass an import check.

    A `RuntimeError` naming the composed task's own `why` is a pass, and is the
    only outcome this scene can give: the model has no objects, so the
    `stop="stuck"` row has nothing to stall against, and `sim.run` turns any
    failed `Result` into that exception. Reaching it means the file imported,
    dispatched, built its program and ran it to completion -- which is what is
    being claimed. Physical success needs an object scene and torque that
    reaches the actuators; neither exists yet (see `sim.py`).
    """
    from cartesian_hand import sim

    with written("t_sim", lambda n: compose.write_program(n, ROWS)) as name:
        task = tasks.make(name, HAND_2, torch.zeros(1, HAND_2.n_dof))
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                sim.run(task, HAND_2, verbose=False)
        except RuntimeError as e:
            assert "composed task failed" in str(e), e


def test_every_number_in_a_composed_task_is_tunable():
    """Compose then tune: the two halves have to meet in the same file.

    A composed task whose numbers were plain defaults would run and then be
    unreachable from the studio's tune panel, which reads `tasks.tunables`. The
    loop the goal describes -- compose it, then move its numbers -- would be
    broken in the middle with nothing failing.

    The zero-goal row is the interesting case. Bounds here are multiplicative,
    and half of zero is zero, so that field is emitted *without* bounds rather
    than with an empty range a sampler would silently draw one value from.
    """
    with written("t_knobs", lambda n: compose.write_program(n, ROWS)) as name:
        knobs = tasks.tunables(name)
        assert "goal_1" not in knobs, "a 0.0 goal has no multiplicative range"
        for row in (0, 2):
            assert f"goal_{row}" in knobs and f"torque_{row}" in knobs
        for field, (value, lo, hi) in knobs.items():
            assert lo < hi and lo <= value <= hi, (field, value, lo, hi)


def test_a_variant_names_only_what_it_changed():
    """A retuned task inherits its parent's defaults and its parent's later edits.

    Freezing every field into the variant is the failure this guards: the
    parent's numbers are bench measurements that get retuned, and a copy that
    pinned all of them would keep running against a hand that had moved on,
    silently. So the written source must mention the changed field and no other.

    It must also *not* inherit the parent's button. `tasks.config` reads `Config`
    off the variant's own namespace, so `from .scissors import Config` would put
    a second copy of the parent's label on the panel -- which is why the writer
    emits `from . import scissors`.
    """
    src = compose.variant_source("v", "scissors", {"open_timeout_s": 11.0})
    assert "from . import scissors" in src
    assert "from .scissors import" not in src
    assert "open_timeout_s=11.0" in src
    assert "grip_torque" not in src, "a variant froze a field it never changed"

    with written("t_variant", lambda n: compose.write_variant(
            n, "scissors", {"open_timeout_s": 11.0})) as name:
        assert tasks.config(name).open_timeout_s == 11.0
        assert tasks.config(name).grip_torque == tasks.config("scissors").grip_torque
        assert not dict(tasks.buttons()).get(name), "a variant inherited a button"
        # And it can be tuned again. A variant is what "save as" produces, so one
        # that was a dead end would break the loop in the middle -- tunable once,
        # then never again, with nothing failing to say so.
        assert tasks.tunables(name)["open_timeout_s"][0] == 11.0


def test_the_writers_refuse_what_would_fail_later():
    """Bad input is refused where it can still be named.

    Each of these otherwise surfaces as a traceback from inside a generated file
    on its first run -- pointing at a line nobody wrote, in a module nobody has
    read. The overwrite refusal is the one that costs something real: a variant
    written this morning is not in git, so clobbering it destroys the only copy
    of whatever it recorded.
    """
    bad = [
        (ValueError, lambda: compose.program_source("x", [])),
        (ValueError, lambda: compose.program_source(
            "x", [compose.Row((0,), 1.0, 1.0, "nope")])),
        (ValueError, lambda: compose.program_source(
            "x", [compose.Row((0,), 1.0, 1.0, "goal", 1.0, "elsewhere")])),
        (ValueError, lambda: compose.program_source(
            "x", [compose.Row((99,), 1.0, 1.0)])),
        (ValueError, lambda: compose.program_source("x", [compose.Row((), 1.0, 1.0)])),
        (ValueError, lambda: compose.variant_source("x", "scissors", {})),
        (ValueError, lambda: compose.variant_source(
            "x", "scissors", {"not_a_field": 1.0})),
        (ValueError, lambda: compose.write_program("not an identifier", ROWS)),
    ]
    for want, call in bad:
        try:
            call()
        except want:
            continue
        raise AssertionError(f"{call} was accepted")

    with written("t_clobber", lambda n: compose.write_program(n, ROWS)):
        try:
            compose.write_program("t_clobber", ROWS)
            raise AssertionError("an existing task file was overwritten")
        except FileExistsError:
            pass


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  ok    {t.__name__}")
        except BaseException as e:
            failed += 1
            print(f"  FAIL  {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

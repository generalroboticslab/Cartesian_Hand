"""Task registry.

A task is a module in this package exposing `run(hand, cfg)`, a `Config`
dataclass whose fields become CLI flags, and `DESCRIPTION` for help text.
Dropping a new module in this directory registers it: there is no list to update
and no per-task parser to write, because tyro derives the flags from Config.
"""

import importlib
import pkgutil

# Support modules, not tasks.
_EXCLUDED = {"primitives", "roles", "sim_zeroing"}


def _discover() -> dict:
    return {
        name: importlib.import_module(f"{__name__}.{name}")
        for _, name, _ in pkgutil.iter_modules(__path__)
        if not name.startswith("_") and name not in _EXCLUDED
    }


def registry() -> dict:
    """{task_name: module} for every module exposing both run() and Config."""
    return {n: m for n, m in _discover().items()
            if hasattr(m, "run") and hasattr(m, "Config")}


def names() -> list:
    return sorted(registry())


def get(name: str):
    tasks = registry()
    if name not in tasks:
        raise KeyError(f"Unknown task {name!r}. Available: {sorted(tasks)}")
    return tasks[name]


def describe(name: str) -> str:
    return getattr(get(name), "DESCRIPTION", "")

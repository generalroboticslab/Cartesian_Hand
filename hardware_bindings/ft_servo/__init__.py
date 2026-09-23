"""hardware_bindings.ft_servo — locate and load the compiled FtServo binding.

`ft_servo_python_only.py` sits alongside as a pyserial reimplementation of the
same protocol, but it is not a fallback: nothing here imports it, and a failed
extension load raises rather than degrading to it. Import it by name if you want
it.
"""
import importlib.util
import os
import pathlib
import sys
from functools import lru_cache


def _candidates(name):
    """Where the compiled extension might be, most specific first.

    Building in place next to this file is only one layout. A consuming project
    may compile these sources into its own wheel, which leaves the .so in that
    package rather than here, so an in-place-only search reports "not built"
    for an extension that is present and importable.
    """
    here = pathlib.Path(__file__).parent
    env = os.environ.get("FT_SERVO_EXT")
    if env:
        yield pathlib.Path(env)
    yield from sorted(here.glob(f"{name}*.so"))
    # Newest first, not path order. The glob reaches one level under every
    # sys.path entry, which also finds a CMake `build/` tree in the repo root --
    # and under `python -m` the repo root joins sys.path as an absolute entry,
    # so path order put a months-old build artifact ahead of the installed
    # extension and silently lost every method added since it was compiled.
    # (`python -c` never hit this: there cwd rides in sys.path as '', which the
    # falsy check skips.) Testing for a sibling __init__.py does not separate
    # the two either -- an editable install leaves the .so alone in its package
    # directory with no __init__.py of its own.
    found = [so for entry in sys.path if entry
             for so in pathlib.Path(entry).glob(f"*/{name}*.so")]
    yield from sorted(found, key=lambda p: p.stat().st_mtime, reverse=True)


@lru_cache(maxsize=None)
def _load_ext(name):
    for path in _candidates(name):
        if not path.exists():
            continue
        spec = importlib.util.spec_from_file_location(name, str(path))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    raise FileNotFoundError(
        f"no {name}*.so found. Build it in place, or point FT_SERVO_EXT at the "
        f"compiled extension. Searched {pathlib.Path(__file__).parent} and "
        f"package directories on sys.path.")


FtServo = _load_ext("ft_servo_ext").FtServo

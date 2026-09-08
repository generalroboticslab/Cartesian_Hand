"""`syringe`, a `pump` variant: draws and dispenses a syringe plunger.

Its own file only so it keeps a GUI button and stays reachable as
`--task syringe` -- see `tasks/__init__.py`'s note on writing a variant.
The implementation, `SyringeConfig`/`build_syringe`, lives in `pump.py` now,
beside the pump dispenser task it shares its jaw geometry and push/rise
mechanism with.
"""
from . import pump

Config = pump.SyringeConfig
build = pump.build_syringe

"""Single entry point: `python -m cartesian_hand <command>`.

This is `__main__.py` rather than a `cli.py` plus a stub, because the stub only
existed to forward to it.

Built with tyro, so the CLI is generated from the same dataclasses the code
already uses. A task declares a `Config` dataclass and its fields become flags,
with types and help text taken from the annotations and docstrings. There is no
parser to write and no place for the flags and the function signature to drift
apart.
"""

import dataclasses
import json
import sys
from dataclasses import dataclass, field
from typing import Annotated, Optional, Union

import tyro

from . import tasks
from .hand import connect
from .hands import DEFAULT_HAND, get_hand, hand_names


class Flat:
    """Splice a nested dataclass's fields in without a prefix, so the flag reads
    `--mock` rather than `--opts.mock`."""

    def __class_getitem__(cls, inner):
        return Annotated[inner, tyro.conf.arg(name="")]


@dataclass
class HandOptions:
    """Which hand to talk to, and how."""

    hand: Optional[str] = None
    """Name of the hand to drive. Defaults to hands.DEFAULT_HAND."""
    port: Optional[str] = None
    """Override the hand's serial port for this run."""
    mock: bool = False
    """Use the simulated backend. No hardware required."""

    def connect(self, **kwargs):
        return connect(hand=self.hand, mock=self.mock, port=self.port, **kwargs)

    def open_bus(self):
        """Raw driver with no controller, for maintenance on an unconfigured hand."""
        from .driver import open_driver
        port = self.port or get_hand(self.hand).port
        print(f"port: {port}")
        return open_driver(port, mock=self.mock)


# ── Non-task commands ─────────────────────────────────────────────────────────

@dataclass
class ListCmd:
    """Show the available hands and tasks."""

    def execute(self):
        print("Hands:")
        for name in hand_names():
            print(f"  {name}{'  (default)' if name == DEFAULT_HAND else ''}")
        print("\nTasks:")
        for name in tasks.names():
            print(f"  {name:<22}{tasks.describe(name)}")
        print("\nRun a task:   python -m cartesian_hand <task> [--mock]")
        print("Run a policy: python -m cartesian_hand policy <rollout.npz|mod:attr>")


@dataclass
class ContractCmd:
    """Export the sim/real policy contract for a digital twin."""

    hand: Optional[str] = None
    """Which hand's contract to export."""
    output: Optional[str] = None
    """Write JSON here instead of stdout."""
    full: bool = False
    """Export the whole config, not only the policy contract."""

    def execute(self):
        config = get_hand(self.hand)
        payload = config.to_dict() if self.full else {
            **config.contract(), "fingerprint": config.fingerprint()}
        text = json.dumps(payload, indent=2)
        if self.output:
            with open(self.output, "w") as f:
                f.write(text + "\n")
            print(f"wrote {self.output}")
        else:
            print(text)


@dataclass
class PublishCmd:
    """Stream DOF state to the terminal."""

    opts: Flat[HandOptions] = field(default_factory=HandOptions)
    hz: float = 1.0
    """Refresh rate."""

    def execute(self):
        with self.opts.connect() as hand:
            hand.publish(hz=self.hz)


@dataclass
class PolicyCmd:
    """Run a recorded rollout or a scripted policy against the hand."""

    source: tyro.conf.Positional[str]
    """Path to a .npz rollout, or module:attr naming a Policy."""
    opts: Flat[HandOptions] = field(default_factory=HandOptions)
    steps: Optional[int] = None
    """Stop after this many steps."""
    duration: Optional[float] = None
    """Stop after this many seconds."""
    hz: Optional[float] = None
    """Control rate. Defaults to the hand's configured rate."""
    max_delta: float = 0.05
    """Slew limit per step in normalized units. 0 disables it."""
    loop: bool = False
    """Replay a rollout on repeat."""
    strict: bool = True
    """Abort on a policy/hand geometry mismatch instead of warning."""
    record: Optional[str] = None
    """Save the executed rollout to this path."""

    def execute(self):
        from .policy import run_policy
        with self.opts.connect() as hand:
            policy = load_policy(self.source, self.loop, hand.config)
            rollout = run_policy(
                hand, policy, steps=self.steps, duration=self.duration,
                hz=self.hz, strict=self.strict,
                max_delta=None if self.max_delta == 0 else self.max_delta)
        print(f"ran {len(rollout)} steps")
        if self.record:
            print(f"saved {rollout.save(self.record)}")


@dataclass
class ScanCmd:
    """List the servos responding on the bus."""

    opts: Flat[HandOptions] = field(default_factory=HandOptions)
    start_id: int = 0
    end_id: int = 20

    def execute(self):
        bus = self.opts.open_bus()
        try:
            found = bus.scan(self.start_id, self.end_id)
            print(f"scanning IDs {self.start_id}-{self.end_id}: found {found}")
            for sid in found:
                volt = bus.get_voltage(sid)
                # read_position, not get_position: the latter serves the async
                # poll cache, which reads 0 until start_poll runs. The mock
                # driver aliases the two, so this only ever showed up on real
                # hardware, as every servo reporting position 0.
                print(f"  ID {sid:>3}: pos={bus.read_position(sid)} "
                      f"volt={'?' if volt is None else f'{volt / 10:.1f}V'} "
                      f"temp={bus.get_temperature(sid)}C")
        finally:
            bus.close()


@dataclass
class SetIdCmd:
    """Reassign the ID of the only servo on the bus."""

    new_id: tyro.conf.Positional[int]
    """The ID to assign."""
    opts: Flat[HandOptions] = field(default_factory=HandOptions)

    def execute(self):
        bus = self.opts.open_bus()
        try:
            # Refuse unless exactly one servo is present: a rename with several
            # connected would collide IDs across the bus.
            found = bus.scan(0, 253)
            if len(found) != 1:
                raise RuntimeError(
                    f"expected exactly 1 servo on the bus, found {found}. "
                    f"Disconnect all but the one you are renaming.")
            old_id = found[0]
            if old_id == self.new_id:
                print(f"servo is already ID {self.new_id}")
                return
            bus.write_id(old_id, self.new_id)
            print(f"ID changed: {old_id} -> {self.new_id}")
            pong = bus.ping(self.new_id)
            if pong is None or pong < 0:
                print(f"warning: no response at new ID {self.new_id}, verify manually")
            else:
                print(f"verified at ID {self.new_id}")
        finally:
            bus.close()


# ── Policy loading ────────────────────────────────────────────────────────────

def load_policy(source: str, loop: bool, config):
    """Build a policy from a recorded rollout or a `module:attr` reference.

    A `module:attr` may name a ready-made Policy, a zero-argument factory, or a
    factory taking the HandConfig. The arity is inspected rather than guessed by
    catching TypeError, which would swallow a genuine TypeError raised inside
    the constructor and retry with the wrong signature.
    """
    from .policy import Policy, ReplayPolicy

    if source.endswith(".npz"):
        return ReplayPolicy(source, loop=loop)
    if ":" not in source:
        raise SystemExit(
            f"cannot load policy from {source!r}: expected a .npz rollout or "
            f"module:attr (for example my_policies:trained)")

    import importlib
    import inspect
    module_name, _, attr = source.partition(":")
    try:
        obj = getattr(importlib.import_module(module_name), attr)
    except (ImportError, AttributeError) as e:
        raise SystemExit(f"cannot load {source!r}: {e}")

    if isinstance(obj, Policy):
        return obj
    if not callable(obj):
        raise SystemExit(f"{source!r} is neither a Policy nor callable")
    takes_config = len(inspect.signature(obj).parameters) >= 1
    return obj(config) if takes_config else obj()


# ── Task commands ─────────────────────────────────────────────────────────────

def _task_command(name: str, module):
    """Build a subcommand dataclass for a task from its Config."""

    def execute(self):
        with self.opts.connect() as hand:
            module.run(hand, self.cfg)

    return dataclasses.make_dataclass(
        f"{name.title().replace('_', '')}Cmd",
        [("opts", Flat[HandOptions], field(default_factory=HandOptions)),
         ("cfg", Flat[module.Config], field(default_factory=module.Config))],
        namespace={"execute": execute, "__doc__": tasks.describe(name)},
    )


def build_cli():
    """Union of every subcommand. tyro turns it into the parser."""
    variants = [
        _annotate(ListCmd, "list"),
        _annotate(ContractCmd, "contract"),
        _annotate(PublishCmd, "publish"),
        _annotate(PolicyCmd, "policy"),
        _annotate(ScanCmd, "scan"),
        _annotate(SetIdCmd, "set-id"),
    ]
    variants += [_annotate(_task_command(name, module), name)
                 for name, module in sorted(tasks.registry().items())]
    return Union[tuple(variants)]


def _annotate(cls, name: str):
    from typing import Annotated
    return Annotated[cls, tyro.conf.subcommand(name=name, description=cls.__doc__ or "")]


def main(argv=None):
    command = tyro.cli(build_cli(), args=argv, prog="cartesian_hand")
    try:
        command.execute()
    except KeyboardInterrupt:
        print("\ninterrupted.")
        return 130
    except (RuntimeError, ValueError, KeyError, NotImplementedError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

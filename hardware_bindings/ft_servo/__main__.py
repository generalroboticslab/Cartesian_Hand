"""Bench tools for a raw servo bus: scan it, rename a servo, drive it by hand.

    python -m hardware_bindings.ft_servo        <port>              # the GUI
    python -m hardware_bindings.ft_servo scan   <port> [--start 0] [--end 20]
    python -m hardware_bindings.ft_servo set-id <port> <new-id> [--end 20]
    python -m hardware_bindings.ft_servo gui    <port> [--ids 7 8 9]

The GUI is the default because it does what the other two do -- it scans on
open and can write an ID -- without needing to know either answer first. The
subcommands stay for scripts and for renaming without a browser.

One file rather than three scripts. All three open a port, ask what answers, and
report health, and the three copies had drifted into three different ways of
showing a dropped read -- one printed `-1C`, one printed `None`, one hid every
servo after the one that glitched. There is one formatter here now.

Bus level, not hand level: this takes a device path and knows nothing about DOFs
or millimetres. `python -m cartesian_hand scan` is the hand-level equivalent and
resolves the port from a hand name.

Built with tyro, like `cartesian_hand.__main__`, so each subcommand is a
dataclass whose fields are its flags and whose field docstrings are its help.
Everything runs on the compiled driver: the pings, the ID write and the register
reads are one C++ call each, and nothing here reimplements the protocol.
"""

import signal
import sys
import time
from dataclasses import dataclass
from typing import Annotated, List, Optional, Union

import tyro

from . import FtServo

NUDGE_COUNTS = 300      # ~2.5mm on a 16mm rack. Visible, and short enough to be safe.
NUDGE_SPEED = 300
NUDGE_TORQUE = 400      # enough to lift a loaded axis; this may be mounted

MIN_VOLT_TENTHS = 90    # below 9.0V an EPROM write risks browning out mid-write
MAX_TEMP_C = 65

# Highest ID a scan tries. A scan is a ping per ID, so this is a cost, not a
# limit of the protocol; 20 covers every bus here (the Cartesian hand uses 7-13)
# without spending a second on IDs nobody assigns.
SCAN_END = 20

# Starting torque for the GUI, 0-1000, written to HLSCL_GOAL_TORQUE with every
# goal. This is a bench tool on an unknown bus, so it cannot read a hand's
# per-DOF torques and has to pick one number for all of them; 50 is what six of
# the Cartesian hand's seven DOFs run at. The seventh, the z stage, needs 300 to
# lift against gravity and will sag at 50 -- raise its own slider in the GUI.
#
# Starting low rather than high because this is the force cap: the tool does not
# know what a servo drives, and a wrong guess at 50 stalls where a wrong guess at
# 300 pushes something over.
DEFAULT_TORQUE = 50

# How long the GUI tolerates a bus where nothing answers before it releases and
# exits. Long enough that a burst of interference does not close the window,
# short enough that pulling the adapter does not leave a panel showing numbers
# from a hand that is no longer connected.
BUS_LOST_AFTER_S = 2.0


def _fmt(value, unit="", scale=1.0, signed=False):
    """Render a register read, or '?' if it did not arrive.

    A dropped read comes back None from the driver, and the SDK reports -1 for
    some failed register reads. Neither is a measurement, so both show as
    unknown rather than printing -1C or dividing None by 10.

    signed=True drops the negative test. Position and load are sign-magnitude
    and go legitimately negative -- a hand parked below its zero offset reads
    -1450 counts, which is data, not a failure. Voltage and temperature cannot
    be negative, so there -1 is only ever the error value.
    """
    if value is None or (value < 0 and not signed):
        return "?"
    return f"{value * scale:.1f}{unit}" if scale != 1.0 else f"{value}{unit}"


def status(drv, sid, label="") -> bool:
    """Print voltage, temperature and load. False if the servo looks unwell.

    Worth checking before an EPROM write: a servo browning out mid-write is how
    you get one that answers to no ID at all.
    """
    volt = drv.get_voltage(sid)
    temp = drv.get_temperature(sid)
    print(f"  {label}ID {sid}: volt={_fmt(volt, 'V', 0.1)} "
          f"temp={_fmt(temp, 'C')} load={_fmt(drv.read_load(sid), signed=True)}")

    if volt is not None and 0 <= volt < MIN_VOLT_TENTHS:
        print(f"  WARNING: {volt / 10:.1f}V is low; an EPROM write may brown out")
        return False
    if temp is not None and temp > MAX_TEMP_C:
        print(f"  WARNING: {temp}C is hot")
        return False
    return True


def rename_refusal(drv, ids, old, new):
    """Why this EPROM write must not happen, as markdown, or None to go ahead.

    Separate from the button that calls it so the branches are testable without
    a browser: every one of them is the last thing standing between a click and
    a permanent write, and the failure mode is a servo that answers to no ID.
    """
    if new == old:
        return f"*already ID {old}*"
    if new in ids:
        return f"**ID {new} is already on the bus**"
    volt = drv.get_voltage(old)
    if volt is None or volt < MIN_VOLT_TENTHS:
        return f"**too low to write EPROM**: volt={_fmt(volt, 'V', 0.1)}"
    temp = drv.get_temperature(old)
    if temp is None or temp > MAX_TEMP_C:
        return f"**too hot to write EPROM**: temp={_fmt(temp, 'C')}"
    return None


def nudge(drv, sid) -> bool:
    """Drive the servo out and back. True if it actually moved.

    A ping only proves something answers to the new ID. Motion proves the servo
    that answers is the one on the bench.
    """
    start = drv.read_position(sid)
    if start is None:
        print(f"  cannot read position of ID {sid}, skipping motion check")
        return False

    print(f"  position {start}, nudging {NUDGE_COUNTS} counts")
    drv.enable_torque(sid, True)
    try:
        drv.set_position(sid, start + NUDGE_COUNTS, NUDGE_SPEED, 50, NUDGE_TORQUE)
        time.sleep(1.0)
        moved = drv.read_position(sid)
        print(f"  moved to {moved}")

        drv.set_position(sid, start, NUDGE_SPEED, 50, NUDGE_TORQUE)
        time.sleep(1.0)
        print(f"  returned to {drv.read_position(sid)}")
    finally:
        # Leave the servo free rather than holding position after a bench test.
        drv.enable_torque(sid, False)

    if moved is None:
        return False
    # Half the commanded distance distinguishes motion from a servo that never
    # left its start, without demanding it fully arrive under load.
    return abs(moved - start) > NUDGE_COUNTS / 2


# ── Subcommands ───────────────────────────────────────────────────────────────

@dataclass
class ScanCmd:
    """List the servos answering on the bus."""

    port: tyro.conf.Positional[str]
    """Serial device, e.g. /dev/ttyACM0."""
    start: int = 0
    """Lowest ID to try."""
    end: int = SCAN_END
    """Highest ID to try."""

    def execute(self, drv):
        found = drv.scan(self.start, self.end)
        print(f"scanning IDs {self.start}-{self.end}: found {found}")
        for sid in found:
            print(f"  ID {sid:>3}: pos={_fmt(drv.read_position(sid), signed=True)} "
                  f"volt={_fmt(drv.get_voltage(sid), 'V', 0.1)} "
                  f"temp={_fmt(drv.get_temperature(sid), 'C')}")


@dataclass
class SetIdCmd:
    """Rename the only servo on the bus, then prove the rename took.

    Finds the servo rather than being told its old ID, so it works on one whose
    ID nobody recorded. Refuses with more than one connected: renaming is
    ambiguous then, and the new ID could collide with one already in use.
    """

    port: tyro.conf.Positional[str]
    """Serial device, e.g. /dev/ttyACM0."""
    new_id: tyro.conf.Positional[int]
    """The ID to assign."""
    end: int = SCAN_END
    """Highest ID to look for when finding the current one."""

    def execute(self, drv):
        found = drv.scan(0, self.end)
        print(f"scanned IDs 0-{self.end}: found {found}")
        if not found:
            sys.exit(f"no servo responded on {self.port}")
        if len(found) > 1:
            sys.exit(f"expected 1 servo, found {found}. "
                     f"Disconnect all but the one you are renaming.")

        old_id = found[0]
        if not status(drv, old_id, "before: "):
            sys.exit("servo health check failed; not writing to EPROM")

        if old_id == self.new_id:
            print(f"already ID {self.new_id}, nothing to write")
        else:
            drv.write_id(old_id, self.new_id)
            print(f"ID changed: {old_id} -> {self.new_id}")
            if drv.ping(self.new_id) < 0:
                sys.exit(f"no response at new ID {self.new_id}; "
                         f"the write may have failed")
            if drv.ping(old_id) >= 0:
                sys.exit(f"ID {old_id} still answers after the rename; "
                         f"there may be two servos on the bus")

        print(f"verifying ID {self.new_id} by moving it")
        status(drv, self.new_id, "after:  ")
        if not nudge(drv, self.new_id):
            sys.exit(f"servo answers at ID {self.new_id} but did not move. "
                     f"Check power and that the output is free to turn.")
        print(f"OK: servo is ID {self.new_id} and responds to motion")


@dataclass
class TorqueLimitCmd:
    """Read, and optionally set, TORQUE_LIMIT (reg 48/49) on the bus.

    The register that bounds output in position mode. `set_positions`' `torque`
    argument does NOT reach it -- that writes GOAL_TORQUE (44/45), the
    constant-force-mode setpoint -- so a hand can press at full output no matter
    what torque the caller passes. Reading it is how you tell which of those two
    stories a unit is living in.

    Writes are SRAM and revert on power cycle, so this is a probe, not a fix:
    set a limit, run the move, decide, and let it lapse.
    """

    port: tyro.conf.Positional[str]
    """Serial device, e.g. /dev/ttyACM0."""
    ids: tyro.conf.Positional[List[int]]
    """Servo IDs to read or write."""
    limit: Optional[int] = None
    """0-1000. Omit to only read what is already there."""

    def execute(self, drv):
        if self.limit is not None:
            if not 0 <= self.limit <= 1000:
                sys.exit(f"limit must be 0-1000, got {self.limit}")
            drv.set_torque_limits(self.ids, [self.limit] * len(self.ids))
        for sid in self.ids:
            got = drv.get_torque_limit(sid)
            # A servo that does not answer reads -1, and a write that the
            # firmware ignored reads back as whatever it was. Both are the
            # answer to the question, so print rather than exit.
            print(f"  ID {sid:>3}: torque_limit={_fmt(got)}"
                  + ("" if self.limit is None or got == self.limit
                     else f"  (wrote {self.limit}, did not take)"))


@dataclass
class GuiCmd:
    """Live position and load readout, target slider per servo, ID writes."""

    port: tyro.conf.Positional[str]
    """Serial device, e.g. /dev/ttyACM0."""
    ids: Optional[List[int]] = None
    """Servo IDs to show. Omit to take whatever the bus reports."""
    web_port: int = 8080
    """Port to serve the GUI on. Taken ports fall through to the next free one."""
    span: int = 4000
    """Counts a slider may travel either side of the servo's starting position.
    4000 is ~49mm on a 16mm rack. Adjustable in the GUI."""
    torque: int = DEFAULT_TORQUE
    """Starting force cap, 0-1000. Per-servo slider in the GUI."""
    speed: int = 300
    """Speed for slider moves. Adjustable in the GUI."""

    def execute(self, drv):
        import viser

        server = viser.ViserServer(port=self.web_port)

        # Replaced wholesale by _build on every (re)scan. A dict rather than
        # locals because the signal handler and the ID-write callback both run
        # on other threads and must see the *current* panel, not the one that
        # existed when they were registered.
        state = {"ids": [], "handles": {}, "rebuild": False}

        # An explicit handler, not `except KeyboardInterrupt`. viser installs
        # signal handling of its own, which swallows SIGINT before the main
        # thread ever raises -- verified against this hand: the process kept
        # running and kept the servos energized. Registering last wins, and
        # covering SIGTERM means a plain `kill` also releases rather than
        # leaving a live hand behind.
        #
        # Torque comes off first, before the poll thread is joined and before
        # viser is stopped, so a hang in either still leaves the servos
        # released. The port itself is closed by main()'s finally, on the way
        # out through SystemExit. flush=True because stdout is block-buffered
        # when this runs in the background, and these are the lines that say
        # whether the hand is safe.
        def release_and_exit(why, code=0):
            print(f"\n{why}: releasing torque", flush=True)
            try:
                drv.enable_torques(state["ids"], False)
            except Exception as e:
                # The bus-lost path arrives here with a port that is gone, so
                # the release is the call most likely to throw -- and throwing
                # here would strand the process holding a live hand, which is
                # the exact failure this exit is trying to avoid.
                print(f"could not release torque: {e}. Cut power if the hand "
                      f"is loaded.", flush=True)
            drv.stop_poll()
            server.stop()
            print("stopped", flush=True)
            sys.exit(code)

        def shutdown(signum, frame):
            release_and_exit(f"signal {signum}")

        signal.signal(signal.SIGINT, shutdown)
        signal.signal(signal.SIGTERM, shutdown)

        # get_port(), not web_port: viser falls back to the next free port when
        # the requested one is taken, so the number asked for is not the number
        # bound. Printing the request sent you to somebody else's server.
        print(f"GUI ready at http://localhost:{server.get_port()}, Ctrl+C to stop")

        ids = self.ids
        while True:
            self._build(drv, server, state, ids)
            silent_since = None
            while not state["rebuild"]:
                # get_positions serves the last good numbers whether the hand is
                # holding still or the adapter has been pulled, so the panel
                # cannot tell the two apart and this loop would spin forever on
                # a dead bus. poll_silence can tell them apart. Timed rather
                # than counted: a silent sweep costs one serial timeout per
                # servo, so sweeps per second depend on how many are configured.
                if drv.poll_silence() == 0:
                    silent_since = None
                elif silent_since is None:
                    silent_since = time.time()
                elif time.time() - silent_since > BUS_LOST_AFTER_S:
                    release_and_exit(
                        f"no servo has answered in {BUS_LOST_AFTER_S:g}s; the "
                        f"bus is gone. Check power and the serial cable", code=1)

                pos = drv.get_positions(state["ids"])
                load = drv.get_loads(state["ids"])
                for i, sid in enumerate(state["ids"]):
                    state["handles"][sid]["pos"].value = pos[i]
                    state["handles"][sid]["load"].value = load[i]
                time.sleep(0.1)

            # An ID write invalidates every label, slider name and poll target
            # on the panel at once, so the panel is thrown away and rebuilt from
            # a fresh scan rather than patched in place.
            drv.enable_torques(state["ids"], False)
            drv.stop_poll()
            server.gui.reset()
            ids = None  # any explicit --ids list is stale once an ID moved

    def _build(self, drv, server, state, ids):
        """Find the servos, energize them, lay out the panel. Mutates `state`.

        Split from execute() so a servo rename can rebuild: the alternative was
        renaming handles in place, which viser cannot do for a slider label.
        """
        # Cleared first, not on the way out: a click that lands mid-build would
        # otherwise be overwritten by the final assignment and silently lost.
        state["rebuild"] = False

        # A scan is the same pings, so listing IDs by hand buys nothing but the
        # chance to name one that is not there. Give them anyway to drive a
        # subset of a bus -- a bench where you want the jaws and not the z stage.
        if ids is None:
            ids = drv.scan(0, SCAN_END)
            print(f"scanned IDs 0-{SCAN_END}: found {ids}")
            if not ids:
                sys.exit(f"no servo responded on {self.port}")
        else:
            for sid in ids:
                print(f"ping({sid}) = {drv.ping(sid)}")

        # Read, then command present position, then energize. Enabling torque
        # first would make each servo snap to whatever goal was left in its
        # register by the previous session -- after a zeroing run that goal is a
        # hard stop, so the hand would slam shut on connect.
        here = drv.read_positions(ids)
        if any(c is None for c in here):
            sys.exit(f"could not read every servo: {dict(zip(ids, here))}. "
                     f"Refusing to energize without knowing where the hand is.")
        initial = dict(zip(ids, here))
        drv.set_positions(ids, here, speed=0, acc=50, torque=self.torque)
        drv.enable_torques(ids, True)
        drv.start_poll(ids, interval_us=5000)
        time.sleep(0.2)

        # The two bus-wide guards, adjustable here as well as on the command
        # line. Both start at their flag's value; the flags only pick where the
        # GUI opens. Added before the per-servo groups so they render above
        # them -- viser lays out in call order.
        span_h = server.gui.add_number("span", initial_value=self.span,
                                       min=1, max=32767, step=100)
        speed_h = server.gui.add_number("speed", initial_value=self.speed,
                                        min=0, max=10000, step=10)
        self._add_rename(drv, server, state, ids)

        handles = {}
        for sid in ids:
            # Sliders span `span` counts either side of where the servo already
            # is, not the encoder's full +-32767. This is a bus tool: it cannot
            # know what the servo drives or how far that thing may travel, and
            # on the Cartesian hand only one end of each rail has a stop -- a
            # full-range slider is a one-drag route to pushing a carriage off
            # its rail.
            lo, hi = initial[sid] - self.span, initial[sid] + self.span
            server.gui.add_markdown(f"**Servo ID {sid}**")
            handles[sid] = {
                "pos":  server.gui.add_slider(f"{sid} pos", min=lo, max=hi, step=1,
                                              initial_value=initial[sid],
                                              disabled=True),
                # Load is sign-magnitude like position: it reports direction, so
                # the range has to straddle zero or half the readings clamp.
                "load": server.gui.add_slider(f"{sid} load", min=-4096, max=4096,
                                              step=1, initial_value=0,
                                              disabled=True),
                "target": server.gui.add_slider(f"{sid} target", min=lo, max=hi,
                                                step=1, initial_value=initial[sid]),
                # Per-servo, because torque is per-servo in the protocol and
                # because one number cannot fit a bus: on the Cartesian hand the
                # z stage needs 300 to lift while the other six work at 50.
                # Live, so finding a servo's number is a drag rather than a
                # restart.
                "torque": server.gui.add_slider(f"{sid} torque", min=0, max=1000,
                                                step=10, initial_value=self.torque),
            }
            # Torque only reaches the servo attached to a goal, so raising the
            # slider does nothing until the next write -- re-issue the current
            # target on both, or a servo stalled at 50 stays stalled while its
            # torque slider reads 300.
            for name in ("target", "torque"):
                handles[sid][name].on_update(
                    lambda _, sid=sid: drv.set_position(
                        sid, handles[sid]["target"].value, speed=speed_h.value,
                        acc=50, torque=handles[sid]["torque"].value))

        # Registered after the loop because it walks every handle. Widening a
        # span moves only the bounds; narrowing one can leave a target outside
        # the window it is meant to enforce, so the value is clamped back in.
        # Bounds still anchor on the position each servo held at startup, not on
        # where it is now -- the window is a guard around a known-good place, and
        # re-anchoring it on the current position would let a servo walk the
        # window along with it one drag at a time.
        @span_h.on_update
        def _(_) -> None:
            for sid in ids:
                lo, hi = initial[sid] - span_h.value, initial[sid] + span_h.value
                for name in ("pos", "target"):
                    h = handles[sid][name]
                    h.min, h.max = lo, hi
                    h.value = min(max(h.value, lo), hi)

        state["ids"], state["handles"] = ids, handles

    def _add_rename(self, drv, server, state, ids):
        """EPROM ID write, folded away because it is rare and permanent.

        The `set-id` subcommand refuses to run with more than one servo on the
        bus: it scans to discover which servo to rename, so two candidates make
        the choice ambiguous. Here the servo is named explicitly from a list of
        what already answered, so that restriction does not apply and a rename
        needs no unplugging. The remaining hazard is a new ID colliding with one
        already on the bus, which is checked before the write.
        """
        with server.gui.add_folder("change a servo ID", expand_by_default=False):
            src = server.gui.add_dropdown("from", [str(i) for i in ids])
            dst = server.gui.add_number("to", initial_value=max(ids) + 1,
                                        min=0, max=253, step=1)
            note = server.gui.add_markdown("*writes EPROM; survives power cycles*")
            btn = server.gui.add_button("write ID")

        @btn.on_click
        def _(_) -> None:
            old, new = int(src.value), int(dst.value)
            refusal = rename_refusal(drv, ids, old, new)
            if refusal:
                note.content = refusal
                return

            # Release the whole bus before the write. An EPROM write to a servo
            # drawing hold current is the case that browns out mid-write, and a
            # servo that browns out mid-write answers to no ID at all afterwards.
            drv.enable_torques(state["ids"], False)
            drv.write_id(old, new)
            ok = drv.ping(new) >= 0
            # Rebuild either way: torque is off and the panel no longer matches
            # the bus, so a rescan is the only honest thing to show next.
            print(f"ID {old} -> {new}: {'OK' if ok else 'NO RESPONSE at new ID'}",
                  flush=True)
            state["rebuild"] = True


def _annotate(cls, name: str):
    # First line only. tyro prints `description` in the subcommand index, where
    # a multi-paragraph docstring wraps into an unreadable wall; the full text
    # still reaches `<subcommand> --help` from the class docstring itself.
    summary = (cls.__doc__ or "").strip().splitlines()[0]
    return Annotated[cls, tyro.conf.subcommand(name=name, description=summary)]


SUBCOMMANDS = ("scan", "set-id", "torque-limit", "gui")


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    # `python -m hardware_bindings.ft_servo <port>` opens the GUI. It is the one
    # subcommand you reach for before knowing what is on the bus, and it scans
    # and renames on its own now, so requiring the word "gui" only made the
    # quickest path the one you had to read the help for. A leading `-` still
    # goes to tyro so `--help` prints the subcommand index.
    if argv and argv[0] not in SUBCOMMANDS and not argv[0].startswith("-"):
        argv.insert(0, "gui")

    command = tyro.cli(
        Union[(_annotate(ScanCmd, "scan"),
               _annotate(SetIdCmd, "set-id"),
               _annotate(TorqueLimitCmd, "torque-limit"),
               _annotate(GuiCmd, "gui"))],
        args=argv, prog="python -m hardware_bindings.ft_servo")
    # The other half of "the hardware went away": unplugged before the tool
    # starts rather than during. A missing device is the ordinary case on a
    # bench, not a defect worth a traceback.
    try:
        drv = FtServo(command.port)
    except RuntimeError as e:
        sys.exit(f"{e}. Check the adapter is plugged in and that "
                 f"{command.port} exists.")

    try:
        command.execute(drv)
    finally:
        drv.close()


if __name__ == "__main__":
    main()

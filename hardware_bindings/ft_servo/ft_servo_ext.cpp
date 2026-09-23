#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/tuple.h>
#include "ft_servo_driver.hpp"

namespace nb = nanobind;

// set_positions is overloaded on its gain arguments, so the member pointers
// need spelling out for nanobind to pick the right one.
using SetPosScalarGains = bool (FtServoDriver::*)(
    const std::vector<int>&, const std::vector<int>&, int, int, int);
using SetPosVectorGains = bool (FtServoDriver::*)(
    const std::vector<int>&, const std::vector<int>&,
    const std::vector<int>&, const std::vector<int>&, const std::vector<int>&);

// Every .def carries a docstring so `help(FtServo)` is the API reference. The
// surface is otherwise only discoverable by reading this file: the class is a
// compiled extension, so nothing in Python names its methods, and the two
// traps here -- get_* returning stale zeros without a poll, read_* returning
// None on a dropped reply -- are invisible from a signature.
NB_MODULE(ft_servo_ext, m) {
    m.doc() = "Feetech HLS-series serial bus servo driver.";

    nb::class_<FtServoDriver>(m, "FtServo",
        "One open serial port and the servos on it.\n\n"
        "Reads come in three flavours and they are not interchangeable:\n"
        "  read_position(id)   one round trip, for a single servo\n"
        "  read_positions(ids) one packet for the whole bus\n"
        "  get_positions(ids)  free, but ONLY while start_poll runs\n\n"
        "read_* return None when a reply does not arrive. get_* read a cache the\n"
        "poll thread fills and return zeros when it is not running, which looks\n"
        "exactly like servos parked at origin.\n\n"
        "Counts are sign-magnitude and go legitimately negative. Torque is\n"
        "0-1000 and is a force cap written with every goal.")

        .def(nb::init<const std::string&, int>(),
             nb::arg("port"), nb::arg("baud") = 1000000,
             "Open a serial port. Raises if it cannot be opened.")

        // ---- Motion. GIL released: these block on a serial write. -----------
        .def("set_position", &FtServoDriver::set_position,
             nb::arg("id"), nb::arg("pos"),
             nb::arg("speed") = 0, nb::arg("acc") = 50, nb::arg("torque") = 500,
             "Command one servo to a position. speed=0 means as fast as it can.\n"
             "torque is the force cap for this move, 0-1000, and only reaches the\n"
             "servo attached to a goal -- changing it alone does nothing.",
             nb::call_guard<nb::gil_scoped_release>())
        // Per-servo gains first: a list argument matches only this one, an int
        // argument only the scalar overload below, so the two never collide.
        .def("set_positions", static_cast<SetPosVectorGains>(&FtServoDriver::set_positions),
             nb::arg("ids"), nb::arg("pos"),
             nb::arg("speed"), nb::arg("acc"), nb::arg("torque"),
             "Command every servo, each with its own gains, in one packet.\n"
             "Costs the same as uniform gains: INST_SYNC_WRITE broadcasts once\n"
             "and each servo reads its own slice. All five lists must match.",
             nb::call_guard<nb::gil_scoped_release>())
        .def("set_positions", static_cast<SetPosScalarGains>(&FtServoDriver::set_positions),
             nb::arg("ids"), nb::arg("pos"),
             nb::arg("speed") = 0, nb::arg("acc") = 50, nb::arg("torque") = 500,
             "Command every servo in one packet, sharing one set of gains.",
             nb::call_guard<nb::gil_scoped_release>())
        .def("set_speed", &FtServoDriver::set_speed,
             "Constant-speed mode for one servo. Needs set_mode(id, 1) first.",
             nb::call_guard<nb::gil_scoped_release>())
        .def("set_speeds", &FtServoDriver::set_speeds,
             nb::arg("ids"), nb::arg("speeds"),
             nb::arg("acc") = 50, nb::arg("torque") = 500,
             "Constant-speed mode for several servos, one packet.",
             nb::call_guard<nb::gil_scoped_release>())
        .def("enable_torque", &FtServoDriver::enable_torque,
             "Energize or release one servo. Released servos backdrive freely.\n"
             "Enabling makes a servo snap to whatever goal is still in its\n"
             "register, so write the present position first.",
             nb::call_guard<nb::gil_scoped_release>())
        .def("enable_torques", &FtServoDriver::enable_torques,
             "enable_torque for several servos. One unicast write each, not a\n"
             "broadcast -- the register is not in the sync-write block.",
             nb::call_guard<nb::gil_scoped_release>())
        .def("set_torque_limit", &FtServoDriver::set_torque_limit,
             nb::arg("id"), nb::arg("limit"),
             "Cap one servo's output, 0-1000, at TORQUE_LIMIT (reg 48/49).\n"
             "NOT the `torque` argument of set_positions -- that one writes\n"
             "GOAL_TORQUE (44/45), the constant-force-mode setpoint, which does\n"
             "not bound a position-mode move. SRAM: reverts on power cycle.",
             nb::call_guard<nb::gil_scoped_release>())
        .def("set_torque_limits", &FtServoDriver::set_torque_limits,
             nb::arg("ids"), nb::arg("limits"),
             "set_torque_limit for several servos. One unicast write each.",
             nb::call_guard<nb::gil_scoped_release>())
        .def("get_torque_limit", &FtServoDriver::get_torque_limit,
             nb::arg("id"),
             "Read back TORQUE_LIMIT (reg 48/49). -1 on no reply. Live read,\n"
             "not cached -- this is how you tell a limit that took from one the\n"
             "firmware ignored.",
             nb::call_guard<nb::gil_scoped_release>())
        .def("set_mode", &FtServoDriver::set_mode,
             "0 = position, 1 = constant speed, 2 = constant torque.",
             nb::call_guard<nb::gil_scoped_release>())
        .def("set_modes", &FtServoDriver::set_modes,
             nb::arg("ids"), nb::arg("mode"),
             "set_mode for several servos.",
             nb::call_guard<nb::gil_scoped_release>())

        // ---- Cached reads. No serial traffic, no GIL release needed. --------
        .def("get_position", &FtServoDriver::get_position,
             "Cached position. ZERO unless start_poll is running.")
        .def("get_positions", &FtServoDriver::get_positions,
             "Cached positions. ZEROS unless start_poll is running -- which reads\n"
             "as a bus parked at origin. Use read_positions when not polling.")
        .def("get_speed", &FtServoDriver::get_speed,
             "Cached speed. Zero unless start_poll is running.")
        .def("get_speeds", &FtServoDriver::get_speeds,
             "Cached speeds. Zeros unless start_poll is running.")
        .def("get_load", &FtServoDriver::get_load,
             "Cached load. Zero unless start_poll is running.")
        .def("get_loads", &FtServoDriver::get_loads,
             "Cached loads. Zeros unless start_poll is running.")

        // ---- Direct reads. GIL released: these block on serial. -------------
        // All return None on a failed read rather than the SDK's error value
        // dressed up as data.
        .def("read_position", &FtServoDriver::read_position,
             "Position of one servo, or None if the reply did not arrive.\n"
             "Sign-magnitude, so negative values are real data.",
             nb::call_guard<nb::gil_scoped_release>())
        .def("read_positions", &FtServoDriver::read_positions,
             nb::arg("ids"),
             "Positions of every listed servo in one sync-read: one packet out,\n"
             "one reply each, against a full round trip per servo for a loop of\n"
             "read_position. None per servo that did not answer.\n"
             "Raises while start_poll runs -- they share the sync-read buffer.",
             nb::call_guard<nb::gil_scoped_release>())
        .def("read_all", &FtServoDriver::read_all,
             nb::arg("ids"),
             "(position, speed, load) per servo in one sync-read, None per servo\n"
             "that did not answer. Registers 56-61 are contiguous, so this is the\n"
             "same packet read_positions already sends -- load costs no extra bus\n"
             "traffic, only two more decodes.\n"
             "Raises while start_poll runs -- they share the sync-read buffer.",
             nb::call_guard<nb::gil_scoped_release>())
        .def("read_speed", &FtServoDriver::read_speed,
             "Speed of one servo, or None. Sign-magnitude.",
             nb::call_guard<nb::gil_scoped_release>())
        .def("read_load", &FtServoDriver::read_load,
             "Load on one servo, or None. Sign-magnitude: the sign is direction.",
             nb::call_guard<nb::gil_scoped_release>())
        .def("get_voltage", &FtServoDriver::get_voltage,
             "Supply voltage in tenths of a volt (115 = 11.5V), or None.",
             nb::call_guard<nb::gil_scoped_release>())
        .def("get_temperature", &FtServoDriver::get_temperature,
             "Case temperature in Celsius, or None.",
             nb::call_guard<nb::gil_scoped_release>())
        .def("get_current", &FtServoDriver::get_current,
             "Motor current in mA, or None. Sign-magnitude: the sign is "
             "direction. This is what the firmware's overload protection "
             "integrates, and unlike load it does not saturate at 1000.",
             nb::call_guard<nb::gil_scoped_release>())
        .def("read_byte", &FtServoDriver::read_byte,
             nb::arg("id"), nb::arg("addr"),
             "One raw register byte, or None. For the registers HLSCL.h does "
             "not map -- 13-25 hold this family's protection settings, which "
             "is where a low overload trip point is actually configured.",
             nb::call_guard<nb::gil_scoped_release>())
        .def("ping", &FtServoDriver::ping,
             "Servo ID if it answers, -1 on timeout.",
             nb::call_guard<nb::gil_scoped_release>())

        // ---- Poll thread ----------------------------------------------------
        .def("start_poll", &FtServoDriver::start_poll,
             nb::arg("ids"), nb::arg("interval_us") = 5000,
             "Refresh position, speed and load in a background thread so the\n"
             "get_* calls are free. Owns the bus while running: read_positions\n"
             "raises until stop_poll.")
        .def("stop_poll", &FtServoDriver::stop_poll,
             "Stop the poll thread and join it. The cache goes stale, not empty.")
        .def("poll_silence", &FtServoDriver::poll_silence,
             "Consecutive poll sweeps in which no servo replied at all, 0 while\n"
             "healthy. The only way to tell a hand holding still from a bus that\n"
             "went away: get_* serves the last good numbers either way. A sweep\n"
             "or two is EMI; climbing is an unplugged adapter or a dead supply.")
        .def("close", &FtServoDriver::close,
             "Close the port. Does NOT release torque -- call enable_torques\n"
             "first or the servos hold position after the process exits.")

        // ---- Bus and EPROM --------------------------------------------------
        .def("scan", &FtServoDriver::scan,
             nb::arg("start_id") = 0, nb::arg("end_id") = 253,
             "IDs that answer, as a list. One ping per ID, so keep the range\n"
             "tight -- the full 0-253 default takes a while.",
             nb::call_guard<nb::gil_scoped_release>())
        .def("write_id", &FtServoDriver::write_id,
             nb::arg("id"), nb::arg("new_id"),
             "Unlock EPROM, write a new ID, re-lock. Survives power cycles.\n"
             "With more than one servo on the bus the new ID may collide, and a\n"
             "brownout mid-write leaves a servo answering to no ID at all.",
             nb::call_guard<nb::gil_scoped_release>());
}

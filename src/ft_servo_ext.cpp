#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>
#include "ft_servo_driver.hpp"

namespace nb = nanobind;

NB_MODULE(ft_servo_ext, m) {
    nb::class_<FtServoDriver>(m, "FtServo")
        .def(nb::init<const std::string&, int>(),
             nb::arg("port"), nb::arg("baud") = 1000000)
        // Motion — release GIL: these block on serial write
        .def("set_position", &FtServoDriver::set_position,
             nb::arg("id"), nb::arg("pos"),
             nb::arg("speed") = 0, nb::arg("acc") = 50, nb::arg("torque") = 500,
             nb::call_guard<nb::gil_scoped_release>())
        .def("set_positions", &FtServoDriver::set_positions,
             nb::call_guard<nb::gil_scoped_release>())
        .def("set_speed", &FtServoDriver::set_speed,
             nb::call_guard<nb::gil_scoped_release>())
        .def("enable_torque", &FtServoDriver::enable_torque,
             nb::call_guard<nb::gil_scoped_release>())
        .def("enable_torques", &FtServoDriver::enable_torques,
             nb::call_guard<nb::gil_scoped_release>())
        .def("set_mode", &FtServoDriver::set_mode,
             nb::call_guard<nb::gil_scoped_release>())
        // Cached reads — no serial, fast
        .def("get_position",  &FtServoDriver::get_position)
        .def("get_speed",     &FtServoDriver::get_speed)
        .def("get_positions", &FtServoDriver::get_positions)
        .def("get_speeds",    &FtServoDriver::get_speeds)
        // Direct reads — release GIL: these block on serial
        .def("read_position",   &FtServoDriver::read_position,
             nb::call_guard<nb::gil_scoped_release>())
        .def("read_speed",      &FtServoDriver::read_speed,
             nb::call_guard<nb::gil_scoped_release>())
        .def("get_voltage",     &FtServoDriver::get_voltage,
             nb::call_guard<nb::gil_scoped_release>())
        .def("get_temperature", &FtServoDriver::get_temperature,
             nb::call_guard<nb::gil_scoped_release>())
        .def("ping",            &FtServoDriver::ping,
             nb::call_guard<nb::gil_scoped_release>())
        // Poll control
        .def("start_poll", &FtServoDriver::start_poll,
             nb::arg("ids"), nb::arg("interval_us") = 5000)
        .def("stop_poll", &FtServoDriver::stop_poll)
        .def("close",     &FtServoDriver::close)
        // Load
        .def("read_load",  &FtServoDriver::read_load,
             nb::call_guard<nb::gil_scoped_release>())
        .def("get_load",   &FtServoDriver::get_load)
        .def("get_loads",  &FtServoDriver::get_loads)
        // Multi-ID speed + mode
        .def("set_speeds", &FtServoDriver::set_speeds,
             nb::arg("ids"), nb::arg("speeds"),
             nb::arg("acc") = 50, nb::arg("torque") = 500,
             nb::call_guard<nb::gil_scoped_release>())
        .def("set_modes",  &FtServoDriver::set_modes,
             nb::arg("ids"), nb::arg("mode"),
             nb::call_guard<nb::gil_scoped_release>())
        // Bus scan
        .def("scan", &FtServoDriver::scan,
             nb::arg("start_id") = 0, nb::arg("end_id") = 253,
             nb::call_guard<nb::gil_scoped_release>())
        // ID programming
        .def("write_id", &FtServoDriver::write_id,
             nb::arg("id"), nb::arg("new_id"),
             nb::call_guard<nb::gil_scoped_release>());
}

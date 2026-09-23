#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/vector.h>
// #include <nanobind/stl/bind_vector.h>


#include <nanobind/eigen/dense.h> // for EIGEN integration
#include <vector>

#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>
#include <string>

#include "imu.hpp"

namespace nb = nanobind;
using namespace nb::literals;



NB_MODULE(imu_nanobind, m)
{
    auto imu = nb::class_<IMU>(m, "IMU", "This is the IMU class.")
        .def("__copy__", [](const IMU&) {
            throw std::runtime_error("Copy not allowed");
        })
        .def(nb::init<>())
        .def(nb::init<std::string>())
        .def("run", &IMU::run, nb::call_guard<nb::gil_scoped_release>())
        // .def("run", &IMU::run)
        .def("close", &IMU::close)
        .def("findPortNameByDescription", &IMU::findPortNameByDescription, nb::arg("portDescription") = "STMicroelectronics Virtual COM Port")
        .def_rw("should_print", &IMU::should_print)
        .def_ro("counter", &IMU::counter)
        // Eigen fields: def_prop_ro returns copies under mutex to prevent torn reads from callback thread
        .def_prop_ro("raw_acc", [](const IMU& s) { std::lock_guard<std::mutex> g(s.data_mutex_); return VectorNd<float,3>(s.raw_acc); })
        .def_prop_ro("ang_vel", [](const IMU& s) { std::lock_guard<std::mutex> g(s.data_mutex_); return VectorNd<float,3>(s.ang_vel); })
        .def_prop_ro("world_space_ang_vel", [](const IMU& s) { std::lock_guard<std::mutex> g(s.data_mutex_); return VectorNd<float,3>(s.world_space_ang_vel); })
        .def_prop_ro("raw_mag", [](const IMU& s) { std::lock_guard<std::mutex> g(s.data_mutex_); return VectorNd<float,3>(s.raw_mag); })
        .def_prop_ro("quat_xyzw", [](const IMU& s) { std::lock_guard<std::mutex> g(s.data_mutex_); return VectorNd<float,4>(s.quat_xyzw); })
        .def_prop_ro("quat_wxyz", [](const IMU& s) { std::lock_guard<std::mutex> g(s.data_mutex_); return VectorNd<float,4>(s.quat_wxyz); })
        .def_prop_ro("rotation_matrix", [](const IMU& s) { std::lock_guard<std::mutex> g(s.data_mutex_); return Eigen::Matrix3f(s.rotation_matrix); })
        .def_prop_ro("euler", [](const IMU& s) { std::lock_guard<std::mutex> g(s.data_mutex_); return VectorNd<float,3>(s.euler); })
        .def_prop_ro("direct_gravity_vec", [](const IMU& s) { std::lock_guard<std::mutex> g(s.data_mutex_); return VectorNd<float,3>(s.direct_gravity_vec); })
        .def_prop_ro("gravity_vec", [](const IMU& s) { std::lock_guard<std::mutex> g(s.data_mutex_); return VectorNd<float,3>(s.gravity_vec); })
        // Scalars: atomic-width on ARM64, no mutex needed
        .def_ro("timeStamp", &IMU::timeStamp)
        .def_ro("qos", &IMU::qos)
        .def_ro("temperature", &IMU::temperature)
        .def_ro("updateRate", &IMU::updateRate)
        // Rotation offset support
        .def_prop_rw("rotation_offset",
            [](const IMU& self) { return self.get_rotation_offset(); },
            [](IMU& self, const Eigen::Matrix3f& offset) { self.set_rotation_offset(offset); },
            "3x3 rotation matrix for sensor mounting compensation")
        // Transformed properties (with rotation offset applied, mutex-protected)
        .def_prop_ro("transformed_quat_xyzw", [](const IMU& s) { std::lock_guard<std::mutex> g(s.data_mutex_); return s.get_transformed_quat_xyzw(); })
        .def_prop_ro("transformed_quat_wxyz", [](const IMU& s) { std::lock_guard<std::mutex> g(s.data_mutex_); return s.get_transformed_quat_wxyz(); })
        .def_prop_ro("transformed_euler", [](const IMU& s) { std::lock_guard<std::mutex> g(s.data_mutex_); return s.get_transformed_euler(); })
        .def_prop_ro("transformed_rotation_matrix", [](const IMU& s) { std::lock_guard<std::mutex> g(s.data_mutex_); return s.get_transformed_rotation_matrix(); })
        .def_prop_ro("transformed_ang_vel", [](const IMU& s) { std::lock_guard<std::mutex> g(s.data_mutex_); return s.get_transformed_ang_vel(); })
        .def_prop_ro("transformed_world_space_ang_vel", [](const IMU& s) { std::lock_guard<std::mutex> g(s.data_mutex_); return s.get_transformed_world_space_ang_vel(); })
        .def_prop_ro("transformed_raw_acc", [](const IMU& s) { std::lock_guard<std::mutex> g(s.data_mutex_); return s.get_transformed_raw_acc(); })
        .def_prop_ro("transformed_gravity_vec", [](const IMU& s) { std::lock_guard<std::mutex> g(s.data_mutex_); return s.get_transformed_gravity_vec(); })
        .def_prop_ro("transformed_raw_mag", [](const IMU& s) { std::lock_guard<std::mutex> g(s.data_mutex_); return s.get_transformed_raw_mag(); })
        // Helper methods
        .def("has_rotation_offset", &IMU::has_rotation_offset,
            "Check if rotation offset is set")
        .def("clear_rotation_offset", &IMU::clear_rotation_offset,
            "Clear rotation offset (revert to identity)");
}
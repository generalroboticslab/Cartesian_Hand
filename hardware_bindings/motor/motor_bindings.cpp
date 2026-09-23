#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/vector.h>
// #include <nanobind/stl/bind_vector.h>


#include <nanobind/eigen/dense.h> // for EIGEN integration
#include <vector>

#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>
#include <string>

// #include "imu.hpp"
#include "motor.hpp"
#include "gl_motor.hpp"


// NB_MAKE_OPAQUE(std::vector<float>);
// NB_MAKE_OPAQUE(std::vector<unsigned int>);

namespace nb = nanobind;
using namespace nb::literals;

int add(int a, int b = 1) { return a + b; }

std::vector<float> get_floats_in_range(int start, int end)
{
    std::vector<float> floats;
    for (int i = start; i < end; i++)
        floats.push_back(float(i));

    return floats;
}

float data[] = {1, 2, 3, 4, 5, 6, 7, 8};

Eigen::Matrix3d mat33{{1, 2, 3},
                      {4, 5, 6},
                      {7, 8, 9}};

// Classes
class Mesh
{
public:
    Mesh(std::vector<float> positions,
         std::vector<uint32_t> indices)
    {   
        printf("Mesh constructor called\n");
        this->positions = std::move(positions);
        this->indices = std::move(indices);
    }

    ~Mesh() { printf("Mesh destructor called\n"); }

    std::vector<float> positions;
    std::vector<uint32_t> indices;
};


Eigen::Matrix<double, 1, 3> vec1{1, 2, 3};

Mesh get_mesh()
{
    std::vector<float> positions = {1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<uint32_t> indices = {1, 2, 3, 4, 5, 6, 7, 8};
    return {positions, indices};
}

Eigen::Matrix<double, 1, 1> eigen_add(Eigen::Matrix<double, 1, 1> a = Eigen::Matrix<double, 1, 1>::Zero(), Eigen::Matrix<double, 1, 1> b = Eigen::Matrix<double, 1, 1>::Zero())
{
    return a + b;
}
class MyClass {
public:
    MyClass() { std::cout << "Constructor\n"; }
    MyClass(const MyClass&) { std::cout << "Copy constructor\n"; }
    ~MyClass() { std::cout << "Destructor\n"; }
};


NB_MODULE(motor_bindings, m)
{
    // // https://nanobind.readthedocs.io/en/latest/exchanging.html#bindings
    // // for binding std vectors you must manually add the bind_vector macro for all types
    // nb::bind_vector<std::vector<float>>(m, "float_vector");
    // nb::bind_vector<std::vector<uint32_t>>(m, "uint32_vector");
    // nb::bind_vector<std::vector<std::string>>(m, "string_vector");

    // nb::bind_vector<std::vector<std::string>>(m, "string_vector");
    // nb::bind_vector<std::vector<CanController>>(m, "can_controller_vector");


    nb::class_<MyClass>(m, "MyClass")
        .def(nb::init<>())
        .def(nb::init<const MyClass &>()) // Expose copy constructor
        
        ;
    m.def("add", &add, "a"_a, "b"_a = 1,
          "This function adds two numbers and increments if only one is provided."); // this need: using namespace nb::literals;

    m.def("add_no_literals", &add, nb::arg("a"), nb::arg("b"),
          "This function adds two numbers and does not use the namespace nb::literals");

    m.def("eigen_add", &eigen_add, "a"_a = Eigen::Matrix<double, 1, 1>::Zero(), "b"_a = Eigen::Matrix<double, 1, 1>::Zero());

    m.def("get_floats_in_range", &get_floats_in_range, "start"_a, "end"_a,
          "This function returns a vector of floats based on a start and end value.");

    m.def("get_numpy_data", []()
          {
        size_t shape[2] = {2, 4};
        return nb::ndarray<nb::numpy, float, nb::shape<2, 4>>(
                data, /* ndim = */ 2, shape); });

    m.def("get_mesh", &get_mesh, "This function returns a Mesh class.");
    nb::class_<Mesh>(std::move(m), "Mesh")
        .def(nb::init<std::vector<float>, std::vector<uint32_t>>())
        .def_rw("positions", &Mesh::positions)
        .def_rw("indices", &Mesh::indices);

    m.attr("vec1") = vec1;

    // m.def("test_release_gil", []() -> bool
    //       { return PyGILState_Check(); }, nb::call_guard<nb::gil_scoped_release>());

    // m.def("test_no_release_gil", []() -> bool
    //       { return PyGILState_Check(); });



    nb::enum_<MotorParams>(m, "MotorParams",nb::is_arithmetic())
        .value("run_mode", MotorParams::run_mode,"run_mode")
        .value("iq_ref", MotorParams::iq_ref,"iq_ref")
        .value("mech_vel_ref", MotorParams::mech_vel_ref,"mech_vel_ref")
        .value("torque_limit", MotorParams::torque_limit,"torque_limit")
        .value("cur_kp", MotorParams::cur_kp,"cur_kp")
        .value("cur_ki", MotorParams::cur_ki,"cur_ki")
        .value("cur_filt_gain", MotorParams::cur_filt_gain,"cur_filt_gain")
        .value("mech_pos_ref", MotorParams::mech_pos_ref,"mech_pos_ref")
        .value("spd_limit", MotorParams::spd_limit,"spd_limit")
        .value("cur_limit", MotorParams::cur_limit,"cur_limit")
        .value("mech_pos", MotorParams::mech_pos,"mech_pos")
        .value("iqf", MotorParams::iqf,"iqf")
        .value("mech_vel", MotorParams::mech_vel,"mech_vel")
        .value("v_bus", MotorParams::v_bus,"v_bus")
        // .value("rotation", MotorParams::rotation,"rotation")
        .value("loc_kp", MotorParams::loc_kp,"loc_kp")
        .value("spd_kp", MotorParams::spd_kp,"spd_kp")
        .value("spd_ki", MotorParams::spd_ki,"spd_ki")
        .value("spd_filt_gain", MotorParams::spd_filt_gain,"spd_filt_gain")
        ;


    auto cls_CanMotorController = nb::class_<CanMotorController>(m, "CanMotorController", "This is the Motor class.")
        .def("__copy__", [](const CanMotorController&) {
            throw std::runtime_error("Copy not allowed");
        })
        .def(nb::init<
            std::vector<std::string>,
            Eigen::Matrix<uint8_t, Eigen::Dynamic, 1>,
            Eigen::Matrix<uint8_t, Eigen::Dynamic, 1>,
            Eigen::Matrix<uint8_t, Eigen::Dynamic, 1>>(),
            nb::call_guard<nb::gil_scoped_release>()
            )
        .def_ro("num_motors", &CanMotorController::num_motors)
        .def_ro("can_interfaces", &CanMotorController::can_interfaces)
        .def_ro("can_interface_ids", &CanMotorController::can_interface_ids)
        .def_ro("can_ids", &CanMotorController::can_ids)
        .def_ro("motor_types", &CanMotorController::motor_types)
        .def_rw("run_mode", &CanMotorController::run_mode)
        .def_rw("iq_ref", &CanMotorController::iq_ref)
        .def_rw("mech_vel_ref", &CanMotorController::mech_vel_ref)
        .def_rw("torque_limit", &CanMotorController::torque_limit)
        .def_rw("cur_kp", &CanMotorController::cur_kp)
        .def_rw("cur_ki", &CanMotorController::cur_ki)
        .def_rw("cur_filt_gain", &CanMotorController::cur_filt_gain)
        .def_rw("mech_pos_ref", &CanMotorController::mech_pos_ref)
        .def_rw("spd_limit", &CanMotorController::spd_limit)
        .def_rw("cur_limit", &CanMotorController::cur_limit)
        .def_rw("mech_pos", &CanMotorController::mech_pos)
        .def_ro("iqf", &CanMotorController::iqf)
        .def_ro("mech_vel", &CanMotorController::mech_vel)
        .def_ro("v_bus", &CanMotorController::v_bus)
        .def_rw("loc_kp", &CanMotorController::loc_kp)
        .def_rw("spd_kp", &CanMotorController::spd_kp)
        .def_rw("spd_ki", &CanMotorController::spd_ki)
        .def_rw("spd_filt_gain", &CanMotorController::spd_filt_gain)
        .def_rw("can_timeout", &CanMotorController::can_timeout)
        .def_rw("mech_torque_ref", &CanMotorController::mech_torque_ref)
        .def_ro("mech_torque", &CanMotorController::mech_torque)
        .def_ro("temperature", &CanMotorController::temperature)
        .def_rw("should_print_send", &CanMotorController::should_print_send)
        .def_rw("should_print_recv", &CanMotorController::should_print_recv)
        .def_rw("response_wait_microsecond", &CanMotorController::response_wait_microsecond)
        .def_rw("recv_timeout_ms", &CanMotorController::recv_timeout_ms)
        .def("get_bus_send_queue_depth", &CanMotorController::get_bus_send_queue_depth)
        .def("getAllParams", &CanMotorController::getAllParams)
        .def("enable", nb::overload_cast<>(&CanMotorController::enable))
        .def("enable", nb::overload_cast<const Eigen::Matrix<bool, Eigen::Dynamic, 1>&>(&CanMotorController::enable), nb::arg("mask"))
        .def("disable", &CanMotorController::disable,nb::arg("clear_fault"))
        .def("getParam", &CanMotorController::getParam,nb::arg("param"))
        .def("setParam", &CanMotorController::setParam<float>,nb::arg("param"),nb::arg("ref"))
        .def("setParam", &CanMotorController::setParam<uint32_t>,nb::arg("param"),nb::arg("ref"))
        .def("setParam", &CanMotorController::setParam<uint8_t>,nb::arg("param"),nb::arg("ref"))
        .def("set_pos_zero", &CanMotorController::set_pos_zero)
        .def("save_data", &CanMotorController::save_data)
        .def("motion_control_once", &CanMotorController::motion_control_once)
        .def("start_motion_control_continuously",
            &CanMotorController::start_motion_control_continuously,
            nb::call_guard<nb::gil_scoped_release>())
        .def("stop", &CanMotorController::stop)
        .def("get_motor_latency_history", &CanMotorController::get_motor_latency_history)

        // .def("motion_control_once", &CanMotorController::motion_control_once, nb::call_guard<nb::gil_scoped_release>())

        // .def("setRunMode", &CanMotorController::setRunMode, nb::call_guard<nb::gil_scoped_release>())
        // .def("run", &CanMotorController::run, nb::call_guard<nb::gil_scoped_release>())
        // .def("close", &CanMotorController::close)
        // .def("setSpeed", &CanMotorController::setSpeed, nb::call_guard<nb::gil_scoped_release>())
        // .def("setDutyCycle", &CanMotorController::setDutyCycle, nb::call_guard<nb::gil_scoped_release>())
        ;

        if (!nb::type<CanMotorController>().is(cls_CanMotorController))
        nb::detail::raise("type lookup failed!");


    // ===================== CubeMars GL II (GL40) =====================
    // Separate controller — standard-frame MIT protocol, no Robstride param model.
    auto cls_GLMotorController = nb::class_<GLMotorController>(m, "GLMotorController", "CubeMars GL II (GL40) controller — MIT / Position-Velocity / Velocity modes.")
        .def("__copy__", [](const GLMotorController&) {
            throw std::runtime_error("Copy not allowed");
        })
        .def(nb::init<
            std::vector<std::string>,
            Eigen::Matrix<uint8_t, Eigen::Dynamic, 1>,
            Eigen::Matrix<uint8_t, Eigen::Dynamic, 1>,
            Eigen::Matrix<uint8_t, Eigen::Dynamic, 1>,
            uint8_t>(),
            nb::call_guard<nb::gil_scoped_release>(),
            nb::arg("can_interfaces"), nb::arg("can_interface_ids"), nb::arg("can_ids"),
            nb::arg("motor_types"), nb::arg("control_mode") = (uint8_t)GL_MODE_MIT
            )
        .def_ro("num_motors", &GLMotorController::num_motors)
        .def_ro("control_mode", &GLMotorController::control_mode)
        .def_ro("can_interfaces", &GLMotorController::can_interfaces)
        .def_ro("can_interface_ids", &GLMotorController::can_interface_ids)
        .def_ro("can_ids", &GLMotorController::can_ids)
        .def_ro("motor_types", &GLMotorController::motor_types)
        .def_rw("mech_pos_ref", &GLMotorController::mech_pos_ref)
        .def_rw("mech_vel_ref", &GLMotorController::mech_vel_ref)
        .def_rw("mech_torque_ref", &GLMotorController::mech_torque_ref)
        .def_rw("kp", &GLMotorController::kp)
        .def_rw("kd", &GLMotorController::kd)
        .def_ro("mech_pos", &GLMotorController::mech_pos)
        .def_ro("mech_vel", &GLMotorController::mech_vel)
        .def_ro("mech_torque", &GLMotorController::mech_torque)
        .def_ro("drive_temp", &GLMotorController::drive_temp)
        .def_ro("motor_temp", &GLMotorController::motor_temp)
        .def_ro("mode_status", &GLMotorController::mode_status)
        .def_ro("error_code", &GLMotorController::error_code)
        .def_rw("should_print_send", &GLMotorController::should_print_send)
        .def_rw("should_print_recv", &GLMotorController::should_print_recv)
        .def_rw("response_wait_microsecond", &GLMotorController::response_wait_microsecond)
        .def_rw("recv_timeout_ms", &GLMotorController::recv_timeout_ms)
        .def("get_bus_send_queue_depth", &GLMotorController::get_bus_send_queue_depth)
        .def("enable", nb::overload_cast<>(&GLMotorController::enable))
        .def("enable", nb::overload_cast<const Eigen::Matrix<bool, Eigen::Dynamic, 1>&>(&GLMotorController::enable), nb::arg("mask"))
        .def("disable", &GLMotorController::disable, nb::arg("clear_fault") = false)
        .def("set_pos_zero", &GLMotorController::set_pos_zero)
        .def("clear_error", &GLMotorController::clear_error)
        .def("motion_control_once", &GLMotorController::motion_control_once)
        .def("start_motion_control_continuously",
            &GLMotorController::start_motion_control_continuously,
            nb::call_guard<nb::gil_scoped_release>())
        .def("stop", &GLMotorController::stop)
        .def("get_motor_latency_history", &GLMotorController::get_motor_latency_history)
        ;

        if (!nb::type<GLMotorController>().is(cls_GLMotorController))
        nb::detail::raise("type lookup failed!");
}
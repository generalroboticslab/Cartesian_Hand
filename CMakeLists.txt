cmake_minimum_required(VERSION 3.15...3.27)
project(cartesian_hand LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(CMAKE_VERSION VERSION_LESS 3.18)
  set(DEV_MODULE Development)
else()
  set(DEV_MODULE Development.Module)
endif()

find_package(Python 3.8
  COMPONENTS Interpreter ${DEV_MODULE} REQUIRED
  OPTIONAL_COMPONENTS Development.SABIModule)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

# Servo driver sources come from the hardware_bindings submodule rather than a
# copy in this repo. They used to be duplicated under src/, which drifted: the
# fork gained write_id and a position-encoding fix that were never pushed back,
# and edited INST.h in a way that changed the RESET and CAL opcodes away from
# the values the vendor SCS.cpp expects. Both fixes are upstream now, so there
# is one copy again.
#
# Only ft_servo/ is compiled here. The submodule's own top-level CMakeLists also
# builds motor/ and imu/ and requires Eigen3, CSerialPort and fmt; this gripper
# uses none of them, so we name the sources directly instead of add_subdirectory.
set(FT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/hardware_bindings/ft_servo)

if(NOT EXISTS ${FT_DIR}/ft_servo_ext.cpp)
  message(FATAL_ERROR
    "hardware_bindings/ft_servo is empty. Run:\n"
    "  git submodule update --init --recursive")
endif()

set(FT_SRC ${FT_DIR}/HLSCL.cpp ${FT_DIR}/SCS.cpp ${FT_DIR}/SCSerial.cpp)

nanobind_add_module(
    ft_servo_ext
    STABLE_ABI
    ${FT_DIR}/ft_servo_ext.cpp
    ${FT_DIR}/ft_servo_driver.hpp
    ${FT_SRC}
)
target_include_directories(ft_servo_ext PRIVATE ${FT_DIR})

# --- wheel install layout (used by scikit-build-core) ---
# The extension lands inside the package so `from .ft_servo_ext import FtServo`
# resolves directly and the search fallback in driver.py stays unused.
# The .py files are NOT installed here: scikit-build-core packages them (see
# wheel.packages in pyproject.toml), which is what keeps -e installs editable.
install(TARGETS ft_servo_ext LIBRARY DESTINATION cartesian_hand)

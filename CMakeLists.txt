cmake_minimum_required(VERSION 3.15...3.27)
project(cartesian_hand LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(CMAKE_VERSION VERSION_LESS 3.18)
  set(DEV_MODULE Development)
else()
  set(DEV_MODULE Development.Module)
endif()

find_package(Python 3.10
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

# Servo driver sources. The vendor SDK (HLSCL, SCS, SCSerial) and the driver
# wrapper live under cartesian_hand/src/ft_servo/ so the build has no
# dependency on an external checkout.
set(FT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/cartesian_hand/src/ft_servo)

if(NOT EXISTS ${FT_DIR}/ft_servo_ext.cpp)
  message(FATAL_ERROR
    "cartesian_hand/src/ft_servo/ is empty. Sources are checked in; restore them.")
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
# The extension lands inside the package so servo.py's
# `from .ft_servo_ext import FtServo` resolves.
# The .py files are NOT installed here: scikit-build-core packages them (see
# wheel.packages in pyproject.toml), which is what keeps -e installs editable.
install(TARGETS ft_servo_ext LIBRARY DESTINATION cartesian_hand)

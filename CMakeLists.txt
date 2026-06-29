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

# Use vcpkg-provided CSerialPort if available; otherwise fetch from source.

set(FT_SRC src/HLSCL.cpp src/SCS.cpp src/SCSerial.cpp)

nanobind_add_module(
    ft_servo_ext
    STABLE_ABI
    src/ft_servo_ext.cpp
    src/ft_servo_driver.hpp
    ${FT_SRC}
)
target_include_directories(ft_servo_ext PRIVATE src)

# --- wheel install layout (used by scikit-build-core) ---
install(TARGETS ft_servo_ext LIBRARY DESTINATION cartesian_hand)
install(FILES __init__.py cartesian_hand.py DESTINATION cartesian_hand)
install(DIRECTORY tasks commons DESTINATION cartesian_hand
        PATTERN "__pycache__" EXCLUDE
        PATTERN "*.pyc" EXCLUDE)

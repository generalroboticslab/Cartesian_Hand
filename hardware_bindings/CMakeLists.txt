cmake_minimum_required(VERSION 3.15...3.27)
project(hardware_bindings LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Python 3.12 REQUIRED COMPONENTS Interpreter Development)
execute_process(COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_DIR)
find_package(nanobind CONFIG REQUIRED)

find_package(Eigen3 CONFIG REQUIRED)
find_package(CSerialPort CONFIG REQUIRED)
find_package(fmt CONFIG REQUIRED)

# .so files land in each package's source dir (next to .cpp) so the Python
# wrapper's _load_ext finds them via `parent / name.abi3.so` — single canonical
# path shared by both dev builds and editable installs.
add_subdirectory(motor)
add_subdirectory(imu)
add_subdirectory(ft_servo)

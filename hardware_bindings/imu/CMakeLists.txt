add_subdirectory(EasyProfile)

nanobind_add_module(
  imu_nanobind
  NB_SHARED
  STABLE_ABI
  imu_nanobind.cpp
  imu.hpp
)
set_target_properties(imu_nanobind PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)
# Copy libnanobind-abi3.so next to our .so so the wrapper can dlopen
# without setting LD_LIBRARY_PATH.
add_custom_command(TARGET imu_nanobind POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    $<TARGET_FILE:nanobind-abi3>
    ${CMAKE_CURRENT_SOURCE_DIR}/
)
target_link_libraries(imu_nanobind PRIVATE Eigen3::Eigen)
target_link_libraries(imu_nanobind PUBLIC EasyProfile)
target_include_directories(imu_nanobind PUBLIC ${EasyProfile_INCLUDE_DIR})
target_link_libraries(imu_nanobind PUBLIC ${CSerialPort_LIBRARY})
target_include_directories(imu_nanobind PUBLIC ${CSerialPort_INCLUDE_DIR})
target_link_libraries(imu_nanobind PUBLIC fmt::fmt)
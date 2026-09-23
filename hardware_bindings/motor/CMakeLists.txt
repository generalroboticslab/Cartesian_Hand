nanobind_add_module(
  motor_bindings
  NB_SHARED
  STABLE_ABI
  motor_bindings.cpp
  motor.hpp
  motor.cpp
  gl_motor.hpp
  gl_motor.cpp
)
set_target_properties(motor_bindings PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)
# Copy libnanobind-abi3.so next to our .so so the wrapper can dlopen
# without setting LD_LIBRARY_PATH.
add_custom_command(TARGET motor_bindings POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    $<TARGET_FILE:nanobind-abi3>
    ${CMAKE_CURRENT_SOURCE_DIR}/
)
target_link_libraries(motor_bindings PRIVATE Eigen3::Eigen)
target_link_libraries(motor_bindings PUBLIC EasyProfile)
target_include_directories(motor_bindings PUBLIC ${EasyProfile_INCLUDE_DIR})
target_link_libraries(motor_bindings PUBLIC ${CSerialPort_LIBRARY})
target_include_directories(motor_bindings PUBLIC ${CSerialPort_INCLUDE_DIR})
target_link_libraries(motor_bindings PUBLIC fmt::fmt)
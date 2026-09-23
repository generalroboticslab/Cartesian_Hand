set(FT_SRC HLSCL.cpp SCS.cpp SCSerial.cpp)

nanobind_add_module(
    ft_servo_ext
    NB_SHARED
    STABLE_ABI
    ft_servo_ext.cpp
    ft_servo_driver.hpp
    ${FT_SRC}
)
set_target_properties(ft_servo_ext PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)
# Copy libnanobind-abi3.so next to our .so so the wrapper can dlopen
# without setting LD_LIBRARY_PATH.
add_custom_command(TARGET ft_servo_ext POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    $<TARGET_FILE:nanobind-abi3>
    ${CMAKE_CURRENT_SOURCE_DIR}/
)
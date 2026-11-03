# Timing of the Enzyme Jacobian for benchmark.py (not installed, not a test).
add_executable(enzyme_jacobian_timing jacobian_timing.cpp)
target_link_libraries(enzyme_jacobian_timing PRIVATE roadrunner-static)
target_include_directories(enzyme_jacobian_timing PRIVATE ${CMAKE_SOURCE_DIR}/source)
set_target_properties(enzyme_jacobian_timing PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)


# integration solver test
set(target "test_integrator_api")
add_executable(${target} test-integrator-api.c)
target_compile_definitions(${target} PRIVATE STATIC_RR)
target_link_libraries (${target} roadrunner_c_api roadrunner-static unit_test-static)
target_include_directories(${target} PRIVATE ${INCLUDE_DIRECTORIES} ${CMAKE_CURRENT_SOURCE_DIR})


# Steady state solver test
set(target "ss-solver-test")
add_executable(${target} ss-solver-test.c)
target_compile_definitions(${target} PRIVATE STATIC_RR)
target_link_libraries (${target} roadrunner_c_api unit_test-static)
target_include_directories(${target} PRIVATE ${INCLUDE_DIRECTORIES} ${CMAKE_CURRENT_SOURCE_DIR})

# install
install(TARGETS ss-solver-test DESTINATION bin COMPONENT testing)
install (TARGETS ${target} DESTINATION bin COMPONENT testing)


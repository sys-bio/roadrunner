cmake_minimum_required(VERSION 2.8)
set(target rr_run_test_suite)
set(rrSrc "../../../../source")

add_executable(${target}
        main.cpp
        Args.cpp
        TestSuiteSimulation.cpp
        Args.h
        TestSuiteSimulation.h
        )
target_compile_definitions(${target} PRIVATE STATIC_RR)
target_link_libraries(${target} PRIVATE roadrunner_c_api roadrunner-static) # poco_foundation-static

target_include_directories(${target} PRIVATE
        ${INCLUDE_DIRECTORIES}
        )

install(
        TARGETS ${target}
        DESTINATION bin COMPONENT apps
)

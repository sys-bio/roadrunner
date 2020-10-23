CMAKE_MINIMUM_REQUIRED(VERSION 2.6.3 FATAL_ERROR)
PROJECT(RR_CXX_API_TESTS)

set(target rr-sbml-benchmark)

add_executable(${target} main.cpp)

target_link_libraries(${target} PRIVATE roadrunner-static ${UNIT_TEST_STATIC_LIBRARY})
target_include_directories(${target} PRIVATE
        ${INCLUDE_DIRECTORIES}
        )
add_dependencies(${target} unit_test-static roadrunner-static)

install(TARGETS ${target}
        DESTINATION bin
        COMPONENT testing
        )



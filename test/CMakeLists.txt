########################################################
#   Add the googletest framework as a subdirectory
#
add_subdirectory(googletest-release-1.10.0)
# we be naughty and use include_directories here to include googletest globally
include_directories(googletest-release-1.10.0/googletest/include)

# set directory here to be used in subdirectories for locating site-packages in build tree
set(RR_PYTHON_TESTING_DIR "${CMAKE_BINARY_DIR}/lib/site-packages/roadrunner/testing")

###########################################################
# CMake function for adding roadrunner test executables
#

# "CMAKE_CURRENT_SOURCE_DIR" changes in the different files were
# add_test_executable gets called. We can use this as a constant
set(ThisDirectory ${CMAKE_CURRENT_SOURCE_DIR})

# Adds a new executable cmake target to cmake.
# Links roadrunner-static and gtest.
# Parameters:
#   - TEST_TARGET: name of the binary
#   - OUT_VARIABLE: list variable to append BINARY. Creates the list if not exist.
#   - All further arguments are added to the binary as sources.
# Usage:
#   To add a new test target called myTestTargetName which is linked to appropriate libraries do
#
#       add_test_executable(myTestTargetName RoadRunnerTestTargetList testFile1.h testFile1.cpp ...)
#
function(add_test_executable TEST_TARGET OUT_VARIABLE)
    add_executable(${TEST_TARGET} ${ARGN})

    target_include_directories(
            ${TEST_TARGET} PRIVATE
            "${ThisDirectory}"
    )

    target_link_libraries(${TEST_TARGET} PUBLIC
            roadrunner-static gtest gtest_main gmock gmock_main
            )

    # This line is a hack for enabling targets built with add_test_executable function
    # access to the TestModelFactory. Eventally, it'd be nice to pull TestModelFactory
    # into its own directory, and this will feel like less of a hack.
    target_include_directories(${TEST_TARGET} PUBLIC "${ThisDirectory}/sundials-tests")

    add_dependencies(${TEST_TARGET} roadrunner-static gtest gtest_main gmock gmock_main)
    set_target_properties(${TEST_TARGET} PROPERTIES LINKER_LANGUAGE CXX)

    # Add to ctest.
    # add_test(NAME ${BINARY} COMMAND "$<TARGET_FILE:${BINARY}>")
    gtest_discover_tests(
            ${TEST_TARGET}
            TEST_LIST sundials-tests
            DISCOVERY_TIMEOUT 500
            )

    set_tests_properties(${sundials-tests} PROPERTIES TIMEOUT 500)


    set(test_targets "${test_targets}" "${testid}")
    if ("${${OUT_VARIABLE}}" STREQUAL "")
        set(${OUT_VARIABLE} "${TEST_LIST}" PARENT_SCOPE)
    else ()
        set(${OUT_VARIABLE} "${${OUT_VARIABLE}}" "${TEST_LIST}" PARENT_SCOPE)
    endif ()

endfunction()




########################################################
#   Tests
#

add_subdirectory(python)
add_subdirectory(rrtest_files)

set(TESTDIRS
        c_api_core
        c_api_rrtests
        model_analysis
        model_editing
        sbml_features
        state_saving
        steady_state
        #List the semantic_STS last, since it takes much longer to run than the others.
        semantic_STS
        )

set(test_targets)
foreach (dirid IN LISTS TESTDIRS)
    #    message(STATUS "dirid = ${dirid}")
    file(GLOB MAIN_FILE ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)
    file(GLOB UTIL_FILE ${CMAKE_CURRENT_SOURCE_DIR}/test_util.cpp)
    file(GLOB UTIL_HEADER ${CMAKE_CURRENT_SOURCE_DIR}/test_util.h)
    file(GLOB CPP_FILES ${CMAKE_CURRENT_SOURCE_DIR}/${dirid}/*.cpp)
    file(GLOB H_FILES ${CMAKE_CURRENT_SOURCE_DIR}/${dirid}/*.h)

    set(TEST_FILES ${MAIN_FILE} ${UTIL_FILE} ${UTIL_HEADER} ${CPP_FILES} ${H_FILES})

    set(testid "test_${dirid}")
    add_executable(${testid} ${TEST_FILES})

    add_compile_definitions(${testid} PRIVATE STATIC_RR)

    if (WIN32)
        add_definitions(/bigobj)
    endif ()

    # This order is important:  on linux, including the core library before the C api library
    # causes the resulting program to crash in Poco.
    target_link_libraries(${testid}
            roadrunner_c_api
            roadrunner-static
            PocoXML::PocoXML
            gtest gtest_main
            )

    set_target_properties(${testid} PROPERTIES ENVIRONMENT
            "testdir=${RR_ROOT}/test")

    target_include_directories(${testid} PRIVATE "$<TARGET_PROPERTY:roadrunner_c_api,INCLUDE_DIRECTORIES>")
    add_test(${testid}_run ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${testid})
    # set the testdir variable for when using ctest to run tests

    set(TEST_ENV_VARS "testdir=${RR_ROOT}/test" "CTEST_OUTPUT_ON_FAILURE=TRUE")
    set_tests_properties(${testid}_run PROPERTIES ENVIRONMENT "${TEST_ENV_VARS}")

    set(test_targets "${test_targets}" "${testid}")

endforeach ()

# other directories to be added to tests
add_subdirectory(sundials-tests)
add_subdirectory(picklable-roadrunner)

install(TARGETS ${test_targets})

add_custom_target(
        run-ctest
        ctest --progress --verbose --extra-verbose ${CMAKE_BINARY_DIR}
)


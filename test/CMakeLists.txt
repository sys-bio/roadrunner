########################################################
#   Add the googletest framework as a subdirectory
#
add_subdirectory(googletest-release-1.10.0)
# we be naughty and use include_directories here to include googletest globally
include_directories(googletest-release-1.10.0/googletest/include)
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

#set(test_targets)
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


endforeach ()


add_custom_target(
        run-ctest
            ctest --progress  --verbose --extra-verbose ${CMAKE_BINARY_DIR}
)


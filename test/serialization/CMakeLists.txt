add_subdirectory(PickledSwig)

if (BUILD_TESTS)
    add_test_executable(
            test_state_saving test_targets
            ${SharedTestFiles}
            ../serialization/state_saving.cpp
    )

    if (BUILD_PYTHON)
        if ("${RR_PYTHON_TESTING_BUILD_PREFIX}" STREQUAL "")
            message(ERROR "No RR_PYTHON_TESTING_BUILD_PREFIX variable defined")
        endif ()

        configure_file(
                "${CMAKE_CURRENT_SOURCE_DIR}/pickle_tests.py"
                "${RR_PYTHON_TESTING_BUILD_PREFIX}/pickle_tests.py"
                COPYONLY
        )
    endif ()
endif ()


# make test_targets list global to all tests
set(test_targets ${test_targets} PARENT_SCOPE)





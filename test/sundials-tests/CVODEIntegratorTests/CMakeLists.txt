

add_test_executable(
        test_sundials_cvode_integration test_targets
        CvodeIntegrationTest.cpp
        ${SharedTestFiles}
)

add_test_executable(
        test_sundials_cvode_unit test_targets
        CvodeUnitTest.cpp
        ${SharedTestFiles}
)

# make test_targets list global to all tests
set(test_targets ${test_targets} PARENT_SCOPE)

if (BUILD_PYTHON AND BUILD_TESTS)
    # copy IntegratorTests into the Build tree
    message(STATUS "RR_PYTHON_TESTING_BUILD_PREFIX ${RR_PYTHON_TESTING_BUILD_PREFIX}")
    configure_file(
            "${CMAKE_CURRENT_SOURCE_DIR}/IntegratorTests.py"
            "${RR_PYTHON_TESTING_BUILD_PREFIX}/IntegratorTests.py"
            COPYONLY
    )

    install(FILES
            "${RR_PYTHON_TESTING_BUILD_PREFIX}/IntegratorTests.py"
            DESTINATION
            "${RR_PYTHON_TESTING_INSTALL_PREFIX}"
            )

endif ()

























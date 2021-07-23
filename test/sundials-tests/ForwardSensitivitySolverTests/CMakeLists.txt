#add_test_executable(cvsRoberts_FSA_dns cvode-example cvsRoberts_FSA_dns.c)

add_test_executable(test_sundials_FFSUnitTests test_targets FFSUnitTests.cpp)
add_test_executable(test_sundials_FFSIntegrationTests test_targets FFSIntegrationTests.cpp ../CVODEIntegratorTests/CvodeIntegrationTest.h)



if (BUILD_PYTHON AND BUILD_TESTS)
    configure_file(
            "${CMAKE_CURRENT_SOURCE_DIR}/ForwardSensitivitySolverTests.py"
            "${RR_PYTHON_TESTING_BUILD_PREFIX}/ForwardSensitivitySolverTests.py"
            COPYONLY
    )

    install(FILES
            "${RR_PYTHON_TESTING_BUILD_PREFIX}/ForwardSensitivitySolverTests.py"
            DESTINATION
            "${RR_PYTHON_TESTING_INSTALL_PREFIX}"
            )

endif ()











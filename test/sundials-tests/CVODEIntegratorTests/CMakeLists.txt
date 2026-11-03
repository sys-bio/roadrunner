

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

if (RR_USE_ENZYME)
    add_test_executable(
            test_sundials_cvode_enzyme_jacobian test_targets
            CvodeEnzymeJacobianTests.cpp
    )
endif ()

# make test_targets list global to all tests
set(test_targets ${test_targets} PARENT_SCOPE)
























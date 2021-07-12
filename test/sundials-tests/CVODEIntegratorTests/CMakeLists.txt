

add_test_executable(
        test_cvode_integration test_targets
        CVODEIntegratorIntegrationTest.cpp
        ${SharedTestFiles}
)

add_test_executable(
        test_cvode_unit test_targets
        CVODEIntegratorUnitTests.cpp
        ${SharedTestFiles}
)

# make test_targets list global to all tests
set(test_targets ${test_targets} PARENT_SCOPE)



























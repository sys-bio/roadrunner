add_test_executable(
        test_sbml_features test_targets
        ${SharedTestFiles}
        boolean_delay.cpp
)

# make test_targets list global to all tests
set(test_targets ${test_targets} PARENT_SCOPE)


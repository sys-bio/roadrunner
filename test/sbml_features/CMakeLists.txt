add_test_executable(
        test_sbml_features test_targets
        ${SharedTestFiles}
        boolean_delay.cpp
        invalid_sbml.cpp
        valid_sbml.cpp
        events.cpp
)

# make test_targets list global to all tests
set(test_targets ${test_targets} PARENT_SCOPE)


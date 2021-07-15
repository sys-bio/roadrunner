set(testid "test_plugin_test_model")

add_test_executable(
        ${testid} test_targets
        ${SharedTestFiles}
        model_check.cpp
)

# make test_targets list global to all tests
set(test_targets ${test_targets} PARENT_SCOPE)

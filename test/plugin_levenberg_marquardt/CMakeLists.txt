set(testid "test_plugin_levenberg_marquardt")

add_test_executable(
        ${testid} test_targets
        ${SharedTestFiles}
        lm_optimize.cpp
)

# make test_targets list global to all tests
set(test_targets ${test_targets} PARENT_SCOPE)

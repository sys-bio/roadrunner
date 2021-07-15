set(testid "test_plugin_monte_carlo_bs")

add_test_executable(
        ${testid} test_targets
        ${SharedTestFiles}
        mc_optimize.cpp
)

# make test_targets list global to all tests
set(test_targets ${test_targets} PARENT_SCOPE)

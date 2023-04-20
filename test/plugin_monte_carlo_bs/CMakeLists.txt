#[[
set(test1_id "test_plugin_monte_carlo_bs_1")

add_test_executable(
        ${test1_id} test_targets
        ${SharedTestFiles}
        mc_optimize_1.cpp PluginMonteCarloTests.cpp
)

# make test_targets list global to all tests
set(test_targets ${test_targets} PARENT_SCOPE)

set(test2_id "test_plugin_monte_carlo_bs_2")

add_test_executable(
        ${test2_id} test_targets
        ${SharedTestFiles}
        mc_optimize_2.cpp PluginMonteCarloTests.cpp
)

# make test_targets list global to all tests
set(test_targets ${test_targets} PARENT_SCOPE)

set(test3_id "test_plugin_monte_carlo_bs_3")

add_test_executable(
        ${test3_id} test_targets
        ${SharedTestFiles}
        mc_optimize_3.cpp PluginMonteCarloTests.cpp
)

# make test_targets list global to all tests
set(test_targets ${test_targets} PARENT_SCOPE)

]]

#[[
set(test1_id "test_plugin_levenberg_marquardt_1")

add_test_executable(
        ${test1_id} test_targets
        ${SharedTestFiles}
        lm_optimize_1.cpp PluginLevenbergMarquardtTests.cpp
)

# make test_targets list global to all tests
set(test_targets ${test_targets} PARENT_SCOPE)

set(test2_id "test_plugin_levenberg_marquardt_2")

add_test_executable(
        ${test2_id} test_targets
        ${SharedTestFiles}
        lm_optimize_2.cpp PluginLevenbergMarquardtTests.cpp
)

# make test_targets list global to all tests
set(test_targets ${test_targets} PARENT_SCOPE)
]]

#[[

set(test1_id "test_plugin_nelder_mead_1")

add_test_executable(
        ${test1_id} test_targets
        ${SharedTestFiles}
        nm_optimize_1.cpp PluginNelderMeadTests.cpp
)

# make test_targets list global to all tests
set(test_targets ${test_targets} PARENT_SCOPE)


set(test2_id "test_plugin_nelder_mead_2")

add_test_executable(
        ${test2_id} test_targets
        ${SharedTestFiles}
        nm_optimize_2.cpp PluginNelderMeadTests.cpp
)

# make test_targets list global to all tests
set(test_targets ${test_targets} PARENT_SCOPE)

]]

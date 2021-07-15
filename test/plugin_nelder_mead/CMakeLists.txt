set(testid "test_plugin_nelder_mead")

add_test_executable(
        ${testid} test_targets
        ${SharedTestFiles}
        nm_optimize.cpp
)

# make test_targets list global to all tests
set(test_targets ${test_targets} PARENT_SCOPE)

set(testid "test_plugin_auto2000")

add_test_executable(
        ${testid} test_targets
        ${SharedTestFiles}
        auto2000.cpp
)

# make test_targets list global to all tests
set(test_targets ${test_targets} PARENT_SCOPE)

set(testid "test_plugin_api")

add_test_executable(
        ${testid} test_targets
        ${SharedTestFiles}
        api.cpp
)

# make test_targets list global to all tests
set(test_targets ${test_targets} PARENT_SCOPE)

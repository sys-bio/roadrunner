set(test_id "test_plugin_test_model")

add_test_executable(
        ${test_id} test_targets
        ${SharedTestFiles}
        model_check.cpp
)

# make test_targets list global to all tests
set(test_targets ${test_targets} PARENT_SCOPE)

#[[

set(test1_id "test_plugin_test_model_1")

add_test_executable(
        ${test1_id} test_targets
        ${SharedTestFiles}
        model_check_1.cpp PluginTestModelTests.cpp
)

# make test_targets list global to all tests
set(test_targets ${test_targets} PARENT_SCOPE)

set(test2_id "test_plugin_test_model_2")

add_test_executable(
        ${test2_id} test_targets
        ${SharedTestFiles}
        model_check_2.cpp PluginTestModelTests.cpp
)

# make test_targets list global to all tests
set(test_targets ${test_targets} PARENT_SCOPE)

set(test3_id "test_plugin_test_model_3")

add_test_executable(
        ${test3_id} test_targets
        ${SharedTestFiles}
        model_check_3.cpp PluginTestModelTests.cpp
)

# make test_targets list global to all tests
set(test_targets ${test_targets} PARENT_SCOPE)

set(test4_id "test_plugin_test_model_4")

add_test_executable(
        ${test4_id} test_targets
        ${SharedTestFiles}
        model_check_4.cpp PluginTestModelTests.cpp
)

# make test_targets list global to all tests
set(test_targets ${test_targets} PARENT_SCOPE)

]]
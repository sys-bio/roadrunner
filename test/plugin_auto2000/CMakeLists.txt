set(test_id "test_plugin_auto2000")

add_test_executable(
        ${test_id} test_targets
        ${SharedTestFiles}
        PluginTests.cpp
)

# make test_targets list global to all tests
set(test_targets ${test_targets} PARENT_SCOPE)

#[[
set(test1_id "test_plugin_auto2000_1")

add_test_executable(
        ${test1_id} test_targets
        ${SharedTestFiles}
        auto2000_1.cpp PluginAuto2000Tests.cpp
)

# make test_targets list global to all tests
set(test_targets ${test_targets} PARENT_SCOPE)

set(test2_id "test_plugin_auto2000_2")

add_test_executable(
        ${test2_id} test_targets
        ${SharedTestFiles}
        auto2000_2.cpp PluginAuto2000Tests.cpp
)

# make test_targets list global to all tests
set(test_targets ${test_targets} PARENT_SCOPE)

set(test3_id "test_plugin_auto2000_3")

add_test_executable(
        ${test3_id} test_targets
        ${SharedTestFiles}
        auto2000_3.cpp PluginAuto2000Tests.cpp
)

# make test_targets list global to all tests
set(test_targets ${test_targets} PARENT_SCOPE)

set(test4_id "test_plugin_auto2000_4")

add_test_executable(
        ${test4_id} test_targets
        ${SharedTestFiles}
        auto2000_4.cpp PluginAuto2000Tests.cpp
)

# make test_targets list global to all tests
set(test_targets ${test_targets} PARENT_SCOPE)
]]

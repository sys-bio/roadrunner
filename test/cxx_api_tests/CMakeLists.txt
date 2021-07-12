#add_test_executable(variant_tests test_targets variant_tests.cpp)

add_test_executable(SettingTests test_targets SettingTests.cpp)
add_test_executable(RoadRunnerAPITests test_targets RoadRunnerAPITests.cpp ${SharedTestFiles})
add_test_executable(GillespieTests test_targets GillespieTests.cpp ${SharedTestFiles})
add_test_executable(LoggerTests test_targets LoggerTests.cpp ${SharedTestFiles})
add_test_executable(BasicDictionaryTests test_targets BasicDictionaryTests.cpp ${SharedTestFiles})
add_test_executable(SelectionRecordTests test_targets SelectionRecordTests.cpp ${SharedTestFiles})
add_test_executable(LoadSBMLOptionsTests test_targets LoadSBMLOptionsTests.cpp ${SharedTestFiles})
add_test_executable(ConfigTests test_targets ConfigTests.cpp ${SharedTestFiles})

if (BUILD_PYTHON)
    add_swig_executable(SettingTestsSwigAPI
            SettingTestsSwigAPI.i
            SettingTestsSwigAPI
            "../../wrappers/Python/roadrunner/PyUtils"
            )
endif ()
















add_test_executable(test_cxx_api_SettingTests test_targets SettingTests.cpp)
add_test_executable(test_cxx_api_RoadRunnerAPITestsMCJit test_targets RoadRunnerAPITests.h RoadRunnerAPITestsWithMCJit.cpp ${SharedTestFiles})
add_test_executable(test_cxx_api_RoadRunnerAPITestsLLJit test_targets RoadRunnerAPITests.h RoadRunnerAPITestsWithLLJit.cpp ${SharedTestFiles})
add_test_executable(test_cxx_api_GillespieTests test_targets GillespieTests.cpp ${SharedTestFiles})
add_test_executable(test_cxx_api_LoggerTests test_targets LoggerTests.cpp ${SharedTestFiles})
add_test_executable(test_cxx_api_BasicDictionaryTests test_targets BasicDictionaryTests.cpp ${SharedTestFiles})
add_test_executable(test_cxx_api_SelectionRecordTests test_targets SelectionRecordTests.cpp ${SharedTestFiles})
add_test_executable(test_cxx_api_LoadSBMLOptionsTests test_targets LoadSBMLOptionsTests.cpp ${SharedTestFiles})
add_test_executable(test_cxx_api_ConfigTests test_targets ConfigTests.cpp ${SharedTestFiles})
add_test_executable(test_cxx_api_MatrixTests test_targets MatrixTests.cpp ${SharedTestFiles})
add_test_executable(test_cxx_api_Matrix3DTests test_targets Matrix3DTests.cpp ${SharedTestFiles})
add_test_executable(test_cxx_api_JacobianTests test_targets JacobianTests.cpp ${SharedTestFiles})
add_test_executable(test_cxx_api_StructuralAnalysisTests test_targets StructuralAnalysisTests.cpp ${SharedTestFiles})
add_test_executable(test_cxx_api_MCATests test_targets MCATests.cpp ${SharedTestFiles})
add_test_executable(test_cxx_api_RoadRunnerMapTests test_targets RoadRunnerMapTests.cpp ${SharedTestFiles})

set(test_targets ${test_targets} PARENT_SCOPE)

if (BUILD_PYTHON AND BUILD_TESTS)

    ###################################################################################
    # When (and if) we want tests for the PyUtils source file, we can use the following commented out code.
    #

    # Test the PyUtils api from C++, enabling us use of the debugger etc.
    # add_test_executable(test_cxx_api_PyUtilsTests test_targets PyUtilsTests.cpp ${SharedTestFiles}
    #         ../../wrappers/Python/roadrunner/PyUtils.h
    #         ../../wrappers/Python/roadrunner/PyUtils.cpp
    #         )
    # target_link_libraries(test_cxx_api_PyUtilsTests PRIVATE
    #         Python::NumPy Python::Python)
    # target_include_directories(test_cxx_api_PyUtilsTests PRIVATE
    #         ../../wrappers/Python/include
    #         )

    add_rr_test_swig_library(SettingTestsSwigAPI
            SettingTestsSwigAPI.i
            SettingTestsSwigAPI
            "../../wrappers/Python/roadrunner/PyUtils.cpp"
            "../../wrappers/Python/roadrunner/PyUtils.h"
            )
endif ()





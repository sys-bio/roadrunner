#add_test_executable(variant_tests test_targets variant_tests.cpp)

add_test_executable(test_cxx_api_SettingTests test_targets SettingTests.cpp)
add_test_executable(test_cxx_api_RoadRunnerAPITests test_targets RoadRunnerAPITests.cpp ${SharedTestFiles})
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

if (BUILD_PYTHON AND BUILD_TESTS)
    # Test the PyUtils api from C++, enabling us use of the debugger etc.
    add_test_executable(test_cxx_api_PyUtilsTests test_targets PyUtilsTests.cpp ${SharedTestFiles}
            ../../wrappers/Python/roadrunner/PyUtils.h
            ../../wrappers/Python/roadrunner/PyUtils.cpp
            )
    target_link_libraries(test_cxx_api_PyUtilsTests PRIVATE
            Python::NumPy Python::Python)
    target_include_directories(test_cxx_api_PyUtilsTests PRIVATE
            ../../wrappers/Python/include
            )

    add_swig_executable(SettingTestsSwigAPI
            SettingTestsSwigAPI.i
            SettingTestsSwigAPI
            "../../wrappers/Python/roadrunner/PyUtils.cpp"
            "../../wrappers/Python/roadrunner/PyUtils.h"
            )

    configure_file(
            "${CMAKE_CURRENT_SOURCE_DIR}/SettingTests.py"
            "${RR_PYTHON_TESTING_BUILD_PREFIX}/SettingTests.py"
            COPYONLY
    )

    configure_file(
            "${CMAKE_CURRENT_SOURCE_DIR}/StructuralAnalysisTests.py"
            "${RR_PYTHON_TESTING_BUILD_PREFIX}/StructuralAnalysisTests.py"
            COPYONLY
    )

    configure_file(
            "${CMAKE_CURRENT_SOURCE_DIR}/JacobianTests.py"
            "${RR_PYTHON_TESTING_BUILD_PREFIX}/JacobianTests.py"
            COPYONLY
    )

    install(FILES
            "${RR_PYTHON_TESTING_BUILD_PREFIX}/SettingTests.py"
            "${RR_PYTHON_TESTING_BUILD_PREFIX}/JacobianTests.py"
            "${RR_PYTHON_TESTING_BUILD_PREFIX}/StructuralAnalysisTests.py"
            DESTINATION
            "${RR_PYTHON_TESTING_INSTALL_PREFIX}"
            )

endif ()
















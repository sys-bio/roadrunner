CMAKE_MINIMUM_REQUIRED(VERSION 2.6.3 FATAL_ERROR)
PROJECT(RR_CXX_API_TESTS)

set(target rr_analysis)

if (BUILD_PYTHON)

    MESSAGE ("THIS IS PYTHON_INCLUDE_PATH " ${PYTHON_INCLUDE_PATH})

    INCLUDE_DIRECTORIES(${PYTHON_INCLUDE_PATH})

endif(BUILD_PYTHON)

set(rr_analysis_src
    main.cpp
    src/tests.cpp
    src/cpplapack.cpp
    src/CSRMatrixTest.cpp
    src/LLVMCSRMatrixTest.cpp
    src/test_compiler.cpp
    src/TestBase.cpp
    src/TestEvalInitialConditions.cpp
    src/TestGetSetValues.cpp
    src/TestEvalReactionRates.cpp
    src/TestEvalModel.cpp
    src/TestRoadRunner.cpp
    src/GetBoundarySpeciesAmountTest.cpp
    src/TestCapabilities.cpp
    src/ConfigurableTest.cpp
    src/TestVariant.cpp
    src/tests.h
    src/cpplapack.h
    src/CSRMatrixTest.h
    src/LLVMCSRMatrixTest.h
    src/test_compiler.h
    src/TestBase.h
    src/TestEvalInitialConditions.h
    src/TestGetSetValues.h
    src/TestEvalReactionRates.h
    src/TestEvalModel.h
    src/TestRoadRunner.h
    src/GetBoundarySpeciesAmountTest.h
    src/TestCapabilities.h
    src/ConfigurableTest.h
    src/TestVariant.h
    )

if(LLVM_FLAGS)
  set_source_files_properties(${rr_analysis_src} PROPERTIES COMPILE_FLAGS ${LLVM_FLAGS})
endif()

add_executable( ${target} ${rr_analysis_src} )

# The additional required flags are set at global level. Here
# we only need to tell it about static rr.
target_compile_definitions(${target} PRIVATE STATIC_RR)

target_link_libraries(${target} PRIVATE unit_test-static roadrunner-static )
target_include_directories(${target} PRIVATE
        src
        ${INCLUDE_DIRECTORIES}
        )

if (BUILD_PYTHON)
    target_link_libraries (${target} PRIVATE ${PYTHON_LIBRARIES})
endif(BUILD_PYTHON)

install (TARGETS ${target}
    DESTINATION bin
    COMPONENT testing
    )

install(
    FILES run_cpp_tests.bat
    DESTINATION bin
    COMPONENT testing
    )



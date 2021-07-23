########################################################
#   Add the googletest framework as a subdirectory
#
add_subdirectory(googletest-master)
# we be naughty and use include_directories here to include googletest globally
include_directories(googletest-master/include)

# configure a text file in binary tree containing location of roadrunner root
# We use this root location to automatically find the testdir location (<rr_root>/test).
# This is not work in every case, so we fall back on testdir environment variable
set(RR_TEST_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
configure_file(RoadRunnerTestDirectory.h.in "${CMAKE_CURRENT_BINARY_DIR}/RoadRunnerTestDirectory.h" @ONLY)
configure_file(RoadRunnerTestDirectory.cpp.in "${CMAKE_CURRENT_BINARY_DIR}/RoadRunnerTestDirectory.cpp" @ONLY)

#file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/RoadRunnerRootLocation.txt" "${RR_ROOT}")
#message(STATUS "RoadRunner root location writen to binary tree \"${CMAKE_CURRENT_BINARY_DIR}/RoadRunnerRootLocation.txt\"")

############################################################
#   Function for creating roadrunner test binaries
#

# "CMAKE_CURRENT_SOURCE_DIR" changes in the different files were
# add_test_executable gets called. We can use this as a constant
set(ThisSourceDirectory "${CMAKE_CURRENT_SOURCE_DIR}")
set(ThisBinaryDirectory "${CMAKE_CURRENT_BINARY_DIR}")
set(MOCKUPS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/mockups")
set(RR_Python_wrapper_dir "${CMAKE_CURRENT_SOURCE_DIR}/../wrappers/Python/roadrunner")

#todo move these to the rr_root/cmake folder
# Adds a new executable cmake target to cmake.
# Links roadrunner-static and gtest.
# Parameters:
#   - TEST_TARGET: name of the binary
#   - OUT_VARIABLE: list variable to append BINARY. Creates the list if not exist.
#   - All further arguments are added to the binary as sources.
function(add_test_executable TEST_TARGET OUT_VARIABLE)
    #message(STATUS "Adding test ${TEST_TARGET} as part of ${OUT_VARIABLE}. Source Files: ARGN: ${ARGN}" )
    add_executable(${TEST_TARGET} ${ARGN} $<TARGET_OBJECTS:rr-mockups>
            ${ThisSourceDirectory}/TestModelFactory)

    target_include_directories(
            ${TEST_TARGET} PRIVATE
            "${ThisSourceDirectory}"
            "${ThisBinaryDirectory}"
            "${MOCKUPS_DIRECTORY}"
            "${RR_Python_wrapper_dir}" # for PyUtils in tests that need it
    )

    target_link_libraries(${TEST_TARGET} PRIVATE
            roadrunner_c_api
            roadrunner-static
            rr-mockups
            gtest gtest_main gmock gmock_main
            )

    # This causes a weird build error whereby it is no longer possible
    # to name a class the name as the binary filename. However, since w
    # link to roadrunner-static, the flags needed should be transitive
    #    add_compile_definitions(${TEST_TARGET} PRIVATE STATIC_RR)

    add_dependencies(${TEST_TARGET} roadrunner-static gtest gtest_main gmock gmock_main rr-mockups)
    set_target_properties(${TEST_TARGET} PROPERTIES LINKER_LANGUAGE CXX)

    # Add to ctest.
    set(TEST_ENV_VARS "testdir=${RR_ROOT}/test" "CTEST_OUTPUT_ON_FAILURE=TRUE")
    gtest_discover_tests(
            ${TEST_TARGET}
            DISCOVERY_TIMEOUT 500
            PROPERTIES
            TIMEOUT 500
            ENVIRONMENT "${TEST_ENV_VARS}"
    )

    set_target_properties(${TEST_TARGET} PROPERTIES ENVIRONMENT
            "testdir=${RR_ROOT}/test")


    if (WIN32)
        add_definitions(/bigobj)
    endif ()

    # helpful for debugging this function
    # message(STATUS "OUT_VARIABLE; ${${OUT_VARIABLE}} ${OUT_VARIABLE} OUT_VARIABLE"  )

    if ("${${OUT_VARIABLE}}" STREQUAL "")
        set(${OUT_VARIABLE} "${TEST_TARGET}" PARENT_SCOPE)
    else ()
        set(${OUT_VARIABLE} "${${OUT_VARIABLE}}" "${TEST_TARGET}" PARENT_SCOPE)
    endif ()

endfunction()


# Add a swig target to cmake build system
# Arguments
# ----------
#   SWIG_TARGET_NAME: What to call the swig target
#   SWIG_INTERFACE_FILE one or more swig interface files
#   Any other argument to this function is assumed to be source files
#   to be added to the swig target.
macro(add_swig_executable SWIG_TARGET_NAME SWIG_INTERFACE_FILE)
    # no harm in finding python here
    find_package(Python COMPONENTS Interpreter Development NumPy REQUIRED)
    find_package(SWIG 4.0.2 REQUIRED
            COMPONENTS python
            )
    include(UseSWIG)

    set_source_files_properties(${SWIG_INTERFACE_FILE}
            PROPERTIES
            CPLUSPLUS ON
            SWIG_FLAGS
            # turns on automatic annotation of python arg and return types
            # BUT does so in c++ types. Therefore I'm not sure how much value this adds.
            "-py3"
            )

    swig_add_library(${SWIG_TARGET_NAME}
            TYPE MODULE
            LANGUAGE python
            SOURCES ${SWIG_INTERFACE_FILE} ${ARGN}
            )

    #    set(CMAKE_SWIG_FLAGS "-py3 -castmode -keyword")
    set_target_properties(${SWIG_TARGET_NAME} PROPERTIES
            SWIG_USE_TARGET_INCLUDE_DIRECTORIES TRUE
            )

    set(thisDir ${CMAKE_CURRENT_SOURCE_DIR})
    target_include_directories(${SWIG_TARGET_NAME} PUBLIC
            ${thisDir}
            "${RR_ROOT}/wrappers/Python/include"
            "${RR_ROOT}/wrappers/Python/roadrunner"
            $<TARGET_PROPERTY:roadrunner-static,INCLUDE_DIRECTORIES>
            $<TARGET_PROPERTY:roadrunner_python,INCLUDE_DIRECTORIES>
            )
    target_link_libraries(${SWIG_TARGET_NAME} PUBLIC
            rr-mockups
            roadrunner-static
            Python::Python # remember, includes will just come with linking Python::Python
            Python::NumPy  # remember, includes will just come with linking Python::NumPy
            )

    message(STATUS "Adding swig executable: ${SWIG_TARGET_NAME}")
    message(STATUS "Swig target \"${SWIG_TARGET_NAME}\" will be copied to \"${RR_PYTHON_TESTING_BUILD_PREFIX}\" in the build tree after the target is built")
    set(SWIG_GENERATED_PYTHON_FILE "${CMAKE_CURRENT_BINARY_DIR}/${SWIG_TARGET_NAME}.py")
    message(STATUS "Swig generated .py for target \"${SWIG_TARGET_NAME}\" will be copied to \"${RR_PYTHON_TESTING_BUILD_PREFIX}\" in the build tree after the target is built")

    # copy Python files to build tree post build
    add_custom_command(
            TARGET ${SWIG_TARGET_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:${SWIG_TARGET_NAME}> "${RR_PYTHON_TESTING_BUILD_PREFIX}"
    )
    add_custom_command(
            TARGET ${SWIG_TARGET_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy "${SWIG_GENERATED_PYTHON_FILE}" "${RR_PYTHON_TESTING_BUILD_PREFIX}"
    )

    # todo "install" commands

endmacro()

#########################################################
#   Build the TestModelFactory
#
message(STATUS "RR_PYTHON_TESTING_BUILD_PREFIX: ${RR_PYTHON_TESTING_BUILD_PREFIX}")

if (BUILD_PYTHON AND BUILD_TESTS)
    message(STATUS "Building TestModelFactory")
    add_swig_executable(TestModelFactory
            "${CMAKE_CURRENT_SOURCE_DIR}/TestModelFactory.i"
            "${CMAKE_CURRENT_SOURCE_DIR}/TestModelFactory.h"
            "${CMAKE_CURRENT_SOURCE_DIR}/TestModelFactory.cpp"
            "../wrappers/Python/roadrunner/PyUtils"
            )


    # configure Python files to binary tree
    configure_file(
            "${CMAKE_CURRENT_SOURCE_DIR}/TestModelFactoryTests.py"
            "${RR_PYTHON_TESTING_BUILD_PREFIX}/TestModelFactoryTests.py" COPYONLY)
    configure_file(
            "${CMAKE_CURRENT_SOURCE_DIR}/RoadRunnerTest.py"
            "${RR_PYTHON_TESTING_BUILD_PREFIX}/RoadRunnerTest.py" COPYONLY)

    # install python test files to install tree
    install(TARGETS TestModelFactory DESTINATION "${RR_PYTHON_TESTING_INSTALL_PREFIX}")
    install(
            FILES
            "${RR_PYTHON_TESTING_BUILD_PREFIX}/TestModelFactory.py"
            "${RR_PYTHON_TESTING_BUILD_PREFIX}/TestModelFactoryTests.py"
            DESTINATION
            "${RR_PYTHON_TESTING_INSTALL_PREFIX}"
    )
endif ()


#########################################################
#   Add Tests
#

# Provide these sources to every test executable
set(
        SharedTestFiles
        # path *must* be absolute to work in cmake functions/macros
        "${CMAKE_CURRENT_BINARY_DIR}/RoadRunnerTestDirectory.h"
        "${CMAKE_CURRENT_BINARY_DIR}/RoadRunnerTestDirectory.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/test_util.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/test_util.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/RoadRunnerTest.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/RoadRunnerTest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/TestModelFactory.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/TestModelFactory.cpp" # need explicit extensions here (for some reason)
)

add_subdirectory(mockups)
add_subdirectory(c_api_core)
add_subdirectory(c_api_rrtests)
add_subdirectory(model_analysis)
add_subdirectory(model_editing)
add_subdirectory(sbml_features)
add_subdirectory(state_saving)
add_subdirectory(sundials-tests)
add_subdirectory(python)
add_subdirectory(cxx_api_tests)

#List the semantic_STS last, since it takes much longer to run than the others.
# (fyi, build the run-ctest-exclude-sbml-test-suite target - see below)
add_subdirectory(semantic_STS)


#######################################################################
#   Add Python unit tests to ctest. Assumes availability of pytest (pip install pytest)
#
if (BUILD_PYTHON AND BUILD_TESTS)
    function(add_pytest_to_ctest TestName TestSourceFile)
        add_test(
                NAME ${TestName}
                COMMAND Python::Interpreter -m pytest ${TestSourceFile}
                WORKING_DIRECTORY "${RR_PYTHON_TESTING_BUILD_PREFIX}"
        )
        set_tests_properties(${TestName} PROPERTIES ENVIRONMENT "PYTHONPATH=${PYTHON_PACKAGE_SITE_DIR_BUILD_TREE}:$PYTHONPATH")
    endfunction()


    add_pytest_to_ctest(PyAutomaticMoietyConservationAnalysisTests AutomaticMoietyConservationAnalysisTests.py)
    add_pytest_to_ctest(PyForwardSensitivitySolverTests ForwardSensitivitySolverTests.py)
    add_pytest_to_ctest(PyIntegratorTests IntegratorTests.py)
    add_pytest_to_ctest(Pypython_api_tests python_api_tests.py)
    add_pytest_to_ctest(PySettingTests SettingTests.py)
    add_pytest_to_ctest(PySteadyStateSolverTests SteadyStateSolverTests.py)
    add_pytest_to_ctest(PyStructuralAnalysisTests StructuralAnalysisTests.py)
    add_pytest_to_ctest(PyTestModelFactoryTests TestModelFactoryTests.py)

endif ()

#########################################################
#   Install tests and create wrapper for ctest command
#
install(TARGETS ${test_targets})

set(ctest_args "--build-config ${CMAKE_BUILD_TYPE} --output-on-failure --parallel 8 --progress --verbose --extra-verbose ${CMAKE_BINARY_DIR}")

# Creates a custom target which runs all tests
add_custom_target(
        run-ctest
        ctest ${ctest_args}
)
add_dependencies(run-ctest ${test_targets})

# Creates a custom target which runs all tests except the sbml test suite
add_custom_target(
        run-ctest-exclude-sbml-test-suite
        ctest ${ctest_args} --exclude-regex SbmlTestSuite
)
add_dependencies(run-ctest-exclude-sbml-test-suite ${test_targets})

# Creates a custom target which runs only the sbml tests suite
add_custom_target(
        run-ctest-only-sbml-test-suite
        ctest ${ctest_args} --tests-regex SbmlTestSuite
)

add_dependencies(run-ctest-only-sbml-test-suite ${test_targets})





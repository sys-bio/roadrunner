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

set(SBMLTestSuiteRoot "${CMAKE_CURRENT_SOURCE_DIR}/sbml-test-suite")
set(SBMLTestSuiteSemantic "${SBMLTestSuiteRoot}/semantic")
set(SBMLTestSuiteStochastic "${SBMLTestSuiteRoot}/stochastic")

# Adds a new executable cmake target to cmake. Links to roadrunner via the c api. Links gtest.
# Parameters:
#   - TEST_TARGET: name of the binary
#   - OUT_VARIABLE: list variable to append BINARY. Creates the list if not exist.
#   - All further arguments are added to the binary as sources.
function(add_test_executable TEST_TARGET OUT_VARIABLE)
    #message(STATUS "Adding test ${TEST_TARGET} as part of ${OUT_VARIABLE}. Source Files: ARGN: ${ARGN}" )
    add_executable(${TEST_TARGET} ${ARGN} $<TARGET_OBJECTS:rr-mockups>
            ${ThisSourceDirectory}/TestModelFactory)

    if (TEST_TARGET MATCHES ".*plugin.*")
        set(TEST_PLUGIN_INCLUDE "$<TARGET_PROPERTY:telplugins_c_api,INCLUDE_DIRECTORIES>")
        set(TEST_PLUGIN_LIB "telplugins_c_api")
    else()
        set(TEST_PLUGIN_LIB "roadrunner_c_api")
    endif()

    target_include_directories(
            ${TEST_TARGET} PRIVATE
            "${ThisSourceDirectory}"
            "${ThisBinaryDirectory}"
            "${MOCKUPS_DIRECTORY}"
            "${TEST_PLUGIN_INCLUDE}"
            ${SBMLTestSuiteRoot}
            #            "${RR_Python_wrapper_dir}" # for PyUtils in tests that need it
    )

    target_link_libraries(${TEST_TARGET} PRIVATE
            #            rr-mockups
            ${TEST_PLUGIN_LIB}
            gtest gtest_main gmock gmock_main
            )

    # Add root of sbml test suite to all test targets.
    target_compile_definitions(${TEST_TARGET} PRIVATE
            "SBMLTestSuiteRoot=\"${SBMLTestSuiteRoot}\""
            "SBMLTestSuiteSemantic=\"${SBMLTestSuiteSemantic}\""
            "SBMLTestSuiteStochastic=\"${SBMLTestSuiteStochastic}\""
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
macro(add_rr_test_swig_library SWIG_TARGET_NAME SWIG_INTERFACE_FILE)
    # no harm in finding python here
    find_package(Python COMPONENTS Interpreter Development NumPy REQUIRED)
    find_package(SWIG 4.0.1 REQUIRED
            COMPONENTS python
            )
    include(UseSWIG)

    set_source_files_properties(${SWIG_INTERFACE_FILE}
            PROPERTIES
            CPLUSPLUS ON
            SWIG_FLAGS
            # turns on automatic annotation of python arg and return types
            # BUT does so in c++ types. Therefore I'm not sure how much value this adds.
            -py3
            -threads
            -doxygen
            )

    swig_add_library(${SWIG_TARGET_NAME}
            TYPE MODULE
            LANGUAGE python
            SOURCES ${SWIG_INTERFACE_FILE} ${ARGN}
            )

    set_target_properties(
            ${SWIG_TARGET_NAME} PROPERTIES

            # On windows with the Visual Studio CMake generators, the pyd file
            # is put into a configuration directory (release / debug etc.)
            # however we use the makefile way of doing things and put them in to
            # one directory on all systems.
            RUNTIME_OUTPUT_DIRECTORY "${RR_PYTHON_TESTING_BUILD_PREFIX}"
            RUNTIME_OUTPUT_DIRECTORY_RELEASE "${RR_PYTHON_TESTING_BUILD_PREFIX}"
            RUNTIME_OUTPUT_DIRECTORY_DEBUG "${RR_PYTHON_TESTING_BUILD_PREFIX}"
            RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${RR_PYTHON_TESTING_BUILD_PREFIX}"

            LIBRARY_OUTPUT_DIRECTORY "${RR_PYTHON_TESTING_BUILD_PREFIX}"
            LIBRARY_OUTPUT_DIRECTORY_DEBUG "${RR_PYTHON_TESTING_BUILD_PREFIX}"
            LIBRARY_OUTPUT_DIRECTORY_RELEASE "${RR_PYTHON_TESTING_BUILD_PREFIX}"
            LIBRARY_OUTPUT_DIRECTORY_RELWITHDEBINFO "${RR_PYTHON_TESTING_BUILD_PREFIX}"

            ARCHIVE_OUTPUT_DIRECTORY "${RR_PYTHON_TESTING_BUILD_PREFIX}"
            ARCHIVE_OUTPUT_DIRECTORY_RELEASE "${RR_PYTHON_TESTING_BUILD_PREFIX}"
            ARCHIVE_OUTPUT_DIRECTORY_DEBUG "${RR_PYTHON_TESTING_BUILD_PREFIX}"
            ARCHIVE_OUTPUT_DIRECTORY_RELWITHDEBINFO "${RR_PYTHON_TESTING_BUILD_PREFIX}"

            # make the command "target_include_directories" work with swig targets
            SWIG_USE_TARGET_INCLUDE_DIRECTORIES TRUE
            OUTPUT_NAME ${SWIG_TARGET_NAME}
    )

    target_include_directories(${SWIG_TARGET_NAME} PUBLIC
            "${ThisSourceDirectory}" # test root
            "${CMAKE_CURRENT_SOURCE_DIR}"
            "${RR_ROOT}/wrappers/Python/include"
            "${RR_ROOT}/wrappers/Python/roadrunner"
            $<TARGET_PROPERTY:roadrunner,INCLUDE_DIRECTORIES>
            "${RR_DEPENDENCIES_INSTALL_PREFIX}/include/rr-libstruct"
            "${Python_INCLUDE_DIRS}"
            "${Python_NumPy_INCLUDE_DIRS}"
            $<TARGET_PROPERTY:roadrunner_python,INCLUDE_DIRECTORIES>
            )
    if (NOT APPLE)
        target_link_libraries(${SWIG_TARGET_NAME} PUBLIC
                rr-mockups
                roadrunner-static
                Python::Python # remember, includes will just come with linking Python::Python
                Python::NumPy  # remember, includes will just come with linking Python::NumPy
                )
    else ()

        set_target_properties(${SWIG_TARGET_NAME}
                PROPERTIES
                LINK_FLAGS "-undefined dynamic_lookup"
                INSTALL_RPATH "@loader_path/../../lib;@loader_path/../../bin;@loader_path/;@loader_path/..@loader_path/../.."
                INSTALL_NAME_DIR "@rpath"
                )
        swig_link_libraries(${SWIG_TARGET_NAME} PUBLIC
                rr-mockups
                roadrunner-static
                # Since we are using "-undefined dynamic_lookup"
                # we do not need to link to Python here - they will
                # be found at runtime.
                #Python::Python
                #Python::NumPy
                )
    endif ()

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

    # install python test files to install tree
    install(TARGETS ${SWIG_TARGET_NAME} DESTINATION "${RR_PYTHON_TESTING_INSTALL_PREFIX}")
    install(
            FILES
            "${RR_PYTHON_TESTING_BUILD_PREFIX}/${SWIG_TARGET_NAME}.py"
            DESTINATION
            "${RR_PYTHON_TESTING_INSTALL_PREFIX}/"
    )
endmacro()

#########################################################
#   Build the TestModelFactory
#
message(STATUS "RR_PYTHON_TESTING_BUILD_PREFIX: ${RR_PYTHON_TESTING_BUILD_PREFIX}")

if (BUILD_PYTHON AND BUILD_TESTS)
    message(STATUS "Building TestModelFactory")
    add_rr_test_swig_library(TestModelFactory
            "${CMAKE_CURRENT_SOURCE_DIR}/TestModelFactory.i"
            "${CMAKE_CURRENT_SOURCE_DIR}/TestModelFactory.h"
            "${CMAKE_CURRENT_SOURCE_DIR}/TestModelFactory.cpp"
            "../wrappers/Python/roadrunner/PyUtils.h"
            "../wrappers/Python/roadrunner/PyUtils.cpp"
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
        # todo make this its own binary and link it.
        "${CMAKE_CURRENT_SOURCE_DIR}/sbml-test-suite/STS.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/sbml-test-suite/STS.h"
)
# doesn't actually contain tests, just an interface into the STS
# add_subdirectory(sbml-test-suite)
add_subdirectory(mockups)
# add_subdirectory(c_api_core)
# add_subdirectory(llvm_tests)
# add_subdirectory(c_api_rrtests)
# add_subdirectory(model_analysis)
# add_subdirectory(model_editing)
# add_subdirectory(sbml_features)
# add_subdirectory(sundials-tests)
# add_subdirectory(cxx_api_tests)
add_subdirectory(serialization)
# add_subdirectory(PerformanceTests)
# add_subdirectory(python)

if (WIN32 AND BUILD_RR_PLUGINS)
   set(
        SharedPluginTestFiles
        # path *must* be absolute to work in cmake functions/macros
        "${CMAKE_CURRENT_BINARY_DIR}/RoadRunnerPluginDirectory.h"
        "${CMAKE_CURRENT_BINARY_DIR}/RoadRunnerPluginDirectory.cpp"
   )

   add_subdirectory(plugin_api)
   add_subdirectory(plugin_auto2000)
   add_subdirectory(plugin_levenberg_marquardt)
   add_subdirectory(plugin_monte_carlo_bs)
   add_subdirectory(plugin_nelder_mead)
   add_subdirectory(plugin_test_model)
endif() #BUILD_RR_PLUGINS

#List the semantic_STS last, since it takes much longer to run than the others.
# add_subdirectory(semantic_STS)


#######################################################################
#   Add Python unit tests to ctest. Assumes availability of pytest (pip install pytest)
#
if (BUILD_PYTHON AND BUILD_TESTS)

    set(PyTestTests)
    macro(add_pytest_to_ctest TestName TestSourceFile)
        # make sure file is up to date
        # run pytest with -rP to always capture stdout regardless of pass/fail
        add_test(
                NAME ${TestName}
                COMMAND Python::Interpreter -m pytest -rPs ${TestSourceFile}
                WORKING_DIRECTORY "${RR_PYTHON_TESTING_BUILD_PREFIX}"
        )
        set_tests_properties(${TestName} PROPERTIES ENVIRONMENT "PYTHONPATH=${PYTHON_PACKAGE_SITE_DIR_BUILD_TREE}:$PYTHONPATH")
        set(PyTestTests ${PyTestTests} ${TestName})
    endmacro()

    #For some reason, the new build of llvm makes importing TestModelFactory fail on MacOS (and causes Python to crash entirely!)
    if (NOT APPLE)
        add_pytest_to_ctest(python_tests_AutomaticMoietyConservationAnalysisTests "${RR_PYTHON_TESTING_BUILD_PREFIX}/test_automaticMoietyConservationAnalysis.py")
        add_pytest_to_ctest(python_tests_ForwardSensitivitySolverTests "${RR_PYTHON_TESTING_BUILD_PREFIX}/test_forwardSensitivitySolver.py")
        add_pytest_to_ctest(python_tests_IntegratorTests "${RR_PYTHON_TESTING_BUILD_PREFIX}/test_integrator.py")
        add_pytest_to_ctest(python_tests_JacobianTests "${RR_PYTHON_TESTING_BUILD_PREFIX}/test_jacobian.py")
        add_pytest_to_ctest(python_tests_TestModelFactoryTests "${RR_PYTHON_TESTING_BUILD_PREFIX}/test_testModelFactory.py")
        add_pytest_to_ctest(python_tests_NamedArrayTests "${RR_PYTHON_TESTING_BUILD_PREFIX}/test_namedArray.py")
        add_pytest_to_ctest(python_tests_Python_api_tests "${RR_PYTHON_TESTING_BUILD_PREFIX}/test_python_api.py")
        add_pytest_to_ctest(python_tests_SteadyStateSolverTests "${RR_PYTHON_TESTING_BUILD_PREFIX}/test_steadyStateSolver.py")
        add_pytest_to_ctest(python_tests_StructuralAnalysisTests "${RR_PYTHON_TESTING_BUILD_PREFIX}/test_structuralAnalysis.py")
        add_pytest_to_ctest(python_tests_RoadRunnerMap "${RR_PYTHON_TESTING_BUILD_PREFIX}/test_roadrunner_map.py")
    endif()
    #add_pytest_to_ctest(python_tests_PickleTests "${RR_PYTHON_TESTING_BUILD_PREFIX}/test_pickle.py")
    #add_pytest_to_ctest(python_tests_RRTests "${RR_PYTHON_TESTING_BUILD_PREFIX}/test_rrtests.py")
    #add_pytest_to_ctest(python_tests_SettingTests "${RR_PYTHON_TESTING_BUILD_PREFIX}/test_settings.py")
    #add_pytest_to_ctest(python_tests_RunStochasticTestSuite "${RR_PYTHON_TESTING_BUILD_PREFIX}/test_runStochasticTestSuite.py")
    
    message(STATUS "PyTestTests ${PyTestTests}")

    add_custom_target(
            run-pytest-tests
            ctest --output-on-failure --verbose --extra-verbose --tests-regex python_tests*
    )
endif ()

#########################################################
#   Install tests and create wrapper for ctest command
#
install(TARGETS ${test_targets})

MESSAGE(STATUS "CMAKE_BINARY_DIR ${CMAKE_BINARY_DIR}")
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







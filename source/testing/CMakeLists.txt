###########################################################
#   Roadrunner testing
#

# add the googletest subdirectory to the build
#add_subdirectory(googletest-release-1.10.0)

set(target cxx_api_tests)

set(tests
        tests/base
        tests/sbml_test_suite
        tests/steady_state
        tests/stoichiometric
        tests/model_editing
        tests/state_saving
        tests/other_tests
        )

add_executable(${target}
        main
        src/Args
        src/rrTestUtils
        ${tests}
        )

#set(CMAKE_CXX_FLAGS "-fno-rtti ${CMAKE_CXX_FLAGS}")

add_compile_definitions(${target} STATIC_RR)

target_include_directories(${target} PRIVATE
        src
        ${UNIT_TEST_INCLUDE_DIR}
        ${INCLUDE_DIRECTORIES}
        )

target_link_libraries(${target} PRIVATE
        roadrunner-static
        ${TEST_TARGETS}
        #        ${LINK_LIBRARIES}
        )

add_dependencies(${target} unit_test-static roadrunner-static)
if (UNIX)
    target_compile_options(${target} PRIVATE -frtti)
endif ()
set(resultsDir "${CMAKE_CURRENT_SOURCE_DIR}/reports")
set(resultsFile "${resultsDir}/cpp_api_tests-${CMAKE_CXX_COMPILER_ID}-${CMAKE_CXX_COMPILER_VERSION}")

# TODO move the models directory
set(sbmlModelsDir "${CMAKE_CURRENT_SOURCE_DIR}/models")
set(compilerFolder "${CMAKE_CURRENT_SOURCE_DIR}/compilers")
set(supportCodeFolder "${CMAKE_CURRENT_SOURCE_DIR}/rr_support")

if (UNIX)
    set(compilerFolder "")
endif ()

Message("SupportCodeFolder is: " ${supportCodeFolder})
# Call like this -mr:/rrl/models -rr:/rrl/wiki/reports/file -tr:/rrTemp -sSupportCodeFolder -lCompilerFolder
#add_custom_command(
#    OUTPUT ${resultsFile}
#    COMMAND ${target} -m${sbmlModelsDir} -l${compilerFolder} -s${supportCodeFolder} -r${resultsFile} -v
#    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
#    COMMENT "=========== Generating CPP-wrappers Wiki reports =============="
#    VERBATIM
#    )
#
set(CMD "${target} -m${sbmlModelsDir} -l${compilerFolder} -s${supportCodeFolder} -r${resultsFile} -v")
MESSAGE(STATUS "CMD ${CMD}")
add_custom_target(cppTestingReport ALL
        COMMAND ${CMD}
        COMMENT "=========== Generating CPP-API Wiki reports =============="
        COMMENT ${CMD}
        COMMENT "Executing ${target} -m${sbmlModelsDir} -l${compilerFolder} -s${supportCodeFolder} -r${resultsFile} -v"
        )

#add_dependencies(cppTestingReport cxx_api_tests)
#
#install(TARGETS ${target}
#        DESTINATION bin
#        COMPONENT testing
#        )
#
#install(
#        FILES run_cpp_tests.bat
#        DESTINATION bin
#        COMPONENT testing
#)
#


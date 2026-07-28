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
            DISCOVERY_MODE PRE_TEST
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


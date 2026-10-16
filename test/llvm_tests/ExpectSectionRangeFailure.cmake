# Runs the LLJitSectionRange invariant test with the contiguous section range turned off and
# succeeds only if the test fails (an assertion failure or a crash). A test that passes here
# could not detect the bug it is meant to guard against.
#
# usage: cmake -DTEST_EXE=<path to test_llvm_LLJitSectionRange> -P ExpectSectionRangeFailure.cmake

set(testName "LLJitSectionRange.SectionsStayWithinTwoGiB")
set(ENV{RR_LLJIT_RESERVE} "0")

execute_process(
        COMMAND "${TEST_EXE}" "--gtest_filter=${testName}"
        RESULT_VARIABLE result
        OUTPUT_VARIABLE output
        ERROR_VARIABLE errors)
message("${output}")
message("${errors}")

string(FIND "${output}" "RUN      ] ${testName}" started)
if (started EQUAL -1)
    message(FATAL_ERROR "${testName} did not start, so its failure cannot be trusted (result: ${result})")
endif ()

if ("${result}" STREQUAL "0")
    message(FATAL_ERROR "${testName} passed with RR_LLJIT_RESERVE=0; it cannot detect sections placed far apart")
endif ()
message(STATUS "${testName} failed with RR_LLJIT_RESERVE=0 as expected (result: ${result})")

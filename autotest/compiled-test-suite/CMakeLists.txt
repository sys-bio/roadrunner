# == PREAMBLE ================================================
# * Licensed under the Apache License, Version 2.0; see README

# == FILEDOC =================================================

# @file autotest/_CMakeLists.txt
# @author JKM(0u812)
# @date 11/17/2015
# @copyright Apache License, Version 2.0
# @brief Compiled unit tests

if(WIN32)
  set(RUN_TEST_SUITE_EXE_NAME "run_test_suite.exe")
else()
  set(RUN_TEST_SUITE_EXE_NAME "run_test_suite")
endif()

add_test(NAME CompiledTestSuite
  COMMAND ${CMAKE_BINARY_DIR}/bin/${RUN_TEST_SUITE_EXE_NAME} -c llvm -m ${CMAKE_SOURCE_DIR}/models/sbml-test-suite -r results.xml >log 2>&1)
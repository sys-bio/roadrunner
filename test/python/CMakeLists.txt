# == PREAMBLE ================================================
# * Licensed under the Apache License, Version 2.0; see README

# == FILEDOC =================================================

# @file autotest/CMakeLists.txt
# @author JKM(0u812)
# @date 07/01/2015
# @copyright Apache License, Version 2.0
# @brief Unit tests
# @details Tests can be run via ctest -VV

if(BUILD_PYTHON)
  # performance benchmark (adapted from published benchmark in bioinformatics, 2015)
  file(COPY python-benchmark-bioinf DESTINATION .)

  # update when source changes
  configure_file(python-benchmark-bioinf/rr_bench_ode.py
    ${CMAKE_CURRENT_BINARY_DIR}/python-benchmark-bioinf/rr_bench_ode.py COPYONLY)

  # update when source changes
  configure_file(python-unit-tests/run.py
    ${CMAKE_CURRENT_BINARY_DIR}/python-unit-tests/run.py COPYONLY)

  add_test(NAME PythonUnitTests
    COMMAND ${PYTHON_EXECUTABLE} run.py
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/python-unit-tests)
  set_tests_properties(PythonUnitTests PROPERTIES ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}/lib/site-packages:$PYTHONPATH")
endif()

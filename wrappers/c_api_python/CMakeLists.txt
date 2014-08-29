CMAKE_MINIMUM_REQUIRED(VERSION 2.6.3 FATAL_ERROR)
PROJECT(rrWrappers)

if(BUILD_TEST_TOOLS)
add_subdirectory(Testing)

if(INSTALL_EXAMPLES)
add_subdirectory(Examples)
endif()
endif(BUILD_TEST_TOOLS)

#=============== INSTALL =====================
install(
    FILES
    rrPython.py test.py test3.py README.txt
    DESTINATION c_api_python
    COMPONENT testing
    )

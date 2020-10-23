CMAKE_MINIMUM_REQUIRED(VERSION 2.6.3 FATAL_ERROR)
PROJECT(rrWrappers)


#=============== install =====================
install(
    FILES 
    rrRunTests.py
    rrTester.py
    DESTINATION c_api_python COMPONENT testing
)


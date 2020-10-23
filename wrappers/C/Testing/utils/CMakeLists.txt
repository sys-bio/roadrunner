CMAKE_MINIMUM_REQUIRED(VERSION 2.6.3 FATAL_ERROR)
PROJECT(RR_C_API_Tests)
include(${CMAKE_SOURCE_DIR}/cmake/SlicerMacroGetOperatingSystemArchitectureBitness.cmake)

message(STATUS "Testing/utils...")

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    execute_process(
        COMMAND ${CMAKE_CXX_COMPILER} --version
        OUTPUT_VARIABLE _compiler_output
        )
    string(REGEX MATCH "[0-9]\\.[0-9]\\.[0-9]" RR_COMPILER_VERSION ${_compiler_output})
	message(STATUS "C++ compiler version: ${RR_COMPILER_VERSION} [${CMAKE_CXX_COMPILER_ID}]")
endif()

SlicerMacroGetOperatingSystemArchitectureBitness("RR")

message(STATUS "bitness: ${RR_BITNESS}")
message(STATUS "bitness: ${RR_OS}")
message(STATUS "bitness: ${RR_ARCHITECTURE}")

message(STATUS ${CMAKE_SYSTEM})

configure_file(run_c_tests.sh.in run_c_tests.sh @ONLY)

if(UNIX)
	install(
		FILES ${CMAKE_CURRENT_BINARY_DIR}/run_c_tests.sh
		DESTINATION bin
		COMPONENT testing
		PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
		)
else()
	install(
		FILES run_c_tests.bat ${CMAKE_CURRENT_BINARY_DIR}/run_c_tests.sh
		DESTINATION bin
		COMPONENT testing
	)
endif()



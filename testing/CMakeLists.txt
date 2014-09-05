cmake_minimum_required(VERSION 2.8)
PROJECT(tests)

set(tests
    README.txt
    Test_1.txt
    ModelSourceTest.h
    ModelSourceTest.c
)
install(
    FILES ${tests}
    DESTINATION testing 	COMPONENT testing
)

file(GLOB datfiles "${CMAKE_CURRENT_SOURCE_DIR}/*.dat")
install(
    FILES ${datfiles}
    DESTINATION testing COMPONENT testing
)


file(GLOB xmlfiles "${CMAKE_CURRENT_SOURCE_DIR}/*.xml")
install(
    FILES ${xmlfiles}
    DESTINATION testing COMPONENT testing
)


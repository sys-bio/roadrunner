cmake_minimum_required(VERSION 2.8)
PROJECT(tests)

set(tests
    README.txt
    NOM_Test.dat
    LibStructTest.dat
    Test_1.dat
    Test_1.xml
    Test_2.dat
    Test_2.xml
    Test_3.dat
    Test_3.xml
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


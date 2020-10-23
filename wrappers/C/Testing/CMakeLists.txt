CMAKE_MINIMUM_REQUIRED(VERSION 2.6.3 FATAL_ERROR)
PROJECT(RR_C_API_Tests)

set(target CAPIRunTestSuite)

set(tests
        Suite_Core_Tests
        Suite_SBML_Tests
        Suite_NOM_Tests
        Suite_LibStruct_Tests
        Suite_Test_Model_Editing
        Suite_State_Saving_Tests
        Suite_Other_Tests
        Suite_TestModel.cpp
        Suite_TestModel.h
        )

add_executable(${target}
        main.cpp
        src/Args.cpp
        src/TestUtils.cpp
        src/rrSBMLTestSuiteSimulation_CAPI.cpp
        src/Args.h
        src/TestUtils.h
        src/rrSBMLTestSuiteSimulation_CAPI.h
        ${tests}
        )

target_compile_definitions(${target} PUBLIC
        STATIC_LIBSTRUCT
        STATIC_RR
        POCO_STATIC
        XML_STATIC
        POCO_NO_AUTOMATIC_LIBS
        )

target_link_libraries(${target} PRIVATE roadrunner_c_api unit_test-static roadrunner-static)
target_include_directories(${target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${INCLUDE_DIRECTORIES})


add_subdirectory(utils)

install(TARGETS ${target}
        DESTINATION bin COMPONENT testing
        )





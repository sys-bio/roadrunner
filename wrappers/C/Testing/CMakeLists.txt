CMAKE_MINIMUM_REQUIRED(VERSION 2.6.3 FATAL_ERROR)
PROJECT(RR_C_API_Tests)

set(target run_test_suite)

set(tests
    Suite_Core_Tests
    Suite_TestModel
    Suite_SBML_Tests
    Suite_NOM_Tests
    Suite_LibStruct_Tests
	Suite_State_Saving_Tests
    )

add_executable( ${target}
    main.cpp
    src/Args.cpp
    src/TestUtils
    src/rrSBMLTestSuiteSimulation_CAPI
    ${tests}
    )

set_property(TARGET ${target}
    PROPERTY  COMPILE_DEFINITIONS
    STATIC_LIBSTRUCT
    STATIC_RR
    POCO_STATIC
    XML_STATIC
    POCO_NO_AUTOMATIC_LIBS
    )

if(UNIX)
    set(staticLibPrefix ".a")
    set(sharedLibPrefix ".so")
endif()

target_link_libraries (${target}
    roadrunner_c_api
    unit_test-static
    PocoXML
    )

link_directories(
    ${THIRD_PARTY_INSTALL_FOLDER}/lib
    )

include_directories(
    "."
    ${RR_ROOT}
    ${THIRD_PARTY_FOLDER}/unit_test/src
${THIRD_PARTY_FOLDER}/rr-libstruct
)

add_subdirectory(utils)

install (TARGETS ${target}
    DESTINATION bin COMPONENT testing
    )





CMAKE_MINIMUM_REQUIRED(VERSION 2.6.3 FATAL_ERROR)
PROJECT(RR_CXX_API_TESTS)

set(target rr-sbml-benchmark)

set(tests
    tests/base
    tests/sbml_test_suite
    tests/steady_state
    tests/stoichiometric
    )

add_executable( ${target}
    main
    )

set_property(TARGET ${target}
    PROPERTY  COMPILE_DEFINITIONS
    LIBSBML_USE_CPP_NAMESPACE
    LIBSBML_STATIC
    STATIC_LIBSTRUCT
    STATIC_PUGI
    STATIC_RR
    STATIC_NLEQ1
    STATIC_NLEQ2
    )

link_directories(
    ${THIRD_PARTY_INSTALL_FOLDER}/lib
    )

include_directories(
    src
    ${RR_ROOT}
    ${THIRD_PARTY_INSTALL_FOLDER}/include/clapack
    )

if(UNIX)
    set(staticLibPrefix ".a")
    set(sharedLibPrefix ".so")
else()
    set(staticLibPrefix "")
    set(sharedLibPrefix "")
endif()

if(WIN32)
    target_link_libraries (${target}
        roadrunner-static
        unit_test-static${staticLibPrefix}
        )
endif()

if(UNIX)
    target_link_libraries (${target}
        roadrunner-static
        libunit_test-static${staticLibPrefix}
        lapack
        blas
        f2c
        dl
        )
endif()


install (TARGETS ${target}
    DESTINATION bin
    COMPONENT testing
    )



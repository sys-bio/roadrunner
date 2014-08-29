set(target roadrunner_c_api)

set(C_API_INCLUDE_ROOT "${RR_ROOT}/wrappers/C")

include_directories(${RR_GENERATED_HEADER_PATH}) #we have to do this to get the generated header

include_directories(
${C_API_INCLUDE_ROOT}
${RR_ROOT}/source
${THIRD_PARTY_INSTALL_FOLDER}/include
)

# TODO: remove all the deprecated definition when we have time...
#if(CMAKE_COMPILER_IS_GNUCXX)
if(UNIX)
    # should work with either gcc or clang
    add_definitions(-Wno-deprecated-declarations)
endif(UNIX)

#The C-api is built as a shared library..
add_library(${target} SHARED
    rrcStringList.cpp
    rrArrayList.cpp
    rrArrayListItem.cpp
    rrArrayListItemBase.cpp
    rrc_api.cpp
    rrc_logging_api.cpp
    rrc_nom_api.cpp
    rrc_libstruct_api.cpp
    rrc_utilities.cpp
    rrc_cpp_support.cpp
    rrcStringList.h
    rrArrayList.h
    rrArrayListItem.h
    rrArrayListItemBase.h
    rrc_api.h
    rrc_logging_api.h
    rrc_nom_api.h
    rrc_libstruct_api.h
    rrc_utilities.h
    rrc_cpp_support.h
)

#Link C_api dll statically to roadrunner and libsbml
if(WIN32)
    set_property(TARGET ${target}
        PROPERTY  COMPILE_DEFINITIONS
        EXPORT_RRC
        LIBSBML_STATIC
        LIBLAX_STATIC
        POCO_STATIC
        STATIC_LIBSTRUCT
        STATIC_RR
        )

    target_link_libraries ( ${target}
        roadrunner-static
        )

else() #Unix
    # OSX flags:
    #"${CMAKE_SHARED_LINKER_FLAGS} -Wl,-no_implicit_dylibs -undefined error")
    # OSX seems to link fine without this flag
    if(APPLE)
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-no_implicit_dylibs -undefined error")
    else()
        set(CMAKE_SHARED_LINKER_FLAGS "-Wl,--no-undefined")
    endif()

    set_property(TARGET ${target}
        PROPERTY  COMPILE_DEFINITIONS
        LIBSBML_STATIC
        LIBLAX_STATIC
        STATIC_LIBSTRUCT
        )

    target_link_libraries ( ${target}
        roadrunner-static
        rr-libstruct-static
        nleq-static
        sbml-static
        lapack
        blas
        f2c
        pthread
        )
endif()

add_subdirectory(Apps)
if(INSTALL_EXAMPLES)
    add_subdirectory(Examples)
endif()

if(BUILD_TEST_TOOLS)
    add_subdirectory(Testing)
endif()

#=============== INSTALL =====================
#Glob all header files
FILE (GLOB headers rrc*.h)

install(
    FILES
    ${headers}
    DESTINATION include/rr/C COMPONENT c_api
    )

install(
    TARGETS ${target}
    RUNTIME DESTINATION bin COMPONENT c_api
    ARCHIVE DESTINATION lib COMPONENT c_api
    LIBRARY DESTINATION lib COMPONENT c_api
    )


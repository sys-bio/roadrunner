# M Totte Karlsson
# J Kyle Medley

set(target telplugins_c_api)

#The C-api is built as a shared library..
add_library(${target} SHARED
        telplugins_c_api.cpp
        telplugins_properties_api.cpp
        telplugins_telluriumdata_api.cpp
        telplugins_matrix_api.cpp
        telplugins_utilities.cpp
        telplugins_logging_api.cpp

        telplugins_c_api.h
        telplugins_properties_api.h
        telplugins_telluriumdata_api.h
        telplugins_matrix_api.h
        telplugins_utilities.h
        telplugins_logging_api.h
        telplugins_types.h
        )

if(BUILD_PYTHON)
    add_custom_command(
        TARGET ${target} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory ${RR_PLUGINS_PYLIB_BUILD_PREFIX}
        COMMAND ${CMAKE_COMMAND} -E copy
        "$<TARGET_FILE:${target}>"
        "${RR_PLUGINS_PYLIB_BUILD_PREFIX}"
        COMMENT "Copying library to build directory for testing"
    )
endif() #BUILD_PYTHON

target_include_directories(${target}
        PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../../"
        )
target_include_directories(${target} PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/rrplugins/c>"
        )

# The static libraries are part of this shared library. Targets in this build tree
# still see them as dependencies; installed consumers do not.
target_link_libraries(${target}
        PUBLIC ${RRPLUGINS_COMMON_SHARED_LIBS}
        PRIVATE telplugins_core-static telplugins_math-static
        )
target_link_libraries(${target} INTERFACE
        "$<BUILD_INTERFACE:telplugins_core-static>"
        "$<BUILD_INTERFACE:telplugins_math-static>"
        )

# remove the "lib" prefix on some platforms
set_target_properties(${target} PROPERTIES PREFIX "")

add_library(rrplugins::${target} ALIAS ${target})

#Link C_api shared lib statically to roadrunner and libsbml
if (WIN32)
    target_compile_definitions(${target} PUBLIC "$<BUILD_INTERFACE:EXPORT_RRP>")
else ()
    if (APPLE)
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-no_implicit_dylibs")
        set_target_properties(${target} PROPERTIES INSTALL_RPATH "@loader_path;@loader_path/../roadrunner")
    else ()
        set(CMAKE_SHARED_LINKER_FLAGS "-Wl,--no-undefined")

        # add rpath for site-packages/rrplugins directory, and for the roadrunner libraries in the
        # install libdir (lib64 on some platforms) when this library is installed in lib
        set_target_properties(${target} PROPERTIES
                INSTALL_RPATH "$ORIGIN/./;$ORIGIN/../${CMAKE_INSTALL_LIBDIR}"
                )
    endif ()
endif ()

install(
        TARGETS ${target}
        EXPORT rrpluginsTargets
        RUNTIME DESTINATION bin COMPONENT rrplugins
        ARCHIVE DESTINATION lib COMPONENT rrplugins
        LIBRARY DESTINATION lib COMPONENT rrplugins
)

if (BUILD_PYTHON AND BUILD_RR_PLUGINS)
    install(TARGETS ${target} DESTINATION "${RR_PLUGINS_PYLIB_INSTALL_PREFIX}")
endif()

# Public headers
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/rrplugins/c"
        COMPONENT rrplugins
        FILES_MATCHING PATTERN "*.h")

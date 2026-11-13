# Debashish
# M Totte Karlsson
# J Kyle Medley

set(target telpluginsBaseClass)

set(sources
        telCPlugin.cpp
        telCPPPlugin.cpp
        telPlugin.cpp
        telVersionInfo.cpp

        telCPlugin.h
        telCPPPlugin.h
        telPlugin.h
        telVersionInfo.h
        )

# ** Shared lib ******

if (RR_PLUGINS_BUILD_SHARED_LIB)
    add_library(${target} SHARED ${sources})
    target_compile_definitions(${target} PUBLIC SHARED_PLUGIN_API)

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

    target_link_libraries(${target} PUBLIC
            telplugins_common-static
            )

    target_include_directories(${target} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${RR_PLUGINS_ROOT})

    # Install static library
    install(TARGETS ${target}
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT rrplugins
            ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT rrplugins
            )
    # disable the auto addition of "lib" onto target name
    set_target_properties(${target} PROPERTIES PREFIX "")
endif ()


# ** Static lib ******

if (RR_PLUGINS_BUILD_STATIC_LIB)
    add_library(${target}-static STATIC ${sources})
    add_library(rrplugins::${target}-static ALIAS ${target}-static)
    target_compile_definitions(${target}-static PUBLIC STATIC_PLUGIN_API)

    target_link_libraries(${target}-static PUBLIC
            telplugins_common-static
            )

    target_include_directories(${target}-static PUBLIC
            "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>"
            "$<BUILD_INTERFACE:${RR_PLUGINS_ROOT}>"
            "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
            "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/rrplugins/pluginBaseClass>"
            )

    # Install static library
    install(TARGETS ${target}-static
            EXPORT rrpluginsTargets
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT rrplugins
            ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT rrplugins
            )
    # disable the auto addition of "lib" onto target name
    set_target_properties(${target}-static PROPERTIES PREFIX "")
endif ()



# Public headers
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/rrplugins/pluginBaseClass"
        COMPONENT rrplugins
        FILES_MATCHING PATTERN "*.h")

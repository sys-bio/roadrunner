# Debashish
# M Totte Karlsson
# J Kyle Medley

set(target telpluginsBaseClass)

set(sources
        telCPlugin
        telCPPPlugin
        telPlugin
        telVersionInfo
        )

# ** Shared lib ******

if (RR_PLUGINS_BUILD_SHARED_LIB)
    add_library(${target} SHARED ${sources})
    target_compile_definitions(${target} PRIVATE EXPORT_CORE_API)

    target_link_libraries(${target} PUBLIC telplugins_common ${RRPLUGINS_COMMON_SHARED_LIBS})
    if (UNIX)
        set_target_properties(telpluginsBaseClass PROPERTIES INSTALL_RPATH "$ORIGIN/./")
    endif ()

    target_include_directories(${target} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${RR_PLUGINS_ROOT})

    # Install shared library
    install(TARGETS telpluginsBaseClass
            RUNTIME DESTINATION bin COMPONENT rrplugins
            LIBRARY DESTINATION lib COMPONENT rrplugins
            ARCHIVE DESTINATION lib COMPONENT rrplugins
            )

    install(TARGETS ${target} DESTINATION site-packages/rrplugins)
endif ()

# ** Static lib ******

if (RR_PLUGINS_BUILD_STATIC_LIB)
    add_library(${target}-static STATIC ${sources})
    target_compile_definitions(${target}-static PUBLIC STATIC_PLUGIN_API)

    target_link_libraries(${target}-static PUBLIC
            telplugins_common-static ${RRPLUGINS_COMMON_STATIC_LIBS})

    target_include_directories(${target}-static PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${RR_PLUGINS_ROOT})

    # Install static library
    install(TARGETS telpluginsBaseClass-static
            LIBRARY DESTINATION lib COMPONENT rrplugins
            ARCHIVE DESTINATION lib COMPONENT rrplugins
            )
endif ()



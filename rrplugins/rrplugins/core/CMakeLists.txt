# M Totte Karlsson
# J Kyle Medley
# Debashish Roy

set(target telplugins_core)

set(
        sources
        ${RR_PLUGINS_ROOT}/rrplugins/pluginBaseClass/telCPlugin
        ${RR_PLUGINS_ROOT}/rrplugins/pluginBaseClass/telCPPPlugin
        ${RR_PLUGINS_ROOT}/rrplugins/pluginBaseClass/telPlugin
        tel_api   #struct of roadrunner function pointer
        telPluginManager
        telVersionInfo
)

# ** Shared lib ******

if (RR_PLUGINS_BUILD_SHARED_LIB)
    add_library(${target} SHARED ${sources})

    target_compile_definitions(${target} PRIVATE EXPORT_CORE_API)

    target_link_libraries(${target} PUBLIC
            telplugins_common
            telpluginsBaseClass
            ${RRPLUGINS_COMMON_SHARED_LIBS}
            roadrunner_c_api
            roadrunner
            )

    target_include_directories(${target} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${RR_PLUGINS_ROOT})

    if (UNIX)
        set_target_properties(${target} PROPERTIES INSTALL_RPATH "$ORIGIN/./")
    endif ()

    # Install shared library
    install(TARGETS ${target}
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
            telplugins_common-static
            telpluginsBaseClass-static
            ${RRPLUGINS_COMMON_STATIC_LIBS}
            roadrunner_c_api
            roadrunner-static
            )
    target_include_directories(${target}-static PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${RR_PLUGINS_ROOT})

    # Install static library
    install(TARGETS ${target}-static
            LIBRARY DESTINATION lib COMPONENT rrplugins
            ARCHIVE DESTINATION lib COMPONENT rrplugins
            )
endif ()



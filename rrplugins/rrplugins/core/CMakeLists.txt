# M Totte Karlsson
# J Kyle Medley
# Debashish Roy

set(target telplugins_core)

set(
        sources
        ${RR_PLUGINS_ROOT}/rrplugins/pluginBaseClass/telCPlugin
        ${RR_PLUGINS_ROOT}/rrplugins/pluginBaseClass/telCPPPlugin
        ${RR_PLUGINS_ROOT}/rrplugins/pluginBaseClass/telPlugin
        ${RR_PLUGINS_ROOT}/rrplugins/common/telException
        tel_api   #struct of roadrunner function pointer
        telPluginManager
        telVersionInfo
)

# ** Shared lib ******

if (RR_PLUGINS_BUILD_SHARED_LIB)
    add_library(${target} SHARED ${sources})
    target_compile_definitions(${target} PUBLIC SHARED_PLUGIN_API)

    target_link_libraries(${target} PUBLIC
            telplugins_common-static
            telpluginsBaseClass-static
            telplugins_c_api
            )
    target_include_directories(${target} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${RR_PLUGINS_ROOT})

    # Install static library
    install(TARGETS ${target}
            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT rrplugins
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT rrplugins
            ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT rrplugins
            )

    # disable the auto addition of "lib" onto target name
    set_target_properties(${target} PROPERTIES PREFIX "")
endif ()

# ** Static lib ******

if (RR_PLUGINS_BUILD_STATIC_LIB)
    add_library(${target}-static STATIC ${sources})
    target_compile_definitions(${target}-static PUBLIC STATIC_PLUGIN_API)

    target_link_libraries(${target}-static PUBLIC
            telplugins_common-static
            telpluginsBaseClass-static
            roadrunner_c_api
            )
    target_include_directories(${target}-static PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${RR_PLUGINS_ROOT})

    # Install static library
    install(TARGETS ${target}-static
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT rrplugins
            ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT rrplugins
            )

    # disable the auto addition of "lib" onto target name
    set_target_properties(${target}-static PROPERTIES PREFIX "")
endif ()



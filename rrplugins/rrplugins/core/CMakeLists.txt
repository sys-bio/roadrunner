# M Totte Karlsson
# J Kyle Medley
# Debashish Roy

set(target telplugins_core)

set(
        sources
        telPluginManager.cpp
        telVersionInfo.cpp
        telplugins_cpp_support.cpp
        telAPIHandleManager.cpp

        tel_api.h   #struct of roadrunner function pointer
        telplugins_cpp_support.h
        telCoreExporter.h
        telPluginManager.h
        telVersionInfo.h
        telAPIHandleManager.h
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
            telpluginsBaseClass-static
            roadrunner_c_api
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



# M Totte Karlsson
# J Kyle Medley

set(target telplugins_common)

set(RRPLUGINS_COMMON_SOURCES
        telArrayList.cpp
        telArrayedParameter.cpp
        telConstants.cpp
        telException.cpp
        telFileName.cpp
        telIniFile.cpp
        telIniKey.cpp
        telIniSection.cpp
        telLogger.cpp
        telProperties.cpp
        telPropertyBase.cpp
        telStringBuilder.cpp
        telStringList.cpp
        telStringListContainer.cpp
        telStringUtils.cpp
        telTelluriumData.cpp
        telUtils.cpp
        telArrayList.h
        telArrayedParameter.h
        telConstants.h
        telException.h
        telFileName.h
        telIniFile.h
        telIniKey.h
        telIniSection.h
        telLogger.h
        telProperties.h
        telPropertyBase.h
        telStringBuilder.h
        telStringList.h
        telStringListContainer.h
        telStringUtils.h
        telTelluriumData.h
        telUtils.h)


# ** Shared lib ******
if (RR_PLUGINS_BUILD_SHARED_LIB)
    add_library(${target} SHARED ${RRPLUGINS_COMMON_SOURCES})
    target_compile_definitions(${target} PRIVATE EXPORT_COMMON_API)

    target_link_libraries(${target} PUBLIC
            ${RRPLUGINS_COMMON_SHARED_LIBS}
            roadrunner roadrunner_c_api
            )
    target_include_directories(${target}
            PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
            "$<TARGET_PROPERTY:roadrunner-static,INCLUDE_DIRECTORIES>"
            )

    if (UNIX)
        set_target_properties(${target} PROPERTIES INSTALL_RPATH "$ORIGIN/./")
    endif ()

    # Install shared library
    install(TARGETS ${target}
            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT rrplugins
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT rrplugins
            ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT rrplugins
            )

    install(TARGETS ${target} DESTINATION site-packages/rrplugins)
endif ()

# ** Static lib ******
if (RR_PLUGINS_BUILD_STATIC_LIB)
    add_library(${target}-static STATIC ${RRPLUGINS_COMMON_SOURCES})
    target_compile_definitions(${target}-static PUBLIC STATIC_PLUGIN_API)

    target_link_libraries(${target}-static PUBLIC $<TARGET_PROPERTY:${target},LINK_LIBRARIES>)
    target_include_directories(${target}-static PUBLIC $<TARGET_PROPERTY:${target},INCLUDE_DIRECTORIES>)

    # Install
    install(TARGETS ${target}-static
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT rrplugins
            ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT rrplugins
            )

endif ()

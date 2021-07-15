# M Totte Karlsson
# J Kyle Medley

set(target telplugins_math)


# question: telMatrix is commented out. Does that mean these source files should be deleted?
set(RRPLUGINS_MATH_SOURCES
        telRandom
        #telMatrix
        telMathUtils
        third_party/mtrand
        )

# ** Shared lib ******
if (RR_PLUGINS_BUILD_SHARED_LIB)
    add_library(${target} SHARED ${RRPLUGINS_MATH_SOURCES})
    target_compile_definitions(${target} PRIVATE EXPORT_MATH_API)

    target_link_libraries(${target} PUBLIC
            telplugins_common-static
            )
    target_include_directories(${target} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

    if (UNIX)
        set_target_properties(${target} PROPERTIES INSTALL_RPATH "$ORIGIN/./")
    endif ()

    # Install shared library
    install(TARGETS ${target}
            RUNTIME DESTINATION bin COMPONENT rrplugins
            LIBRARY DESTINATION lib COMPONENT rrplugins
            ARCHIVE DESTINATION lib COMPONENT rrplugins
            )

    install(TARGETS ${target} DESTINATION "${RR_PLUGINS_PYLIB_INSTALL_PREFIX}")

    # disable the auto addition of "lib" onto target name
    set_target_properties(${target} PROPERTIES PREFIX "")
endif ()

# ** Static lib ******

if (RR_PLUGINS_BUILD_STATIC_LIB)
    add_library(${target}-static STATIC ${RRPLUGINS_MATH_SOURCES})
    target_compile_definitions(${target}-static PRIVATE STATIC_PLUGIN_API)

    target_link_libraries(${target}-static PUBLIC
            telplugins_common-static
            )

    target_include_directories(${target}-static PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

    # Install static library
    install(TARGETS ${target}-static
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT rrplugins
            ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT rrplugins
            )
    # disable the auto addition of "lib" onto target name
    set_target_properties(${target}-static PROPERTIES PREFIX "")
endif ()

if (${MSVC})
    # Disable some VS Warnings
    # 4251 About exporting std classes
    # 4018 Comparing unsigned/signed ints
    # 4996 Deprecated functions
    add_definitions("/W3 /D_CRT_SECURE_NO_WARNINGS /wd4251 /wd4018 /wd4996 /nologo")
endif ()

#[[
if(BUILD_TESTS)
    add_subdirectory(testing)
endif()
]]
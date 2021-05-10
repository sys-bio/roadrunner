# M Totte Karlsson
# J Kyle Medley

set(target telplugins_c_api)

#The C-api is built as a shared library..
add_library(${target} SHARED
        telplugins_c_api.cpp
        telplugins_properties_api.cpp
        telplugins_telluriumdata_api.cpp
        telplugins_matrix_api.cpp
        telplugins_cpp_support.cpp
        telplugins_utilities.cpp
        telplugins_logging_api.cpp
        telAPIHandleManager.cpp

        telplugins_c_api.h
        telplugins_properties_api.h
        telplugins_telluriumdata_api.h
        telplugins_matrix_api.h
        telplugins_cpp_support.h
        telplugins_utilities.h
        telplugins_logging_api.h
        telAPIHandleManager.h
        )

target_link_libraries(${target} PUBLIC
        telplugins_common-static
        telplugins_core-static
        telplugins_math-static
        telpluginsBaseClass-static
        )

# remove the "lib" prefix on some platforms
set_target_properties(${target} PROPERTIES PREFIX "")

#Link C_api shared lib statically to roadrunner and libsbml
if (WIN32)
    target_compile_definitions(${target} PUBLIC EXPORT_RRP)
else ()
    if (APPLE)
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-no_implicit_dylibs -undefined error")
        set_target_properties(${target} PROPERTIES INSTALL_RPATH "@loader_path;@loader_path/../roadrunner")
    else ()
        set(CMAKE_SHARED_LINKER_FLAGS "-Wl,--no-undefined")

        # add rpath for site-packages/rrplugins directory
        set_target_properties(${target} PROPERTIES
                INSTALL_RPATH "$ORIGIN/./"
                )
    endif ()
endif ()

install(
        TARGETS ${target}
        RUNTIME DESTINATION bin COMPONENT rrplugins
        ARCHIVE DESTINATION lib COMPONENT rrplugins
        LIBRARY DESTINATION lib COMPONENT rrplugins
)

if (BUILD_PYTHON AND BUILD_RR_PLUGINS)
    install(TARGETS ${target} DESTINATION "${RR_PLUGINS_PYLIB_INSTALL_PREFIX}")
endif()
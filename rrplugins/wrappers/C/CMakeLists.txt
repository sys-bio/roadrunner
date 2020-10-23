# M Totte Karlsson
# J Kyle Medley

set(target telplugins_c_api)

#The C-api is built as a shared library..
add_library(${target} SHARED
        telplugins_c_api
        telplugins_properties_api
        telplugins_telluriumdata_api
        telplugins_matrix_api
        telplugins_cpp_support
        telplugins_utilities
        telplugins_logging_api
        telAPIHandleManager
        )

target_link_libraries(${target} PUBLIC
        telplugins_common
        telplugins_core
        telplugins_math
        telpluginsBaseClass
        ${RRPLUGINS_COMMON_SHARED_LIBS}
        )
message(STATUS "C API: RRPLUGINS_COMMON_SHARED_LIBS = ${RRPLUGINS_COMMON_SHARED_LIBS}")

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
        set_target_properties(${target} PROPERTIES INSTALL_RPATH "$ORIGIN/./")
    endif ()

endif ()

install(
        TARGETS ${target}
        RUNTIME DESTINATION bin COMPONENT rrplugins
        ARCHIVE DESTINATION lib COMPONENT rrplugins
        LIBRARY DESTINATION lib COMPONENT rrplugins
)

install(TARGETS ${target} DESTINATION site-packages/rrplugins)

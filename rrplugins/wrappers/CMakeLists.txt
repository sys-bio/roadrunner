
if (${RR_PLUGINS_BUILD_C_API})
    add_subdirectory("C")
endif ()

if (${BUILD_RR_PLUGINS} AND ${BUILD_PYTHON})
    add_subdirectory("python")
endif ()


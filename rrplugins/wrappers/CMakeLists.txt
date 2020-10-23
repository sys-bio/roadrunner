
if (${RR_PLUGINS_BUILD_C_API})
    add_subdirectory("C")
endif ()

if (${RR_PLUGINS_BUILD_PYTHON_API})
    add_subdirectory("python")
endif ()


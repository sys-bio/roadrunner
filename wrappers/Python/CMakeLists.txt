set(CMAKE_SWIG_FLAGS "") # todo why is this here? Comment out and see if anything breaks.

if (SWIG_FOUND)
    add_subdirectory(roadrunner)
else (SWIG_FOUND)
    message("could not find SWIG in your system")
endif (SWIG_FOUND)


install(FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/INSTALL.txt"
        "${CMAKE_CURRENT_SOURCE_DIR}/setup.py"
        "${CMAKE_CURRENT_SOURCE_DIR}/MANIFEST.in")

if (WITH_CONDA_BUILDER)
    configure_file(conda/meta.yaml.in conda/meta.yaml @ONLY)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/conda/meta.yaml conda/build.sh conda/bld.bat DESTINATION .)
endif ()

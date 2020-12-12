set(CMAKE_SWIG_FLAGS "")

if (SWIG_FOUND)
    add_subdirectory(roadrunner)
else (SWIG_FOUND)
    message("could not find SWIG in your system")
endif (SWIG_FOUND)

if (SWIG_VERSION STRGREATER "4.0")
    message(SEND_ERROR "SWIG version too high:  unable to use SWIG version 4.0 and higher.")
endif (SWIG_VERSION STRGREATER "4.0")

install(FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/INSTALL.txt
        ${CMAKE_CURRENT_SOURCE_DIR}/setup.py
        ${CMAKE_CURRENT_SOURCE_DIR}/MANIFEST.in
        "${CMAKE_BINARY_DIR}/VERSION.txt" DESTINATION . COMPONENT Python)

if (WITH_CONDA_BUILDER)
    configure_file(conda/meta.yaml.in conda/meta.yaml @ONLY)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/conda/meta.yaml conda/build.sh conda/bld.bat DESTINATION . COMPONENT Python)
endif ()

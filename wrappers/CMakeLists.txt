# defines the following:
#
# SWIG_FOUND
# SWIG_EXECUTABLE
# SWIG_DIR
# SWIG_VERSION
# SWIG_<lang>_FOUND
# Note: If swig is installed in a non-standard location users can specify the following during execution of the cmake command:
#   -DSWIG_DIR=/full/path/to/top/level/swig/install/tree -DSWIG_EXECUTABLE=/full/path/to/swig/executable
if (${BUILD_PYTHON} OR ${BUILD_JAVA})
    find_package(SWIG 3.0.0 REQUIRED
            COMPONENTS python
            OPTIONAL_COMPONENTS java
            )
    if (NOT SWIG_FOUND)
        #todo consider automating downloading of swig if it cannot be found (use ExternalProject_Add should make this easy)
        message(FATAL_ERROR "Swig was not found")
    endif ()

    message("SWIG found: ${SWIG_EXECUTABLE}")
    if (NOT SWIG_java_FOUND)
        message(WARNING "SWIG java bindings cannot be generated")
    endif ()

    message(STATUS "swig_version ${SWIG_VERSION}")

    include(UseSWIG)
endif ()

set(RR_CAPI_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/C)
if (BUILD_RR_CAPI)
    add_subdirectory("C")
endif ()

if (BUILD_PYTHON)
    add_subdirectory("Python")
endif ()


#if (BUILD_PYTHON_CTYPES_API)
#    add_subdirectory("c_api_python")
#endif ()

#if (INSTALL_DELPHI)
#    add_subdirectory("Delphi")
#endif ()
#
#
#if (BUILD_JAVA)
#    add_subdirectory("Java")
#endif ()


# copy to /lib so we can import roadrunner in the build tree
configure_file(__init__.py.in "${PYTHON_PACKAGE_BUILD_PREFIX}/__init__.py" COPYONLY)

set_source_files_properties(roadrunner.i PROPERTIES
        CPLUSPLUS ON
        )

set_property(SOURCE roadrunner.i PROPERTY
        SWIG_FLAGS
        # Enable threading
        "-threads"
        "-doxygen" # converts doxygen to pydoc
        #            "-debug-symtabs"
        #            "-debug-csymbols"
        #            "-debug-lsymbols"

        # turns on automatic annotation of python arg and return types
        # BUT does so in c++ types. Therefore I'm not sure how much value this adds
        # and we might want to just keep this off.
        "-py3"


        #    Python Options (available with -python) (run swig.exe -help)
        # "-builtin"                        # Create Python built-in types rather than proxy classes, for better performance
        # "-castmode"                       # Enable the casting mode, which allows implicit cast between types in Python
        # "-debug-doxygen-parser"           # Display doxygen parser module debugging information
        # "-debug-doxygen-translator"       # Display doxygen translator module debugging information
        # -"dirvtable"                      # Generate a pseudo virtual table for directors for faster dispatch
        # -"doxygen"                        # Convert C++ doxygen comments to pydoc comments in proxy classes
        # -"extranative"                    # Return extra native wrappers for C++ std containers wherever possible
        # -"fastproxy"                      # Use fast proxy mechanism for member methods
        # -globals <name>                   # Set <name> used to access C global variable (default: 'cvar')
        # -interface <mod>                  # Set low-level C/C++ module name to <mod> (default: module name prefixed by '_')
        "-keyword"                      # Use keyword arguments
        # -"nofastunpack"                   # Use traditional UnpackTuple method to parse the argument functions
        # -"noh"                            # Don't generate the output header file
        # -"noproxy"                        # Don't generate proxy classes
        # -"nortti"                         # Disable the use of the native C++ RTTI with directors
        # -"nothreads"                      # Disable thread support for the entire interface
        # -"olddefs"                        # Keep the old method definitions when using -fastproxy
        # -"py3"                            # Generate code with Python 3 specific features and syntax
        # -"relativeimport"                 # Use relative Python imports
        # -"threads"                        # Add thread support for all the interface
        # -"O"                              # Enable the following optimization options:
        #                                   #   -fastdispatch -fastproxy -fvirtual
        )

if (USE_TR1_CXX_NS)
    message(STATUS "Using tr1 for SWIG smart pointer namespace")
    set(CMAKE_SWIG_FLAGS "-DSWIG_SHARED_PTR_SUBNAMESPACE=tr1")
endif ()

# we make this target a module library (dll only on windows, so extension on mac and linux).
swig_add_library(roadrunner_python
        TYPE MODULE LANGUAGE python
        SOURCES roadrunner.i rr_docstrings.txt PyUtils.h PyUtils.cpp PyLoggerStream.h PyLoggerStream.cpp PyEventListener.h PyIntegratorListener.h)

# None of these worked, so I renamed rr_docstrings.i to rr_docstrings.txt, to prevent it from being compiled on its own:
#set_source_files_properties(rr_docstrings.i PROPERTIES HEADER_FILE_ONLY TRUE)
#set_source_files_properties(rr_docstrings.i PROPERTIES SKIP_PRECOMPILE_HEADERS TRUE)

message("PYTHON_PACKAGE_BUILD_PREFIX ${PYTHON_PACKAGE_BUILD_PREFIX}")
set_target_properties(roadrunner_python PROPERTIES
        # On windows with the Visual Studio CMake generators, the pyd file
        # is put into a configuration directory (release / debug etc.)
        # however we use the makefile way of doing things and put them in to
        # one directory on all systems.
        #

        RUNTIME_OUTPUT_DIRECTORY                "${PYTHON_PACKAGE_BUILD_PREFIX}"
        RUNTIME_OUTPUT_DIRECTORY_RELEASE        "${PYTHON_PACKAGE_BUILD_PREFIX}"
        RUNTIME_OUTPUT_DIRECTORY_DEBUG          "${PYTHON_PACKAGE_BUILD_PREFIX}"
        RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${PYTHON_PACKAGE_BUILD_PREFIX}"
        LIBRARY_OUTPUT_DIRECTORY                "${PYTHON_PACKAGE_BUILD_PREFIX}"
        LIBRARY_OUTPUT_DIRECTORY_DEBUG          "${PYTHON_PACKAGE_BUILD_PREFIX}"
        LIBRARY_OUTPUT_DIRECTORY_RELEASE        "${PYTHON_PACKAGE_BUILD_PREFIX}"
        LIBRARY_OUTPUT_DIRECTORY_RELWITHDEBINFO "${PYTHON_PACKAGE_BUILD_PREFIX}"
        ARCHIVE_OUTPUT_DIRECTORY                "${PYTHON_PACKAGE_BUILD_PREFIX}"
        ARCHIVE_OUTPUT_DIRECTORY_RELEASE        "${PYTHON_PACKAGE_BUILD_PREFIX}"
        ARCHIVE_OUTPUT_DIRECTORY_DEBUG          "${PYTHON_PACKAGE_BUILD_PREFIX}"
        ARCHIVE_OUTPUT_DIRECTORY_RELWITHDEBINFO "${PYTHON_PACKAGE_BUILD_PREFIX}"
        OUTPUT_NAME roadrunner
        SWIG_USE_TARGET_INCLUDE_DIRECTORIES TRUE # Needed for target_include_directories to have any effect on swig target
        DEBUG_POSTFIX "" # prevent _roadrunnerd
        )
# Move the roadrunner.py produced by swig in the binary tree to the site-pacakges/lib folder with the binaries
add_custom_command(
        TARGET roadrunner_python POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_CURRENT_BINARY_DIR}/roadrunner.py" "${PYTHON_PACKAGE_BUILD_PREFIX}/roadrunner.py"
)
get_filename_component(OneUp ${CMAKE_CURRENT_SOURCE_DIR} DIRECTORY)

target_include_directories(roadrunner_python PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}"
        "${OneUp}/include"
        "${RR_SOURCE_ROOT}"
        "$<TARGET_PROPERTY:roadrunner-static,INCLUDE_DIRECTORIES>"
        "${RR_DEPENDENCIES_INSTALL_PREFIX}/include/rr-libstruct"
        "${Python_INCLUDE_DIRS}"
        "${Python_NumPy_INCLUDE_DIRS}"
        )

if (WIN32)
    target_compile_definitions(roadrunner_python PRIVATE EXPORT_RRC STATIC_RR)

    swig_link_libraries(roadrunner_python
            roadrunner-static
            "${Python_LIBRARIES}"
            )
    # changes dll name to pyd sop that it is compatible with new Python standard
    set_target_properties(roadrunner_python PROPERTIES SUFFIX ".pyd")
elseif (UNIX AND NOT APPLE)
    # Linux specific code
    swig_link_libraries(roadrunner_python
            roadrunner-static

            ${Python_LIBRARIES}
            )
elseif (APPLE)

    # OSX section.
    # we do not link directly with python at at link time, so that at run time,
    # the hosting python would provide all the symbols, we only
    # link with the external libs that we use. This behaviour is enbaled by the -undefined dynamic_lookup flag
    set_target_properties(roadrunner_python
            PROPERTIES
            LINK_FLAGS "-undefined dynamic_lookup"
            INSTALL_RPATH "@loader_path/../../lib;@loader_path/"
            INSTALL_NAME_DIR "@rpath")
    # we use roadrunner not roadrunner-static so we can use shared libraries at runtime
    swig_link_libraries(roadrunner_python
            roadrunner-static
            ##            ${Python_LIBRARIES}
            )
else ()
    message(STATIC "system not supported")
endif ()

add_dependencies(roadrunner_python TestModelFactory)

install(
        TARGETS roadrunner_python
        DESTINATION "${PYTHON_PACKAGE_INSTALL_PREFIX}"
)

install(FILES
        "${PYTHON_PACKAGE_BUILD_PREFIX}/__init__.py"
        "${CMAKE_CURRENT_BINARY_DIR}/roadrunner.py"
        DESTINATION "${PYTHON_PACKAGE_INSTALL_PREFIX}")


install(
        FILES
        testing/__init__.py
        testing/__main__.py
        testing/RunTesterWithUnitTest.py
        testing/README.txt
        testing/tester.py
        testing/testfiles.py
        testing/SBMLTest.py
        testing/feedback.xml
        testing/EcoliCore.xml
        testing/linearPathwayClosed.xml
        testing/linearPathwayOpen.xml
        testing/test_1.xml
        testing/python_api_tests.py
        DESTINATION ${PYTHON_PACKAGE_INSTALL_PREFIX}/testing
)

install(
        DIRECTORY stochastic
        DESTINATION ${PYTHON_PACKAGE_INSTALL_PREFIX}
        FILES_MATCHING PATTERN "*.*"
)

file(GLOB rrtest_files1 "${RR_ROOT}/test/rrtest_files/*.rrtest")
install(
        FILES ${rrtest_files1}
        DESTINATION ${PYTHON_PACKAGE_INSTALL_PREFIX}/testing/test_data
)

file(GLOB rrtest_files2 "${RR_ROOT}/test/rrtest_files/*.xml")
install(
        FILES ${rrtest_files2}
        DESTINATION ${PYTHON_PACKAGE_INSTALL_PREFIX}/testing/test_data
)









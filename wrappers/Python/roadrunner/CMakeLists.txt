#we have to do this to get the generated header
include_directories(${RR_GENERATED_HEADER_PATH})

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${RR_SOURCE_ROOT}
    ${THIRD_PARTY_INSTALL_FOLDER}/include
    ${THIRD_PARTY_INSTALL_FOLDER}/include/rr-libstruct
    ${THIRD_PARTY_INSTALL_FOLDER}/include/sbml
    ${THIRD_PARTY_INSTALL_FOLDER}/include/cvode
    )


# configure compilation of modules for old and new player. Old player modules are compiled with ("threads"=0) option and new are compiled with ("threads"=1) option
# set(thread_option \(\"threads\"=1\))



# copy to /lib so we can import roadrunner in the build tree
configure_file(__init__.py ${CMAKE_BINARY_DIR}/lib/site-packages/roadrunner/__init__.py COPYONLY)
# configure_file(roadrunner.py ${CMAKE_BINARY_DIR}/lib/site-packages/roadrunner/roadrunner.py COPYONLY)

SET_SOURCE_FILES_PROPERTIES(roadrunner.i PROPERTIES CPLUSPLUS ON)
SET_SOURCE_FILES_PROPERTIES(roadrunner.i PROPERTIES SWIG_FLAGS "-threads")


if(USE_TR1_CXX_NS)
  message(STATUS "Using tr1 for SWIG smart pointer namespace")
  set(CMAKE_SWIG_FLAGS "-DSWIG_SHARED_PTR_SUBNAMESPACE=tr1")
endif()

set(CMAKE_SWIG_OUTDIR ${CMAKE_BINARY_DIR}/lib/site-packages/roadrunner)

SWIG_ADD_MODULE(roadrunner python roadrunner.i PyUtils PyLoggerStream)



set_target_properties(_roadrunner PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/python/roadrunner LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/site-packages/roadrunner ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/site-packages/roadrunner)

if(WIN32)
    # changes dll name to pyd sop that it is compatible with new Python standard
    SET_TARGET_PROPERTIES(_roadrunner PROPERTIES SUFFIX ".pyd")


    set_property(TARGET _roadrunner
        PROPERTY  COMPILE_DEFINITIONS
        EXPORT_RRC
        LIBSBML_STATIC
        LIBLAX_STATIC
        STATIC_PUGI
        POCO_STATIC
        STATIC_LIBSTRUCT
        STATIC_RR
        )

    SWIG_LINK_LIBRARIES ( roadrunner
        roadrunner-static
        ${PYTHON_LIBRARIES}
        )


else(WIN32)

    #    set(CMAKE_SHARED_LINKER_FLAGS "-Wl, -z,defs --no-undefined")
    #    set(CMAKE_SHARED_LINKER_FLAGS "-Wl,--no-undefined")

    if(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
        # Linux specific code
        #SET_TARGET_PROPERTIES(_roadrunner PROPERTIES LINK_FLAGS "-Wl,--no-undefined")

    else()
        # OSX section.
        # we do not link directly with python at at link time, so that at run time,
        # the hosting python would provice all the sybols, we only
        # link with the external libs that we uses.
        SET_TARGET_PROPERTIES(_roadrunner PROPERTIES LINK_FLAGS "-undefined dynamic_lookup" INSTALL_RPATH "@loader_path/../../lib;@loader_path/" INSTALL_NAME_DIR "@rpath")
    endif(${CMAKE_SYSTEM_NAME} MATCHES "Linux")

    set_property(TARGET _roadrunner
        PROPERTY  COMPILE_DEFINITIONS
        LIBSBML_STATIC
        LIBLAX_STATIC
        STATIC_PUGI
        STATIC_LIBSTRUCT
        )

    if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
        SWIG_LINK_LIBRARIES ( roadrunner
            roadrunner
            rr-libstruct-static
            nleq1-static
            nleq2-static
            ${LIBSBML_STATIC_LIBRARY}
            lapack
            blas
            f2c
    #        ${PYTHON_LIBRARIES}
            )
    else()
        SWIG_LINK_LIBRARIES ( roadrunner
            roadrunner-static
            rr-libstruct-static
            nleq1-static
            nleq2-static
            ${LIBSBML_STATIC_LIBRARY}
            lapack
            blas
            f2c
    #        ${PYTHON_LIBRARIES}
            )
    endif()

    #        lapack
    #        blas
    #        f2c
    #        "-Wl,--whole-archive"
    #        rr-libstruct-static
    #        "-Wl,--no-whole-archive"
    #        lmfit-static
    #        sbml-static
    #        roadrunner-static


endif(WIN32)

install(
    TARGETS _roadrunner
    DESTINATION ${PYTHON_PACKAGE_DEST_DIR}
)

set(python_files_path ${CMAKE_BINARY_DIR}/wrappers/Python/roadrunner/)

#THIS IS EXAMPLE OF WORKING COMMAND
# ADD_CUSTOM_COMMAND(
#     OUTPUT CompuCell.py.mod
#     COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_SOURCE_DIR}/core/pyinterface/swigmodifier.py ${python_files_path}/CompuCell.py
#     DEPENDS _CompuCell
# )
#
# ADD_CUSTOM_TARGET(_swigmodifier ALL DEPENDS CompuCell.py.mod)

#FILE(GLOB Mac_CPP "*_Mac.cpp")

#INSTALL(FILES ${python_files_path}/roadrunner.py
#    DESTINATION site-packages/roadrunner )

install(
    FILES ${CMAKE_BINARY_DIR}/lib/site-packages/roadrunner/roadrunner.py
    DESTINATION ${PYTHON_PACKAGE_DEST_DIR}
)

install(
    FILES __init__.py
    DESTINATION ${PYTHON_PACKAGE_DEST_DIR}
)

install(
    DIRECTORY testing
    DESTINATION ${PYTHON_PACKAGE_DEST_DIR}
    FILES_MATCHING PATTERN "*.*"
)

install(
    DIRECTORY stochastic
    DESTINATION ${PYTHON_PACKAGE_DEST_DIR}
    FILES_MATCHING PATTERN "*.*"
)

file(GLOB rrtest_files "${RR_ROOT}/testing/*.rrtest")
install(
    FILES ${rrtest_files}
    DESTINATION ${PYTHON_PACKAGE_DEST_DIR}/testing/test_data
)

file(GLOB rrtest_files "${RR_ROOT}/testing/*.xml")
install(
    FILES ${rrtest_files}
    DESTINATION ${PYTHON_PACKAGE_DEST_DIR}/testing/test_data
)

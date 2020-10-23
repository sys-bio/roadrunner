

if(WIN32)
    message(FATAL_ERROR "Java build on Windows not supported")
    # changes dll name to pyd sop that it is compatible with new Python standard
#     set_property(TARGET jroadrunner
#         PROPERTY  COMPILE_DEFINITIONS
#         EXPORT_RRC
#         LIBSBML_STATIC
#         LIBLAX_STATIC
#         STATIC_PUGI
#         POCO_STATIC
#         STATIC_LIBSTRUCT
#         STATIC_RR
#         )
#
#     swig_link_libraries ( jroadrunner
#         jroadrunner-static
#         ${Java_LIBRARIES}
#         )
else()
  #    set(CMAKE_SHARED_LINKER_FLAGS "-Wl, -z,defs --no-undefined")
  #    set(CMAKE_SHARED_LINKER_FLAGS "-Wl,--no-undefined")

  if(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
      # Linux specific code
      #SET_TARGET_PROPERTIES(_jroadrunner PROPERTIES LINK_FLAGS "-Wl,--no-undefined")
  else()
#         SET_TARGET_PROPERTIES(_jroadrunner PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")
  endif(${CMAKE_SYSTEM_NAME} MATCHES "Linux")

  #        lapack
  #        blas
  #        f2c
  #        "-Wl,--whole-archive"
  #        rr-libstruct-static
  #        "-Wl,--no-whole-archive"
  #        lmfit-static
  #        sbml-static
  #        jroadrunner-static


endif()

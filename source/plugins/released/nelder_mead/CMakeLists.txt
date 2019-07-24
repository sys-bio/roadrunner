# M Totte Karlsson
# J Kyle Medley

project( nelder_mead )

set(RRP_NELDER_MEAD_SOURCES
  nelder_mead_doc.cpp
  nmWorker.cpp
  nmUtils.cpp
  nmObjectiveFunction.cpp
  nmNelderMead.cpp
  lib/nmsimplex2.cpp
  nelder_mead_doc.h
  nmWorker.h
  nmUtils.h
  nmObjectiveFunction.h
  nmNelderMead.h
)

add_definitions(
  -DPOCO_STATIC
  -DLIBSBML_STATIC # FIXME: using shared libsbml
  -DEXPORT_TEL_PLUGIN
  -DSTATIC_LIBSTRUCT
)

add_library(tel_nelder_mead MODULE ${RRP_NELDER_MEAD_SOURCES})

target_link_libraries (tel_nelder_mead
  #telplugins_core
  #telplugins_c_api
  #${RRPLUGINS_COMMON_SHARED_LIBS}
  roadrunner
)

if(UNIX)
  if(APPLE)
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-no_implicit_dylibs -undefined error")
  else()
    set(CMAKE_SHARED_LINKER_FLAGS "-Wl,--no-undefined")
  endif()

  set_property(TARGET tel_nelder_mead
    PROPERTY COMPILE_DEFINITIONS
    LIBSBML_STATIC # FIXME: using shared libsbml
    LIBLAX_STATIC
    STATIC_PUGI
    STATIC_LIBSTRUCT
  )
endif()

if(${TLP_BUILD_EXAMPLES})
  add_subdirectory(Examples)
endif()

install (TARGETS tel_nelder_mead
  DESTINATION ${RRP_PLUGINS_DIR}
  COMPONENT plugins
)

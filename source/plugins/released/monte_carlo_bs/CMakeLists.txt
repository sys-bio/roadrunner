# M Totte Karlsson
# J Kyle Medley

project( monte_carlo_bs )

set(RRP_MONTE_CARLO_BS_SOURCES
  bsMonteCarlo.cpp
  bsWorker.cpp
  monte_carlo_bs_doc.cpp
  bsMonteCarlo.h
  bsWorker.h
  monte_carlo_bs_doc.h
)

add_definitions(
  -DPOCO_STATIC
  -DLIBSBML_STATIC # FIXME: using shared libsbml
  -DEXPORT_TEL_PLUGIN
  -DSTATIC_LIBSTRUCT
)

add_library(tel_monte_carlo_bs MODULE ${RRP_MONTE_CARLO_BS_SOURCES})

target_link_libraries (tel_monte_carlo_bs
  #telplugins_core
  #telplugins_math
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

  set_property(TARGET tel_monte_carlo_bs
    PROPERTY  COMPILE_DEFINITIONS
    LIBSBML_STATIC # FIXME: using shared libsbml
    LIBLAX_STATIC
    STATIC_PUGI
    STATIC_LIBSTRUCT
  )
endif()

install (TARGETS tel_monte_carlo_bs
  DESTINATION ${RRP_PLUGINS_DIR}
  COMPONENT plugins
)

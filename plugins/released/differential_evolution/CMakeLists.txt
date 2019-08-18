# M Totte Karlsson
# J Kyle Medley

project( differential_evolution )

set(RRP_DIFFERENTIAL_EVOLUTION_SOURCES
    deWorker.cpp
    deUtils.cpp
    deObjectiveFunction.cpp
    deDifferentialEvolution.cpp
    desimplex.h
    deWorker.h
    deUtils.h
    deObjectiveFunction.h
    deDifferentialEvolution.h
)

add_definitions(
  -DPOCO_STATIC
  -DLIBSBML_STATIC # FIXME: using shared libsbml
  -DEXPORT_TEL_PLUGIN
  -DSTATIC_LIBSTRUCT
)

add_library(tel_differential_evolution MODULE ${RRP_DIFFERENTIAL_EVOLUTION_SOURCES})

target_link_libraries (tel_differential_evolution
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

install (TARGETS tel_differential_evolution
  DESTINATION ${RRP_PLUGINS_DIR}
  COMPONENT plugins
)

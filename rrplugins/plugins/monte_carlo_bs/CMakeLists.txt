# M Totte Karlsson
# J Kyle Medley

project( monte_carlo_bs )
set(target tel_monte_carlo_bs )

set(RRP_MONTE_CARLO_BS_SOURCES
  bsMonteCarlo.cpp
  bsWorker.cpp
  monte_carlo_bs_doc.cpp
  bsMonteCarlo.h
  bsWorker.h
  monte_carlo_bs_doc.h
)

add_library(${target} MODULE ${RRP_MONTE_CARLO_BS_SOURCES})

target_link_libraries (${target} PRIVATE
  telpluginsBaseClass-static
  telplugins_math-static
  #telplugins_common
  ${RRPLUGINS_COMMON_SHARED_LIBS}
)

if(UNIX)
  if(APPLE)
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-no_implicit_dylibs -undefined error")
  else()
    set(CMAKE_SHARED_LINKER_FLAGS "-Wl,--no-undefined")
  endif()
endif()

if(BUILD_PYTHON)
    install (TARGETS ${target}
    DESTINATION ${RR_PLUGINS_PYLIB_INSTALL_PREFIX}
    COMPONENT plugins
    )
    file(GLOB EXAMPLES docs/*)
    install(FILES ${EXAMPLES} DESTINATION "${RR_PLUGINS_PYTHON_INSTALL_PREFIX}" COMPONENT plugins)
    unset(EXAMPLES)
endif()

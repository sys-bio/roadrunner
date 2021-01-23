# M Totte Karlsson
# J Kyle Medley

project( levenberg_marquardt )
set(target tel_levenberg_marquardt )

set(RRP_LEVENBERG_MARQUARDT_SOURCES
  lm.cpp
  lmWorker.cpp
  lmUtils.cpp
  lmObjectiveFunction.cpp
  levenberg_marquardt_doc.cpp
  lm.h
  lmWorker.h
  lmUtils.h
  lmObjectiveFunction.h
  levenberg_marquardt_doc.h
)

# This is where the Levenberg lib is built
add_subdirectory(lib)

add_library(${target} MODULE ${RRP_LEVENBERG_MARQUARDT_SOURCES})

target_link_libraries (${target} PRIVATE
  #lmfit-static
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

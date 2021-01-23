# M Totte Karlsson
# J Kyle Medley

project( nelder_mead )
set(target tel_nelder_mead )

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

add_library(${target} MODULE ${RRP_NELDER_MEAD_SOURCES})

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

target_compile_definitions(${target} PRIVATE EXPORT_NELDER_MEAD)

install(
        TARGETS ${target}
        RUNTIME DESTINATION bin COMPONENT rrplugins
        ARCHIVE DESTINATION lib COMPONENT rrplugins
        LIBRARY DESTINATION lib COMPONENT rrplugins
)

if(BUILD_PYTHON)
    install (TARGETS ${target}
    DESTINATION ${RR_PLUGINS_PYLIB_INSTALL_PREFIX}
    COMPONENT plugins
    )
    file(GLOB EXAMPLES docs/*)
    install(FILES ${EXAMPLES} DESTINATION "${RR_PLUGINS_PYTHON_INSTALL_PREFIX}" COMPONENT plugins)
    unset(EXAMPLES)
endif()

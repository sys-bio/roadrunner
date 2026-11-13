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
  telplugins_c_api
)

target_compile_definitions(${target} PRIVATE
        -DEXPORT_TEL_PLUGIN
        -DEXPORT_MONTE_CARLO_BS
        )

if(UNIX)
  if(APPLE)
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-no_implicit_dylibs")
  else()
    set(CMAKE_SHARED_LINKER_FLAGS "-Wl,--no-undefined")
  endif()
endif()

target_compile_definitions(${target} PRIVATE EXPORT_MONTE_CARLO_BS)

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
    file(GLOB EXAMPLES docs/Examples/*)
    install(FILES ${EXAMPLES} DESTINATION "${RR_PLUGINS_PYTHON_INSTALL_PREFIX}" COMPONENT plugins)
    unset(EXAMPLES)
endif()

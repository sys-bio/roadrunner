# M Totte Karlsson
# J Kyle Medley
# Debashish Roy

project( levenberg_marquardt )
set(target tel_levenberg_marquardt )

# This is where the Levenberg lib is built
add_subdirectory(lib)


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

add_library(${target} MODULE ${RRP_LEVENBERG_MARQUARDT_SOURCES})

target_link_libraries (${target} PRIVATE
    lmfit-static
    telplugins_c_api
)

target_compile_definitions(${target} PRIVATE
    -DEXPORT_TEL_PLUGIN
    -DEXPORT_LEVENBERG_MARQUARDT
    )

if(UNIX)
  if(APPLE)
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-no_implicit_dylibs")
  else()
    set(CMAKE_SHARED_LINKER_FLAGS "-Wl,--no-undefined")
  endif()
endif()

target_compile_definitions(${target} PRIVATE EXPORT_LEVENBERG_MARQUARDT)

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

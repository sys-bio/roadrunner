# Debashish Roy

project(chisquare)
set(target tel_chisquare )

set(RRP_CHISQUARED_SOURCES
  chisquare_doc.cpp
  csChiSquare.cpp
  csChiWorker.cpp
  chisquare_doc.h
  csChiSquare.h
  csChiWorker.h
)

add_library(${target} MODULE ${RRP_CHISQUARED_SOURCES})

target_link_libraries (${target} PRIVATE
  telpluginsBaseClass-static
  telplugins_math-static
  #telplugins_common
)

if(UNIX)
  if(APPLE)
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-no_implicit_dylibs -undefined error")
  else()
    set(CMAKE_SHARED_LINKER_FLAGS "-Wl,--no-undefined")
  endif()
endif()

target_compile_definitions(${target} PRIVATE EXPORT_CHI_SQUARE)


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

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
  telpluginsBaseClass
  telplugins_math
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

install (TARGETS ${target}
  DESTINATION ${RR_PLUGINS_INSTALL_PREFIX}
  COMPONENT plugins
)

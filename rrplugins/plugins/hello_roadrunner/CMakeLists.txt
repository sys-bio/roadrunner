# Debashish Roy

set(target tel_hello)

set(RRP_HELLO_SOURCES
  hello.cpp
  hello.h
)

add_library(${target} MODULE ${RRP_HELLO_SOURCES})

target_link_libraries ( ${target} PUBLIC
  telpluginsBaseClass
  ${RRPLUGINS_COMMON_SHARED_LIBS}
)

if (UNIX)
  set_target_properties( ${target} PROPERTIES INSTALL_RPATH "$ORIGIN/./" )
endif ()

install (TARGETS ${target}
  DESTINATION ${RR_PLUGINS_INSTALL_PREFIX}
  COMPONENT plugins
)

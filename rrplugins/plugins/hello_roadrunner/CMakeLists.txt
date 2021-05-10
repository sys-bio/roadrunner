# Debashish Roy

project(hello)
set(target tel_hello)

set(RRP_HELLO_SOURCES
  hello.cpp
  hello.h
)

add_library(${target} MODULE ${RRP_HELLO_SOURCES})

target_link_libraries ( ${target} PUBLIC
  telpluginsBaseClass-static
  telplugins_math-static
  telplugins_common-static
)

target_compile_definitions(${target} PRIVATE
     -DEXPORT_TEL_PLUGIN
     -DEXPORT_HELLO
     )

if (UNIX)
  set_target_properties( ${target} PROPERTIES INSTALL_RPATH "$ORIGIN/./" )
endif ()

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

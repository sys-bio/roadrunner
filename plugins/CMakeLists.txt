# M Totte Karlsson
# J Kyle Medley

project(plugins)

#Give all plugins access to headers in the following folders
include_directories(
  ${CMAKE_SOURCE_DIR}/source/rrplugins/core
  ${CMAKE_SOURCE_DIR}/source/rrplugins/common
  ${CMAKE_SOURCE_DIR}/source/c
  ${CMAKE_SOURCE_DIR}/source
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_INSTALL_PREFIX}/include
  ${CMAKE_SOURCE_DIR}/source/rrplugins/math
)

add_subdirectory(released)

if (TLP_BUILD_BETA_PLUGINS)
  add_subdirectory(beta)
endif()

install(
  FILES README.txt
  DESTINATION plugins
)

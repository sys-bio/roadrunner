# M Totte Karlsson
# J Kyle Medley

project(plugins)

#Give all plugins access to headers in the following folders
include_directories(
  ${RR_PLUGINS_ROOT}/rrplugins/core
  ${RR_PLUGINS_ROOT}/rrplugins/common
  ${RR_PLUGINS_ROOT}/rrplugins/math
  ${RR_PLUGINS_ROOT}/wrappers/C
)

if (RR_PLUGINS_BUILD_TEST_MODEL_PLUGIN)
    add_subdirectory(test_model)
endif()

if (RR_PLUGINS_BUILD_ADD_NOISE_PLUGIN)
    add_subdirectory(add_noise)
endif()

if (RR_PLUGINS_BUILD_CHISQUARE_PLUGIN)
    add_subdirectory(chisquare)
endif()

if (RR_PLUGINS_BUILD_AUTO2000_PLUGIN)
    add_subdirectory(auto2000)
endif()

if (RR_PLUGINS_BUILD_HELLO_PLUGIN)
    add_subdirectory(hello_roadrunner)
endif()


install(
  FILES README.txt
  DESTINATION ${RR_PLUGINS_INSTALL_PREFIX}
)

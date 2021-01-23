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


#The beta plugins (off by default, since they're currently broken)

if (RR_PLUGINS_BUILD_LEVENBERG_MARQUARDT_PLUGIN)
    add_subdirectory(levenberg_marquardt)
endif()

if (RR_PLUGINS_BUILD_NELDER_MEAD_PLUGIN)
    add_subdirectory(nelder_mead)
endif()

if (RR_PLUGINS_BUILD_MONTE_CARLO_PLUGIN)
    add_subdirectory(monte_carlo_bs)
endif()


#Python
if (WITH_PYTHON)
    install(
        FILES README.txt
        DESTINATION ${RR_PLUGINS_PYTHON_INSTALL_PREFIX}
    )
endif()

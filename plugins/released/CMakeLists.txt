# M Totte Karlsson
# J Kyle Medley

project(released_plugins)

if (TLP_BUILD_TEST_MODEL_PLUGIN)
    add_subdirectory(test_model)
endif()

if (TLP_BUILD_ADD_NOISE_PLUGIN)
    add_subdirectory(add_noise)
endif()

if (TLP_BUILD_CHISQUARE_PLUGIN)
    add_subdirectory(chisquare)
endif()

if (TLP_BUILD_LEVENBERG_MARQUARDT_PLUGIN)
    add_subdirectory(levenberg_marquardt)
endif()

if (TLP_BUILD_NELDER_MEAD_PLUGIN)
    add_subdirectory(nelder_mead)
endif()

#if (TLP_BUILD_MONTE_CARLO_BS_PLUGIN)
#    add_subdirectory(monte_carlo_bs)
#endif()

if (TLP_BUILD_AUTO2000_PLUGIN)
add_subdirectory(auto2000)
endif()

#if (TLP_BUILD_STAT_PLUGIN)
#   add_subdirectory(statistics)
#endif()

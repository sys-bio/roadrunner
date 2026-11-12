# DEPRECATED: use find_package(roadrunner CONFIG REQUIRED) and link roadrunner::roadrunner-static
# (or roadrunner::roadrunner, roadrunner::roadrunner_c_api). roadrunnerConfig.cmake finds all of
# the dependencies itself, so this script is no longer needed. It is kept as a thin wrapper for
# existing projects and will be removed in a future release.
message(DEPRECATION "ImportRoadrunnerAndDependencies.cmake is deprecated and will be removed in a future release. Use find_package(roadrunner CONFIG REQUIRED) and link roadrunner::roadrunner-static instead.")

find_package(roadrunner-static CONFIG REQUIRED)
find_package(roadrunner CONFIG REQUIRED)
find_package(roadrunner_c_api CONFIG REQUIRED)

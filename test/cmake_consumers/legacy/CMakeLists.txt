cmake_minimum_required(VERSION 3.20)
project(UseInstalledRoadRunnerLegacy CXX)

# The deprecated way of finding roadrunner: it must keep working until it is removed.
# Pass -DROADRUNNER_INSTALL_PREFIX=<roadrunner install folder> and -DLLVM_INSTALL_PREFIX=<LLVM folder>.
if (NOT ROADRUNNER_INSTALL_PREFIX)
    message(FATAL_ERROR "Set ROADRUNNER_INSTALL_PREFIX to the folder RoadRunner was installed to.")
endif ()
set(CMAKE_PREFIX_PATH
        "${ROADRUNNER_INSTALL_PREFIX}/cmake"
        "${ROADRUNNER_INSTALL_PREFIX}/lib/cmake"
        )
set(CMAKE_MODULE_PATH "${ROADRUNNER_INSTALL_PREFIX}/lib/cmake")
include(ImportRoadrunnerAndDependencies)

add_executable(legacy_static main.cpp)
target_link_libraries(legacy_static PRIVATE roadrunner-static::roadrunner-static)

add_executable(legacy_c_shared main_c.cpp)
target_link_libraries(legacy_c_shared PRIVATE roadrunner_c_api::roadrunner_c_api)

# Not loaded by the script above; the forwarder must work when asked for explicitly.
find_package(roadrunner_c_api-static CONFIG REQUIRED)
add_executable(legacy_c_static main_c.cpp)
target_link_libraries(legacy_c_static PRIVATE roadrunner_c_api-static::roadrunner_c_api-static)

add_executable(legacy_shared main.cpp)
target_link_libraries(legacy_shared PRIVATE roadrunner::roadrunner)

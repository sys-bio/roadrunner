cmake_minimum_required(VERSION 3.20)
project(UseRoadRunnerFromCxx CXX)

set(CMAKE_CXX_STANDARD 17)

# Pass these on the command line:
#   -DROADRUNNER_INSTALL_PREFIX=<the folder RoadRunner was installed to>
#   -DLLVM_INSTALL_PREFIX=<the folder LLVM was installed to>
if (NOT ROADRUNNER_INSTALL_PREFIX)
    message(FATAL_ERROR "Set ROADRUNNER_INSTALL_PREFIX to the folder RoadRunner was installed to.")
endif ()

# Tell cmake where roadrunner was installed. The roadrunner package finds the libraries it
# depends on (zlib, libSBML, Poco, SUNDIALS, ...) itself, as they are in the same install tree.
# LLVM is not, so it is found through LLVM_INSTALL_PREFIX.
list(APPEND CMAKE_PREFIX_PATH "${ROADRUNNER_INSTALL_PREFIX}")

find_package(roadrunner CONFIG REQUIRED)

# add an executable
add_executable(UseRoadRunnerFromCxx main.cpp)

# link roadrunner-static. The include directories and all of the libraries that roadrunner-static
# needs come with the target, so nothing else has to be linked or included. Other targets are
# roadrunner::roadrunner (shared), roadrunner::roadrunner_c_api and roadrunner::roadrunner_c_api-static.
target_link_libraries(UseRoadRunnerFromCxx PRIVATE roadrunner::roadrunner-static)

# RoadRunner plugins are a second package in the same install folder. If it is there, also build
# a program that loads the plugins. Plugins are loaded while the program runs, so nothing links to them.
find_package(rrplugins CONFIG QUIET)
if (rrplugins_FOUND)
    add_executable(UseRRPluginsFromCxx plugins_main.cpp)
    target_link_libraries(UseRRPluginsFromCxx PRIVATE rrplugins::telplugins_core-static)

    # The plugins are in the lib folder of the install, or in the bin folder on Windows.
    file(TO_CMAKE_PATH "${ROADRUNNER_INSTALL_PREFIX}" plugin_install_prefix)
    if (WIN32)
        set(plugin_folder "${plugin_install_prefix}/bin")
    else ()
        set(plugin_folder "${plugin_install_prefix}/lib")
    endif ()
    target_compile_definitions(UseRRPluginsFromCxx PRIVATE "RR_PLUGIN_FOLDER=\"${plugin_folder}\"")
else ()
    message(STATUS "The rrplugins package was not found in ${ROADRUNNER_INSTALL_PREFIX}: not building UseRRPluginsFromCxx.")
endif ()

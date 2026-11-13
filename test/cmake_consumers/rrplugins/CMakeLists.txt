cmake_minimum_required(VERSION 3.20)
project(UseInstalledRRPlugins CXX C)

set(CMAKE_CXX_STANDARD 17)

# Pass -DROADRUNNER_INSTALL_PREFIX=<roadrunner install folder> and -DLLVM_INSTALL_PREFIX=<LLVM folder>.
if (NOT ROADRUNNER_INSTALL_PREFIX)
    message(FATAL_ERROR "Set ROADRUNNER_INSTALL_PREFIX to the folder RoadRunner was installed to.")
endif ()
list(APPEND CMAKE_PREFIX_PATH "${ROADRUNNER_INSTALL_PREFIX}")

find_package(rrplugins CONFIG REQUIRED)

add_executable(plugins_cpp main.cpp)
target_link_libraries(plugins_cpp PRIVATE rrplugins::telplugins_core-static rrplugins::telplugins_math-static)

add_executable(plugins_c main.c)
target_link_libraries(plugins_c PRIVATE rrplugins::telplugins_c_api)

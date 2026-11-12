cmake_minimum_required(VERSION 3.20)
project(UseInstalledRoadRunner CXX)

set(CMAKE_CXX_STANDARD 17)

# Pass -DROADRUNNER_INSTALL_PREFIX=<roadrunner install folder> and -DLLVM_INSTALL_PREFIX=<LLVM folder>.
if (NOT ROADRUNNER_INSTALL_PREFIX)
    message(FATAL_ERROR "Set ROADRUNNER_INSTALL_PREFIX to the folder RoadRunner was installed to.")
endif ()
list(APPEND CMAKE_PREFIX_PATH "${ROADRUNNER_INSTALL_PREFIX}")

find_package(roadrunner CONFIG REQUIRED)
message(STATUS "Found roadrunner ${roadrunner_VERSION}")

# One executable per target the package exports.
add_executable(use_static main.cpp)
target_link_libraries(use_static PRIVATE roadrunner::roadrunner-static)

add_executable(use_shared main.cpp)
target_link_libraries(use_shared PRIVATE roadrunner::roadrunner)

add_executable(use_c_static main_c.cpp)
target_link_libraries(use_c_static PRIVATE roadrunner::roadrunner_c_api-static)

add_executable(use_c_shared main_c.cpp)
target_link_libraries(use_c_shared PRIVATE roadrunner::roadrunner_c_api)

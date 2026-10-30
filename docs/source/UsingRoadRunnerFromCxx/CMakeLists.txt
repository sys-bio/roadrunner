cmake_minimum_required(VERSION 3.22)
project(UseRoadRunnerFromC)

set(CMAKE_CXX_STANDARD 17)

# could be command line variables (-DROADRUNNER_INSTALL_PREFIX=...)
set(ROADRUNNER_INSTALL_PREFIX "D:/roadrunner/roadrunner/roadrunner-install-rel")
set(LLVM_INSTALL_PREFIX "D:/roadrunner/llvm-23.1.4-release")
set(CMAKE_PREFIX_PATH
        "${ROADRUNNER_INSTALL_PREFIX}/cmake" # needed because this is where poco installs config files to
        "${ROADRUNNER_INSTALL_PREFIX}/lib/cmake" # all other normal packages install config files to here
        )
# for registering location of ImportRoadRunnerAndDependencies
set(CMAKE_MODULE_PATH "${ROADRUNNER_INSTALL_PREFIX}/lib/cmake")

# Import roadrunner and all necessary dependencies for linking roadrunner.
# This command is a roadrunner convenience cmake script which does all of the
# calls to find_package(xxx, CONFIG) for us. If we didn't include this
# script, we would manually need to call find_package for roadrunner-static
# as well as roadrunner dependencies. Checkout The script
# under <rr_install_prefix>/lib/cmake.
include(ImportRoadRunnerAndDependencies)

# add an executable
add_executable(UseRoadRunnerFromCxx main.cpp)

# link roadrunner-static. Note that we have configured roadrunner-static target (which is imported
# under the alias roadrunner-static::roadrunner-static) with PUBLIC visibility, which means
# all includes and transitive linked libraries are automatically handled by cmake.
# without this, you would have to manually link roadrunner dependencies and manually
# include roadrunner include directories with something like:
#   - target_include_directories(UseRoadRunnerFromCxx PRIVATE "${ROADRUNNER_INSTALL_PREFIX}/include")
target_link_libraries(UseRoadRunnerFromCxx PRIVATE roadrunner-static::roadrunner-static)

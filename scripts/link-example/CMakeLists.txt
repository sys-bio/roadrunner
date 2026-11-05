# Minimal project that links a C++ program against an installed roadrunner,
# for example one built by scripts/build_roadrunner.sh.
#
#   cmake -B build -DROADRUNNER_WORKSPACE=<the folder the script built in>
#   cmake --build build
#   ./build/simulate
#
# ROADRUNNER_WORKSPACE is a shortcut for the layout that build_roadrunner.sh
# produces. The three trees can also be given one by one, which is what to do
# when they live elsewhere:
#
#   -DROADRUNNER_INSTALL_PREFIX=<workspace>/install-roadrunner
#   -DRR_DEPENDENCIES_INSTALL_PREFIX=<workspace>/libroadrunner-deps-binaries
#   -DLLVM_INSTALL_PREFIX=<workspace>/llvm-<version>-binaries
#
# All three are needed, and they have to stay where they were when roadrunner was
# built: roadrunner links LLVM and the dependencies statically, and the exported
# cmake targets name those archives by absolute path. Link against the install
# tree, not against the cmake build tree: only the install step writes the cmake
# package files, the headers and the merged include directory.

cmake_minimum_required(VERSION 3.16)
project(roadrunner-link-example CXX)

# roadrunner's headers need C++17 or later, and RTTI has to stay enabled (the
# default) because the LLVM back end inside roadrunner is built with it.
# Poco::Foundation, which comes in with roadrunner, requires cxx_std_20, so the
# compile ends up at C++20 whatever is asked for here. That is expected.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- the trees that the roadrunner build produced ----------------------------
if (DEFINED ROADRUNNER_WORKSPACE)
    if (NOT DEFINED ROADRUNNER_INSTALL_PREFIX)
        set(ROADRUNNER_INSTALL_PREFIX "${ROADRUNNER_WORKSPACE}/install-roadrunner")
    endif ()
    if (NOT DEFINED RR_DEPENDENCIES_INSTALL_PREFIX)
        set(RR_DEPENDENCIES_INSTALL_PREFIX "${ROADRUNNER_WORKSPACE}/libroadrunner-deps-binaries")
    endif ()
    if (NOT DEFINED LLVM_INSTALL_PREFIX)
        # The directory carries the LLVM version, so look it up.
        file(GLOB _llvm_trees "${ROADRUNNER_WORKSPACE}/llvm-*-binaries")
        list(GET _llvm_trees 0 LLVM_INSTALL_PREFIX)
    endif ()
endif ()

foreach (_tree ROADRUNNER_INSTALL_PREFIX RR_DEPENDENCIES_INSTALL_PREFIX LLVM_INSTALL_PREFIX)
    if (NOT DEFINED ${_tree} OR NOT IS_DIRECTORY "${${_tree}}")
        message(FATAL_ERROR "${_tree} is not set to an existing directory. Pass "
                "-DROADRUNNER_WORKSPACE=<the folder roadrunner was built in>, or set "
                "the three trees one by one. See the comment at the top of this file.")
    endif ()
endforeach ()

# --- import roadrunner -------------------------------------------------------
# One include() brings in roadrunner and every package its targets refer to
# (libsbml, Poco, SUNDIALS, clapack, LLVM, ...). It reads the two prefixes set
# above to find them.
include("${ROADRUNNER_INSTALL_PREFIX}/lib/cmake/ImportRoadrunnerAndDependencies.cmake")

# --- the program -------------------------------------------------------------
add_executable(simulate simulate.cpp)

# roadrunner-static::roadrunner-static gives a self contained binary: the
# include directory, the STATIC_RR definition and all of the static archives
# come with the target, and nothing has to be found at run time.
# For the shared library use roadrunner::roadrunner instead and keep
# <install>/lib reachable at run time (RPATH or LD_LIBRARY_PATH).
target_link_libraries(simulate PRIVATE roadrunner-static::roadrunner-static)

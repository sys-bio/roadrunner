# CMake module for importing the roadrunner targets, with their dependencies,
# into a consuming project. It is installed to <prefix>/lib/cmake and used as
#
#   set(LLVM_INSTALL_PREFIX "/path/to/llvm-<version>")
#   set(RR_DEPENDENCIES_INSTALL_PREFIX "/path/to/libroadrunner-deps")
#   include("<prefix>/lib/cmake/ImportRoadrunnerAndDependencies.cmake")
#   target_link_libraries(your_target PRIVATE roadrunner-static::roadrunner-static)
#
# after which roadrunner-static::roadrunner-static, roadrunner::roadrunner and
# roadrunner_c_api::roadrunner_c_api are available. See scripts/link-example for
# a project that does nothing else.
#
# What a consumer has to know:
#
#  * Use the install tree, the one CMAKE_INSTALL_PREFIX pointed at, not the
#    cmake build tree. Only the install step writes these package files, the
#    headers and the merged include directory.
#  * roadrunner links LLVM and the dependencies statically and the exported
#    targets name those archives by absolute path, so the LLVM and
#    libroadrunner-deps trees that roadrunner was built against have to stay
#    where they were. Point LLVM_INSTALL_PREFIX and, unless the dependency tree
#    was installed into the roadrunner prefix, RR_DEPENDENCIES_INSTALL_PREFIX at
#    them. Both default to this install tree.
#  * Compile with C++17 or later and with RTTI enabled, as roadrunner is built.
#    Poco::Foundation, which arrives with roadrunner, requires cxx_std_20, so
#    the compile lands on C++20 whatever the consumer asks for.

# This file is installed next to the roadrunner package files and FindLLVM.cmake,
# in <prefix>/lib/cmake (lib64 on some systems), so both paths follow from it.
get_filename_component(_rr_cmake_dir "${CMAKE_CURRENT_LIST_DIR}" ABSOLUTE)
get_filename_component(_rr_install_prefix "${_rr_cmake_dir}/../.." ABSOLUTE)

if (NOT DEFINED RR_DEPENDENCIES_INSTALL_PREFIX)
    set(RR_DEPENDENCIES_INSTALL_PREFIX "${_rr_install_prefix}")
endif ()
if (NOT DEFINED LLVM_INSTALL_PREFIX)
    set(LLVM_INSTALL_PREFIX "${_rr_install_prefix}")
endif ()

# The dependency package files sit directly in these directories rather than in
# one <name>/ subdirectory each, so every directory has to be named as a prefix
# of its own. Whether the tree uses lib or lib64 depends on the system it was
# built on, so both are listed. This mirrors what roadrunner's own CMakeLists.txt
# assembles for its build, and without it a consumer gets "Could not find a
# package configuration file provided by sbml-static" and the like.
list(APPEND CMAKE_PREFIX_PATH
        "${_rr_install_prefix}"
        "${_rr_cmake_dir}"
        "${RR_DEPENDENCIES_INSTALL_PREFIX}"
        "${RR_DEPENDENCIES_INSTALL_PREFIX}/cmake"
        "${RR_DEPENDENCIES_INSTALL_PREFIX}/lib/cmake"
        "${RR_DEPENDENCIES_INSTALL_PREFIX}/lib/cmake/Poco"
        "${RR_DEPENDENCIES_INSTALL_PREFIX}/lib64/cmake"
        "${RR_DEPENDENCIES_INSTALL_PREFIX}/lib64/cmake/Poco")
list(REMOVE_DUPLICATES CMAKE_PREFIX_PATH)

# FindLLVM.cmake is a find module, not a package config, and is installed here.
list(APPEND CMAKE_MODULE_PATH "${_rr_cmake_dir}")
list(REMOVE_DUPLICATES CMAKE_MODULE_PATH)

# Clients need roadrunner but must also import the dependencies
# with find_package. Its easy to bundle these together
# and document it.
#
find_package(Threads) # for libxml2, FindThreads.cmake is shipped with cmake
find_package(LibLZMA) # for libxml2, LibLZMA.cmake is shipped with cmake
# Spelled ZLIB, the way the dependency tree ships it (ZLIBConfig.cmake) and the
# way roadrunner's own build looks for it. find_package(zlib) only resolves on a
# case insensitive file system, so this line used to fail on Linux.
find_package(ZLIB CONFIG REQUIRED COMPONENTS static)
find_package(bzip2 CONFIG REQUIRED)
#find_package(iconv CONFIG REQUIRED)
#find_package(LibXml2 CONFIG REQUIRED)
find_package(expat CONFIG REQUIRED)
find_package(sbml-static CONFIG REQUIRED)
find_package(rr-libstruct CONFIG REQUIRED)
find_package(clapack CONFIG REQUIRED)
find_package(nleq1 CONFIG REQUIRED)
find_package(nleq2 CONFIG REQUIRED)
find_package(PocoFoundation CONFIG REQUIRED)
find_package(PocoNet CONFIG REQUIRED)
find_package(PocoXML CONFIG REQUIRED)
find_package(SUNDIALS CONFIG REQUIRED)
find_package(LLVM REQUIRED)
find_package(roadrunner-static CONFIG REQUIRED)
find_package(roadrunner CONFIG REQUIRED)
find_package(roadrunner_c_api CONFIG REQUIRED)

unset(_rr_cmake_dir)
unset(_rr_install_prefix)

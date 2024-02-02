cmake_minimum_required(VERSION 3.16)


# libRoadRunner semantic versioning (http://semver.org/) - used to generate
# source files expose library version information

#########################################################################
# Version information and include modules

set(ROADRUNNER_VERSION_MAJOR 2)
set(ROADRUNNER_VERSION_MINOR 6)
set(ROADRUNNER_VERSION_PATCH 0)

set(ROADRUNNER_VERSION "${ROADRUNNER_VERSION_MAJOR}.${ROADRUNNER_VERSION_MINOR}.${ROADRUNNER_VERSION_PATCH}")

if (APPLE AND NOT CMAKE_OSX_ARCHITECTURES)
    # Some CMake features like object libraries get confused if you don't
    # explicitly specify an architecture setting with the Xcode generator.
    #    set(CMAKE_OSX_ARCHITECTURES "arm64;x86_64")
    set(CMAKE_OSX_ARCHITECTURES "x86_64" CACHE INTERNAL "Compile for intel chips on apple" FORCE)
    message(STATUS "building on mac for intel chips only.")
endif()

project(
        roadrunner
        DESCRIPTION "roadrunner project"
        VERSION ${ROADRUNNER_VERSION}
        LANGUAGES CXX C
)

if (${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.20")
    # Source file extensions (.h/.cpp) do not need to be explicitly
    #   specified if CMP0115 set to OLD.
    #   New behavior requires all extensions be explicit
    cmake_policy(SET CMP0115 OLD)
endif ()


if ("${CMAKE_SIZEOF_VOID_P}" STREQUAL "4")
    message(STATUS "Build on 32-bit platform")
else ()
    message(STATUS "Build on 64-bit platform")
endif ()


# do some bitness checks as its an easy trap to fall into on windows.
if (WIN32)
    if ("${CMAKE_SIZEOF_VOID_P}" STREQUAL "4")
        message(WARNING "You are building 32-bit binaries \
this is unlikely what you want. If you intented to build x64-bit ensure you are using the x64-bit
visual studio developers command prompt for running the cmake command" .)
    endif ()

    string(FIND "${CMAKE_SYSTEM_PROCESSOR}" "64" IS_64_BIT)
    if ((${IS_64_BIT} GREATER 0) AND ("${CMAKE_SIZEOF_VOID_P}" STREQUAL "4"))
        message(WARNING "You are generating a 32-bit version of roadrunner on a \
x64 bit machine")
    endif ()

endif ()

set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}" "${CMAKE_CURRENT_SOURCE_DIR}/cmake")


# External project. Try to avoid
include(ExternalProject)

# macro for setting a variable with cross platform independent values
include(SetCrossPlatform)

# convenience function for querying presently defined cmake variables
include(QueryCMakeVariables) # debbugging

# GNU standard install locations
include(GNUInstallDirs)

# generate automatic export headers
include(GenerateExportHeader)

# Custom macro for containing the git related stuff.
include(WriteGitInfo)
write_git_info() # write the git branch and commit hash to console

# load cmake_dependent_options
include(CMakeDependentOption)

include(ConfigurationSummary)
include(ConfigurationSummaryPython)

#############################################################
#   Configuration Options
#

# any of these directories are accepted as default directories.
# this supports multi platform builds on a single system
set(RR_DEPENDENCIES_INSTALL_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}/dependencies"
        CACHE PATH "Path to dependency package"
        )

# This variable provides users a way of providing their own LLVM installation.
# The two options are: 1) build llvm along with the other dependencies or
# 2) build the other dependencies and llvm separate. If the latter is chosen,
# users will need to provide their own argument to LLVM_INSTALL_PREFIX
set(LLVM_INSTALL_PREFIX ${RR_DEPENDENCIES_INSTALL_PREFIX} CACHE PATH "Location of LLVM Root directory, which contains bin, lib, include and share directories")
set(LLVM_ROOT_DIR "${LLVM_ROOT_DIR}")

#set(USE_UNIVERSAL_BINARIES "ON" CACHE BOOL "Build with universal binaries on MacOS")

# todo difference between build tests and build test tools option? (cw)
option(BUILD_TEST_TOOLS "Build and install test tools" OFF)
option(BUILD_TESTS "Build the SBML C API test suite" OFF)
option(BUILD_LLVM "Build the LLVM back end" ON)

# note I've tried using the C backend and it no longer compiles. Could be useful to keep though
option(BUILD_LEGACY_C "Build the legacy C code generating backend (deprecated)" OFF)

# should we build the swig python wrapper?
option(BUILD_PYTHON "Build the SWIG generated python wrapper" OFF)
if (BUILD_PYTHON)
    find_package(Python COMPONENTS Interpreter Development NumPy)
    if (NOT ${Python_FOUND})
        # sets a variable called PythonSummary
        ConfigurationSummaryPython()
        message(FATAL_ERROR "Could not find Python:\n ${PythonSummary}")
    endif ()

    if ("${Python_ROOT_DIR}" STREQUAL "")
        get_filename_component(PRD ${Python_EXECUTABLE} DIRECTORY)
        set(Python_ROOT_DIR "${PRD}" CACHE PATH "Path to python executable and libraries (optional).")
    endif ()

    # build directories
    set(PYTHON_PACKAGE_SITE_DIR_BUILD_TREE "${CMAKE_BINARY_DIR}/lib/site-packages" CACHE PATH "Path to Python site packages directory in the build tree")
    set(PYTHON_PACKAGE_BUILD_PREFIX "${PYTHON_PACKAGE_SITE_DIR_BUILD_TREE}/roadrunner")
    set(RR_PYTHON_TESTING_BUILD_PREFIX "${PYTHON_PACKAGE_BUILD_PREFIX}/tests")

    # install directories
    set(PYTHON_PACKAGE_SITE_DIR "${CMAKE_INSTALL_PREFIX}/site-packages" CACHE PATH "Path to Python site packages directory in the install tree")
    set(PYTHON_PACKAGE_INSTALL_PREFIX "${PYTHON_PACKAGE_SITE_DIR}/roadrunner")#CACHE PATH "Destination for roadrunner python bindings.")
    set(RR_PYTHON_TESTING_INSTALL_PREFIX "${PYTHON_PACKAGE_INSTALL_PREFIX}/tests")


    # plugin directories
    set(RR_PLUGINS_PYTHON_INSTALL_PREFIX "${PYTHON_PACKAGE_SITE_DIR}/rrplugins")#CACHE PATH "Destination for rrplugins python bindings.")
    set(RR_PLUGINS_PYLIB_INSTALL_PREFIX "${PYTHON_PACKAGE_SITE_DIR}/rrplugins/libs")#CACHE PATH "Destination for rrplugins python binding libraries.")

    cmake_dependent_option(RR_PLUGINS_BUILD_PYTHON_API "Build the RR plugin PYTHON API" ON "BUILD_RR_PLUGINS" OFF)

    #	cmake_dependent_option(BUILD_PYTHON_CTYPES_API "Build the ctypes python API for roadrunner" ON
    #	        "BUILD_PYTHON" OFF)
    #	cmake_dependent_option(WITH_CONDA_BUILDER "Include files required to build the conda package" OFF
    #	        "BUILD_PYTHON" OFF)
endif ()

#Setup so that roadrunner is always built as a dll and linked statically with 'as much as possible'
option(BUILD_RR_CAPI "Generate C API" ON)

#TODO this option yet supported in new buildsystem (cw).
option(BUILD_APPS "Build and install Apps" OFF)

option(BUILD_EXAMPLES "Build and install Examples" OFF)
option(INSTALL_SBML_MODELS "Install SBML Models" OFF)
option(BUILD_DOCS "Build the roadrunner documentation" OFF)

# TODO this options is not yet suppoerted in the new build system (cw)
option(BUILD_JAVA "build the SWIG generated Java wrapper" OFF)

option(BUILD_PACKAGING "Package roadrunner into binary distributions" OFF)
option(WITH_ADDRESS_SANITIZER "Build roadrunner with memory sanitizer tool (assumes address sanitizer is available)" OFF)

# telplugin options
option(BUILD_RR_PLUGINS "Build the telplugins package" OFF)

#####################################################
# Configure Asan build
# https://stackoverflow.com/a/64294837/3059024
#   and
# Professional CMake: A Practical Guide.
# Usage:
#   -DCMAKE_BUILD_TYPE=Asan ...
#

if ("${CMAKE_BUILD_TYPE}" STREQUAL "Asan")
    # these variables only get used in Asan builds
    # and they will only work if CMake can find address sanitizer
    # executable.
    #
    # Only works on macos.
    #
    # The Asan build is based on the Debug build, but with a few extra flags
    #
    get_property(isMultiConfig GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)

    if (isMultiConfig)
        if (NOT "Asan" IN_LIST CMAKE_CONFIGURATION_TYPES)
            list(APPEND CMAKE_CONFIGURATION_TYPES Asan)
        endif ()
    else ()
        set(allowedBuildTypes Asan Debug Release RelWithDebInfo MinSizeRel)
        set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "${allowedBuildTypes}")

        if (CMAKE_BUILD_TYPE AND NOT CMAKE_BUILD_TYPE IN_LIST allowedBuildTypes)
            message(FATAL_ERROR "Invalid build type: ${CMAKE_BUILD_TYPE}")
        endif ()
    endif ()

    set(WHICH_SANITIZER "address" CACHE STRING "which sanitizer to use. address, memory, thread, leak, undefined")
    set(SANITIZER_OPTIONS "-fno-omit-frame-pointer" CACHE STRING "options to pass to command -fsanitize")

    message(STATUS "WHICH_SANITIZER ${WHICH_SANITIZER}")

    set(CMAKE_C_FLAGS_ASAN
            "${CMAKE_C_FLAGS_DEBUG} -fsanitize=${WHICH_SANITIZER} ${SANITIZER_OPTIONS}" CACHE STRING
            "Flags used by the C compiler for Asan build type or configuration." FORCE)

    set(CMAKE_CXX_FLAGS_ASAN
            "${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=${WHICH_SANITIZER} ${SANITIZER_OPTIONS}" CACHE STRING
            "Flags used by the C++ compiler for Asan build type or configuration." FORCE)

    set(CMAKE_EXE_LINKER_FLAGS_ASAN
            "${CMAKE_SHARED_LINKER_FLAGS_DEBUG} -fsanitize=${WHICH_SANITIZER}" CACHE STRING
            "Linker flags to be used to create executables for Asan build type." FORCE)

    set(CMAKE_SHARED_LINKER_FLAGS_ASAN
            "${CMAKE_SHARED_LINKER_FLAGS_DEBUG} -fsanitize=${WHICH_SANITIZER}" CACHE STRING
            "Linker lags to be used to create shared libraries for Asan build type." FORCE)
endif ()


## Feature selection

# These options are dependent on BUILD_RR_PLUGINS option. This reads: when BUILD_RR_PLUGINS=ON set the value of
# RR_PLUGINS_BUILD_SHARED_LIB default value to ON, otherwise set to OFF
cmake_dependent_option(RR_PLUGINS_BUILD_SHARED_LIB "Build the RR plugin shared libraries" ON "BUILD_RR_PLUGINS" OFF)
cmake_dependent_option(RR_PLUGINS_BUILD_STATIC_LIB "Build the RR plugin static libraries" ON "BUILD_RR_PLUGINS" OFF)
cmake_dependent_option(RR_PLUGINS_BUILD_C_API "Build the RR plugin C API" ON "BUILD_RR_PLUGINS" OFF)

# Plugin selection: Released
# These options default to ON when BUILD_RR_PLUGINS=ON and OFF otherwise
cmake_dependent_option(RR_PLUGINS_BUILD_TEST_MODEL_PLUGIN "Build the Test Model Plugin" ON "BUILD_RR_PLUGINS" OFF)
cmake_dependent_option(RR_PLUGINS_BUILD_ADD_NOISE_PLUGIN "Build the Add Noise Plugin" ON "BUILD_RR_PLUGINS" OFF)
cmake_dependent_option(RR_PLUGINS_BUILD_CHISQUARE_PLUGIN "Build the ChiSquare Plugin" ON "BUILD_RR_PLUGINS" OFF)
cmake_dependent_option(RR_PLUGINS_BUILD_AUTO2000_PLUGIN "Build the Auto 2000 Plugin" ON "BUILD_RR_PLUGINS" OFF)
cmake_dependent_option(RR_PLUGINS_BUILD_HELLO_PLUGIN "Build the Demo Plugin" ON "BUILD_RR_PLUGINS" OFF)
cmake_dependent_option(RR_PLUGINS_BUILD_LEVENBERG_MARQUARDT_PLUGIN "Build the Levenberg-Marquardt Plugin" ON "BUILD_RR_PLUGINS" OFF)
cmake_dependent_option(RR_PLUGINS_BUILD_NELDER_MEAD_PLUGIN "Build the Nelder-Mead Plugin" ON "BUILD_RR_PLUGINS" OFF)
cmake_dependent_option(RR_PLUGINS_BUILD_MONTE_CARLO_PLUGIN "Build the Monte Carlo Plugin" ON "BUILD_RR_PLUGINS" OFF)

#mark_as_advanced(RR_PLUGINS_BUILD_LEVENBERG_MARQUARDT_PLUGIN)
#mark_as_advanced(RR_PLUGINS_BUILD_NELDER_MEAD_PLUGIN)
#mark_as_advanced(RR_PLUGINS_BUILD_MONTE_CARLO_PLUGIN)

set(CXX_STANDARD 17 CACHE STRING "set the cmake standard to use")

##################################################
#   Settings
#

#handle rpath
set(CMAKE_MACOSX_RPATH TRUE)
set(CMAKE_BUILD_RPATH_USE_ORIGIN TRUE)

if (APPLE AND NOT UNIX)
    set(RPATH_BASE @loader_path)
else ()
    set(RPATH_BASE $ORIGIN)
endif ()
file(RELATIVE_PATH relDir
        ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}
        ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
set(CMAKE_INSTALL_RPATH ${RPATH_BASE} ${RPATH_BASE}/${relDir})

if (BUILD_TESTS OR BUILD_JAVA)
    enable_testing()
endif ()


set(RR_USE_CXX14 TRUE CACHE BOOL "Set to TRUE to enable C++14 features")

# new clang on OSX and msvc use std namespace.
if (RR_USE_CXX14)
    option(USE_TR1_CXX_NS "Use the tr1 c++ namespace for swig smart pointer" OFF)
else ()
    option(USE_TR1_CXX_NS "Use the tr1 c++ namespace for swig smart pointer" ${CMAKE_COMPILER_IS_GNUCXX})
endif ()

set(CMAKE_BUILD_TYPE "Release" CACHE STRING
        "Choose the build type. The options are: None (CMAKE_CXX_FLAGS or
     CMAKE_C_FLAGS are used), Debug, Release, RelWithDebInfo, MinSizeRel.")

message(STATUS "PROJECT_BINARY_DIR ${PROJECT_BINARY_DIR}")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin")

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if (BUILD_LLVM)
    # todo needed?
    add_definitions(-DBUILD_LLVM)
else ()
    message(STATUS "NOT using LLVM")
endif (BUILD_LLVM)

if (BUILD_LEGACY_C)
    message(STATUS "building legacy C backend")
    add_definitions(-DBUILD_LEGACY_C)
else ()
    message(STATUS "not building legacy C backend")
endif (BUILD_LEGACY_C)


# Enable cmake printing out commands
set(CMAKE_VERBOSE_MAKEFILE TRUE)

if (${MSVC})
    # Here we set all flags that are needed for dependency packages.
    # Since we link to everything statically, we need these flags.
    # Individual targets will need RR_STATIC or RR_EXPORT but otherwise
    # these flags are set globally.
    # todo consider whether this is desirable behaviour in the scenario that
    #   roadrunner is being used as a embedded package
    add_definitions(
            -DLIBSBML_USE_CPP_NAMESPACE
            -DLIBSBML_STATIC
            -DLIBLAX_STATIC
            -DSTATIC_LIBSTRUCT
            -DSTATIC_PUGI
            -DSTATIC_NLEQ1
            -DSTATIC_NLEQ2
            -DPOCO_STATIC
            -DPOCO_NO_AUTOMATIC_LIBS
            -DCRT_NONSTDC_NO_DEPRECATE
            -D_CRT_SECURE_NO_WARNINGS    # MS wanting you to changes basically all standard C functions :(
            -D_WIN32                     # used as switch in zlib macro
            -DXML_STATIC                 # for expat inside poco. The code that uses this define (poco_xml) is only used in tests.
    )

endif ()

if (${MINGW})
    # todo test on mingw (cw)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++0x")
endif ()

if (${BORLAND})
    # todo test on borland (cw)
    add_definitions(
            -DUSE_PCH                   #Enable pre-compiled headers
            -H=${PROJECT_BINARY_DIR}/rr_pch.csm
            -w-8012                     #Comparing signed /unsigned
            -w-8057                     #Parameter never used
            -w-8004                     #'var' is assigned a value that is never used
    )
endif ()

set(CMAKE_DEBUG_POSTFIX "d")

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX "../install-${CMAKE_BUILD_TYPE}" CACHE PATH "..." FORCE)
endif()


####################################################
#   Dependencies

if (NOT EXISTS ${RR_DEPENDENCIES_INSTALL_PREFIX})
    message(FATAL_ERROR "The path given to -DRR_DEPENDENCIES_INSTALL_PREFIX (\"${RR_DEPENDENCIES_INSTALL_PREFIX}\") does not exist. Please \
build the dependency package (https://github.com/sys-bio/libroadrunner-deps) and pass in
    -DRR_DEPENDENCIES_INSTALL_PREFIX=\"/full/path/to/dependency/install/tree\"
to the roadrunner cmake command.")
endif ()
set(DEPENDENCY_INCLUDE_DIR "${RR_DEPENDENCIES_INSTALL_PREFIX}/include")

if (NOT EXISTS ${DEPENDENCY_INCLUDE_DIR})
    message(FATAL_ERROR "Cannot find the dependency include directory in your \
dependency install tree. Please ensure the path you have given to -DRR_DEPENDENCIES_INSTALL_PREFIX \
exists and is the full path to the installed dependency tree.")
endif ()

set(RR_ROOT "${CMAKE_CURRENT_SOURCE_DIR}")
set(DEPENDENCY_CMAKE_CONFIG_PATHS
        "${RR_DEPENDENCIES_INSTALL_PREFIX}/lib/cmake"
        "${RR_DEPENDENCIES_INSTALL_PREFIX}/lib64/cmake"
        "${RR_DEPENDENCIES_INSTALL_PREFIX}/cmake"
        "${RR_DEPENDENCIES_INSTALL_PREFIX}/lib/cmake/Poco"
        )
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "${DEPENDENCY_CMAKE_CONFIG_PATHS}")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${DEPENDENCY_CMAKE_CONFIG_PATHS}")
message(STATUS "CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH}")
if (WIN32)
    set(SBML_TARGET_NAME "libsbml-static")
else ()
    set(SBML_TARGET_NAME "sbml-static")
endif (WIN32)


#find_package(Threads) # for libxml2, FindThreads.cmake is shipped with cmake
#find_package(LibLZMA) # for libxml2, LibLZMA.cmake is shipped with cmake
find_package(zlib CONFIG REQUIRED)
#find_package(bzip2 CONFIG REQUIRED)
#find_package(iconv CONFIG REQUIRED)

find_package(expat CONFIG REQUIRED)

## https://stackoverflow.com/questions/32183975/how-to-print-all-the-properties-of-a-target-in-cmake/56738858#56738858
## https://stackoverflow.com/a/56738858/3743145

## Get all properties that cmake supports
execute_process(COMMAND cmake --help-property-list OUTPUT_VARIABLE CMAKE_PROPERTY_LIST)
## Convert command output into a CMake list
STRING(REGEX REPLACE ";" "\\\\;" CMAKE_PROPERTY_LIST "${CMAKE_PROPERTY_LIST}")
STRING(REGEX REPLACE "\n" ";" CMAKE_PROPERTY_LIST "${CMAKE_PROPERTY_LIST}")

list(REMOVE_DUPLICATES CMAKE_PROPERTY_LIST)

function(print_target_properties tgt)
    if(NOT TARGET ${tgt})
      message("There is no target named '${tgt}'")
      return()
    endif()

    foreach (prop ${CMAKE_PROPERTY_LIST})
        string(REPLACE "<CONFIG>" "${CMAKE_BUILD_TYPE}" prop ${prop})
        get_target_property(propval ${tgt} ${prop})
        if (propval)
            message ("${tgt} ${prop} = ${propval}")
        endif()
    endforeach(prop)
endfunction(print_target_properties)

cmake_language(CALL print_target_properties zlib::zlibstatic)
cmake_language(CALL print_target_properties expat::expat)

#libsbml now needs to know that we found expat, so we set the relevant flags
set(EXPAT_LIBRARY "expat::expat")
get_target_property(EXPAT_INCLUDE_DIR expat::expat INTERFACE_INCLUDE_DIRECTORIES)
get_target_property(EXPAT_LIBRARY expat::expat LOCATION)
message(STATUS "EXPAT_LIBRARY: ${EXPAT_LIBRARY}")
set(ZLIB_LIBRARY "zlib::zlibstatic")
get_target_property(ZLIB_LIBRARY zlib::zlibstatic LOCATION)
set(ZLIB_INCLUDE_DIR "${RR_DEPENDENCIES_INSTALL_PREFIX}/include/")

message(STATUS "ZLIB_LIBRARY: ${ZLIB_LIBRARY}")
message(STATUS "ZLIB_INCLUDE_DIR: ${ZLIB_INCLUDE_DIR}")

#For some reason, something in the bowels of the dependencies needs ZLIB::ZLIB, and won't accept zlib::zlib.
# I stole this from libsbml's CMakeLists.txt file, since they apparently also suffer the same problem
# (and are the probable cause, at that.)  I've only seen this needed on Ubuntu, and even there, it's
# not needed for the azure ubuntu build, just when actually running Ubuntu.
if(NOT TARGET ZLIB::ZLIB)
  add_library(ZLIB::ZLIB UNKNOWN IMPORTED)
  set_target_properties(ZLIB::ZLIB PROPERTIES
    IMPORTED_LINK_INTERFACE_LANGUAGES "C"
    IMPORTED_LOCATION "${ZLIB_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${ZLIB_INCLUDE_DIR}")
endif()


find_package(${SBML_TARGET_NAME} CONFIG REQUIRED)
find_package(rr-libstruct CONFIG REQUIRED)
find_package(clapack CONFIG REQUIRED)
find_package(nleq1 CONFIG REQUIRED)
find_package(nleq2 CONFIG REQUIRED)
find_package(PocoFoundation CONFIG REQUIRED)
find_package(PocoNet CONFIG REQUIRED)
find_package(PocoXML CONFIG REQUIRED)
find_package(LLVM REQUIRED)

find_package(SUNDIALS CONFIG REQUIRED)

# not for windows. So not required, but we throw error if not win and not found.
find_package(Curses)

#if (NOT WIN32 AND NOT ${CURSES_FOUND})
#    message(FATAL_ERROR "ncurses library not found. Please install it with your native package manager")
#endif ()


set(BUILD_SHARED_LIBS OFF)
set(BUILD_STATIC_LIBS ON)

# install dependencies so clients can use roadrunner from c++/cmake
message(STATUS "CMAKE_INSTALL_LIBDIR ${CMAKE_INSTALL_LIBDIR}")
install(DIRECTORY "${RR_DEPENDENCIES_INSTALL_PREFIX}/lib" DESTINATION "${CMAKE_INSTALL_PREFIX}")
install(DIRECTORY "${RR_DEPENDENCIES_INSTALL_PREFIX}/bin" DESTINATION "${CMAKE_INSTALL_PREFIX}")
install(DIRECTORY "${RR_DEPENDENCIES_INSTALL_PREFIX}/include" DESTINATION "${CMAKE_INSTALL_PREFIX}")
install(DIRECTORY "${RR_DEPENDENCIES_INSTALL_PREFIX}/share" DESTINATION "${CMAKE_INSTALL_PREFIX}")
if (WIN32)
    install(DIRECTORY "${RR_DEPENDENCIES_INSTALL_PREFIX}/cmake" DESTINATION "${CMAKE_INSTALL_PREFIX}")
endif()
# install a cmake script for consumer libraries to easily import roadrunner targets
install(FILES "cmake/ImportRoadrunnerAndDependencies.cmake" DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake")

if (${MINGW})
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++0x")
endif ()

if (${BORLAND})
    link_directories(${THIRD_PARTY_INSTALL_FOLDER}/lib)
    add_definitions(
            -DUSE_PCH                   #Enable pre-compiled headers
            -H=${PROJECT_BINARY_DIR}/rr_pch.csm
            -w-8012                     #Comparing signed /unsigned
            -w-8057                     #Parameter never used
            -w-8004                     #'var' is assigned a value that is never used
    )
endif ()

##########################################################
# Building
#


# Set standard
set(CMAKE_CXX_STANDARD 17)
if (NOT WIN32)
    # some cmake (or gcc) bug means that we also need to set -std=c++17
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
endif ()

# We always have to build the roadrunner core, everything depends on this
add_subdirectory(source)

###########################################################
## API's

add_subdirectory(wrappers)

if (BUILD_RR_PLUGINS)
    add_subdirectory(rrplugins)
endif ()


##########################################################
# Documentation
#

if (${BUILD_DOCS})
    add_subdirectory(docs)
endif ()

##########################################################
# Packaging
#

if (${BUILD_PACKAGING})
    # it doesn't look that useful and we now have the contents of
    # packaging that will automate production of binary releases.
    add_subdirectory(packaging)
endif ()


##########################################################
# Configuration summary
#

if (BUILD_TESTS)
    # these two must be present at top level cmake
    # so that we can use gtest_discover_tests as opposed
    # to add_test.
    enable_testing()
    include(GoogleTest)

    # defined here, passed on to running the Python tests via CTest
    set(PYTHON_API_IN_BINARY_TREE "${CMAKE_CURRENT_BINARY_DIR}/wrappers/Python")

    # these sources are needed in some tests
    set(
            RR_PYTHON_PYUTILS_SRC
            "${CMAKE_CURRENT_SOURCE_DIR}/wrappers/Python/roadrunner/PyUtils.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/wrappers/Python/roadrunner/PyUtils.h"
    )

    add_subdirectory(test)

    message(STATUS "building tests")

    file(GLOB PYTHON_TEST_FILES "test/python/*.*")

    file(GLOB PYTHON_TEST_DATA "${CMAKE_CURRENT_SOURCE_DIR}/test/rrtest_files/*.rrtest")

    file(GLOB TEST_DATA_XML "${CMAKE_CURRENT_SOURCE_DIR}/test/rrtest_files/*.xml")

    foreach (f ${PYTHON_TEST_FILES})
        if (NOT IS_DIRECTORY ${f})
            get_filename_component(FILE_NAME ${f} NAME)
            configure_file(${f} lib/site-packages/roadrunner/tests/${FILE_NAME} COPYONLY)
        endif ()
    endforeach ()

    foreach (f ${PYTHON_TEST_DATA})
        if (NOT IS_DIRECTORY ${f})
            get_filename_component(FILE_NAME ${f} NAME)
            configure_file(${f} lib/site-packages/roadrunner/tests/test_data/${FILE_NAME} COPYONLY)
        endif ()
    endforeach ()

    foreach (f ${TEST_DATA_XML})
        if (NOT IS_DIRECTORY ${f})
            get_filename_component(FILE_NAME ${f} NAME)
            configure_file(${f} lib/site-packages/roadrunner/tests/test_data/${FILE_NAME} COPYONLY)
        endif ()
    endforeach ()

else ()
    message(STATUS "NOT building tests")
endif ()


#======================= INSTALL STUFF ========================================================
#What sub folders to move into...?
if (${INSTALL_SBML_MODELS})
    add_subdirectory(models)
endif ()


install(FILES README.md LICENSE.txt AUTHORS.txt FUNDING.txt
        DESTINATION .
        COMPONENT info
        )

# make or copy installers into root directory
add_subdirectory(installer)

#=== ThirdParties
if (${BORLAND})
    set(CG_RUNTIMES cc32110MT.dll)
    foreach (runtime ${CG_RUNTIMES})
        install(FILES
                ${THIRD_PARTY_INSTALL_FOLDER}/cg/xe/${runtime}
                DESTINATION bin
                COMPONENT rr_core)
    endforeach ()
endif ()

configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/VERSION.txt.in"
        "${CMAKE_CURRENT_BINARY_DIR}/VERSION.txt" @ONLY
)

configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/requirements.txt"
        "${CMAKE_BINARY_DIR}/requirements.txt" @ONLY
)
configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/test-requirements.txt"
        "${CMAKE_BINARY_DIR}/test-requirements.txt" @ONLY
)
install(
        FILES
            "${CMAKE_CURRENT_SOURCE_DIR}/requirements.txt"
            "${CMAKE_CURRENT_SOURCE_DIR}/test-requirements.txt"
        DESTINATION .
)

install(
        FILES
        NOTICE.txt
        NEWS.txt
        DESTINATION .
)

# add FindLLVM to install cmake module path so consuming libraries can use it
install(FILES cmake/FindLLVM.cmake DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake)


# print out configuration summary
include(ConfigurationSummary)

ConfigurationSummary()



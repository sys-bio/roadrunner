# Find LLVM
#
# It defines the following variables
#  LLVM_FOUND        - True if llvm found.
#  LLVM_INCLUDE_DIRS  - where to find llvm include files
#  LLVM_LIBRARY_DIRS  - where to find llvm libs
#  LLVM_LDFLAGS       - llvm linker flags
#  LLVM_LIBRARIES     - list of llvm libs for working with modules. These are LLVM's
#                      imported targets (LLVMCore, ...) when LLVM's CMake package is used, and
#                      absolute path file names when the llvm-config fallback is used.
#  LLVM_FLAGS     - flags to add to the CXX compiler for llvm support
#  LLVM_VERSION_MAJOR
#  LLVM_VERSION_MINOR
#  LLVM_VERSION_PATCH
#
# LLVM is located with its own CMake package (<prefix>/lib/cmake/llvm/LLVMConfig.cmake):
# from LLVM_INSTALL_PREFIX or LLVM_ROOT_DIR, then LLVM_DIR, CMAKE_PREFIX_PATH and the
# system paths. Only if no such package is found is llvm-config used.
#
# The LLVM that is found must have been built the way roadrunner needs it: version 23.1 or
# newer, with RTTI and exceptions enabled, and with the native target. A small test program
# is also compiled and linked against it, which catches compiler, standard library and
# runtime mismatches.
#
# Variables that can be set before the search:
#  LLVM_INSTALL_PREFIX / LLVM_ROOT_DIR - prefix of the LLVM installation
#  RR_LLVM_USE_LLVM_CONFIG             - ON: skip the CMake package, use llvm-config
#  RR_LLVM_SKIP_PROBE                  - ON: skip the test compile
#################################################################################

set(RR_LLVM_MIN_VERSION 23.1)

# this function borrowed from PlPlot, Copyright (C) 2006  Alan W. Irwin
function(TRANSFORM_VERSION numerical_result version)
    # internal_version ignores everything in version after any character that
    # is not 0-9 or ".".  This should take care of the case when there is
    # some non-numerical data in the patch version.
    #message(STATUS "DEBUG: version = ${version}")
    string(REGEX REPLACE "^([0-9.]+).*$" "\\1" internal_version ${version})

    # internal_version is normally a period-delimited triplet string of the form
    # "major.minor.patch", but patch and/or minor could be missing.
    # Transform internal_version into a numerical result that can be compared.
    string(REGEX REPLACE "^([0-9]*).+$" "\\1" major ${internal_version})
    string(REGEX REPLACE "^[0-9]*\\.([0-9]*).*$" "\\1" minor ${internal_version})
    string(REGEX REPLACE "^[0-9]*\\.[0-9]*\\.([0-9]*)$" "\\1" patch ${internal_version})

    if (NOT patch MATCHES "^[0-9]+$")
        set(patch 0)
    endif (NOT patch MATCHES "^[0-9]+$")

    if (NOT minor MATCHES "[0-9]+")
        set(minor 0)
    endif (NOT minor MATCHES "[0-9]+")

    if (NOT major MATCHES "[0-9]+")
        set(major 0)
    endif (NOT major MATCHES "[0-9]+")
    #message(STATUS "DEBUG: internal_version = ${internal_version}")
    #message(STATUS "DEBUG: major = ${major}")
    #message(STATUS "DEBUG: minor= ${minor}")
    #message(STATUS "DEBUG: patch = ${patch}")
    math(EXPR internal_numerical_result
            "${major}*1000000 + ${minor}*1000 + ${patch}"
            #"${major}*1000000 + ${minor}*1000"
            )
    #message(STATUS "DEBUG: ${numerical_result} = ${internal_numerical_result}")
    set(${numerical_result} ${internal_numerical_result} PARENT_SCOPE)
endfunction(TRANSFORM_VERSION)

# Preferred: LLVM's own CMake package.
if (NOT RR_LLVM_USE_LLVM_CONFIG)
    set(_rr_llvm_hints)
    foreach (_rr_llvm_root IN ITEMS "${LLVM_INSTALL_PREFIX}" "${LLVM_ROOT_DIR}")
        if (_rr_llvm_root AND EXISTS "${_rr_llvm_root}/lib/cmake/llvm/LLVMConfig.cmake")
            list(APPEND _rr_llvm_hints "${_rr_llvm_root}/lib/cmake/llvm")
        endif ()
    endforeach ()

    set(_rr_llvm_find_args)
    if (_rr_llvm_hints)
        list(GET _rr_llvm_hints 0 _rr_llvm_first_hint)
        # A cached LLVM_DIR from an earlier configure would win over the prefix that is set now.
        if (LLVM_DIR AND NOT "${LLVM_DIR}" STREQUAL "${_rr_llvm_first_hint}")
            unset(LLVM_DIR CACHE)
        endif ()
        list(APPEND _rr_llvm_find_args HINTS ${_rr_llvm_hints})
    endif ()

    # An LLVM built with the DIA SDK looks for it through VSINSTALLDIR, which only a Visual Studio
    # command prompt sets. Without it, take the DIA SDK folder from the Visual Studio in use.
    if (WIN32 AND MSVC AND NOT LLVM_WINSYSROOT AND NOT DEFINED MSVC_DIA_SDK_DIR AND NOT DEFINED ENV{VSINSTALLDIR})
        set(_rr_vs_dir "${CMAKE_GENERATOR_INSTANCE}")
        if (NOT _rr_vs_dir)
            string(REGEX MATCH "^(.*)/VC/Tools/MSVC/" _rr_vs_match "${CMAKE_CXX_COMPILER}")
            set(_rr_vs_dir "${CMAKE_MATCH_1}")
        endif ()
        if (_rr_vs_dir AND IS_DIRECTORY "${_rr_vs_dir}/DIA SDK")
            set(MSVC_DIA_SDK_DIR "${_rr_vs_dir}/DIA SDK" CACHE PATH "Path to the DIA SDK")
        endif ()
    endif ()

    find_package(LLVM ${RR_LLVM_MIN_VERSION} CONFIG QUIET ${RR_FIND_PACKAGE_GLOBAL} ${_rr_llvm_find_args})

    if (LLVM_FOUND)
        message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}: ${LLVM_DIR}")

        if (LLVM_ENABLE_DIA_SDK AND NOT TARGET DIASDK::Diaguids)
            message(FATAL_ERROR "The LLVM at ${LLVM_DIR} was built with the Microsoft DIA SDK, and the DIA SDK was not found. "
                    "Set MSVC_DIA_SDK_DIR to the \"DIA SDK\" folder of your Visual Studio installation (it is part of the "
                    "\"Desktop development with C++\" workload), or configure from a Visual Studio command prompt.")
        endif ()

        # Options that the LLVM build must have.
        set(_rr_llvm_problems)
        if (NOT LLVM_ENABLE_RTTI)
            list(APPEND _rr_llvm_problems "it was built without RTTI (LLVM_ENABLE_RTTI=OFF), but roadrunner is compiled with RTTI")
        endif ()
        if (NOT LLVM_ENABLE_EH)
            list(APPEND _rr_llvm_problems "it was built without exception handling (LLVM_ENABLE_EH=OFF), but roadrunner is compiled with exceptions")
        endif ()
        if (LLVM_NATIVE_ARCH)
            list(FIND LLVM_TARGETS_TO_BUILD "${LLVM_NATIVE_ARCH}" _rr_llvm_native_index)
            if (_rr_llvm_native_index EQUAL -1)
                list(APPEND _rr_llvm_problems "it was not built with its native target ${LLVM_NATIVE_ARCH} (LLVM_TARGETS_TO_BUILD=${LLVM_TARGETS_TO_BUILD})")
            endif ()
        endif ()
        if (_rr_llvm_problems)
            string(REPLACE ";" "\n  - " _rr_llvm_problem_text "${_rr_llvm_problems}")
            message(FATAL_ERROR "The LLVM at ${LLVM_DIR} cannot be used by roadrunner:\n  - ${_rr_llvm_problem_text}\n"
                    "Use an LLVM built with the options roadrunner needs (the one from libroadrunner-deps), "
                    "or point LLVM_INSTALL_PREFIX at it.")
        endif ()

        if (LLVM_VERSION_MAJOR GREATER 23)
            message(WARNING "Roadrunner is built and tested against LLVM 23.1; found LLVM ${LLVM_PACKAGE_VERSION}.")
        endif ()

        # Same variables as the llvm-config path below.
        set(LLVM_VERSION "${LLVM_PACKAGE_VERSION}")
        set(LLVM_FLAGS "${LLVM_DEFINITIONS}")
        llvm_map_components_to_libnames(LLVM_LIBRARIES core mcjit orcjit support native)

        # Compile and link a small program against these libraries with the compiler and
        # flags of this project. The result is remembered until one of them changes.
        if (NOT RR_LLVM_SKIP_PROBE)
            set(_rr_llvm_probe_key "${LLVM_DIR}|${CMAKE_CXX_COMPILER}|${CMAKE_CXX_COMPILER_VERSION}|${CMAKE_CXX_FLAGS}|${CMAKE_BUILD_TYPE}")
            if (NOT "${RR_LLVM_PROBE_PASSED_FOR}" STREQUAL "${_rr_llvm_probe_key}")
                set(_rr_llvm_probe_dir "${CMAKE_BINARY_DIR}/CMakeFiles/rr_llvm_probe")
                file(WRITE "${_rr_llvm_probe_dir}/probe.cpp" [[
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/ExecutionEngine/ExecutionEngine.h>
#include <llvm/ExecutionEngine/MCJIT.h>
#include <llvm/Support/TargetSelect.h>
#include <memory>
#include <typeinfo>
struct Probe : llvm::LLVMContext {};
int main() {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    Probe context;
    auto module = std::make_unique<llvm::Module>("probe", context);
    try { throw 1; } catch (int) {}
    return typeid(context) == typeid(Probe) ? 0 : 1;
}
]])
                set(_rr_llvm_std 17)
                if (CMAKE_CXX_STANDARD)
                    set(_rr_llvm_std ${CMAKE_CXX_STANDARD})
                endif ()
                # Test in the configuration that will be built: the one LLVM was built in with a
                # multi-configuration generator, and CMAKE_BUILD_TYPE with any other generator.
                set(_rr_llvm_saved_try_config "${CMAKE_TRY_COMPILE_CONFIGURATION}")
                get_property(_rr_llvm_multi GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
                if (_rr_llvm_multi AND LLVM_BUILD_TYPE)
                    set(CMAKE_TRY_COMPILE_CONFIGURATION "${LLVM_BUILD_TYPE}")
                elseif (NOT _rr_llvm_multi AND CMAKE_BUILD_TYPE)
                    set(CMAKE_TRY_COMPILE_CONFIGURATION "${CMAKE_BUILD_TYPE}")
                endif ()
                # try_compile copies these into the probe project without escaping backslashes.
                set(_rr_llvm_saved_module_path "${CMAKE_MODULE_PATH}")
                set(_rr_llvm_saved_prefix_path "${CMAKE_PREFIX_PATH}")
                if (WIN32)
                    file(TO_CMAKE_PATH "${CMAKE_MODULE_PATH}" CMAKE_MODULE_PATH)
                    file(TO_CMAKE_PATH "${CMAKE_PREFIX_PATH}" CMAKE_PREFIX_PATH)
                endif ()
                try_compile(_rr_llvm_probe_ok "${_rr_llvm_probe_dir}/build" "${_rr_llvm_probe_dir}/probe.cpp"
                        CMAKE_FLAGS "-DINCLUDE_DIRECTORIES=${LLVM_INCLUDE_DIRS}"
                        LINK_LIBRARIES ${LLVM_LIBRARIES}
                        CXX_STANDARD ${_rr_llvm_std}
                        OUTPUT_VARIABLE _rr_llvm_probe_log)
                set(CMAKE_TRY_COMPILE_CONFIGURATION "${_rr_llvm_saved_try_config}")
                set(CMAKE_MODULE_PATH "${_rr_llvm_saved_module_path}")
                set(CMAKE_PREFIX_PATH "${_rr_llvm_saved_prefix_path}")
                if (NOT _rr_llvm_probe_ok)
                    file(WRITE "${_rr_llvm_probe_dir}/probe.log" "${_rr_llvm_probe_log}")
                    # The end of the output holds the compiler or linker errors.
                    string(LENGTH "${_rr_llvm_probe_log}" _rr_llvm_probe_log_length)
                    set(_rr_llvm_probe_log_tail "${_rr_llvm_probe_log}")
                    if (_rr_llvm_probe_log_length GREATER 6000)
                        math(EXPR _rr_llvm_probe_log_start "${_rr_llvm_probe_log_length} - 6000")
                        string(SUBSTRING "${_rr_llvm_probe_log}" ${_rr_llvm_probe_log_start} -1 _rr_llvm_probe_log_tail)
                        set(_rr_llvm_probe_log_tail "...${_rr_llvm_probe_log_tail}")
                    endif ()
                    message(FATAL_ERROR "The LLVM at ${LLVM_DIR} does not work with this compiler "
                            "(${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}): a test program that uses "
                            "${LLVM_LIBRARIES} failed to compile or link. The usual causes are an LLVM built "
                            "with a different compiler, C++ standard library or runtime library setting. "
                            "The whole output is in ${_rr_llvm_probe_dir}/probe.log. Set RR_LLVM_SKIP_PROBE=ON to skip this test.\n"
                            "The end of the output:\n${_rr_llvm_probe_log_tail}")
                endif ()
                set(RR_LLVM_PROBE_PASSED_FOR "${_rr_llvm_probe_key}" CACHE INTERNAL "LLVM test compile passed for this setup")
            endif ()
        endif ()

        return()
    endif ()
    message(STATUS "LLVM CMake package ${RR_LLVM_MIN_VERSION} or newer not found, looking for llvm-config")
endif ()

# Fallback: llvm-config
find_program(LLVM_CONFIG_EXECUTABLE
        NAMES llvm-config-${LLVM_MIN_VERSION_TEXT} llvm-config
        PATHS /opt/local/bin /usr/local/bin
        HINTS
        ${LLVM_INSTALL_PREFIX}/bin
        $ENV{LLVM_DIR}/bin
        C:/LLVM/bin
        C:/LLVM6.0.1/bin
        C:/llvm/bin
        C:/llvm6.0.1/bin
        D:/LLVM/bin
        D:/LLVM6.0.1/bin
        D:/llvm/bin
        D:/llvm6.0.1/bin
        ~/LLVM/bin
        ~/LLVM6.0.1/bin
        ~/llvm/bin
        ~/llvm6.0.1/bin
        DOC "llvm-config executable"
        )


if (LLVM_CONFIG_EXECUTABLE)
    message(STATUS "LLVM llvm-config found at: ${LLVM_CONFIG_EXECUTABLE}")

    #    execute_process(
    #            COMMAND chmod -x ${LLVM_CONFIG_EXECUTABLE}
    #            #            OUTPUT_VARIABLE LLVM_VERSION
    #            #            OUTPUT_STRIP_TRAILING_WHITESPACE
    #    )

    execute_process(
            COMMAND ${LLVM_CONFIG_EXECUTABLE} --version
            OUTPUT_VARIABLE LLVM_VERSION
            OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    #    message(STATUS "LLVM_VERSION ${LLVM_VERSION}")

    # Version Info
    execute_process(COMMAND ${LLVM_CONFIG_EXECUTABLE} --version OUTPUT_VARIABLE LLVM_STRING_VERSION)
    string(REGEX REPLACE "^([0-9]+)\\.([0-9]+)([^\n\r\t ]*)" "\\1" LLVM_VERSION_MAJOR "${LLVM_VERSION}")
    string(REGEX REPLACE "^([0-9]+)\\.([0-9]+)([^\n\r\t ]*)" "\\2" LLVM_VERSION_MINOR "${LLVM_VERSION}")
    string(REGEX REPLACE "^([0-9]+)\\.([0-9]+)([^\n\r\t ]*)" "\\3" LLVM_VERSION_PATCH "${LLVM_VERSION}")

    # Roadrunner is written against the LLVM 23.1 API.
    if (LLVM_VERSION VERSION_LESS "23.1")
        message(FATAL_ERROR "Roadrunner requires LLVM 23.1 or newer, but found LLVM ${LLVM_VERSION} (${LLVM_CONFIG_EXECUTABLE}).")
    elseif (LLVM_VERSION_MAJOR GREATER 23)
        message(WARNING "Roadrunner is built and tested against LLVM 23.1; found LLVM ${LLVM_VERSION}.")
    endif ()
#    message(STATUS "LLVM_VERSION: ${LLVM_VERSION_MAJOR}.${LLVM_VERSION_MINOR}.${LLVM_VERSION_PATCH}")

    execute_process(COMMAND ${LLVM_CONFIG_EXECUTABLE} --bindir OUTPUT_VARIABLE LLVM_BIN_DIR)
    execute_process(COMMAND ${LLVM_CONFIG_EXECUTABLE} --libdir OUTPUT_VARIABLE LLVM_LIB_DIR)
    #    MESSAGE(STATUS "LLVM_BIN_DIR: " ${LLVM_BIN_DIR})
    #    MESSAGE(STATUS "LLVM_LIB_DIR: " ${LLVM_LIB_DIR})

    # Include Dir
    execute_process(
            COMMAND ${LLVM_CONFIG_EXECUTABLE} --includedir
            OUTPUT_VARIABLE LLVM_INCLUDE_DIRS
            OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    #    message(STATUS "LLVM_INCLUDE_DIRS: ${LLVM_INCLUDE_DIRS}")

    # Lib Dir
    execute_process(
            COMMAND ${LLVM_CONFIG_EXECUTABLE} --libdir
            OUTPUT_VARIABLE LLVM_LIBRARY_DIRS
            OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    #    message(STATUS "LLVM_LIBRARY_DIRS:  ${LLVM_LIBRARY_DIRS}")

    # System libs
    execute_process(
            COMMAND ${LLVM_CONFIG_EXECUTABLE} --system-libs
            OUTPUT_VARIABLE LLVM_SYSTEM_LIBS
            OUTPUT_STRIP_TRAILING_WHITESPACE
    )
#    message(STATUS "LLVM_SYSTEM_LIBS:  ${LLVM_SYSTEM_LIBS}")
    # For some reason, the return value for system-libs for macos is of the
    # form '-llib1 -llib2 ...", but the return value for system-libs for windows
    # is of the form "lib1.lib lib2.lib ..." which doesn't work the same way.
    # The only system it was failing for was macos, so we'll just set it for that.
    if (APPLE)
        set(LLVM_SYSTEM_LIBS_THISOS ${LLVM_SYSTEM_LIBS})
    endif ()
#    message(STATUS "LLVM_SYSTEM_LIBS_THISOS:  ${LLVM_SYSTEM_LIBS_THISOS}")

    # C++ Flags, strip out stuff that CMake build adds
    execute_process(
            COMMAND ${LLVM_CONFIG_EXECUTABLE} --cxxflags
            OUTPUT_VARIABLE LLVM_FLAGS
            RESULT_VARIABLE LLVM_CONFIG_RESULT
            ERROR_VARIABLE LLVM_CONFIG_ERROR
            OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if (NOT LLVM_FLAGS)
        message(STATUS "LLVM_INSTALL_PREFIX ${LLVM_INSTALL_PREFIX}")
        if (EXISTS "${LLVM_CONFIG_EXECUTABLE}")
            message(FATAL_ERROR "llvm-config exists at ${LLVM_CONFIG_EXECUTABLE} but failed to run \
(result: ${LLVM_CONFIG_RESULT}, stderr: ${LLVM_CONFIG_ERROR}). Check that it has execute \
permissions (on Windows, that its ACLs are valid) and that the runtime libraries it needs are installed.")
        else ()
            message(FATAL_ERROR "llvm-config was not found at ${LLVM_CONFIG_EXECUTABLE}. \
Check that LLVM_INSTALL_PREFIX (${LLVM_INSTALL_PREFIX}) points to the correct location.")
        endif ()
    endif ()
#    message(STATUS "LLVM_FLAGS ${LLVM_FLAGS}")
    # strip this from llvm's version, we should add this ourselves in
    # production mode to main CFLAGS
    STRING(REPLACE "-DNDEBUG" "" LLVM_FLAGS ${LLVM_FLAGS})
    STRING(REPLACE "-D_DEBUG" "" LLVM_FLAGS ${LLVM_FLAGS})
    # remove optimization from flags, our cmake build will decide what optimization to use
    STRING(REGEX REPLACE "-O[0-9]" "" LLVM_FLAGS ${LLVM_FLAGS})
    STRING(REPLACE "-fno-rtti" "" LLVM_FLAGS ${LLVM_FLAGS})
    STRING(REPLACE "-fno-exceptions" "" LLVM_FLAGS ${LLVM_FLAGS})
    # this causes problems as most other libs compiled without this.
    # should be OK linking to LLVM as this just results in a slightly larger lib (I think, I hope...)
    string(REPLACE "-fvisibility-inlines-hidden" "" LLVM_FLAGS ${LLVM_FLAGS})

    # remove extra warnings that the llvm-config added
    STRING(REPLACE "-Wcast-qual" "" LLVM_FLAGS ${LLVM_FLAGS})
    STRING(REPLACE "-Woverloaded-virtual" "" LLVM_FLAGS ${LLVM_FLAGS})

    # some LLVMs add these options
    # -Wall -W -Wno-unused-parameter -Wwrite-strings -Wno-missing-field-initializers -pedantic
    # -Wno-long-long -Wno-uninitialized -Wnon-virtual-dtor


    STRING(REPLACE "-Wall" "" LLVM_FLAGS ${LLVM_FLAGS})
    STRING(REPLACE "-W " "" LLVM_FLAGS ${LLVM_FLAGS})
    STRING(REPLACE "-Wno-unused-parameter" "" LLVM_FLAGS ${LLVM_FLAGS})
    STRING(REPLACE "-Wwrite-strings" "" LLVM_FLAGS ${LLVM_FLAGS})
    STRING(REPLACE "-Wno-missing-field-initializers" "" LLVM_FLAGS ${LLVM_FLAGS})
    STRING(REPLACE "-pedantic" "" LLVM_FLAGS ${LLVM_FLAGS})
    STRING(REPLACE "-Wno-long-long" "" LLVM_FLAGS ${LLVM_FLAGS})
    STRING(REPLACE "-Wno-uninitialized" "" LLVM_FLAGS ${LLVM_FLAGS})
    STRING(REPLACE "-Wnon-virtual-dtor" "" LLVM_FLAGS ${LLVM_FLAGS})


    #    MESSAGE(STATUS "LLVM_FLAGS: " ${LLVM_FLAGS})


    # link libraries, currently only need core, jit and native.
    # TODO: in future, replace this with something like LLVM_CORE_LIBS, LLVM_JIT_LIBS...
    # Replaced this with manually set variable LLVM_LIBRARIES
    #   The advantage is that we can pass these in as targets to
    # the llvm build and not have to build everything.
    execute_process(
            COMMAND ${LLVM_CONFIG_EXECUTABLE} --libfiles core mcjit native Core OrcJIT Support native
            OUTPUT_VARIABLE LLVM_LIBRARIES
            OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if (WIN32)
        # llvm-config returns .a library names, even on Windows, so manually convert .a -> .lib and remove lib prefix.
        STRING(REGEX REPLACE "lib([^/\\]*)\\.a" "\\1.lib" LLVM_LIBRARIES ${LLVM_LIBRARIES})
    endif ()
    # we get a space sep list from llvm-config, make it a cmake ; separated list.
    STRING(REGEX REPLACE "[\n\t\r ]+" ";" LLVM_LIBRARIES ${LLVM_LIBRARIES})
    #    message(STATUS "LLVM_LIBRARIES: ${LLVM_LIBRARIES}")

    # LLVMSupport needs system libraries on Windows (ntdll, psapi, shell32, ...).
    if (WIN32)
        string(REGEX REPLACE "[\n\t\r ]+" ";" LLVM_SYSTEM_LIBS_WIN "${LLVM_SYSTEM_LIBS}")
        list(APPEND LLVM_LIBRARIES ${LLVM_SYSTEM_LIBS_WIN})
    endif ()

    # starting with LLVM 3.4 (at least on Ubuntu) it requres functions in
    # ncurses for console IO formatting. So, we find ncurses here.
    if (UNIX)
        #message("UNIX true")
        #        message("LLVM VERSION >= 3.4, looking for curses library")
        # sudo apt-get install libncurses5-dev libncursesw5-dev
        #find_package(Curses REQUIRED)
        #        message(STATUS "curses: ${CURSES_FOUND}")
        #        message(STATUS "curdir: ${CURSES_INCLUDE_DIR}")
        #        message(STATUS "curlib: ${CURSES_LIBRARIES}")
        #        message(STATUS "LLVM_LIBRARIES: ${LLVM_LIBRARIES}")

        #set(LLVM_LIBRARIES "${LLVM_LIBRARIES};${CURSES_LIBRARIES}")


        # LLVM 3.5 seems to require zlib, at least on OSX 10.9.
        # no big deal to just add it on UNIX in general as it already there.
        if (LLVM_VERSION_MINOR GREATER 4)
            message("LLVM > 3.4, looking for zlib")
            find_package(ZLIB REQUIRED)
            set(LLVM_LIBRARIES "${LLVM_LIBRARIES};${ZLIB_LIBRARY}")
        endif ()

    endif ()


    if (LLVM_INCLUDE_DIRS)
        set(LLVM_FOUND TRUE)
        message(STATUS "Found LLVM: ${LLVM_INCLUDE_DIRS}")
    endif (LLVM_INCLUDE_DIRS)


else (LLVM_CONFIG_EXECUTABLE)
    message(STATUS "Could NOT find LLVM executable")
endif (LLVM_CONFIG_EXECUTABLE)

if (NOT LLVM_FOUND AND LLVM_FIND_REQUIRED)
    message(FATAL_ERROR "Could NOT find LLVM ${RR_LLVM_MIN_VERSION} or newer. Set LLVM_INSTALL_PREFIX to the "
            "LLVM install prefix (it contains lib/cmake/llvm), or LLVM_DIR to its lib/cmake/llvm directory.")
endif (NOT LLVM_FOUND AND LLVM_FIND_REQUIRED)

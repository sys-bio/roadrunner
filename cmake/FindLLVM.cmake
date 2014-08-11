# Find LLVM
#
# It defines the following variables
#  LLVM_FOUND        - True if llvm found.
#  LLVM_INCLUDE_DIRS  - where to find llvm include files
#  LLVM_LIBRARY_DIRS  - where to find llvm libs
#  LLVM_LDFLAGS       - llvm linker flags
#  LLVM_LIBRARIES          - list of llvm libs for working with modules.
#                      This is a list of absolute path file names, which is
#                      evidently the cmake convention.
#  LLVM_FLAGS     - flags to add to the CXX compiler for llvm support
#  LLVM_VERSION_MAJOR
#  LLVM_VERSION_MINOR
#  LLVM_VERSION_PATCH
#################################################################################

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

    if(NOT patch MATCHES "^[0-9]+$")
        set(patch 0)
    endif(NOT patch MATCHES "^[0-9]+$")

    if(NOT minor MATCHES "[0-9]+")
        set(minor 0)
    endif(NOT minor MATCHES "[0-9]+")

    if(NOT major MATCHES "[0-9]+")
        set(major 0)
    endif(NOT major MATCHES "[0-9]+")
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


find_program(LLVM_CONFIG_EXECUTABLE
    NAMES llvm-config-${LLVM_MIN_VERSION_TEXT} llvm-config
    PATHS /opt/local/bin /usr/local/bin
    HINTS "$ENV{LLVM_DIR}/bin"
    DOC "llvm-config executable"
    )

if (LLVM_CONFIG_EXECUTABLE)
    message(STATUS "LLVM llvm-config found at: ${LLVM_CONFIG_EXECUTABLE}")

    execute_process(
        COMMAND ${LLVM_CONFIG_EXECUTABLE} --version
        OUTPUT_VARIABLE LLVM_VERSION
        OUTPUT_STRIP_TRAILING_WHITESPACE
        )

    # Version Info
    execute_process(COMMAND ${LLVM_CONFIG_EXECUTABLE} --version OUTPUT_VARIABLE LLVM_STRING_VERSION )
    string(REGEX REPLACE "^([0-9]+)\\.([0-9]+)([^\n\r\t ]*)" "\\1" LLVM_VERSION_MAJOR
        "${LLVM_VERSION}")
    string(REGEX REPLACE "^([0-9]+)\\.([0-9]+)([^\n\r\t ]*)" "\\2" LLVM_VERSION_MINOR
        "${LLVM_VERSION}")
    string(REGEX REPLACE "^([0-9]+)\\.([0-9]+)([^\n\r\t ]*)" "\\3" LLVM_VERSION_PATCH
        "${LLVM_VERSION}")
    message(STATUS "LLVM_VERSION_MAJOR: ${LLVM_VERSION_MAJOR}")
    message(STATUS "LLVM_VERSION_MINOR: ${LLVM_VERSION_MINOR}")
    message(STATUS "LLVM_VERSION_PATCH: ${LLVM_VERSION_PATCH}")

    execute_process(COMMAND ${LLVM_CONFIG_EXECUTABLE} --bindir OUTPUT_VARIABLE LLVM_BIN_DIR )
    execute_process(COMMAND ${LLVM_CONFIG_EXECUTABLE} --libdir OUTPUT_VARIABLE LLVM_LIB_DIR )
    MESSAGE(STATUS "LLVM_BIN_DIR: " ${LLVM_BIN_DIR})
    MESSAGE(STATUS "LLVM_LIB_DIR: " ${LLVM_LIB_DIR})

    # Include Dir
    execute_process(
        COMMAND ${LLVM_CONFIG_EXECUTABLE} --includedir
        OUTPUT_VARIABLE LLVM_INCLUDE_DIRS
        OUTPUT_STRIP_TRAILING_WHITESPACE
        )
    message(STATUS "LLVM_INCLUDE_DIRS: ${LLVM_INCLUDE_DIRS}")

    # Lib Dir
    execute_process(
        COMMAND ${LLVM_CONFIG_EXECUTABLE} --libdir
        OUTPUT_VARIABLE LLVM_LIBRARY_DIRS
        OUTPUT_STRIP_TRAILING_WHITESPACE
        )
    message(STATUS "LLVM_LIBRARY_DIRS:  ${LLVM_LIBRARY_DIRS}")

    # C++ Flags, strip out stuff that CMake build adds
    execute_process(
        COMMAND ${LLVM_CONFIG_EXECUTABLE} --cxxflags
        OUTPUT_VARIABLE LLVM_FLAGS
        OUTPUT_STRIP_TRAILING_WHITESPACE
        )
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


    MESSAGE(STATUS "LLVM_FLAGS: " ${LLVM_FLAGS})



    # link libraries, currently only need core, jit and native.
    # TODO: in future, replace this with something like LLVM_CORE_LIBS, LLVM_JIT_LIBS...
    execute_process(
        COMMAND ${LLVM_CONFIG_EXECUTABLE} --libfiles core jit mcjit native
        OUTPUT_VARIABLE LLVM_LIBRARIES
        OUTPUT_STRIP_TRAILING_WHITESPACE
        )
    # we get a space sep list from llvm-config, make it a cmake ; separated list.
    STRING(REGEX REPLACE "[\n\t\r ]+" ";" LLVM_LIBRARIES ${LLVM_LIBRARIES})
    message(STATUS "LLVM_LIBRARIES: ${LLVM_LIBRARIES}")

    # starting with LLVM 3.4 (at least on Ubuntu) it requres functions in
    # ncurses for console IO formatting. So, we find ncurses here.

    if (((LLVM_VERSION_MAJOR GREATER 3) OR (LLVM_VERSION_MAJOR EQUAL 3)) AND
            (LLVM_VERSION_MINOR GREATER 4) OR (LLVM_VERSION_MINOR EQUAL 4))
        if (UNIX)
            #message("UNIX true")
            message("LLVM VERSION >= 3.4, looking for curses library")
            find_package(Curses REQUIRED)
            #message("curses: ${CURSES_FOUND}")
            #message("curdir: ${CURSES_INCLUDE_DIR}")
            #message("curlib: ${CURSES_LIBRARIES}")
            #message("LLVM_LIBRARIES: ${LLVM_LIBRARIES}")

            set(LLVM_LIBRARIES "${LLVM_LIBRARIES};${CURSES_LIBRARIES}")
            message("LLVM_LIBRARIES: ${LLVM_LIBRARIES}")

        else()
            #message("NOT UNIX")
        endif()    

    else()
        #message("LLVM VERSION < 3.4")
    endif()

    if(LLVM_INCLUDE_DIRS)
        set(LLVM_FOUND TRUE)
        message(STATUS "Found LLVM: ${LLVM_INCLUDE_DIRS}")
    endif(LLVM_INCLUDE_DIRS)


else (LLVM_CONFIG_EXECUTABLE)
    message(STATUS "Could NOT find LLVM executable")
endif (LLVM_CONFIG_EXECUTABLE)

if(NOT LLVM_FOUND AND LLVM_FIND_REQUIRED)
    message(FATAL_ERROR "Could NOT find LLVM")
endif(NOT LLVM_FOUND AND LLVM_FIND_REQUIRED)

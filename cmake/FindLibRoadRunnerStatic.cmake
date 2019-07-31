#.rst:
# FindLibRoadRunnerStatic
# -----------
#
# Try to find libRoadRunner, an SBML-compliant simulation library.
# Find static versions of libs.
# Set LIBROADRUNNERSTATIC_PATH to give a hint about where to search.
#
# Once done this will define
#
# ::
#
#   LIBROADRUNNERSTATIC_FOUND - System has libRoadRunner static libs
#   LIBROADRUNNERSTATIC_INCLUDE_DIR - The libRoadRunner include directory
#   LIBROADRUNNERSTATIC_LIBRARIES - The libraries needed to use libRoadRunner
#   LIBROADRUNNERSTATIC_DEFINITIONS - Compiler switches required for using libRoadRunner
#   LIBROADRUNNERSTATIC_VERSION_STRING - the version of libRoadRunner found

#=============================================================================
# Copyright 2015 J. Kyle Medley
#
# Licensed under the Apache License, Version 2.0
#
# This software is distributed WITHOUT ANY WARRANTY; without even the
# implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the License for more information.
#
# Based on FindLibXml2.cmake
#
#=============================================================================

# find version info file
find_file(LIBROADRUNNERSTATIC_VERSION_FILE NAME "VERSION.txt"
  HINTS
  ${LIBROADRUNNERSTATIC_PATH}
  )

if(LIBROADRUNNERSTATIC_VERSION_FILE AND EXISTS "${LIBROADRUNNERSTATIC_VERSION_FILE}" AND NOT IS_DIRECTORY ${LIBROADRUNNERSTATIC_VERSION_FILE})
  file(STRINGS "${LIBROADRUNNERSTATIC_VERSION_FILE}" LIBROADRUNNERSTATIC_VERSION_STRING
      REGEX "^[0-9]+.[0-9]+.[0-9]+")
endif()

# include dirs
find_path(LIBROADRUNNERSTATIC_INCLUDE_DIR NAMES rrRoadRunner.h
  HINTS
  ${LIBROADRUNNERSTATIC_PATH}/include/rr
  )

# libraries
find_library(LIBROADRUNNERSTATIC_LIBRARY NAMES roadrunner-static
  HINTS
  ${LIBROADRUNNERSTATIC_PATH}/lib
  )

find_library(SUNDIALS_CVODE_LIBRARY NAMES sundials_cvode
  HINTS
  ${LIBROADRUNNERSTATIC_PATH}/lib
  )

find_library(SUNDIALS_NVECSERIAL_LIBRARY NAMES sundials_nvecserial
  HINTS
  ${LIBROADRUNNERSTATIC_PATH}/lib
  )

find_library(F2C_LIBRARY NAMES f2c
  HINTS
  ${LIBROADRUNNERSTATIC_PATH}/lib
  )

find_library(BLAS_LIBRARY NAMES blas
  HINTS
  ${LIBROADRUNNERSTATIC_PATH}/lib
  )

find_library(LAPACK_LIBRARY NAMES lapack
  HINTS
  ${LIBROADRUNNERSTATIC_PATH}/lib
  )

find_library(NLEQ_LIBRARY_STATIC NAMES nleq-static nleq1-static
  HINTS
  ${LIBROADRUNNERSTATIC_PATH}/lib
  )

find_library(RR_LIBSTRUCT_LIBRARY_STATIC NAMES rr-libstruct-static
  HINTS
  ${LIBROADRUNNERSTATIC_PATH}/lib
  )

find_library(POCO_FOUNDATION_LIBRARY_STATIC NAMES PocoFoundation
  HINTS
  ${LIBROADRUNNERSTATIC_PATH}/lib
  )

message(STATUS "Searching for libsbml-static in ${LIBROADRUNNERSTATIC_PATH}/lib")
find_library(LIBROADRUNNERSTATIC_SBML_LIBRARY
  NAMES
  sbml-static
  libsbml-static.lib # Windows
  HINTS
  ${LIBROADRUNNERSTATIC_PATH}/lib
  )
  message(STATUS "lisbml-static = ${LIBROADRUNNERSTATIC_SBML_LIBRARY}")
get_filename_component(LIBROADRUNNERSTATIC_SBML_LIBRARY_BARE ${LIBROADRUNNERSTATIC_SBML_LIBRARY} NAME)
get_filename_component(LIBROADRUNNERSTATIC_SBML_LIBRARY_DIR  ${LIBROADRUNNERSTATIC_SBML_LIBRARY} DIRECTORY)

set ( LIBROADRUNNERSTATIC_LIBRARIES ${LIBROADRUNNERSTATIC_LIBRARY};${SUNDIALS_CVODE_LIBRARY};
      ${SUNDIALS_NVECSERIAL_LIBRARY};${F2C_LIBRARY};${BLAS_LIBRARY};${LAPACK_LIBRARY};
      ${NLEQ_LIBRARY_STATIC};${RR_LIBSTRUCT_LIBRARY_STATIC};
      ${POCO_FOUNDATION_LIBRARY_STATIC};${LIBROADRUNNERSTATIC_SBML_LIBRARY_BARE} )
message( STATUS ${LIBROADRUNNERSTATIC_LIBRARIES} )

# handle the QUIETLY and REQUIRED arguments and set LIBROADRUNNERSTATIC_FOUND to TRUE if
# all listed variables are TRUE
include(${CMAKE_CURRENT_LIST_DIR}/FindPackageHandleStandardArgs.cmake)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(LibRoadRunnerStatic
                                  REQUIRED_VARS LIBROADRUNNERSTATIC_LIBRARIES LIBROADRUNNERSTATIC_INCLUDE_DIR
                                  VERSION_VAR LIBROADRUNNERSTATIC_VERSION_STRING)

mark_as_advanced(LIBROADRUNNERSTATIC_INCLUDE_DIR LIBROADRUNNERSTATIC_LIBRARIES)

#.rst:
# FindLibRoadRunner
# -----------
#
# Try to find libRoadRunner, an SBML-compliant simulation library.
# Set LIBROADRUNNER_PATH to give a hint about where to search.
#
# Once done this will define
#
# ::
#
#   LIBROADRUNNER_FOUND - System has libRoadRunner
#   LIBROADRUNNER_INCLUDE_DIR - The libRoadRunner include directory
#   LIBROADRUNNER_LIBRARIES - The libraries needed to use libRoadRunner
#   LIBROADRUNNER_DEFINITIONS - Compiler switches required for using libRoadRunner
#   LIBROADRUNNER_VERSION_STRING - the version of libRoadRunner found

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
find_file(LIBROADRUNNER_VERSION_FILE NAME "VERSION.txt"
  HINTS
  ${LIBROADRUNNER_PATH}
  )

if(LIBROADRUNNER_VERSION_FILE AND EXISTS "${LIBROADRUNNER_VERSION_FILE}" AND NOT IS_DIRECTORY ${LIBROADRUNNER_VERSION_FILE})
  file(STRINGS "${LIBROADRUNNER_VERSION_FILE}" LIBROADRUNNER_VERSION_STRING
      REGEX "^[0-9]+.[0-9]+.[0-9]+")
endif()

# include dirs
find_path(LIBROADRUNNER_INCLUDE_DIR NAMES rrRoadRunner.h
  HINTS
  ${LIBROADRUNNER_PATH}/include/rr
  )

# libraries
find_library(LIBROADRUNNER_LIBRARY NAMES roadrunner
  HINTS
  ${LIBROADRUNNER_PATH}/lib
  )

find_library(SUNDIALS_CVODE_LIBRARY NAMES sundials_cvode
  HINTS
  ${LIBROADRUNNER_PATH}/lib
  )

find_library(SUNDIALS_NVECSERIAL_LIBRARY NAMES sundials_nvecserial
  HINTS
  ${LIBROADRUNNER_PATH}/lib
  )

find_library(F2C_LIBRARY
  NAMES
  f2c
  libf2c.lib # Windows
  HINTS
  ${LIBROADRUNNER_PATH}/lib
  )

find_library(BLAS_LIBRARY NAMES blas
  HINTS
  ${LIBROADRUNNER_PATH}/lib
  )

find_library(LAPACK_LIBRARY NAMES lapack
  HINTS
  ${LIBROADRUNNER_PATH}/lib
  )

find_library(NLEQ_LIBRARY_STATIC NAMES nleq-static nleq1-static
  HINTS
  ${LIBROADRUNNER_PATH}/lib
  )

find_library(RR_LIBSTRUCT_LIBRARY_STATIC NAMES rr-libstruct-static
  HINTS
  ${LIBROADRUNNER_PATH}/lib
  )

find_library(POCO_FOUNDATION_LIBRARY_STATIC NAMES PocoFoundation
  HINTS
  ${LIBROADRUNNER_PATH}/lib
  )

find_library(SBML_LIBRARY
  NAMES
  sbml-static
  libsbml-static.lib # Windows
  HINTS
  ${LIBROADRUNNER_PATH}/lib
  )
get_filename_component(SBML_LIBRARY_BARE ${SBML_LIBRARY} NAME)
get_filename_component(SBML_LIBRARY_DIR  ${SBML_LIBRARY} DIRECTORY)

set ( LIBROADRUNNER_LIBRARIES ${LIBROADRUNNER_LIBRARY};${SUNDIALS_CVODE_LIBRARY};
      ${SUNDIALS_NVECSERIAL_LIBRARY};${F2C_LIBRARY};${BLAS_LIBRARY};${LAPACK_LIBRARY};
      ${NLEQ_LIBRARY_STATIC};${RR_LIBSTRUCT_LIBRARY_STATIC};
      ${POCO_FOUNDATION_LIBRARY_STATIC};${SBML_LIBRARY_BARE};
      ${SBML_LIBRARY})

if (${CMAKE_SYSTEM_NAME} MATCHES "Windows")
  find_library(ICONV_LIBRARY
    NAMES
    libiconv.lib
    HINTS
    ${LIBROADRUNNER_PATH}/lib
    )
  find_library(ZLIB_LIBRARY
    NAMES
    zdll.lib
    HINTS
    ${LIBROADRUNNER_PATH}/lib
    )
  list ( APPEND LIBROADRUNNER_LIBRARIES ${ICONV_LIBRARY};${ZLIB_LIBRARY};ws2_32.lib;Iphlpapi.lib )
endif ()
message( STATUS "LIBROADRUNNER_LIBRARIES: ${LIBROADRUNNER_LIBRARIES}" )

# handle the QUIETLY and REQUIRED arguments and set LIBROADRUNNER_FOUND to TRUE if
# all listed variables are TRUE
include(${CMAKE_CURRENT_LIST_DIR}/FindPackageHandleStandardArgs.cmake)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(LibRoadRunner
                                  REQUIRED_VARS LIBROADRUNNER_LIBRARIES LIBROADRUNNER_INCLUDE_DIR
                                  VERSION_VAR LIBROADRUNNER_VERSION_STRING)

mark_as_advanced(LIBROADRUNNER_INCLUDE_DIR LIBROADRUNNER_LIBRARIES)

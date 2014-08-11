## @file    CMakeLists.txt
## @brief   CMake build script for example programs
## @author  Frank T. Bergmann
##
## <!--------------------------------------------------------------------------
## This file is part of libSBML.  Please visit http://sbml.org for more
## information about SBML, and the latest version of libSBML.
##
## Copyright (C) 2013-2014 jointly by the following organizations:
##     1. California Institute of Technology, Pasadena, CA, USA
##     2. EMBL European Bioinformatics Institute (EMBL-EBI), Hinxton, UK
##     3. University of Heidelberg, Heidelberg, Germany
##
## Copyright (C) 2009-2013 jointly by the following organizations: 
##     1. California Institute of Technology, Pasadena, CA, USA
##     2. EMBL European Bioinformatics Institute (EMBL-EBI), Hinxton, UK
##  
## Copyright (C) 2006-2008 by the California Institute of Technology,
##     Pasadena, CA, USA 
##  
## Copyright (C) 2002-2005 jointly by the following organizations: 
##     1. California Institute of Technology, Pasadena, CA, USA
##     2. Japan Science and Technology Agency, Japan
## 
## This library is free software; you can redistribute it and/or modify it
## under the terms of the GNU Lesser General Public License as published by
## the Free Software Foundation.  A copy of the license agreement is provided
## in the file named "LICENSE.txt" included with this software distribution
## and also available online as http://sbml.org/software/libsbml/license.html
## ------------------------------------------------------------------------ -->


include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(BEFORE ${LIBSBML_ROOT_SOURCE_DIR}/src)
include_directories(${LIBSBML_ROOT_SOURCE_DIR}/include)
include_directories(BEFORE ${LIBSBML_ROOT_BINARY_DIR}/src)


if (EXTRA_INCLUDE_DIRS) 
include_directories(${EXTRA_INCLUDE_DIRS})
endif(EXTRA_INCLUDE_DIRS)

foreach(example 

	flattenModel
	flattenModelAdvanced
	spec_example1
	spec_example2
	spec_example3
	spec_example4
	
)
	add_executable(example_comp_cpp_${example} ${example}.cpp ../util.c)
	set_target_properties(example_comp_cpp_${example} PROPERTIES  OUTPUT_NAME ${example})
	target_link_libraries(example_comp_cpp_${example} ${LIBSBML_LIBRARY}-static)
	
	if (WITH_LIBXML)
		target_link_libraries(example_comp_cpp_${example} ${LIBXML_LIBRARY} ${EXTRA_LIBS})
	endif()

	if (WITH_ZLIB)
		target_link_libraries(example_comp_cpp_${example} ${LIBZ_LIBRARY})
	endif(WITH_ZLIB)
	if (WITH_BZIP2)
		target_link_libraries(example_comp_cpp_${example} ${LIBBZ_LIBRARY})
	endif(WITH_BZIP2)

endforeach()


# build web resolver example
# for which we need libcurl

find_library(LIBCURL_LIBRARY
  NAMES libcurl.lib curl
  PATHS /usr/lib /usr/local/lib
        ${LIBSBML_DEPENDENCY_DIR}/lib
  DOC "The file name of the curl library."
)
find_path(LIBCURL_INCLUDE_DIR
  NAMES curl/curl.h
  PATHS /usr/include /usr/local/include
        ${LIBSBML_DEPENDENCY_DIR}/include
  DOC "The directory containing the curl include files."
)

if (${LIBCURL_LIBRARY} AND ${LIBCURL_INCLUDE_DIR})
  include_directories(${CMAKE_CURRENT_SOURCE_DIR})
  include_directories(${LIBCURL_INCLUDE_DIR})
  include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
  add_definitions(-DCURL_STATICLIB)
  add_executable(example_comp_cpp_SBMLHttpResolverExample SBMLHttpResolverExample.cpp ../util.c)
  set_target_properties(example_comp_cpp_SBMLHttpResolverExample PROPERTIES  OUTPUT_NAME SBMLHttpResolverExample)
  target_link_libraries(example_comp_cpp_SBMLHttpResolverExample ${LIBSBML_LIBRARY}-static ${LIBCURL_LIBRARY})
  if (WITH_LIBXML)
  	target_link_libraries(example_comp_cpp_SBMLHttpResolverExample ${LIBXML_LIBRARY} ${EXTRA_LIBS})
  endif()
else()
  message(WARNING "comp example: The SBML HTTP Resolver example cannot be built as libcurl was not found.")
endif (${LIBCURL_LIBRARY} AND ${LIBCURL_INCLUDE_DIR})

# install c++ examples
file(GLOB cpp_samples "${CMAKE_CURRENT_SOURCE_DIR}/*.c"
                    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
                    "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
install(FILES ${cpp_samples} DESTINATION ${MISC_PREFIX}examples/c++/comp)


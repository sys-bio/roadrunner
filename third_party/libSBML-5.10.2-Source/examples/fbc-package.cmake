###############################################################################
#
# Description       : CMake macros for SBML Flux Balance Constraints package
# Original author(s): Lucian Smith <lpsmith@u.washington.edu>
# Organization      : California Institute of Technology
#
# This file is part of libSBML.  Please visit http://sbml.org for more
# information about SBML, and the latest version of libSBML.
#
# Copyright (C) 2013-2014 jointly by the following organizations:
#     1. California Institute of Technology, Pasadena, CA, USA
#     2. EMBL European Bioinformatics Institute (EMBL-EBI), Hinxton, UK
#     3. University of Heidelberg, Heidelberg, Germany
#
# Copyright (C) 2009-2013 jointly by the following organizations: 
#     1. California Institute of Technology, Pasadena, CA, USA
#     2. EMBL European Bioinformatics Institute (EMBL-EBI), Hinxton, UK
#  
# Copyright (C) 2006-2008 by the California Institute of Technology,
#     Pasadena, CA, USA 
#  
# Copyright (C) 2002-2005 jointly by the following organizations: 
#     1. California Institute of Technology, Pasadena, CA, USA
#     2. Japan Science and Technology Agency, Japan
# 
# This library is free software; you can redistribute it and/or modify it
# under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation.  A copy of the license agreement is provided
# in the file named "LICENSE.txt" included with this software distribution
# and also available online as http://sbml.org/software/libsbml/license.html
#
###############################################################################

if (ENABLE_FBC)
  # add_subdirectory(c/fbc)
  add_subdirectory(c++/fbc)
  
  if(WITH_CSHARP)
    add_subdirectory(csharp/fbc)
  endif(WITH_CSHARP)
  
  if(WITH_JAVA)
    add_subdirectory(java/fbc)
  endif(WITH_JAVA)

  
  if(WITH_PERL)
  # install perl examples
  file(GLOB perl_fbc_samples "${CMAKE_CURRENT_SOURCE_DIR}/perl/fbc/*.pl"
                         "${CMAKE_CURRENT_SOURCE_DIR}/perl/fbc/README.txt")
  install(FILES ${perl_fbc_samples} DESTINATION ${MISC_PREFIX}examples/perl/fbc)
  endif()
  
  if (WITH_PYTHON)
  # install python examples
  file(GLOB python_fbc_samples "${CMAKE_CURRENT_SOURCE_DIR}/python/fbc/*.py"
                           "${CMAKE_CURRENT_SOURCE_DIR}/python/fbc/README.txt")
  install(FILES ${python_fbc_samples} DESTINATION ${MISC_PREFIX}examples/python/fbc)
  endif()
  
  
  if(WITH_RUBY)
  # install ruby examples
  file(GLOB ruby_fbc_samples "${CMAKE_CURRENT_SOURCE_DIR}/ruby/fbc/*.rb"
                         "${CMAKE_CURRENT_SOURCE_DIR}/ruby/fbc/README.txt")
  install(FILES ${ruby_fbc_samples} DESTINATION ${MISC_PREFIX}examples/ruby/fbc)
  endif()
  
  if (WITH_R)
  # install r examples
  file(GLOB r_fbc_samples "${CMAKE_CURRENT_SOURCE_DIR}/r/fbc/*.R"
                           "${CMAKE_CURRENT_SOURCE_DIR}/r/fbc/README.txt")
  install(FILES ${r_fbc_samples} DESTINATION ${MISC_PREFIX}examples/r/fbc)
  endif()
  
endif(ENABLE_FBC)

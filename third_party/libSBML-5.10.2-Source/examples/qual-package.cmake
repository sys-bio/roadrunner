###############################################################################
#
# Description       : CMake macros for SBML Hierarchical Composition package
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

if (ENABLE_QUAL)
  # add_subdirectory(c/qual)
  add_subdirectory(c++/qual)
  
  if(WITH_CSHARP)
    add_subdirectory(csharp/qual)
  endif(WITH_CSHARP)
  
  if(WITH_JAVA)
    add_subdirectory(java/qual)
  endif(WITH_JAVA)

  if(WITH_PERL)
  # install perl examples
  file(GLOB perl_qual_samples "${CMAKE_CURRENT_SOURCE_DIR}/perl/qual/*.pl"
                         "${CMAKE_CURRENT_SOURCE_DIR}/perl/qual/README.txt")
  install(FILES ${perl_qual_samples} DESTINATION ${MISC_PREFIX}examples/perl/qual)
  endif()
  
  if (WITH_PYTHON)
  # install python examples
  file(GLOB python_qual_samples "${CMAKE_CURRENT_SOURCE_DIR}/python/qual/*.py"
                           "${CMAKE_CURRENT_SOURCE_DIR}/python/qual/README.txt")
  install(FILES ${python_qual_samples} DESTINATION ${MISC_PREFIX}examples/python/qual)
  endif()
  
  
  if(WITH_RUBY)
  # install ruby examples
  file(GLOB ruby_qual_samples "${CMAKE_CURRENT_SOURCE_DIR}/ruby/qual/*.rb"
                         "${CMAKE_CURRENT_SOURCE_DIR}/ruby/qual/README.txt")
  install(FILES ${ruby_qual_samples} DESTINATION ${MISC_PREFIX}examples/ruby/qual)
  endif()
  
  if (WITH_R)
  # install r examples
  file(GLOB r_qual_samples "${CMAKE_CURRENT_SOURCE_DIR}/r/qual/*.R"
                           "${CMAKE_CURRENT_SOURCE_DIR}/r/qual/README.txt")
  install(FILES ${r_qual_samples} DESTINATION ${MISC_PREFIX}examples/r/qual)
  endif()
  
endif(ENABLE_QUAL)

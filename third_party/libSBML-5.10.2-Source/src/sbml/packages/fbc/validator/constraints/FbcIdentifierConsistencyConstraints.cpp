/**
 * @cond doxygenLibsbmlInternal
 *
 * @file    IdentifierConsistencyConstraints.cpp
 * @brief   IdentifierConsistency check constraints.  See SBML Wiki
 * @author  Sarah Keating
 * 
 * <!--------------------------------------------------------------------------
 * This file is part of libSBML.  Please visit http://sbml.org for more
 * information about SBML, and the latest version of libSBML.
 * 
 * Copyright (C) 2013-2014 jointly by the following organizations:
 *     1. California Institute of Technology, Pasadena, CA, USA
 *     2. EMBL European Bioinformatics Institute (EMBL-EBI), Hinxton, UK
 *     3. University of Heidelberg, Heidelberg, Germany
 * 
 * Copyright (C) 2009-2013 jointly by the following organizations: 
 *     1. California Institute of Technology, Pasadena, CA, USA
 *     2. EMBL European Bioinformatics Institute (EMBL-EBI), Hinxton, UK
 *
 * This library is free software; you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation.  A copy of the license agreement is provided
 * in the file named "LICENSE.txt" included with this software distribution
 * and also available online as http://sbml.org/software/libsbml/license.html
 * ---------------------------------------------------------------------- -->*/

#ifndef AddingConstraintsToValidator

#include <sbml/validator/VConstraint.h>
//#include <sbml/packages/fbc/sbml/Objective.h>
//#include <sbml/packages/fbc/sbml/FluxBound.h>
//
#include <sbml/packages/fbc/validator/FbcSBMLError.h>
//
#include "UniqueModelWideIds.h"
#endif

#include <sbml/validator/ConstraintMacros.h>

/** @cond doxygenIgnored */

using namespace std;
// 10301
EXTERN_CONSTRAINT(FbcDuplicateComponentId, UniqueModelWideIds);
// 10302 - caught at read
/** @endcond */



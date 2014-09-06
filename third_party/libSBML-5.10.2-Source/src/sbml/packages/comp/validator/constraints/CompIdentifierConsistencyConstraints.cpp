/**
 * @cond doxygenLibsbmlInternal
 *
 * @file    CompIdentifierConsistencyConstraints.cpp
 * @brief   CompIdentifierConsistency check constraints.
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
 * Copyright 2011-2012 jointly by the following organizations:
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
#include <sbml/packages/comp/validator/CompSBMLError.h>
#include "UniqueModelIds.h"
#include "UniqueComponentIds.h"
#include "UniquePortIds.h"
#endif

#include <sbml/validator/ConstraintMacros.h>

/** @cond doxygenIgnored */

using namespace std;

/** @endcond */


// General Identifier validation 
//1010301
EXTERN_CONSTRAINT( CompDuplicateComponentId, UniqueComponentIds             )

//1010302
EXTERN_CONSTRAINT( CompUniqueModelIds, UniqueModelIds             )

//1010303 
EXTERN_CONSTRAINT( CompUniquePortIds, UniquePortIds             )

//1010304 syntax of id - caught at read

// 1010305- 1010307 non existant

//1010308 - submodelRef syntax - caught at read
//1010309 - deletion syntax - caught at read
//1010310 - conversionfactor syntax - caught at read
//1010311 - string syntax - since an attribute value is a string 
            // by the time it is read in we cannot check this


/** @endcond */


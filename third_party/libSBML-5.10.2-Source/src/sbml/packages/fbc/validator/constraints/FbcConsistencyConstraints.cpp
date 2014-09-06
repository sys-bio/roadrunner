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
#include <sbml/packages/fbc/sbml/Objective.h>
#include <sbml/packages/fbc/sbml/FluxBound.h>

#include <sbml/packages/fbc/validator/FbcSBMLError.h>

#include "FluxBoundsConsistent.h"
#endif

#include <sbml/validator/ConstraintMacros.h>

/** @cond doxygenIgnored */

using namespace std;
// 20101 - caught at read
// 20102 - caught at read
// 20103 - caught at read

// 20201 - caught at read
// 20202 - caught at read
// 20203 - caught at read
// 20204 - caught at read
// 20205 - caught at read
// 20206 - caught at read
// 20207 - caught at read

// 20208 - TO DO
START_CONSTRAINT (FbcActiveObjectiveRefersObjective, ListOfObjectives, loObj)
{
  pre (loObj.isSetActiveObjective());

  bool fail = false;

  msg = "<lisOfObjectives> has an activeObjective '";
  msg += loObj.getActiveObjective() ;
  msg += "' that does not refer to an <objective> within the <model>.";

  std::string activeObj = loObj.getActiveObjective();
  
  if (loObj.get(activeObj) == NULL)
  {
    fail = true;
  }

  inv(fail == false);
}
END_CONSTRAINT

// 20301 - caught at read
// 20302 - caught at read
// 20303 - TO DO

// 20401 - caught at read
// 20402 - caught at read
// 20403 - caught at read
// 20404 - caught at read
// 20405 - caught at read
// 20406 - caught at read
// 20407 - caught at read

// 20408
START_CONSTRAINT (FbcFluxBoundReactionMustExist, FluxBound, fb)
{
  pre (fb.isSetReaction());

  bool fail = false;

  msg = "<fluxBound> '";
  msg += fb.getId() ;
  msg += "' refers to reaction with id '";
  msg += fb.getReaction();
  msg += "' that does not exist within the <model>.";

  std::string reaction = fb.getReaction();
  
  if (m.getReaction(reaction) == NULL)
  {
    fail = true;
  }

  inv(fail == false);
}
END_CONSTRAINT

// 20409
EXTERN_CONSTRAINT( FbcFluxBoundsForReactionConflict, FluxBoundsConsistent)

// 20501 - caught at read
// 20502 - caught at read
// 20503 - caught at read
// 20505 - caught at read
// 20505 - caught at read
// 20506
START_CONSTRAINT (FbcObjectiveOneListOfObjectives, Objective, obj)
{
  bool fail = false;

  msg = "<objective> '";
  msg += obj.getId() ;
  msg += "' has no listOfFluxObjectives.";

  if (obj.getIsSetListOfFluxObjectives() == false)
  {
    fail = true;
  }
  else if (obj.getNumFluxObjectives() == 0)
  {
    fail = true;
  }

  inv(fail == false);
}
END_CONSTRAINT

// 20507 - caught at read
// 20508 - caught at read
// 20509 - caught at read

// 20601 - caught at read
// 20602 - caught at read
// 20603 - caught at read
// 20606 - caught at read
// 20606 - caught at read

// 20606
START_CONSTRAINT (FbcFluxObjectReactionMustExist, FluxObjective, fo)
{
  pre (fo.isSetReaction());

  bool fail = false;

  msg = "<fluxObjective> '";
  msg += fo.getId() ;
  msg += "' refers to reaction with id '";
  msg += fo.getReaction();
  msg += "' that does not exist within the <model>.";

  std::string reaction = fo.getReaction();
  
  if (m.getReaction(reaction) == NULL)
  {
    fail = true;
  }

  inv(fail == false);
}
END_CONSTRAINT

// 20607 - caught at read
/** @endcond */



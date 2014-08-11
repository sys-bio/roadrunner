/**
 * @cond doxygenLibsbmlInternal
 *
 * @file    QualConsistencyConstraints.cpp
 * @brief   QualConsistency check constraints.  See SBML Wiki
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
 * Copyright (C) 2006-2008 by the California Institute of Technology,
 *     Pasadena, CA, USA 
 *  
 * Copyright (C) 2002-2005 jointly by the following organizations: 
 *     1. California Institute of Technology, Pasadena, CA, USA
 *     2. Japan Science and Technology Agency, Japan
 * 
 * This library is free software; you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation.  A copy of the license agreement is provided
 * in the file named "LICENSE.txt" included with this software distribution
 * and also available online as http://sbml.org/software/libsbml/license.html
 * ---------------------------------------------------------------------- -->*/

#ifndef AddingConstraintsToValidator

#include <sbml/validator/VConstraint.h>
#include <sbml/packages/qual/sbml/Output.h>
#include <sbml/packages/qual/sbml/QualitativeSpecies.h>
#include <sbml/packages/qual/extension/QualModelPlugin.h>
#include <sbml/packages/qual/extension/QualSBMLDocumentPlugin.h>

#include <sbml/packages/qual/validator/QualSBMLError.h>

#include "ResultExceedsMaximum.h"
#include "ResultBecomesNegative.h"
#include "QSAssignedOnce.h"

#endif

#include <sbml/validator/ConstraintMacros.h>

/** @cond doxygenIgnored */

using namespace std;

/** @endcond */

// 20101 - 20102 - caught at read

//20103
START_CONSTRAINT (QualRequiredTrueIfTransitions, Model, x)
{
  bool fail = false;

  // do not ever log this error as the interpretation of required
  // changed

  //QualSBMLDocumentPlugin * plug = (QualSBMLDocumentPlugin*)(x.getSBMLDocument()
  //  ->getPlugin("qual"));
  //QualModelPlugin *plug1 = (QualModelPlugin*)(x.getPlugin("qual"));

  //pre (plug->isSetRequired());

  //if (plug->getRequired() == false
  //  && plug1->getNumTransitions() > 0)
  //{
  //  fail = true;
  //}

  inv(fail == false );
}
END_CONSTRAINT

// 20201 - 20206 - caught at read

// 20301 - 20307 - caught at read
//20308
START_CONSTRAINT (QualCompartmentMustReferExisting, QualitativeSpecies, qs)
{
  pre( qs.isSetCompartment() );

  bool fail = false;

  msg =
    "Compartment '" + qs.getCompartment() + "' is undefined. ";

  if (m.getCompartment(qs.getCompartment()) == NULL )
  {
    fail = true;
  }

  inv(fail == false );
}
END_CONSTRAINT

//20309
START_CONSTRAINT (QualInitialLevelCannotExceedMax, QualitativeSpecies, qs)
{
  pre( qs.isSetInitialLevel() );
  pre( qs.isSetMaxLevel() );

  bool fail = false;

  if (qs.getInitialLevel() > qs.getMaxLevel())
  {
    fail = true;
  }

  inv(fail == false );
}
END_CONSTRAINT

//20310 - same as 20608

//20311
EXTERN_CONSTRAINT( QualQSAssignedOnlyOnce, QSAssignedOnce            )

// 20312 - 20313 - caught at read

// 20401 - 20404 - caught at read

//20405 - some caught at read but need to catch missing function terms
START_CONSTRAINT (QualTransitionLOElements, Transition, t)
{
  bool fail = false;

  const ListOfFunctionTerms *loft = t.getListOfFunctionTerms();

  if (loft->size() == 0 && loft->isSetDefaultTerm() == false)
  {
    fail = true;
  }

  inv(fail == false);
}
END_CONSTRAINT

// 20406 - 20408 - caught at read

//20409 - some caught at read but need to catch missing defaultTerm
START_CONSTRAINT (QualTransitionLOFuncTermElements, ListOfFunctionTerms, loft)
{
  pre (loft.size() > 0);

  bool fail = false;

  if (loft.isSetDefaultTerm() == false)
  {
    fail = true;
  }

  inv(fail == false);
}
END_CONSTRAINT

// 20410 - 20412 - caught at read

//20413
EXTERN_CONSTRAINT( QualTransitionLOFuncTermExceedMax,  ResultExceedsMaximum )

//20414
EXTERN_CONSTRAINT( QualTransitionLOFuncTermNegative,  ResultBecomesNegative  )

// 20501 - 20507 caught at read

//20508
START_CONSTRAINT (QualInputQSMustBeExistingQS, Input, input)
{
  pre( input.isSetQualitativeSpecies() );

  bool fail = false;

  msg =
    "QualitativeSpecies '" + input.getQualitativeSpecies() + "' is undefined. ";

  QualModelPlugin *plug1 = (QualModelPlugin*)(m.getPlugin("qual"));

  if (plug1->getQualitativeSpecies(input.getQualitativeSpecies()) == NULL )
  {
    fail = true;
  }

  inv(fail == false );
}
END_CONSTRAINT

//20509
START_CONSTRAINT (QualInputConstantCannotBeConsumed, Input, input)
{
  pre( input.isSetQualitativeSpecies() );
  pre( input.isSetTransitionEffect() );
  pre (input.getTransitionEffect() == INPUT_TRANSITION_EFFECT_CONSUMPTION);

  bool fail = false;

  msg =
    "The QualitativeSpecies '" + input.getQualitativeSpecies() + "'";
  msg += "referred to by this Input has constant set to true";
  msg += "and the transitionEffect set to consumption.";

  QualModelPlugin *plug1 = (QualModelPlugin*)(m.getPlugin("qual"));

  QualitativeSpecies *qs = plug1->getQualitativeSpecies
                                  (input.getQualitativeSpecies());
  pre (qs != NULL);
  pre (qs->isSetConstant() == true);

  
  if (qs->getConstant() == true)
  {
    fail = true;
  }

  inv(fail == false );
}
END_CONSTRAINT

// 20510 - caught at read

// 20601 - 20606 caught at read

// 20607
START_CONSTRAINT (QualOutputQSMustBeExistingQS, Output, input)
{
  pre( input.isSetQualitativeSpecies() );

  bool fail = false;

  msg =
    "QualitativeSpecies '" + input.getQualitativeSpecies() + "' is undefined. ";

  QualModelPlugin *plug1 = (QualModelPlugin*)(m.getPlugin("qual"));

  if (plug1->getQualitativeSpecies(input.getQualitativeSpecies()) == NULL )
  {
    fail = true;
  }

  inv(fail == false );
}
END_CONSTRAINT

// 20608
START_CONSTRAINT (QualOutputConstantMustBeFalse, Output, input)
{
  pre( input.isSetQualitativeSpecies() );

  bool fail = false;

  msg =
    "The QualitativeSpecies '" + input.getQualitativeSpecies() + "'";
  msg += "referred to by this Output has constant set to true.";

  QualModelPlugin *plug1 = (QualModelPlugin*)(m.getPlugin("qual"));

  QualitativeSpecies *qs = plug1->getQualitativeSpecies
                                  (input.getQualitativeSpecies());
  pre (qs != NULL);
  
  if (qs->isSetConstant() && qs->getConstant() == true)
  {
    fail = true;
  }

  inv(fail == false );
}
END_CONSTRAINT

// 20609
START_CONSTRAINT (QualOutputProductionMustHaveLevel, Output, input)
{
  pre (input.isSetTransitionEffect());

  bool fail = false;

  OutputTransitionEffect_t transition = input.getTransitionEffect();
  
  if (transition == OUTPUT_TRANSITION_EFFECT_PRODUCTION)
  {
    if (input.isSetOutputLevel() == false)
    {
      fail = true;
    }
  }

  inv(fail == false);
}
END_CONSTRAINT

// 20610 - caught at read

// 20701 - 20705 - caught at read

// 20801 - 20803 - caught at read

// 20804 - some caught at read but also need to catch missing math
START_CONSTRAINT (QualFuncTermOnlyOneMath, FunctionTerm, ft)
{
  bool fail = false;

  if (ft.isSetMath() == false)
  {
    fail = true;
  }

  inv(fail == false);
}
END_CONSTRAINT


// 20805 - 20806 - caught at read

  /** @endcond */



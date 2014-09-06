/**
 * @cond doxygenLibsbmlInternal
 *
 * @file    ResultBecomesNegative.cpp
 * @brief   Checks rule ordering for l2v1 and l1
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

#include <cstring>

#include <sbml/Model.h>
#include <sbml/packages/qual/extension/QualModelPlugin.h>

#include "ResultBecomesNegative.h"

/** @cond doxygenIgnored */

using namespace std;

/** @endcond */

LIBSBML_CPP_NAMESPACE_BEGIN

/**
 * Creates a new Constraint with the given constraint id.
 */
ResultBecomesNegative::ResultBecomesNegative (unsigned int id, Validator& v) :
  TConstraint<Model>(id, v)
{
}


/**
 * Destroys this Constraint.
 */
ResultBecomesNegative::~ResultBecomesNegative ()
{
}


/**
 * Checks that all ids on the following Model objects are unique:
 * event assignments and assignment rules.
 */
void
ResultBecomesNegative::check_ (const Model& m, const Model& object)
{

  const QualModelPlugin * plug = 
                 static_cast<const QualModelPlugin*>(m.getPlugin("qual"));
  
  if (plug == NULL) 
  {
    return;
  }

  for (unsigned int i = 0; i < plug->getNumQualitativeSpecies(); i++)
  {
    if (plug->getQualitativeSpecies(i)->isSetMaxLevel() == true)
    {
      checkResult(plug->getQualitativeSpecies(i), plug);
    }
  }

}
 
void
ResultBecomesNegative::checkResult(const QualitativeSpecies* qs, 
                                          const QualModelPlugin* plug)
{
  for (unsigned int i = 0; i < plug->getNumTransitions(); i++)
  {
    const Transition* tr = plug->getTransition(i);

    // is this qualitative species an output of this transition
    const Output* out = tr->getOutputBySpecies(qs->getId());
    if (out == NULL)
    {
      return;
    }

    // check defaultTerm does not cause max
    if (tr->isSetDefaultTerm())
    {
      if (tr->getDefaultTerm()->isSetResultLevel())
      {
        if (tr->getDefaultTerm()->getResultLevel() < 0)
        {
          logNegativeResult(*tr, *qs);
        }
      }
    }

    // loop thru the function terms
    for (unsigned int j = 0; j < tr->getNumFunctionTerms(); j++)
    {
      if (tr->getFunctionTerm(j)->isSetResultLevel())
      {
        if (tr->getFunctionTerm(j)->getResultLevel() < 0)
        {
          logNegativeResult(*tr, *qs);
        }
      }
    }
  }
}


void
ResultBecomesNegative::logNegativeResult (const Transition& tr,
                          const SBase& object)
{
  msg =
    "The Transition with id '";
  msg += tr.getId();
  msg += "' includes a resultLevel";
  msg += " that may cause the QualitativeSpecies '";
  msg += object.getId();
  msg += "' to become negative.";
  
  logFailure(object);

}

LIBSBML_CPP_NAMESPACE_END

/** @endcond */

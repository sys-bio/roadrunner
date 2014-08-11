/**
 * @cond doxygenLibsbmlInternal
 *
 * @file    FluxBoundsConsistent.cpp
 * @brief   Checks for fluxBounds consistency
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

#include <sbml/Model.h>
#include <sbml/packages/fbc/sbml/FluxBound.h>
#include <sbml/packages/fbc/extension/FbcModelPlugin.h>

#include "FluxBoundsConsistent.h"

/** @cond doxygenIgnored */

using namespace std;

/** @endcond */

LIBSBML_CPP_NAMESPACE_BEGIN
#ifdef __cplusplus


/*
 * Creates a new Constraint with the given constraint id.
 */
FluxBoundsConsistent::FluxBoundsConsistent (unsigned int id, Validator& v) :
  TConstraint<Model>(id, v)
{
}


/*
 * Destroys this Constraint.
 */
FluxBoundsConsistent::~FluxBoundsConsistent ()
{
}


/*
 * Checks whether all annotations have duplicate top level namespaces
 */
void
FluxBoundsConsistent::check_ (const Model& m, const Model& object)
{
  for (unsigned int n = 0; n < m.getNumReactions(); n++)
  {
    ListOfFluxBounds *lofb = static_cast<const FbcModelPlugin*>(m.getPlugin("fbc"))->
      getFluxBoundsForReaction(m.getReaction(n)->getId());
    if (lofb != NULL)
    {
      checkBounds(*(lofb));
    }
  }
}

/*
  * Check the annotation
  */
void FluxBoundsConsistent::checkBounds(ListOfFluxBounds & object)
{
  double upper = util_NaN();
  double lower = util_NaN();

  for (unsigned int n = 0; n < object.size(); n++)
  {
    FluxBound * fb = object.get(n);
    std::string bound = fb->getOperation();
    if (bound == "lessEqual")
    {
      if (util_isNaN(upper) == 1)
      {
        upper = fb->getValue();
      }
      else if (util_isEqual(upper, fb->getValue()) == 0)
      {
        logInconsistency("upper", *(fb));
      }
    }
    else if (bound == "greaterEqual")
    {
      if (util_isNaN(lower) == 1)
      {
        lower = fb->getValue();
      }
      else if (util_isEqual(lower, fb->getValue()) == 0)
      {
        logInconsistency("lower", *(fb));
      }
    }
    else
    {
      if (util_isNaN(upper) == 1)
      {
        upper = fb->getValue();
      }
      else if (util_isEqual(upper, fb->getValue()) == 0)
      {
        logInconsistency("upper", *(fb));
      }
      if (util_isNaN(lower) == 1)
      {
        lower = fb->getValue();
      }
      else if (util_isEqual(lower, fb->getValue()) == 0)
      {
        logInconsistency("lower", *(fb));
      }
    }
  }
}

/*
  * Logs a message about dupliacte top level annotations.
  */
void
FluxBoundsConsistent::logInconsistency (std::string name, const FluxBound& object )
{

  msg = "The ListOfFluxBounds already contains a fluxBound for reaction '";
  msg += static_cast<FluxBound>(object).getReaction();
  msg += "' with the operation '";
  msg += static_cast<FluxBound>(object).getOperation();
  msg += "' that assigns the ";
  msg += name;
  msg +=" bound.";

  
  logFailure(object);
}

#endif /* __cplusplus */
LIBSBML_CPP_NAMESPACE_END

/** @endcond */

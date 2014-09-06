/**
 * @cond doxygenLibsbmlInternal
 *
 * @file    UniquePortReferences.cpp
 * @brief   Ensures the appropriate ids within a Model are unique
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

#include <sbml/Model.h>
#include "UniquePortReferences.h"

/** @cond doxygenIgnored */

using namespace std;

/** @endcond */

LIBSBML_CPP_NAMESPACE_BEGIN
#ifdef __cplusplus

/*
 * Creates a new Constraint with the given constraint id.
 */
 UniquePortReferences::UniquePortReferences (unsigned int id, CompValidator& v):
  TConstraint<Model>(id, v)
{
}


/*
 * Destroys this Constraint.
 */
UniquePortReferences::~UniquePortReferences ()
{
}


/*
 * Checks that all ids on the following Model objects are unique:
 * FunctionDefinitions, Species, Compartments, global Parameters,
 * Reactions, and Events.
 */
void
UniquePortReferences::check_ (const Model& m, const Model& object)
{
  unsigned int n, size;

  mReferencedElements = new List();

  const CompModelPlugin * plug = 
    static_cast <const CompModelPlugin*>(m.getPlugin("comp"));
  if (plug == NULL)
  {
    return;
  }

  size = plug->getNumPorts();
  for (n = 0; n < size; ++n) 
  {
    checkReferencedElement(*(const_cast<Port*>(plug->getPort(n))));
  }

  delete mReferencedElements;
}

int ObjectsSame(SBase * obj1, SBase* obj2)
{
  if (obj1 == obj2)
    return 0;
  else
    return 1;
}

void 
UniquePortReferences::checkReferencedElement(Port& p)
{
  unsigned int numErrsB4 = p.getSBMLDocument()->getNumErrors();
  
  SBase* refElem = p.getReferencedElement();
  
  // if there is an issue with references the getReferencedElement
  // code may log errors
  // we dont want them - since we will log any  errors here
  // so remove them
  unsigned int numErrsAfter = p.getSBMLDocument()->getNumErrors();
  for (unsigned int i = numErrsAfter; i > numErrsB4; i--)
  {
    p.getSBMLDocument()->getErrorLog()->remove(
      p.getSBMLDocument()->getError(i-1)->getErrorId());
  }
  
  if (mReferencedElements->find(refElem, 
                           (ListItemComparator) (ObjectsSame)) != NULL)
  {
    logReferenceExists (p);
  }
  else
  {

    mReferencedElements->add(refElem);
  }
}

void 
UniquePortReferences::logReferenceExists (Port& p)
{
  msg = "Port with id '";
  msg += p.getId();
  msg += "' references the object ";
  if (p.isSetIdRef() == true)
  {
    msg += "with id '";
    msg += p.getIdRef();
  }
  else if (p.isSetMetaIdRef() == true)
  {
    msg += "with metaid '";
    msg += p.getMetaIdRef();
  }
  else if (p.isSetUnitRef() == true)
  {
    msg += "with unitId '";
    msg += p.getUnitRef();
  }
  msg += "' that has already been referenced by a <port> ";
  msg += "in the containing model."; 

  logFailure(p);
}

#endif /* __cplusplus */
/** @cond doxygenIgnored */

/** @endcond */

LIBSBML_CPP_NAMESPACE_END

/** @endcond */

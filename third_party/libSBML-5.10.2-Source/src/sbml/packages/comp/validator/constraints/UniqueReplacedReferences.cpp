/**
 * @cond doxygenLibsbmlInternal
 *
 * @file    UniqueReplacedReferences.cpp
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
#include <sbml/util/ElementFilter.h>
#include "UniqueReplacedReferences.h"


/** @cond doxygenIgnored */

using namespace std;

/** @endcond */

LIBSBML_CPP_NAMESPACE_BEGIN
#ifdef __cplusplus
class ReplacedFilter : public ElementFilter
{
public:
  ReplacedFilter() : ElementFilter()
  {
  }

  virtual bool filter(const SBase* element)
  {
    // return in case we don't have a valid element with a comp plugin
        if (element == NULL)
        {
            return false;
        }
        const CompSBasePlugin * plug = static_cast<const CompSBasePlugin*>
                                            (element->getPlugin("comp"));
        if (plug == NULL)
        {
            return false;
        }
        // otherwise we have a comp plugin
        // we want to keep the object if it has ReplacedElements
        if (plug->getNumReplacedElements() == 0)
        {
          return false;
        }


        return true;
  }

};


/*
 * Creates a new Constraint with the given constraint id.
 */
 UniqueReplacedReferences::UniqueReplacedReferences (unsigned int id, CompValidator& v):
  TConstraint<Model>(id, v)
{
}


/*
 * Destroys this Constraint.
 */
UniqueReplacedReferences::~UniqueReplacedReferences ()
{
}


/*
 * Checks that all ids on the following Model objects are unique:
 * FunctionDefinitions, Species, Compartments, global Parameters,
 * Reactions, and Events.
 */
void
UniqueReplacedReferences::check_ (const Model& m, const Model& object)
{
  unsigned int n, size;
  const CompSBasePlugin * plug;
  ReplacedFilter filter;

  mReferencedElements = new List();

  /* get all elements that have replaced elements */
  List* allElements = const_cast<Model *>(&m)->getAllElements(&filter);

  size = allElements->getSize();


  for (n = 0; n < size; ++n) 
  {
    SBase *sb = static_cast<SBase*>(allElements->get(n));
    plug = static_cast<const CompSBasePlugin*>(sb->getPlugin("comp"));

    for (unsigned int i = 0; i < plug->getNumReplacedElements(); i++)
    {
      checkReferencedElement(*(const_cast<ReplacedElement*>
                                        (plug->getReplacedElement(i))));
    }
  }

  delete allElements;
  delete mReferencedElements;
}

int ObjectsSame1(SBase * obj1, SBase* obj2)
{
  if (obj1 == obj2)
    return 0;
  else
    return 1;
}

void 
UniqueReplacedReferences::checkReferencedElement(ReplacedElement& repE)
{
  unsigned int numErrsB4 = repE.getSBMLDocument()->getNumErrors();
  
  SBase* refElem = repE.getReferencedElement();
  
  // if there is an issue with references the getReferencedElement
  // code may log errors
  // we dont want them - since we will log any  errors here
  // so remove them
  unsigned int numErrsAfter = repE.getSBMLDocument()->getNumErrors();
  for (unsigned int i = numErrsAfter; i > numErrsB4; i--)
  {
    repE.getSBMLDocument()->getErrorLog()->remove(
      repE.getSBMLDocument()->getError(i-1)->getErrorId());
  }
  
  if (mReferencedElements->find(refElem, 
                           (ListItemComparator) (ObjectsSame1)) != NULL)
  {
    if (refElem->getTypeCode() != SBML_COMP_DELETION)
    {
      logReferenceExists (repE);
    }
  }
  else
  {

    mReferencedElements->add(refElem);
  }
}

void 
UniqueReplacedReferences::logReferenceExists (ReplacedElement& repE)
{
  std::string id = (repE.getParentSBMLObject())->getParentSBMLObject()->getId();
  msg = "ReplacedElement on object with id '";
  msg += id;
  msg += "' references the object ";
  if (repE.isSetIdRef() == true)
  {
    msg += "with id '";
    msg += repE.getIdRef();
  }
  else if (repE.isSetMetaIdRef() == true)
  {
    msg += "with metaid '";
    msg += repE.getMetaIdRef();
  }
  else if (repE.isSetUnitRef() == true)
  {
    msg += "with unitId '";
    msg += repE.getUnitRef();
  }
  else if (repE.isSetPortRef() == true)
  {
    msg += "with portId '";
    msg += repE.getPortRef();
  }
  msg += "' in the submodel '";
  msg += repE.getSubmodelRef();
  msg += "' that has already been referenced by a <replacedElement>."; 

  logFailure(repE);
}

#endif /* __cplusplus */

LIBSBML_CPP_NAMESPACE_END

/** @endcond */

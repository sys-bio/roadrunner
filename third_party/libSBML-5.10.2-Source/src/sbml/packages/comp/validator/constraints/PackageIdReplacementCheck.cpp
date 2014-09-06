/**
 * @cond doxygenLibsbmlInternal
 *
 * @file    PackageIdReplacementCheck.cpp
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
#include "PackageIdReplacementCheck.h"


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


class ReplacedByFilter : public ElementFilter
{
public:
  ReplacedByFilter() : ElementFilter()
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
        if (plug->getReplacedBy() == NULL)
        {
          return false;
        }


        return true;
  }

};


/*
 * Creates a new Constraint with the given constraint id.
 */
 PackageIdReplacementCheck::PackageIdReplacementCheck (unsigned int id, CompValidator& v):
  TConstraint<Model>(id, v)
{
}


/*
 * Destroys this Constraint.
 */
PackageIdReplacementCheck::~PackageIdReplacementCheck ()
{
}


/*
 * Checks that all ids on the following Model objects are unique:
 * FunctionDefinitions, Species, Compartments, global Parameters,
 * Reactions, and Events.
 */
void
PackageIdReplacementCheck::check_ (const Model& m, const Model& object)
{
  unsigned int n, size;
  const CompSBasePlugin * plug;
  ReplacedFilter filter;
  ReplacedByFilter repByFilter;

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

  /* get all elements that have replaced elements */
  allElements = const_cast<Model *>(&m)->getAllElements(&repByFilter);

  size = allElements->getSize();


  for (n = 0; n < size; ++n) 
  {
    SBase *sb = static_cast<SBase*>(allElements->get(n));
    plug = static_cast<const CompSBasePlugin*>(sb->getPlugin("comp"));

    checkReferencedElement(*(const_cast<ReplacedBy*>
                                        (plug->getReplacedBy())));
  }

  delete allElements;

}

void 
PackageIdReplacementCheck::checkReferencedElement(ReplacedBy& repBy)
{
  unsigned int numErrsB4 = repBy.getSBMLDocument()->getNumErrors();
  
  SBase* refElem = repBy.getReferencedElement();
  
  // if there is an issue with references the getReferencedElement
  // will log errors and possibly fail
  // we dont want to try any further
  unsigned int numErrsAfter = repBy.getSBMLDocument()->getNumErrors();

  if (numErrsB4 != numErrsAfter || refElem == NULL)
  {
    return;
  }

  SBase * parent = repBy.getParentSBMLObject();

  if (parent->isSetId() == true && refElem->isSetId() == false)
  {
    logMissingIdAttribute(repBy, refElem, parent);
  }

}


void 
PackageIdReplacementCheck::checkReferencedElement(ReplacedElement& repE)
{
  // if the deletion attribute is set then it does not point
  // to another package element
  if (repE.isSetDeletion() == true)
  {
    return;
  }

  unsigned int numErrsB4 = repE.getSBMLDocument()->getNumErrors();
  
  SBase* refElem = repE.getReferencedElement();
  
  // if there is an issue with references the getReferencedElement
  // will log errors and possibly fail
  // we dont want to try any further
  unsigned int numErrsAfter = repE.getSBMLDocument()->getNumErrors();

  if (numErrsB4 != numErrsAfter || refElem == NULL)
  {
    return;
  }

  SBase * parent = repE.getParentSBMLObject()->getParentSBMLObject();

  if (refElem->isSetId() == true && parent->isSetId() == false)
  {
    logMissingIdAttribute(repE, refElem, parent);
  }


}


void 
PackageIdReplacementCheck::logMissingIdAttribute (ReplacedBy& repBy, 
                                           SBase* refElem, SBase* parent)
{
  std::string id = parent->getId();
  msg = "A ReplacedBy object on the ";
  msg += SBMLTypeCode_toString(refElem->getTypeCode(), 
                               refElem->getPackageName().c_str());
  msg += " object with id attribute '";
  msg += id;
  msg += "' refers to the ";
  msg += SBMLTypeCode_toString(refElem->getTypeCode(), 
                               refElem->getPackageName().c_str());
  msg += " object that does not have an id attribute.";

  logFailure(repBy);
}

void 
PackageIdReplacementCheck::logMissingIdAttribute (ReplacedElement& repE, 
                                           SBase* refElem, SBase* parent)
{
  std::string id = refElem->getId();
  msg = "A ReplacedElement object on the ";
  msg += SBMLTypeCode_toString(refElem->getTypeCode(), 
                               refElem->getPackageName().c_str());
  msg += " object with no id attribute refers to the ";
  msg += SBMLTypeCode_toString(refElem->getTypeCode(), 
                               refElem->getPackageName().c_str());
  msg += " object that does has an id '";
  msg += id;
  msg += "'.";


  logFailure(repE);
}


#endif /* __cplusplus */

LIBSBML_CPP_NAMESPACE_END

/** @endcond */

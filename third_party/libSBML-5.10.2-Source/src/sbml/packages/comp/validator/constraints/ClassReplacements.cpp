/**
 * @cond doxygenLibsbmlInternal
 *
 * @file    ClassReplacements.cpp
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
#include "ClassReplacements.h"


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
 ClassReplacements::ClassReplacements (unsigned int id, CompValidator& v):
  TConstraint<Model>(id, v)
{
}


/*
 * Destroys this Constraint.
 */
ClassReplacements::~ClassReplacements ()
{
}


/*
 * Checks that all ids on the following Model objects are unique:
 * FunctionDefinitions, Species, Compartments, global Parameters,
 * Reactions, and Events.
 */
void
ClassReplacements::check_ (const Model& m, const Model& object)
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
ClassReplacements::checkReferencedElement(ReplacedBy& repBy)
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

  if (refElem->getTypeCode() != parent->getTypeCode())
  {
    int refElemTC = refElem->getTypeCode();
    // exception is anything with mathematical meaning
    // may replace a parameter
    if (parent->getTypeCode() == SBML_PARAMETER)
    {
      if (refElemTC == SBML_COMPARTMENT || refElemTC == SBML_SPECIES ||
        refElemTC == SBML_SPECIES_REFERENCE || refElemTC == SBML_REACTION
        || refElemTC == SBML_LOCAL_PARAMETER)
      {
        return;
      }

    }
    else if (parent->getTypeCode() == SBML_LOCAL_PARAMETER)
    {
      if (refElemTC == SBML_PARAMETER)
      {
        return;
      }
    }

    logBadClassReplacement(repBy, refElem, parent);
  }


}


void 
ClassReplacements::checkReferencedElement(ReplacedElement& repE)
{
  // if the deletion attribute is set then we don't care what 
  // it points to
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

  if (refElem->getTypeCode() != parent->getTypeCode())
  {
    int parentTC = parent->getTypeCode();
    // exception is anything with mathematical meaning
    // may replace a parameter
    if (refElem->getTypeCode() == SBML_PARAMETER)
    {
      if (parentTC == SBML_COMPARTMENT || parentTC == SBML_SPECIES ||
        parentTC == SBML_SPECIES_REFERENCE || parentTC == SBML_REACTION
        || parentTC == SBML_LOCAL_PARAMETER)
      {
        return;
      }
    }
    else if (refElem->getTypeCode() == SBML_LOCAL_PARAMETER)
    {
      if (parentTC == SBML_PARAMETER)
      {
        return;
      }
    }



    logBadClassReplacement(repE, refElem, parent);
  }


}


void 
ClassReplacements::logBadClassReplacement (ReplacedBy& repBy, 
                                           SBase* refElem, SBase* parent)
{
  std::string id = parent->getId();
  msg = "ReplacedBy on object with id '";
  msg += id;
  msg += "' refers to an object of type '";
  msg += SBMLTypeCode_toString(refElem->getTypeCode(), 
                               refElem->getPackageName().c_str());
  msg += "' but expects an object of type '";
  msg += SBMLTypeCode_toString(parent->getTypeCode(), 
                               parent->getPackageName().c_str());
  msg += "'.";


  logFailure(repBy);
}

void 
ClassReplacements::logBadClassReplacement (ReplacedElement& repE, 
                                           SBase* refElem, SBase* parent)
{
  std::string id = parent->getId();
  msg = "ReplacedElement on object with id '";
  msg += id;
  msg += "' refers to an object of type '";
  msg += SBMLTypeCode_toString(refElem->getTypeCode(), 
                               refElem->getPackageName().c_str());
  msg += "' but expects an object of type '";
  msg += SBMLTypeCode_toString(parent->getTypeCode(), 
                               parent->getPackageName().c_str());
  msg += "'.";


  logFailure(repE);
}


#endif  /* __cplusplus */

LIBSBML_CPP_NAMESPACE_END

/** @endcond */

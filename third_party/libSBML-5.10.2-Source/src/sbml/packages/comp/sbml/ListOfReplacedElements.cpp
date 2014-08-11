/**
 * @file    ReplacedElement.cpp
 * @brief   Implementation of ReplacedElement, the Subelement derived class of replacedElements package.
 * @author  Lucian Smith
 *
 *<!---------------------------------------------------------------------------
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
 *------------------------------------------------------------------------- -->
 */

#include <iostream>

#include <sbml/SBMLVisitor.h>
#include <sbml/xml/XMLNode.h>
#include <sbml/xml/XMLToken.h>
#include <sbml/xml/XMLAttributes.h>
#include <sbml/xml/XMLInputStream.h>
#include <sbml/xml/XMLOutputStream.h>

#include <sbml/packages/comp/extension/CompExtension.h>
#include <sbml/packages/comp/sbml/ListOfReplacedElements.h>

using namespace std;

LIBSBML_CPP_NAMESPACE_BEGIN
#ifdef __cplusplus

ListOfReplacedElements::ListOfReplacedElements(CompPkgNamespaces* compns)
  : ListOf(compns)
{
  // set the element namespace of this object
  setElementNamespace(compns->getURI());
  loadPlugins(compns);
}


ListOfReplacedElements::ListOfReplacedElements(unsigned int level, unsigned int version, unsigned int pkgVersion)
  : ListOf(level,version)
{
  setSBMLNamespacesAndOwn(new CompPkgNamespaces(level,version,pkgVersion));
  loadPlugins(mSBMLNamespaces);
};


ListOfReplacedElements*
ListOfReplacedElements::clone () const
{
  return new ListOfReplacedElements(*this);
}


ReplacedElement *
ListOfReplacedElements::get(unsigned int n)
{
  return static_cast<ReplacedElement*>(ListOf::get(n));
}


const ReplacedElement *
ListOfReplacedElements::get(unsigned int n) const
{
  return static_cast<const ReplacedElement*>(ListOf::get(n));
}


ReplacedElement*
ListOfReplacedElements::remove (unsigned int n)
{
  return static_cast<ReplacedElement*>(ListOf::remove(n));
}


int
ListOfReplacedElements::getItemTypeCode () const
{
  return SBML_COMP_REPLACEDELEMENT;
}


const std::string&
ListOfReplacedElements::getElementName () const
{
  static const std::string name = "listOfReplacedElements";
  return name;
}


/** @cond doxygenLibsbmlInternal */
SBase*
ListOfReplacedElements::createObject (XMLInputStream& stream)
{
  const std::string& name   = stream.peek().getName();
  SBase*        object = 0;


  if (name == "replacedElement")
  {
    COMP_CREATE_NS(compns, getSBMLNamespaces());
    object = new ReplacedElement(compns);
    appendAndOwn(object);
    delete compns;
    //mItems.push_back(object);
  }

  return object;
}
/** @endcond */

/** @cond doxygenLibsbmlInternal */
void 
ListOfReplacedElements::writeXMLNS (XMLOutputStream& stream) const
{
  XMLNamespaces xmlns;

  std::string prefix = getPrefix();

  if (prefix.empty())
  {
    XMLNamespaces* thisxmlns = getNamespaces();
    if (thisxmlns && thisxmlns->hasURI(CompExtension::getXmlnsL3V1V1()))
    {
      xmlns.add(CompExtension::getXmlnsL3V1V1(),prefix);
    }
  }

  stream << xmlns;
}
/** @endcond */


#endif /* __cplusplus */

LIBSBML_CPP_NAMESPACE_END


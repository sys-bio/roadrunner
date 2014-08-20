/**
 * @file    ListOfSubmodels.cpp
 * @brief   Implementation of ListOfSubmodels.
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
#include <sbml/packages/comp/sbml/ListOfSubmodels.h>
#include <sbml/Model.h>

using namespace std;

LIBSBML_CPP_NAMESPACE_BEGIN
#ifdef __cplusplus

ListOfSubmodels::ListOfSubmodels(CompPkgNamespaces* compns)
  : ListOf(compns)
{
  // set the element namespace of this object
  setElementNamespace(compns->getURI());
  loadPlugins(compns);
}


ListOfSubmodels::ListOfSubmodels(unsigned int level, unsigned int version, unsigned int pkgVersion)
  : ListOf(level,version)
{
  setSBMLNamespacesAndOwn(new CompPkgNamespaces(level,version,pkgVersion));
  loadPlugins(mSBMLNamespaces);
}


ListOfSubmodels*
ListOfSubmodels::clone () const
{
  return new ListOfSubmodels(*this);
}


Submodel*
ListOfSubmodels::get(unsigned int n)
{
  return static_cast<Submodel*>(ListOf::get(n));
}


const Submodel*
ListOfSubmodels::get(unsigned int n) const
{
  return static_cast<const Submodel*>(ListOf::get(n));
}


Submodel*
ListOfSubmodels::get (const std::string& symbol)
{
  return const_cast<Submodel*>(static_cast<const ListOfSubmodels&>(*this).get(symbol) );
}


const Submodel*
ListOfSubmodels::get (const std::string& symbol) const
{
  vector<SBase*>::const_iterator result;

  result = find_if( mItems.begin(), mItems.end(), IdEq<Model>(symbol) );
  return (result == mItems.end()) ? 0 : static_cast <Submodel*> (*result);
}


Submodel*
ListOfSubmodels::remove (unsigned int n)
{
  return static_cast<Submodel*>(ListOf::remove(n));
}


Submodel*
ListOfSubmodels::remove (const std::string& sid)
{
  SBase* item = NULL;
  vector<SBase*>::iterator result;

  if (&(sid) != NULL)
  {
    result = find_if( mItems.begin(), mItems.end(), IdEq<Submodel>(sid) );

    if (result != mItems.end())
    {
      item = *result;
      mItems.erase(result);
    }
  }
  return static_cast<Submodel*>(item);
}


int
ListOfSubmodels::getItemTypeCode () const
{
  return SBML_COMP_SUBMODEL;
}


const std::string&
ListOfSubmodels::getElementName () const
{
  static const std::string name = "listOfSubmodels";
  return name;
}


/** @cond doxygenLibsbmlInternal */
SBase*
ListOfSubmodels::createObject (XMLInputStream& stream)
{
  const std::string& name   = stream.peek().getName();
  SBase*        object = 0;


  if (name == "submodel")
  {
    COMP_CREATE_NS(compns, getSBMLNamespaces());
    object = new Submodel(compns);
    appendAndOwn(object);
    delete compns;
  }

  return object;
}
/** @endcond */

/** @cond doxygenLibsbmlInternal */
void 
ListOfSubmodels::writeXMLNS (XMLOutputStream& stream) const
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


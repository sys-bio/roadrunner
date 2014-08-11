/**
 * @file    ListOfExternalModelDefinitions.cpp
 * @brief   Implementation of ListOfExternalModelDefinitions.
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
#include <sbml/packages/comp/sbml/ListOfExternalModelDefinitions.h>
#include <sbml/Model.h>

using namespace std;

LIBSBML_CPP_NAMESPACE_BEGIN
#ifdef __cplusplus

ListOfExternalModelDefinitions::ListOfExternalModelDefinitions(CompPkgNamespaces* compns)
  : ListOf(compns)
{
  // set the element namespace of this object
  setElementNamespace(compns->getURI());
  loadPlugins(compns);
}


ListOfExternalModelDefinitions::ListOfExternalModelDefinitions(unsigned int level, unsigned int version, unsigned int pkgVersion)
  : ListOf(level,version)
{
  setSBMLNamespacesAndOwn(new CompPkgNamespaces(level,version,pkgVersion));
  loadPlugins(mSBMLNamespaces);
};


ListOfExternalModelDefinitions*
ListOfExternalModelDefinitions::clone () const
{
  return new ListOfExternalModelDefinitions(*this);
}


/* return nth item in list */
ExternalModelDefinition*
ListOfExternalModelDefinitions::get(unsigned int n)
{
  return static_cast<ExternalModelDefinition*>(ListOf::get(n));
}


/* return nth item in list */
const ExternalModelDefinition*
ListOfExternalModelDefinitions::get(unsigned int n) const
{
  return static_cast<const ExternalModelDefinition*>(ListOf::get(n));
}


/* return item by symbol */
ExternalModelDefinition*
ListOfExternalModelDefinitions::get (const std::string& symbol)
{
  return const_cast<ExternalModelDefinition*>( 
    static_cast<const ListOfExternalModelDefinitions&>(*this).get(symbol) );
}


/* return item by symbol */
const ExternalModelDefinition*
ListOfExternalModelDefinitions::get (const std::string& symbol) const
{
  vector<SBase*>::const_iterator result;

  result = find_if( mItems.begin(), mItems.end(), IdEq<Model>(symbol) );
  return (result == mItems.end()) ? NULL : static_cast <ExternalModelDefinition*> (*result);
}


/* Removes the nth item from this list */
ExternalModelDefinition*
ListOfExternalModelDefinitions::remove (unsigned int n)
{
  return static_cast<ExternalModelDefinition*>(ListOf::remove(n));
}

ExternalModelDefinition* 
ListOfExternalModelDefinitions::remove (const std::string &sid)
{
  ListItemIter result = find_if( mItems.begin(), mItems.end(), IdEq<ExternalModelDefinition>(sid) );
  if (result == mItems.end()) return NULL;

  mItems.erase(result);
  return static_cast<ExternalModelDefinition*>(*result);
}


/*
 * @return the typecode (int) of SBML objects contained in this ListOf 
 */
int
ListOfExternalModelDefinitions::getItemTypeCode () const
{
  return SBML_COMP_EXTERNALMODELDEFINITION;
}


const std::string&
ListOfExternalModelDefinitions::getElementName () const
{
  static const std::string name = "listOfExternalModelDefinitions";
  return name;
}


/** @cond doxygenLibsbmlInternal */
SBase*
ListOfExternalModelDefinitions::createObject (XMLInputStream& stream)
{
  const std::string& name   = stream.peek().getName();
  SBase*        object = 0;


  if (name == "externalModelDefinition")
  {
    COMP_CREATE_NS(compns, getSBMLNamespaces());
    object = new ExternalModelDefinition(compns);
    appendAndOwn(object);
    delete compns;
  }

  return object;
}
/** @endcond */

/** @cond doxygenLibsbmlInternal */
void 
ListOfExternalModelDefinitions::writeXMLNS (XMLOutputStream& stream) const
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

/** @cond doxygenLibsbmlInternal */

bool
ListOfExternalModelDefinitions::accept(SBMLVisitor& v) const
{
  v.visit(*this);
  for (unsigned int n = 0 ; n < mItems.size(); n++)
  {
    static_cast<ExternalModelDefinition*>(mItems[n])->accept(v);
  }
  v.leave(*this);

  return true;
}

/** @endcond */

#endif /* __cplusplus */
LIBSBML_CPP_NAMESPACE_END


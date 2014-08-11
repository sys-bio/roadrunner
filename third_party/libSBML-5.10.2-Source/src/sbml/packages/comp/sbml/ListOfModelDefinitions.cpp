/**
 * @file    ModelDefinition.cpp
 * @brief   Implementation of ModelDefinition, the Subelement derived class of modelDefinitions package.
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
#include <sbml/packages/comp/sbml/ListOfModelDefinitions.h>

using namespace std;

LIBSBML_CPP_NAMESPACE_BEGIN
#ifdef __cplusplus

ListOfModelDefinitions::ListOfModelDefinitions(CompPkgNamespaces* compns)
  : ListOf(compns)
{
  // set the element namespace of this object
  setElementNamespace(compns->getURI());
  loadPlugins(compns);
}


ListOfModelDefinitions::ListOfModelDefinitions(unsigned int level, unsigned int version, unsigned int pkgVersion)
  : ListOf(level,version)
{
  setSBMLNamespacesAndOwn(new CompPkgNamespaces(level,version,pkgVersion));
  loadPlugins(mSBMLNamespaces);
};


ListOfModelDefinitions*
ListOfModelDefinitions::clone () const
{
  return new ListOfModelDefinitions(*this);
}


/* return nth item in list */
ModelDefinition*
ListOfModelDefinitions::get(unsigned int n)
{
  return static_cast<ModelDefinition*>(ListOf::get(n));
}


/* return nth item in list */
const ModelDefinition*
ListOfModelDefinitions::get(unsigned int n) const
{
  return static_cast<const ModelDefinition*>(ListOf::get(n));
}


/* return item by symbol */
ModelDefinition*
ListOfModelDefinitions::get (const std::string& symbol)
{
  return const_cast<ModelDefinition*>( 
    static_cast<const ListOfModelDefinitions&>(*this).get(symbol) );
}


/* return item by symbol */
const ModelDefinition*
ListOfModelDefinitions::get (const std::string& symbol) const
{
  vector<SBase*>::const_iterator result;

  result = find_if( mItems.begin(), mItems.end(), IdEq<Model>(symbol) );
  return (result == mItems.end()) ? 0 : static_cast <ModelDefinition*> (*result);
}


/* Removes the nth item from this list */
ModelDefinition*
ListOfModelDefinitions::remove (unsigned int n)
{
  return static_cast<ModelDefinition*>(ListOf::remove(n));
}

ModelDefinition*
ListOfModelDefinitions::remove (const std::string &sid)
{
  ListItemIter result;

  result = find_if( mItems.begin(), mItems.end(), IdEq<ModelDefinition>(sid) );
  if (result == mItems.end()) return NULL;

  mItems.erase(result);
  return static_cast<ModelDefinition*>(*result);
}

/*
 * @return the typecode (int) of SBML objects contained in this ListOf 
 */
int
ListOfModelDefinitions::getItemTypeCode () const
{
  return SBML_COMP_MODELDEFINITION;
}


const std::string&
ListOfModelDefinitions::getElementName () const
{
  static const std::string name = "listOfModelDefinitions";
  return name;
}


/** @cond doxygenLibsbmlInternal */
SBase*
ListOfModelDefinitions::createObject (XMLInputStream& stream)
{
  const std::string& name   = stream.peek().getName();
  SBase*        object = 0;


  if (name == "modelDefinition")
  {
    COMP_CREATE_NS(compns, getSBMLNamespaces());
    object = new ModelDefinition(compns);
    appendAndOwn(object);
    delete compns;
  }

  return object;
}
/** @endcond */

/** @cond doxygenLibsbmlInternal */
void 
ListOfModelDefinitions::writeXMLNS (XMLOutputStream& stream) const
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


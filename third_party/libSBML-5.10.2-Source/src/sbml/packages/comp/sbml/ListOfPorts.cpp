/**
 * @file    ListOfPorts.cpp
 * @brief   Implementation of ListOfPorts.
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
#include <sbml/packages/comp/sbml/ListOfPorts.h>

using namespace std;

LIBSBML_CPP_NAMESPACE_BEGIN
#ifdef __cplusplus

ListOfPorts::ListOfPorts(CompPkgNamespaces* compns)
  : ListOf(compns)
{
  // set the element namespace of this object
  setElementNamespace(compns->getURI());
  loadPlugins(compns);
}


ListOfPorts::ListOfPorts(unsigned int level, unsigned int version, unsigned int pkgVersion)
  : ListOf(level,version)
{
  setSBMLNamespacesAndOwn(new CompPkgNamespaces(level,version,pkgVersion));
  loadPlugins(mSBMLNamespaces);
};


ListOfPorts*
ListOfPorts::clone () const
{
  return new ListOfPorts(*this);
}


Port *
ListOfPorts::get(unsigned int n)
{
  return static_cast<Port*>(ListOf::get(n));
}


const Port *
ListOfPorts::get(unsigned int n) const
{
  return static_cast<const Port*>(ListOf::get(n));
}


Port*
ListOfPorts::get (const std::string& symbol)
{
  return const_cast<Port*>( 
                           static_cast<const ListOfPorts&>(*this).get(symbol) );
}


const Port*
ListOfPorts::get (const std::string& symbol) const
{
  for (size_t it=0; it<mItems.size(); it++) {
    const Port* port = static_cast<Port*>(mItems[it]);
    if (port->getId() == symbol) return port;
  }
  return NULL;
}

Port*
ListOfPorts::remove (unsigned int n)
{
  return static_cast<Port*>(ListOf::remove(n));
}


Port*
ListOfPorts::remove (const std::string& sid)
{
  SBase* item = NULL;
  vector<SBase*>::iterator result;

  if (&(sid) != NULL)
  {
    result = find_if( mItems.begin(), mItems.end(), IdEq<Port>(sid) );

    if (result != mItems.end())
    {
      item = *result;
      mItems.erase(result);
    }
  }
  return static_cast<Port*>(item);
}


int
ListOfPorts::getItemTypeCode () const
{
  return SBML_COMP_PORT;
}


const std::string&
ListOfPorts::getElementName () const
{
  static const std::string name = "listOfPorts";
  return name;
}


/** @cond doxygenLibsbmlInternal */
SBase*
ListOfPorts::createObject (XMLInputStream& stream)
{
  const std::string& name   = stream.peek().getName();
  SBase*        object = 0;


  if (name == "port")
  {
    COMP_CREATE_NS(compns, getSBMLNamespaces());
    object = new Port(compns);
    appendAndOwn(object);
    delete compns;
  }

  return object;
}
/** @endcond */

/** @cond doxygenLibsbmlInternal */
void 
ListOfPorts::writeXMLNS (XMLOutputStream& stream) const
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

SBase*
ListOfPorts::getElementBySId(const std::string& id)
{
  for (unsigned int i = 0; i < size(); i++)
  {
    SBase* obj = get(i);
    //Ports are not in the SId namespace, so don't check 'getId'.  However, their children (through plugins) may have the element we are looking for, so we still need to check all of them.
    obj = obj->getElementBySId(id);
    if (obj != NULL) return obj;
  }

  return getElementFromPluginsBySId(id);
}
  

#endif /* __cplusplus */

LIBSBML_CPP_NAMESPACE_END

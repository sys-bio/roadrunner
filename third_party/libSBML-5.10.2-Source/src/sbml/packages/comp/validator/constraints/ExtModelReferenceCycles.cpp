/**
 * @cond doxygenLibsbmlInternal
 *
 * @file    ExtModelReferenceCycles.cpp
 * @brief   Ensures unique variables assigned by rules and events
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

#include <cstring>

#include <sbml/Model.h>
#include <sbml/packages/comp/extension/CompModelPlugin.h>
#include <sbml/packages/comp/extension/CompSBMLDocumentPlugin.h>
#include <sbml/packages/comp/util/SBMLResolverRegistry.h>

#include "ExtModelReferenceCycles.h"
#include <sbml/util/IdList.h>

/** @cond doxygenIgnored */

using namespace std;

/** @endcond */

LIBSBML_CPP_NAMESPACE_BEGIN
#ifdef __cplusplus

/**
 * Creates a new Constraint with the given constraint id.
 */
ExtModelReferenceCycles::ExtModelReferenceCycles (unsigned int id, Validator& v) :
  TConstraint<Model>(id, v)
{
}


/**
 * Destroys this Constraint.
 */
ExtModelReferenceCycles::~ExtModelReferenceCycles ()
{
}


/**
 * Checks that all ids on the following Model objects are unique:
 * event assignments and assignment rules.
 */
void
ExtModelReferenceCycles::check_ (const Model& m, const Model& object)
{
  mIdMap.clear();

  mDocumentsHandled.clear();

  addAllReferences(m.getSBMLDocument(), "");

  determineAllDependencies();

  determineCycles(m);
}

void
ExtModelReferenceCycles::addAllReferences(const SBMLDocument* doc, 
                                          std::string location)
{
  if (doc == NULL) return;

  const Model * m = doc->getModel();

  if (m == NULL)
  {
    return;
  }

  const CompSBMLDocumentPlugin* docPlug = (CompSBMLDocumentPlugin*)
                                  (doc->getPlugin("comp"));
  const CompModelPlugin* modelPlug = (CompModelPlugin*)(m->getPlugin("comp"));

  if (modelPlug == NULL || docPlug == NULL)
  {
    return;
  }

  if (docPlug->getNumExternalModelDefinitions() == 0)
  {
    return;
  }

  string locationURI = doc->getLocationURI();

  if (locationURI.empty()) {
    return;
  }

  if (location.empty() == true)
  {
    location = locationURI.substr(locationURI.find(':')+1, string::npos);
  }
  
  if (mDocumentsHandled.contains(location) == false)
  {
    addModelReferences(location, docPlug, modelPlug);
    mDocumentsHandled.append(location);


    const SBMLResolverRegistry& registry = SBMLResolverRegistry::getInstance();

    for (unsigned int i = 0; i < docPlug->getNumExternalModelDefinitions(); i++)
    {
      string uri = docPlug->getExternalModelDefinition(i)->getSource();

      SBMLDocument* newDoc = registry.resolve(uri, locationURI);

      addAllReferences(newDoc, uri);
    }
  }
}
 
void 
ExtModelReferenceCycles::addModelReferences(const std::string &location, 
                          const CompSBMLDocumentPlugin* docPlug,
                          const CompModelPlugin* modelPlug)
{
  for (unsigned int i = 0; i < docPlug->getNumExternalModelDefinitions(); i++)
  {
    const ExternalModelDefinition * ext = 
                           docPlug->getExternalModelDefinition(i);
    std::string refDoc = ext->getSource() + "_" + ext->getModelRef();
    std::string thisDoc = location + "_" + ext->getId();
    mIdMap.insert(pair<const std::string, std::string>(thisDoc, refDoc));
  }
}


void 
ExtModelReferenceCycles::determineAllDependencies()
{
  IdIter iterator;
  IdIter inner_it;
  IdRange range;

  /* for each pair in the map (x, y)
   * retrieve all other pairs where y is first (e.g. (y, s))
   * and create pairs showing that x depends on these e.g. (x, s)
   * check whether the pair already exists in the map
   * and add it if not
   */
  for (iterator = mIdMap.begin(); iterator != mIdMap.end(); iterator++)
  {
    range = mIdMap.equal_range((*iterator).second);
    for (inner_it = range.first; inner_it != range.second; inner_it++)
    {
      const pair<const std::string, std::string> &depend = 
            pair<const std::string, std::string>((*iterator).first, (*inner_it).second);
      if (!alreadyExistsInMap(mIdMap, depend))
        mIdMap.insert(depend);
    }
  }
}


bool 
ExtModelReferenceCycles::alreadyExistsInMap(IdMap map, 
                                     pair<const std::string, std::string> dependency)
{
  bool exists = false;

  IdIter it;
  
  for (it = map.begin(); it != map.end(); it++)
  {
    if (((*it).first == dependency.first)
      && ((*it).second == dependency.second))
      exists = true;
  }

  return exists;
}

  
void 
ExtModelReferenceCycles::determineCycles(const Model& m)
{
  IdIter it;
  IdRange range;
  IdList variables;
  IdMap logged;
  std::string id;
  variables.clear();

  /* create a list of variables that are cycles ie (x, x) */
  for (it = mIdMap.begin(); it != mIdMap.end(); it++)
  {
    if ((*it).first == (*it).second)
    {
      id = (*it).first;
      if (!variables.contains(id))
      {
        variables.append(id);
      }
    }
  }

  /* loop thru other dependencies for each; if the dependent is also
   * in the list then this is the cycle
   * keep a record of logged dependencies to avoid logging twice
   */
   
  for (unsigned int n = 0; n < variables.size(); n++)
  {
    id = variables.at(n);
    range = mIdMap.equal_range(id);
    for (it = range.first; it != range.second; it++)
    {
      if (((*it).second == id)
        && !alreadyExistsInMap(logged, 
                    pair<const std::string, std::string>(id, (*it).second))
        && !alreadyExistsInMap(logged, 
                    pair<const std::string, std::string>((*it).second, id)))
      {
        logCycle(m, id, (*it).second);
        logged.insert(pair<const std::string, std::string>(id, (*it).second));
      }
    }
  }
}
 

/**
  * Logs a message about an undefined &lt;ci&gt; element in the given
  * FunctionDefinition.
  */
void
ExtModelReferenceCycles::logCycle (const Model& m, std::string id,
                                std::string id1)
{
  size_t under = id.find(".xml_",0);
  std::string file1 = id.substr(0, under);
  std::string model1 = id.substr(under+5, string::npos);

  size_t under2 = id1.find(".xml_",0);
  std::string file2 = id1.substr(0, under2);
  std::string model2 = id1.substr(under2+5, string::npos);
  
  msg = "ExternalModelDefinition with id '";
  msg += model1;
  msg += "' in file '";
  msg += file1;
  msg += "' creates a circular reference with the externalModelDefinition ";
  msg += "with id '"; 
  msg += model2;
  msg += "' in file '";
  msg += file2;
  msg += "'.";

  // want to log the error on a comp object
  // this is the main model - if there was no docPlugin/externalMDefs 
  // it would have quite by now
  //const CompSBMLDocumentPlugin* docPlug = (CompSBMLDocumentPlugin*)
  //  (m.getSBMLDocument()->getPlugin("comp"));
  COMP_CREATE_NS(compns, m.getSBMLNamespaces());
  ExternalModelDefinition *extMD = new ExternalModelDefinition(compns);
  delete compns;
  logFailure(*(extMD));
}  


void
ExtModelReferenceCycles::logCycle ( const SBase* object,
                                       const SBase* conflict )
{
  msg = "The ";
  msg += SBMLTypeCode_toString( object->getTypeCode(), object->getPackageName().c_str());
  msg += " with id '";
  msg += object->getId();
  msg += "' creates a cycle with the ";
  msg += SBMLTypeCode_toString( conflict->getTypeCode(), object->getPackageName().c_str());
  msg += " with id '";
  msg += conflict->getId();
  msg += "'.";

  
  logFailure(*object);
}

#endif  /* __cplusplus */

LIBSBML_CPP_NAMESPACE_END

/** @endcond */


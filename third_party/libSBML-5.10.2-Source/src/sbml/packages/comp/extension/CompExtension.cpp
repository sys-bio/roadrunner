/**
 * @file    CompExtension.cpp
 * @brief   Implementation of CompExtension, the core module of comp package.
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

#include <sbml/extension/SBMLExtensionRegister.h>
#include <sbml/extension/SBMLExtensionRegistry.h>
#include <sbml/extension/SBasePluginCreator.h>
#include <sbml/extension/SBMLDocumentPlugin.h>

#include <sbml/packages/comp/common/compfwd.h>
#include <sbml/packages/comp/extension/CompExtension.h>
#include <sbml/packages/comp/extension/CompSBasePlugin.h>
#include <sbml/packages/comp/extension/CompModelPlugin.h>
#include <sbml/packages/comp/extension/CompSBMLDocumentPlugin.h>
#include <sbml/packages/comp/validator/CompSBMLErrorTable.h>


#include <sbml/packages/comp/util//CompFlatteningConverter.h>
#include <sbml/conversion/SBMLConverterRegistry.h>


#include <iostream>

LIBSBML_CPP_NAMESPACE_BEGIN
#ifdef __cplusplus
// -------------------------------------------------------------------------
//
// This block is global initialization code which should be automatically 
// executed before invoking main() block.
//
// -------------------------------------------------------------------------

//------------- (START) -----------------------------------

// The name of this package

const std::string& CompExtension::getPackageName ()
{
  static const std::string pkgName = "comp";
  return pkgName;
}

//
// Default SBML level, version, and package version
//
unsigned int CompExtension::getDefaultLevel()
{
  return 3;
}  

unsigned int CompExtension::getDefaultVersion()
{
  return 1; 
}

unsigned int CompExtension::getDefaultPackageVersion()
{
  return 1;
} 

//
// XML namespaces of (1) package versions of groups extension, and 
// (2) another XML namespace(XMLSchema-instance) required in the groups 
//  extension.
//

const std::string& CompExtension::getXmlnsL3V1V1 ()
{
  static const std::string xmlns = "http://www.sbml.org/sbml/level3/version1/comp/version1";
  return xmlns;
}

//
// Adds this CompExtension object to the SBMLExtensionRegistry class.
// CompExtension::init() function is automatically invoked when this
// object is instantiated.
//
static SBMLExtensionRegister<CompExtension> compExtensionRegistry;

template class LIBSBML_EXTERN SBMLExtensionNamespaces<CompExtension>;



CompExtension::CompExtension ()
  : SBMLExtension()
{
}


CompExtension::CompExtension(const CompExtension& orig)
  : SBMLExtension(orig)
{
}


CompExtension::~CompExtension ()
{
}


CompExtension& 
CompExtension::operator=(const CompExtension& orig)
{
  SBMLExtension::operator=(orig);
  return *this;
}


/*
 * Creates and returns a deep copy of this CompExtension object.
 */
CompExtension* 
CompExtension::clone () const
{
  return new CompExtension(*this);  
}


const std::string&
CompExtension::getName() const
{
  return getPackageName();
}


/*
 * Returns the URI (namespace) of the package corresponding to the combination of the given sbml level,
 * sbml version, and package version.
 * Empty string will be returned if no corresponding URI exists.
 */
const std::string& 
CompExtension::getURI(unsigned int sbmlLevel, unsigned int sbmlVersion, unsigned int pkgVersion) const
{
  if (sbmlLevel == 3)
  {
    if (sbmlVersion == 1)
    {
      if (pkgVersion == 1)
      {
        return getXmlnsL3V1V1();
      }
    }
  }

  static std::string empty = "";

  return empty;
}


/*
 * Returns the SBML level with the given URI of this package.
 */
unsigned int 
CompExtension::getLevel(const std::string &uri) const
{
  if (uri == getXmlnsL3V1V1())
  {
    return 3;
  }
  
  return 0;
}


/*
 * Returns the SBML version with the given URI of this package.
 */
unsigned int 
CompExtension::getVersion(const std::string &uri) const
{
  if (uri == getXmlnsL3V1V1())
  {
    return 1;
  }

  return 0;
}


/*
 * Returns the package version with the given URI of this package.
 */
unsigned int
CompExtension::getPackageVersion(const std::string &uri) const
{
  if (uri == getXmlnsL3V1V1())
  {
    return 1;
  }

  return 0;
}


/*
 * Returns an SBMLExtensionNamespaces<class SBMLExtensionType> object 
 * (e.g. SBMLExtensionNamespaces<CompExtension> whose alias type is 
 * CompPkgNamespaces) corresponding to the given uri.
 * NULL will be returned if the given uri is not defined in the corresponding package.
 */
SBMLNamespaces*
CompExtension::getSBMLExtensionNamespaces(const std::string &uri) const
{
  CompPkgNamespaces* pkgns = 0;
  if ( uri == getXmlnsL3V1V1())
  {
    pkgns = new CompPkgNamespaces(3,1,1);    
  }  
  return pkgns;
}


static
const char* SBML_COMP_TYPECODE_STRINGS[] =
{
    "Submodel"
  , "Model Definition"
  , "External Model Definition"
  , "SBaseRef"
  , "Deletion"
  , "Replaced Element"
  , "Replaced By"
  , "Port"
  , "(Unknown SBML Comp Type)"
};

/*
 * This method takes a type code of comp package and returns a string representing
 * the code.
 */
const char* 
CompExtension::getStringFromTypeCode(int typeCode) const
{
  int min = SBML_COMP_SUBMODEL;
  int max = SBML_COMP_PORT;

  if ( typeCode < min || typeCode > max)
  {
    typeCode = SBML_COMP_PORT + 1;
  }

  return SBML_COMP_TYPECODE_STRINGS[typeCode - min];
}


/** @cond doxygenLibsbmlInternal */
/*
 * Initialization function of comp extension module which is automatically invoked 
 * by SBMLExtensionRegister class before main() function invoked.
 */
void 
CompExtension::init()
{
  // 1. Checks if the comp package has already been registered.
  if (SBMLExtensionRegistry::getInstance().isRegistered(getPackageName()))
  {
    // do nothing;
    return;
  }

  // 2. Creates an SBMLExtension derived object.
  CompExtension compExtension;

  // 3. Creates SBasePluginCreatorBase derived objects required for this 
  //    extension.

  std::vector<std::string> packageURIs;
  packageURIs.push_back(getXmlnsL3V1V1());

  SBaseExtensionPoint sbmldocExtPoint("core",SBML_DOCUMENT);
  SBaseExtensionPoint modelExtPoint("core",SBML_MODEL);
  
  //SBaseExtensionPoint compExtPoint("core",SBML_COMPARTMENT);
  //SBaseExtensionPoint constraintExtPoint("core",SBML_CONSTRAINT);
  //SBaseExtensionPoint eventExtPoint("core",SBML_EVENT);
  //SBaseExtensionPoint eventAssgnExtPoint("core",SBML_EVENT_ASSIGNMENT);
  //SBaseExtensionPoint funcDefnExtPoint("core",SBML_FUNCTION_DEFINITION);
  //SBaseExtensionPoint initAssgnExtPoint("core",SBML_INITIAL_ASSIGNMENT);
  //SBaseExtensionPoint kLawExtPoint("core",SBML_KINETIC_LAW);
  //SBaseExtensionPoint listOfExtPoint("core",SBML_LIST_OF);
  //SBaseExtensionPoint paramExtPoint("core",SBML_PARAMETER);
  //SBaseExtensionPoint reactionExtPoint("core",SBML_REACTION);
  //SBaseExtensionPoint ruleExtPoint("core",SBML_RULE);
  //SBaseExtensionPoint speciesExtPoint("core",SBML_SPECIES);
  //SBaseExtensionPoint spRefExtPoint("core",SBML_SPECIES_REFERENCE);
  //SBaseExtensionPoint modSpRefExtPoint("core",SBML_MODIFIER_SPECIES_REFERENCE);
  //SBaseExtensionPoint unitDefnExtPoint("core",SBML_UNIT_DEFINITION);
  //SBaseExtensionPoint unitExtPoint("core",SBML_UNIT);
  //SBaseExtensionPoint algebraicRuleExtPoint("core",SBML_ALGEBRAIC_RULE);
  //SBaseExtensionPoint assgnRuleExtPoint("core",SBML_ASSIGNMENT_RULE);
  //SBaseExtensionPoint rateRuleExtPoint("core",SBML_RATE_RULE);
  //SBaseExtensionPoint triggerExtPoint("core",SBML_TRIGGER);
  //SBaseExtensionPoint delayExtPoint("core",SBML_DELAY);
  //SBaseExtensionPoint priorityExtPoint("core",SBML_PRIORITY);
  //SBaseExtensionPoint stoichMathExtPoint("core",SBML_STOICHIOMETRY_MATH);//LS DEBUG switch to priority
  //SBaseExtensionPoint localParamExtPoint("core",SBML_LOCAL_PARAMETER);
  //SBaseExtensionPoint modelDefintionExtPoint("comp", SBML_COMP_MODELDEFINITION);
  //SBaseExtensionPoint deletionExtPoint("comp", SBML_COMP_DELETION);
  //SBaseExtensionPoint externalModelDefintionExtPoint("comp", SBML_COMP_EXTERNALMODELDEFINITION);
  //SBaseExtensionPoint portExtPoint("comp", SBML_COMP_PORT);
  //SBaseExtensionPoint replacedElementExtPoint("comp", SBML_COMP_REPLACEDELEMENT);
  //SBaseExtensionPoint replacedByExtPoint("comp", SBML_COMP_REPLACEDBY);
  //SBaseExtensionPoint sBaseRefExtPoint("comp", SBML_COMP_SBASEREF);
  //SBaseExtensionPoint submodelExtPoint("comp", SBML_COMP_SUBMODEL);
  
  SBaseExtensionPoint sbaseExtPoint("all", SBML_GENERIC_SBASE);


  //CompSBMLDocumentPlugin only extends SBMLDocument:
  SBasePluginCreator<CompSBMLDocumentPlugin, CompExtension> sbmldocPluginCreator(sbmldocExtPoint,packageURIs);

  //CompSBasePlugin extends absolutely everything created with SBase:
  SBasePluginCreator<CompModelPlugin, CompExtension> modelPluginCreator(modelExtPoint,packageURIs);
  
  //SBasePluginCreator<CompSBasePlugin, CompExtension> compPluginCreator(compExtPoint,packageURIs);
  //SBasePluginCreator<CompSBasePlugin, CompExtension> constraintPluginCreator(constraintExtPoint,packageURIs);
  //SBasePluginCreator<CompSBasePlugin, CompExtension> eventPluginCreator(eventExtPoint,packageURIs);
  //SBasePluginCreator<CompSBasePlugin, CompExtension> eventAssgnPluginCreator(eventAssgnExtPoint,packageURIs);
  //SBasePluginCreator<CompSBasePlugin, CompExtension> funcDefnPluginCreator(funcDefnExtPoint,packageURIs);
  //SBasePluginCreator<CompSBasePlugin, CompExtension> initAssgnPluginCreator(initAssgnExtPoint,packageURIs);
  //SBasePluginCreator<CompSBasePlugin, CompExtension> kLawPluginCreator(kLawExtPoint,packageURIs);
  //SBasePluginCreator<CompSBasePlugin, CompExtension> listOfPluginCreator(listOfExtPoint,packageURIs);
  //SBasePluginCreator<CompSBasePlugin, CompExtension> paramPluginCreator(paramExtPoint,packageURIs);
  //SBasePluginCreator<CompSBasePlugin, CompExtension> reactionPluginCreator(reactionExtPoint,packageURIs);
  //SBasePluginCreator<CompSBasePlugin, CompExtension> rulePluginCreator(ruleExtPoint,packageURIs);
  //SBasePluginCreator<CompSBasePlugin, CompExtension> speciesPluginCreator(speciesExtPoint,packageURIs);
  //SBasePluginCreator<CompSBasePlugin, CompExtension> spRefPluginCreator(spRefExtPoint,packageURIs);
  //SBasePluginCreator<CompSBasePlugin, CompExtension> modeSpRefPluginCreator(modSpRefExtPoint,packageURIs);
  //SBasePluginCreator<CompSBasePlugin, CompExtension> unitDefnPluginCreator(unitDefnExtPoint,packageURIs);
  //SBasePluginCreator<CompSBasePlugin, CompExtension> unitPluginCreator(unitExtPoint,packageURIs);
  //SBasePluginCreator<CompSBasePlugin, CompExtension> algebraicRulePluginCreator(algebraicRuleExtPoint,packageURIs);
  //SBasePluginCreator<CompSBasePlugin, CompExtension> assignRulePluginCreator(assgnRuleExtPoint,packageURIs);
  //SBasePluginCreator<CompSBasePlugin, CompExtension> rateRulePluginCreator(rateRuleExtPoint,packageURIs);
  //SBasePluginCreator<CompSBasePlugin, CompExtension> triggerPluginCreator(triggerExtPoint,packageURIs);
  //SBasePluginCreator<CompSBasePlugin, CompExtension> delayPluginCreator(delayExtPoint,packageURIs);
  //SBasePluginCreator<CompSBasePlugin, CompExtension> priorityPluginCreator(priorityExtPoint,packageURIs);
  //SBasePluginCreator<CompSBasePlugin, CompExtension> stoichMathPluginCreator(stoichMathExtPoint,packageURIs);
  //SBasePluginCreator<CompSBasePlugin, CompExtension> localParamPluginCreator(localParamExtPoint,packageURIs);
  //SBasePluginCreator<CompModelPlugin, CompExtension> modelDefinitonPluginCreator(modelDefintionExtPoint,packageURIs);
  //SBasePluginCreator<CompSBasePlugin, CompExtension> deletionPluginCreator(deletionExtPoint,packageURIs);
  //SBasePluginCreator<CompSBasePlugin, CompExtension> externalModelDefintionPluginCreator(externalModelDefintionExtPoint,packageURIs);
  //SBasePluginCreator<CompSBasePlugin, CompExtension> portPluginCreator(portExtPoint,packageURIs);
  //SBasePluginCreator<CompSBasePlugin, CompExtension> replacedByPluginCreator(replacedByExtPoint,packageURIs);
  //SBasePluginCreator<CompSBasePlugin, CompExtension> replacedElementPluginCreator(replacedElementExtPoint,packageURIs);
  //SBasePluginCreator<CompSBasePlugin, CompExtension> sBaseRefPluginCreator(sBaseRefExtPoint,packageURIs);
  //SBasePluginCreator<CompSBasePlugin, CompExtension> submodelPluginCreator(submodelExtPoint,packageURIs);

  SBasePluginCreator<CompSBasePlugin, CompExtension> sbasePluginCreator(sbaseExtPoint,packageURIs);

  // 4. Adds the above SBasePluginCreatorBase derived objects to the SBMLExtension derived object.

  compExtension.addSBasePluginCreator(&sbmldocPluginCreator);
  compExtension.addSBasePluginCreator(&modelPluginCreator);
  
  //compExtension.addSBasePluginCreator(&compPluginCreator);
  //compExtension.addSBasePluginCreator(&constraintPluginCreator);
  //compExtension.addSBasePluginCreator(&eventPluginCreator);
  //compExtension.addSBasePluginCreator(&eventAssgnPluginCreator);
  //compExtension.addSBasePluginCreator(&funcDefnPluginCreator);
  //compExtension.addSBasePluginCreator(&initAssgnPluginCreator);
  //compExtension.addSBasePluginCreator(&kLawPluginCreator);
  //compExtension.addSBasePluginCreator(&listOfPluginCreator);
  //compExtension.addSBasePluginCreator(&paramPluginCreator);
  //compExtension.addSBasePluginCreator(&reactionPluginCreator);
  //compExtension.addSBasePluginCreator(&rulePluginCreator);
  //compExtension.addSBasePluginCreator(&speciesPluginCreator);
  //compExtension.addSBasePluginCreator(&spRefPluginCreator);
  //compExtension.addSBasePluginCreator(&modeSpRefPluginCreator);
  //compExtension.addSBasePluginCreator(&unitDefnPluginCreator);
  //compExtension.addSBasePluginCreator(&unitPluginCreator);
  //compExtension.addSBasePluginCreator(&algebraicRulePluginCreator);
  //compExtension.addSBasePluginCreator(&assignRulePluginCreator);
  //compExtension.addSBasePluginCreator(&rateRulePluginCreator);
  //compExtension.addSBasePluginCreator(&triggerPluginCreator);
  //compExtension.addSBasePluginCreator(&delayPluginCreator);
  //compExtension.addSBasePluginCreator(&priorityPluginCreator);
  //compExtension.addSBasePluginCreator(&stoichMathPluginCreator);
  //compExtension.addSBasePluginCreator(&localParamPluginCreator);
  //compExtension.addSBasePluginCreator(&modelDefinitonPluginCreator);
  //compExtension.addSBasePluginCreator(&deletionPluginCreator);
  //compExtension.addSBasePluginCreator(&externalModelDefintionPluginCreator);
  //compExtension.addSBasePluginCreator(&portPluginCreator);
  //compExtension.addSBasePluginCreator(&replacedByPluginCreator);
  //compExtension.addSBasePluginCreator(&replacedElementPluginCreator);
  //compExtension.addSBasePluginCreator(&sBaseRefPluginCreator);
  //compExtension.addSBasePluginCreator(&submodelPluginCreator);

  compExtension.addSBasePluginCreator(&sbasePluginCreator);

  // 5. Registers the SBMLExtension derived object to SBMLExtensionRegistry

  int result = SBMLExtensionRegistry::getInstance().addExtension(&compExtension);

  if (result != LIBSBML_OPERATION_SUCCESS)
  {
#if 0
    std::cerr << "[Error] CompExtension::init() failed." << std::endl;
#endif
  }


  // 6. Register the flattening converter

  CompFlatteningConverter c1;
  SBMLConverterRegistry::getInstance().addConverter(&c1);

}
/** @endcond */


/** @cond doxygenLibsbmlInternal */
packageErrorTableEntry
CompExtension::getErrorTable(unsigned int index) const
{
  return compErrorTable[index];
}
/** @endcond */

/** @cond doxygenLibsbmlInternal */
unsigned int 
CompExtension::getErrorTableIndex(unsigned int errorId) const
{
  unsigned int tableSize = sizeof(compErrorTable)/sizeof(compErrorTable[0]);
  unsigned int index = 0;

  for ( unsigned int i = 0; i < tableSize; i++ )
  {
    if ( errorId == compErrorTable[i].code )
    {
      index = i;
      break;
    }
  }

  return index;
}
/** @endcond */


/** @cond doxygenLibsbmlInternal */
unsigned int
CompExtension::getErrorIdOffset() const
{
  return 1000000;
}
/** @endcond */



#endif  /* __cplusplus */
LIBSBML_CPP_NAMESPACE_END


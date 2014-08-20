/**
 * @file    FbcExtension.cpp
 * @brief   Implementation of FbcExtension, the core module of fbc package.
 * @author  Akiya Jouraku
 * @author  Frank Bergmann
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
 * Copyright (C) 2009-2013 jointly by the following organizations: 
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
#include <sbml/packages/fbc/extension/FbcSBMLDocumentPlugin.h>
#include <sbml/conversion/SBMLConverterRegistry.h>
#include <sbml/packages/fbc/util/CobraToFbcConverter.h>
#include <sbml/packages/fbc/util/FbcToCobraConverter.h>

#include <sbml/packages/fbc/extension/FbcExtension.h>
#include <sbml/packages/fbc/extension/FbcModelPlugin.h>
#include <sbml/packages/fbc/extension/FbcSpeciesPlugin.h>
#include <sbml/packages/fbc/validator/FbcSBMLErrorTable.h>

#ifdef __cplusplus

#include <iostream>

LIBSBML_CPP_NAMESPACE_BEGIN

// -------------------------------------------------------------------------
//
// This block is global initialization code which should be automatically 
// executed before invoking main() block.
//
// -------------------------------------------------------------------------

//------------- (START) -----------------------------------

// The name of this package

const std::string& FbcExtension::getPackageName ()
{
  static const std::string pkgName = "fbc";
  return pkgName;
}

//
// Default SBML level, version, and package version
//
unsigned int FbcExtension::getDefaultLevel()
{
  return 3;
}  

unsigned int FbcExtension::getDefaultVersion()
{
  return 1; 
}

unsigned int FbcExtension::getDefaultPackageVersion()
{
  return 1;
} 

//
// XML namespaces of (1) package versions of fbc extension, and 
// (2) another XML namespace(XMLSchema-instance) required in the fbc 
//  extension.
//

const std::string& FbcExtension::getXmlnsL3V1V1 ()
{
  static const std::string xmlns = "http://www.sbml.org/sbml/level3/version1/fbc/version1";
  return xmlns;
}

//
// Adds this FbcExtension object to the SBMLExtensionRegistry class.
// FbcExtension::init() function is automatically invoked when this
// object is instantiated.
//
static SBMLExtensionRegister<FbcExtension> fbcExtensionRegistry;

static
const char* SBML_FBC_TYPECODE_STRINGS[] =
{
    "Association"
  , "FluxBound"
  , "FluxObjective"
  , "GeneAssociation"
  , "Objective"

};

//------------- (END) -----------------------------------

// --------------------------------------------------------
//
// Instantiate SBMLExtensionNamespaces<FbcExtension>
// (FbcPkgNamespaces) for DLL.
//
// --------------------------------------------------------

template class LIBSBML_EXTERN SBMLExtensionNamespaces<FbcExtension>;



FbcExtension::FbcExtension ()
{
}


/*
 * Copy constructor.
 */
FbcExtension::FbcExtension(const FbcExtension& orig)
: SBMLExtension(orig)
{
}


/*
 * Destroy this object.
 */
FbcExtension::~FbcExtension ()
{
}


/*
 * Assignment operator for FbcExtension.
 */
FbcExtension& 
FbcExtension::operator=(const FbcExtension& orig)
{
  SBMLExtension::operator=(orig);
  return *this;
}


/*
 * Creates and returns a deep copy of this FbcExtension object.
 * 
 * @return a (deep) copy of this FbcExtension object
 */
FbcExtension* 
FbcExtension::clone () const
{
  return new FbcExtension(*this);  
}


const std::string&
FbcExtension::getName() const
{
  return getPackageName();
}


/*
 * Returns the URI (namespace) of the package corresponding to the combination of the given sbml level,
 * sbml version, and package version.
 * Empty string will be returned if no corresponding URI exists.
 *
 * @return a string of the package URI
 */
const std::string& 
FbcExtension::getURI(unsigned int sbmlLevel, unsigned int sbmlVersion, unsigned int pkgVersion) const
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
 *
 *  (NOTICE) Package developers MUST OVERRIDE this pure virtual function in their derived class.
 *
 */
unsigned int 
FbcExtension::getLevel(const std::string &uri) const
{
  if (uri == getXmlnsL3V1V1())
  {
    return 3;
  }
  
  return 0;
}


/*
 * Returns the SBML version with the given URI of this package.
 *
 *  (NOTICE) Package developers MUST OVERRIDE this pure virtual function in their derived class.
 *
 */
unsigned int 
FbcExtension::getVersion(const std::string &uri) const
{
  if (uri == getXmlnsL3V1V1())
  {
    return 1;
  }

  return 0;
}


/*
 * Returns the package version with the given URI of this package.
 *
 *  (NOTICE) Package developers MUST OVERRIDE this pure virtual function in their derived class.
 *
 */
unsigned int
FbcExtension::getPackageVersion(const std::string &uri) const
{
  if (uri == getXmlnsL3V1V1())
  {
    return 1;
  }

  return 0;
}


/*
 * Returns an SBMLExtensionNamespaces<class SBMLExtensionType> object 
 * (e.g. SBMLExtensionNamespaces<FbcExtension> whose alias type is 
 * FbcPkgNamespaces) corresponding to the given uri.
 * NULL will be returned if the given uri is not defined in the corresponding package.
 *
 *  (NOTICE) Package developers MUST OVERRIDE this pure virtual function in their derived class.
 *
 */
SBMLNamespaces*
FbcExtension::getSBMLExtensionNamespaces(const std::string &uri) const
{
  FbcPkgNamespaces* pkgns = NULL;
  if ( uri == getXmlnsL3V1V1())
  {
    pkgns = new FbcPkgNamespaces(3,1,1);    
  }  
  return pkgns;
}


/*
 * This method takes a type code of the fbc package and returns a string representing
 * the code.
 */
const char* 
FbcExtension::getStringFromTypeCode(int typeCode) const
{
  int min = SBML_FBC_ASSOCIATION;
  int max = SBML_FBC_OBJECTIVE;

  if ( typeCode < min || typeCode > max)
  {
    return "(Unknown SBML Fbc Type)";  
  }

  return SBML_FBC_TYPECODE_STRINGS[typeCode - min];
}

/** @cond doxygenLibsbmlInternal */
/*
 *
 * Initialization function of the fbc extension module which is automatically invoked 
 * by SBMLExtensionRegister class before main() function invoked.
 *
 */
void 
FbcExtension::init()
{
  //-------------------------------------------------------------------------
  //
  // 1. Checks if the fbc package has already been registered.
  //
  //-------------------------------------------------------------------------

  if (SBMLExtensionRegistry::getInstance().isRegistered(getPackageName()))
  {
    // do nothing;
    return;
  }

  //-------------------------------------------------------------------------
  //
  // 2. Creates an SBMLExtension derived object.
  //
  //-------------------------------------------------------------------------

  FbcExtension fbcExtension;

  //-------------------------------------------------------------------------------------
  //
  // 3. Creates SBasePluginCreatorBase derived objects required for this 
  //    extension. The derived classes can be instantiated by using the following 
  //     template class.
  //
  //    temaplate<class SBasePluginType> class SBasePluginCreator
  //
  //    The constructor of the creator class has two arguments:
  //
  //        (1) SBaseExtensionPoint : extension point to which the plugin object connected
  //        (2) std::vector<std::string> : a std::vector object that contains a list of URI
  //                                       (package versions) supported by the plugin object.
  //---------------------------------------------------------------------------------------

  std::vector<std::string> packageURIs;
  packageURIs.push_back(getXmlnsL3V1V1());

  SBaseExtensionPoint sbmldocExtPoint("core",SBML_DOCUMENT);
  SBaseExtensionPoint modelExtPoint("core",SBML_MODEL);
  SBaseExtensionPoint speciesExtPoint("core",SBML_SPECIES);

  SBasePluginCreator<FbcSBMLDocumentPlugin, FbcExtension> sbmldocPluginCreator(sbmldocExtPoint,packageURIs);
  SBasePluginCreator<FbcModelPlugin,     FbcExtension> modelPluginCreator(modelExtPoint,packageURIs);
  SBasePluginCreator<FbcSpeciesPlugin,   FbcExtension> speciesPluginCreator(speciesExtPoint,packageURIs);

  //--------------------------------------------------------------------------------------
  //
  // 3. Adds the above SBasePluginCreatorBase derived objects to the SBMLExtension derived object.
  //
  //--------------------------------------------------------------------------------------

  fbcExtension.addSBasePluginCreator(&sbmldocPluginCreator);
  fbcExtension.addSBasePluginCreator(&modelPluginCreator);
  fbcExtension.addSBasePluginCreator(&speciesPluginCreator);
  
  //-------------------------------------------------------------------------
  //
  // 4. Registers the SBMLExtension derived object to SBMLExtensionRegistry
  //
  //-------------------------------------------------------------------------

  int result = SBMLExtensionRegistry::getInstance().addExtension(&fbcExtension);

  if (result != LIBSBML_OPERATION_SUCCESS)
  {
#if 0
    std::cerr << "[Error] FbcExtension::init() failed." << std::endl;
#endif
  }
  
  
  // 5. Register the cobra converter
  
  CobraToFbcConverter c1;
  SBMLConverterRegistry::getInstance().addConverter(&c1);
  FbcToCobraConverter c2;
  SBMLConverterRegistry::getInstance().addConverter(&c2);
  
}
/** @endcond */


/** @cond doxygenLibsbmlInternal */
packageErrorTableEntry
FbcExtension::getErrorTable(unsigned int index) const
{
  return fbcErrorTable[index];
}
/** @endcond */

/** @cond doxygenLibsbmlInternal */
unsigned int 
FbcExtension::getErrorTableIndex(unsigned int errorId) const
{
  unsigned int tableSize = sizeof(fbcErrorTable)/sizeof(fbcErrorTable[0]);
  unsigned int index = 0;

  for ( unsigned int i = 0; i < tableSize; i++ )
  {
    if ( errorId == fbcErrorTable[i].code )
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
FbcExtension::getErrorIdOffset() const
{
  return 2000000;
}
/** @endcond */


LIBSBML_CPP_NAMESPACE_END

#endif  /* __cplusplus */

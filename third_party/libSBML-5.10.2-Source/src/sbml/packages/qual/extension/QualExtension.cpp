/**
 * @file    QualExtension.cpp
 * @brief   Implementation of QualExtension, the core module of qual package.
 * @author  Akiya Jouraku
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
 * Copyright (C) 2009-2013 jointly by the following organizations:
 *     1. California Institute of Technology, Pasadena, CA, USA
 *     2. EMBL European Bioinformatics Institute (EMBL-EBI), Hinxton, UK
 *  
 * Copyright (C) 2006-2008 by the California Institute of Technology,
 *     Pasadena, CA, USA 
 *  
 * Copyright (C) 2002-2005 jointly by the following organizations: 
 *     1. California Institute of Technology, Pasadena, CA, USA
 *     2. Japan Science and Technology Agency, Japan
 * 
 * This library is free software; you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation.  A copy of the license agreement is provided
 * in the file named "LICENSE.txt" included with this software distribution
 * and also available online as http://sbml.org/software/libsbml/license.html
 * ------------------------------------------------------------------------ -->
 */

#include <sbml/extension/SBMLExtensionRegister.h>
#include <sbml/extension/SBMLExtensionRegistry.h>
#include <sbml/extension/SBasePluginCreator.h>
#include <sbml/extension/SBMLDocumentPlugin.h>

#include <sbml/packages/qual/extension/QualExtension.h>
#include <sbml/packages/qual/extension/QualModelPlugin.h>
#include <sbml/packages/qual/extension/QualSBMLDocumentPlugin.h>
#include <sbml/packages/qual/validator/QualSBMLErrorTable.h>

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

const std::string& QualExtension::getPackageName ()
{
  static const std::string pkgName = "qual";
  return pkgName;
}

//
// Default SBML level, version, and package version
//
unsigned int
QualExtension::getDefaultLevel ()
{
  return 3;
}  

/*
 * Returns the default SBML Version this extension.
 */
unsigned int
QualExtension::getDefaultVersion ()
{
  return 1; 
}

/*
 * Returns the default SBML version this extension.
 */
unsigned int
QualExtension::getDefaultPackageVersion ()
{
  return 1;
} 

//
// XML namespaces of (1) package versions of qual extension, and 
// (2) another XML namespace(XMLSchema-instance) required in the qual 
//  extension.
//
const std::string&
QualExtension::getXmlnsL3V1V1 ()
{
  static const std::string xmlns = "http://www.sbml.org/sbml/level3/version1/qual/version1";
  return xmlns;
}

//
// Adds this QualExtension object to the SBMLExtensionRegistry class.
// QualExtension::init() function is automatically invoked when this
// object is instantiated.
//
static SBMLExtensionRegister<QualExtension> qualExtensionRegistry;

static
const char* SBML_QUAL_TYPECODE_STRINGS[] =
{
    "QualitativeSpecies"
  , "Transition"
  , "Input"
  , "Output"
  , "FunctionTerm"
  , "DefaultTerm"
};

//------------- (END) -----------------------------------

// --------------------------------------------------------
//
// Instantiate SBMLExtensionNamespaces<QualExtension>
// (QualPkgNamespaces) for DLL.
//
// --------------------------------------------------------

template class LIBSBML_EXTERN SBMLExtensionNamespaces<QualExtension>;



QualExtension::QualExtension ()
{
}


/*
 * Copy constructor.
 */
QualExtension::QualExtension(const QualExtension& orig)
: SBMLExtension(orig)
{
}


/*
 * Destroy this object.
 */
QualExtension::~QualExtension ()
{
}


/*
 * Assignment operator for QualExtension.
 */
QualExtension&
QualExtension::operator=(const QualExtension& rhs)
 {
  if (&rhs != this)
  {
    SBMLExtension::operator=(rhs);
  }
  return *this;
}


/*
 * Creates and returns a deep copy of this QualExtension object.
 * 
 * @return a (deep) copy of this SBase object
 */
QualExtension* 
QualExtension::clone () const
{
  return new QualExtension(*this);  
}


const std::string&
QualExtension::getName() const
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
QualExtension::getURI(unsigned int sbmlLevel, unsigned int sbmlVersion, unsigned int pkgVersion) const
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
QualExtension::getLevel(const std::string &uri) const
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
QualExtension::getVersion(const std::string &uri) const
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
QualExtension::getPackageVersion(const std::string &uri) const
{
  if (uri == getXmlnsL3V1V1())
  {
    return 1;
  }

  return 0;
}


/*
 * Returns an SBMLExtensionNamespaces<class SBMLExtensionType> object 
 * (e.g. SBMLExtensionNamespaces<QualExtension> whose alias type is 
 * QualPkgNamespaces) corresponding to the given uri.
 * NULL will be returned if the given uri is not defined in the corresponding package.
 *
 *  (NOTICE) Package developers MUST OVERRIDE this pure virtual function in their derived class.
 *
 */
SBMLNamespaces*
QualExtension::getSBMLExtensionNamespaces(const std::string &uri) const
{
  QualPkgNamespaces* pkgns = NULL;
  if ( uri == getXmlnsL3V1V1())
  {
    pkgns = new QualPkgNamespaces(3,1,1);    
  }  
  return pkgns;
}


/*
 * This method takes a type code of qual package and returns a string representing
 * the code.
 */
const char* 
QualExtension::getStringFromTypeCode(int typeCode) const
{
  int min = SBML_QUAL_QUALITATIVE_SPECIES;
  int max = SBML_QUAL_DEFAULT_TERM;

  if ( typeCode < min || typeCode > max)
  {
    return "(Unknown SBML Qual Type)";  
  }

  return SBML_QUAL_TYPECODE_STRINGS[typeCode - min];
}


/** @cond doxygenLibsbmlInternal */
/*
 *
 * Initialization function of qual extension module which is automatically invoked 
 * by SBMLExtensionRegister class before main() function invoked.
 *
 */
void 
QualExtension::init()
{
  //-------------------------------------------------------------------------
  //
  // 1. Checks if the qual package has already been registered.
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

  QualExtension qualExtension;

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
  //
  //    For example, two plugin objects (plugged in SBMLDocument and Model elements) are 
  //    required for the qual extension.
  //
  //    Since only 'required' attribute is used in SBMLDocument by the qual package, existing
  //    SBMLDocumentPlugin class can be used as-is for the plugin.
  //
  //    Since the lists of supported package versions (currently only L3V1-qual-V1 supported )
  //    are equal in the both plugin objects, the same vector object is given to each 
  //    constructor.
  //
  //---------------------------------------------------------------------------------------

  std::vector<std::string> packageURIs;
  packageURIs.push_back(getXmlnsL3V1V1());

  SBaseExtensionPoint sbmldocExtPoint("core",SBML_DOCUMENT);
  SBaseExtensionPoint modelExtPoint("core",SBML_MODEL);

  SBasePluginCreator<QualSBMLDocumentPlugin, QualExtension> sbmldocPluginCreator(sbmldocExtPoint,packageURIs);
  SBasePluginCreator<QualModelPlugin,   QualExtension> modelPluginCreator(modelExtPoint,packageURIs);

  //--------------------------------------------------------------------------------------
  //
  // 3. Adds the above SBasePluginCreatorBase derived objects to the SBMLExtension derived object.
  //
  //--------------------------------------------------------------------------------------

  qualExtension.addSBasePluginCreator(&sbmldocPluginCreator);
  qualExtension.addSBasePluginCreator(&modelPluginCreator);

  //-------------------------------------------------------------------------
  //
  // 4. Registers the SBMLExtension derived object to SBMLExtensionRegistry
  //
  //-------------------------------------------------------------------------

  int result = SBMLExtensionRegistry::getInstance().addExtension(&qualExtension);

  if (result != LIBSBML_OPERATION_SUCCESS)
  {
#if 0
    std::cerr << "[Error] QualExtension::init() failed." << std::endl;
#endif
  }
}
/** @endcond */

/** @cond doxygenLibsbmlInternal */
packageErrorTableEntry
QualExtension::getErrorTable(unsigned int index) const
{
  return qualErrorTable[index];
}
/** @cond doxygenLibsbmlInternal */
unsigned int 
QualExtension::getErrorTableIndex(unsigned int errorId) const
{
  unsigned int tableSize = sizeof(qualErrorTable)/sizeof(qualErrorTable[0]);
  unsigned int index = 0;

  for ( unsigned int i = 0; i < tableSize; i++ )
  {
    if ( errorId == qualErrorTable[i].code )
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
QualExtension::getErrorIdOffset() const
{
  return 3000000;
}
/** @endcond */




LIBSBML_CPP_NAMESPACE_END


#endif /* __cplusplus */


/**
 * @file    CompExtension.h
 * @brief   Definition of CompExtension, the core module of comp package. 
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
 *
 * @class CompExtension
 * @sbmlbrief{comp} Base extension class for the package.
 *
 * @class CompPkgNamespaces
 * @sbmlbrief{comp} SBMLNamespaces extension for the package.
 */

#ifndef CompExtension_h
#define CompExtension_h

#include <sbml/common/extern.h>
#include <sbml/SBMLTypeCodes.h>

#ifdef __cplusplus

#include <sbml/extension/SBMLExtension.h>
#include <sbml/extension/SBMLExtensionNamespaces.h>
#include <sbml/extension/SBMLExtensionRegister.h>


#ifndef COMP_CREATE_NS
#define COMP_CREATE_NS(variable,sbmlns)\
  EXTENSION_CREATE_NS(CompPkgNamespaces,variable,sbmlns);
#endif

#include <vector>

LIBSBML_CPP_NAMESPACE_BEGIN

class LIBSBML_EXTERN CompExtension : public SBMLExtension
{
public:

  //---------------------------------------------------------------
  //
  // Required class variables
  //
  //---------------------------------------------------------------

  /**
   * Returns the nickname of the SBML Level&nbsp;3 package implemented by
   * this libSBML extension.
   *
   * @return the package nickname, as a string.
   *
   * @copydetails doc_note_static_methods
   */
  static const std::string& getPackageName ();


  /**
   * Returns the default SBML Level used by this libSBML package extension.
   *
   * @return the SBML Level.
   *
   * @copydetails doc_note_static_methods
   */
  static unsigned int getDefaultLevel();


  /**
   * Returns the default SBML Version used by this libSBML package extension.
   *
   * @return the Version within the default SBML Level.
   *
   * @copydetails doc_note_static_methods
   */
  static unsigned int getDefaultVersion();


  /**
   * Returns the default version of the SBML Level&nbsp;3 package implemented
   * by this libSBML extension.
   *
   * @return the default version number of the SBML Level&nbsp;3 package
   * definition.
   *
   * @copydetails doc_note_static_methods
   */
  static unsigned int getDefaultPackageVersion();


  /**
   * Returns the XML namespace URI of the SBML Level&nbsp;3 package
   * implemented by this libSBML extension.
   *
   * @return the XML namespace as a string.
   *
   * @copydetails doc_note_static_methods
   */
  static const std::string&  getXmlnsL3V1V1();


  //
  // Other URI needed in this package (if any)
  //

  //---------------------------------------------------------------


  /**
   * Constructor
   */
  CompExtension ();


  /**
   * Copy constructor.
   */
  CompExtension(const CompExtension&);


  /**
   * Destroy this object.
   */
  virtual ~CompExtension ();


  /**
   * Assignment operator for CompExtension.
   */
  CompExtension& operator=(const CompExtension&);


  /**
   * Creates and returns a deep copy of this CompExtension object.
   *
   * @return a (deep) copy of this CompExtension object
   */
  virtual CompExtension* clone () const;


  /**
   * Returns the name of this package as a short-form label
   * (&quot;<code>comp</code>&quot;).
   *
   * @return the name of this package.
   */
  virtual const std::string& getName() const;


  /**
   * Returns the namespace URI corresponding to the combination of the given
   * SBML Level, Version, and package version.
   *
   * @param sbmlLevel the level of SBML
   * @param sbmlVersion the version of SBML
   * @param pkgVersion the version of package
   *
   * @return a string of the package URI, or an empty string if no
   * corresponding URI exists.
   */
  virtual const std::string& getURI(unsigned int sbmlLevel,
                                    unsigned int sbmlVersion, 
                                    unsigned int pkgVersion) const;


  /**
   * Returns the SBML Level for the given URI of this package.
   *
   * @param uri the string of URI that represents one of versions of the
   * &ldquo;comp&rdquo; package
   *
   * @return the SBML Level with the given URI of this package, or @c 0 if
   * the given URI is invalid.
   */
  virtual unsigned int getLevel(const std::string &uri) const;


  /**
   * Returns the SBML Version for the given URI of this package.
   *
   * @param uri the string of URI that represents one of versions of the
   * &ldquo;comp&rdquo; package
   *
   * @return the SBML version with the given URI of this package, or @c 0 if
   * the given URI is invalid.
   */
  virtual unsigned int getVersion(const std::string &uri) const;


  /**
   * Returns the package version for the given URI of this package.
   *
   * @param uri the string of URI that represents one of versions of the
   * &ldquo;comp&rdquo; package
   *
   * @return the package version with the given URI of this package, or @c 0
   * if the given URI is invalid.
   */
  virtual unsigned int getPackageVersion(const std::string &uri) const;


  /**
   * Returns an CompPkgNamespaces object.
   *
   * @param uri the string of URI that represents one of versions of the
   * &ldquo;comp&rdquo; package
   *
   * @return an CompPkgNamespace object corresponding to the given @p uri, or
   * @c NULL if the URI is not defined in the Hierarchical %Model Composition
   * package.
   */
  virtual SBMLNamespaces* getSBMLExtensionNamespaces(const std::string &uri) const;


  /**
   * Takes a type code of the &ldquo;comp&rdquo; package and returns a string
   * describing the code.
   */
  virtual const char* getStringFromTypeCode(int typeCode) const;


  /** @cond doxygenLibsbmlInternal */
  /**
   * Initializes comp extension by creating an object of this class with 
   * required SBasePlugin derived objects and registering the object 
   * to the SBMLExtensionRegistry class.
   *
   * (NOTE) This function is automatically invoked when creating the following
   *        global object in CompExtension.cpp
   *
   *        static SBMLExtensionRegister<CompExtension> compExtensionRegistry;
   *
   */

  static void init();

  virtual packageErrorTableEntry getErrorTable(unsigned int index) const;
  
  virtual unsigned int getErrorTableIndex(unsigned int errorId) const;

  virtual unsigned int getErrorIdOffset() const;

  /** @endcond */

};


// --------------------------------------------------------------------
//
// Required typedef definitions 
//
// CompPkgNamespaces is derived from the SBMLNamespaces class and
// used when creating an object of SBase derived classes defined in
// comp package.
//
// --------------------------------------------------------------------

//
// (NOTE) 
//
// SBMLExtensionNamespaces<CompExtension> must be instantiated
// in CompExtension.cpp for DLL.
//
typedef SBMLExtensionNamespaces<CompExtension> CompPkgNamespaces; 

#endif  /* __cplusplus */

/**
 * @enum  SBMLCompTypeCode_t
 * @brief SBMLCompTypeCode_t is the enumeration of possible types from the 'comp' package.
 *
 * @copydetails doc_what_are_typecodes
 *
 * @copydetails doc_additional_typecode_details
 */
typedef enum
{
    SBML_COMP_SUBMODEL                = 250 /*!< Submodel */
  , SBML_COMP_MODELDEFINITION         = 251 /*!< ModelDefinition */
  , SBML_COMP_EXTERNALMODELDEFINITION = 252 /*!< ExternalModelDefinition */
  , SBML_COMP_SBASEREF                = 253 /*!< SBaseRef */
  , SBML_COMP_DELETION                = 254 /*!< Deletion */
  , SBML_COMP_REPLACEDELEMENT         = 255 /*!< ReplacedElement */
  , SBML_COMP_REPLACEDBY              = 256 /*!< ReplacedBy */
  , SBML_COMP_PORT                    = 257 /*!< Port */
} SBMLCompTypeCode_t;


LIBSBML_CPP_NAMESPACE_END

#endif  /* CompExtension_h */

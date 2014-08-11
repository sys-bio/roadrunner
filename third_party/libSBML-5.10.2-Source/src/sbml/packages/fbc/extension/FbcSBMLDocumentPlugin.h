/**
 * @file    FbcSBMLDocumentPlugin.h
 * @brief   Definition of FbcSBMLDocumentPlugin, the plugin class of
 *          Fbc package for the Model element.
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
 *
 * @class FbcSBMLDocumentPlugin
 * @sbmlbrief{fbc} Extension of SBMLDocument.
 *
 * The FbcSBMLDocumentPlugin class inherits from the SBMLDocumentPlugin
 * class, and codifies the extentions to the SBMLDocument class defined in
 * the SBML Level&nbsp;3 @ref fbc "Flux Balance Constraints" package ('fbc').
 *
 * The FbcSBMLDocumentPlugin defines a
 * required flag named <code>required</code>, which indicates whether the
 * 'fbc' constructs can be used to change the core mathematics of the
 * <code>&lt;model&gt;</code> child of the <code>&lt;sbml&gt;</code> element.
 * Because they can not, this attribute must be set @c false.
 */

#ifndef FbcSBMLDocumentPlugin_h
#define FbcSBMLDocumentPlugin_h


#ifdef __cplusplus

#include <sbml/common/extern.h>
#include <sbml/extension/SBMLDocumentPlugin.h>

#include <iostream>
#include <string>
#include <map>

#include <sbml/packages/fbc/extension/FbcExtension.h>
#include <sbml/packages/fbc/validator/FbcValidator.h>

LIBSBML_CPP_NAMESPACE_BEGIN

class FbcValidator;

class LIBSBML_EXTERN FbcSBMLDocumentPlugin : public SBMLDocumentPlugin
{
public:

  /**
   * Constructor
   */
  FbcSBMLDocumentPlugin (const std::string &uri, const std::string &prefix, FbcPkgNamespaces *fbcns);


  /**
   * Copy constructor. Creates a copy of this FbcSBMLDocumentPlugin object.
   */
  FbcSBMLDocumentPlugin(const FbcSBMLDocumentPlugin& orig);


  /**
   * Destroy this object.
   */
  virtual ~FbcSBMLDocumentPlugin ();


  /**
   * Assignment operator for FbcSBMLDocumentPlugin.
   */
  FbcSBMLDocumentPlugin& operator=(const FbcSBMLDocumentPlugin& orig);


  /**
   * Creates and returns a deep copy of this FbcSBMLDocumentPlugin object.
   * 
   * @return a (deep) copy of this FbcSBMLDocumentPlugin object
   */
  virtual FbcSBMLDocumentPlugin* clone () const;


#ifndef SWIG

  /** @cond doxygenLibsbmlInternal */
  /**
   * Reads the attributes of corresponding package in SBMLDocument element.
   */
  virtual void readAttributes (const XMLAttributes& attributes,
                               const ExpectedAttributes& expectedAttributes);
  /** @endcond */


#endif //SWIG
 
  /** @cond doxygenLibsbmlInternal */
  virtual bool isCompFlatteningImplemented() const;
  /** @endcond */


  /** @cond doxygenLibsbmlInternal */
  /**
   * Check consistency function.
   */
  virtual unsigned int checkConsistency();
  /** @endcond */


  /** @cond doxygenLibsbmlInternal */

  virtual bool accept(SBMLVisitor& v) const;

  /** @endcond */






};

LIBSBML_CPP_NAMESPACE_END

#endif  /* __cplusplus */
#endif  /* FbcSBMLDocumentPlugin_h */

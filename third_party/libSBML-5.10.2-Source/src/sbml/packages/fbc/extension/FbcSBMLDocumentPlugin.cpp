/**
 * @file    FbcSBMLDocumentPlugin.cpp
 * @brief   Implementation of FbcSBMLDocumentPlugin, the plugin class of
 *          fbc package for the SBase element.
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
 */

#include <ostream>

#include <sbml/common/libsbml-version.h>
#include <sbml/packages/fbc/common/fbcfwd.h>
#include <sbml/packages/fbc/extension/FbcSBMLDocumentPlugin.h>
#include <sbml/packages/fbc/validator/FbcConsistencyValidator.h>
#include <sbml/packages/fbc/validator/FbcIdentifierConsistencyValidator.h>
#include <sbml/packages/fbc/validator/FbcValidator.h>
#include <sbml/packages/fbc/validator/FbcSBMLError.h>

#include <iostream>

#ifdef __cplusplus

using namespace std;

LIBSBML_CPP_NAMESPACE_BEGIN

FbcSBMLDocumentPlugin::FbcSBMLDocumentPlugin (const string &uri, const string &prefix, FbcPkgNamespaces *fbcns)
  : SBMLDocumentPlugin(uri,prefix, fbcns)
{
}


FbcSBMLDocumentPlugin::FbcSBMLDocumentPlugin(const FbcSBMLDocumentPlugin& orig)
  : SBMLDocumentPlugin(orig)
{
}


FbcSBMLDocumentPlugin& 
FbcSBMLDocumentPlugin::operator=(const FbcSBMLDocumentPlugin& orig)
{
  if(&orig!=this)
  {
    SBMLDocumentPlugin::operator =(orig);
  }    
  return *this;
}


/*
 * Creates and returns a deep copy of this FbcSBMLDocumentPlugin object.
 */
FbcSBMLDocumentPlugin* 
FbcSBMLDocumentPlugin::clone () const
{
  return new FbcSBMLDocumentPlugin(*this);  
}

/*
 * Destroy this object.
 */
FbcSBMLDocumentPlugin::~FbcSBMLDocumentPlugin () 
{
}


/** @cond doxygenLibsbmlInternal */
void 
FbcSBMLDocumentPlugin::readAttributes (const XMLAttributes& attributes,
                            const ExpectedAttributes& expectedAttributes)
{
  // for now don't read the required flag for L2 models 
  if (getSBMLDocument() != NULL && getSBMLDocument()->getLevel() < 3) return;
  
  // do not need to call this as we are going to read the required attribute here
  //SBMLDocumentPlugin::readAttributes(attributes, expectedAttributes);
  unsigned int numErrs = getErrorLog()->getNumErrors();
  XMLTriple tripleRequired("required", mURI, getPrefix());
  bool assigned = attributes.readInto(tripleRequired, mRequired, 
                  getErrorLog(), false, getLine(), getColumn());
  if (assigned == false)
  {
    if (getErrorLog()->getNumErrors() == numErrs + 1 && 
        getErrorLog()->contains(XMLAttributeTypeMismatch))
    {
      getErrorLog()->logPackageError("fbc", FbcAttributeRequiredMustBeBoolean,
        getPackageVersion(), getLevel(), getVersion());
    }
    else
    {
      getErrorLog()->logPackageError("fbc", FbcAttributeRequiredMissing,
        getPackageVersion(), getLevel(), getVersion());
    }
  }
  else
  {
    mIsSetRequired = true;
    if (mRequired == true)
    {
      getErrorLog()->logPackageError("fbc", FbcRequiredFalse,
        getPackageVersion(), getLevel(), getVersion());
    }
  }
}
/** @endcond */


/** @cond doxygenLibsbmlInternal */
bool
FbcSBMLDocumentPlugin::isCompFlatteningImplemented() const
{
  return true;
}
/** @endcond */


/** @cond doxygenLibsbmlInternal */
unsigned int 
FbcSBMLDocumentPlugin::checkConsistency()
{
  unsigned int nerrors = 0;
  unsigned int total_errors = 0;

  SBMLDocument* doc = static_cast<SBMLDocument *>(this->getParentSBMLObject());
  SBMLErrorLog *log = doc->getErrorLog();

  unsigned char applicableValidators = doc->getApplicableValidators();

  /* determine which validators to run */
  bool id    = ((applicableValidators & 0x01) == 0x01);
  bool sbml  = ((applicableValidators & 0x02) == 0x02);

  FbcIdentifierConsistencyValidator id_validator;
  FbcConsistencyValidator validator;

  if (id)
  {
    id_validator.init();
    nerrors = id_validator.validate(*doc);
    total_errors += nerrors;
    if (nerrors > 0) 
    {
      log->add(id_validator.getFailures() );
      /* only want to bail if errors not warnings */
      if (log->getNumFailsWithSeverity(LIBSBML_SEV_ERROR) > 0)
      {
        return total_errors;
      }
    }
  }

  if (sbml)
  {
    validator.init();
    nerrors = validator.validate(*doc);
    total_errors += nerrors;
    if (nerrors > 0) 
    {
      log->add( validator.getFailures() );
    }
  }

  return total_errors;  
}
/** @endcond */

/** @cond doxygenLibsbmlInternal */
bool 
FbcSBMLDocumentPlugin::accept(SBMLVisitor& v) const
{
  const SBMLDocument *doc = static_cast<const SBMLDocument *>(this->getParentSBMLObject());
  v.visit(*doc);

  v.leave(*doc);

  return true;
}

/** @endcond */



LIBSBML_CPP_NAMESPACE_END

#endif  /* __cplusplus */

/**
 * @file    FbcSpeciesPlugin.cpp
 * @brief   Implementation of FbcSpeciesPlugin, the plugin class of
 *          the fbc package for the Species element.
 * @author  Frank T. Bergmann
 *
 * $Id$
 * $HeadURL$
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

#include <sbml/packages/fbc/extension/FbcSpeciesPlugin.h>
#include <sbml/packages/fbc/validator/FbcSBMLError.h>
#include <iostream>
using namespace std;


LIBSBML_CPP_NAMESPACE_BEGIN
#ifdef __cplusplus

static void
parseChemicalFormula(std::string& chemicalFormula, 
                     SBMLErrorLog& errLog, unsigned int packageVersion, 
                     unsigned int level, unsigned int version)
{
  std::vector< std::pair< std::string, int > > chemicalSymbols;

  int chemicalNumber;

  size_t index = 0;
  size_t sizeStr = chemicalFormula.size();
  char c = chemicalFormula[index];
  
  while (index < sizeStr)
  {

    stringstream elementName;
    stringstream elementCount;


    if (isupper(c) == 0)
    {
      std::string message = "Encountered '";
      message += c;
      message += "' when expecting a capital letter.";
      errLog.logPackageError("fbc", FbcSpeciesFormulaMustBeString, 
        packageVersion, level, version, message);

      // at this point we already *know* the formula is bad, at this point
      // the map generated woudl be invalid in any case, so we quit here.
      return;

    }
    else
    {
      elementName << c;
      index++;
    }

    // next char
    if (index < sizeStr) 
    {
      c = chemicalFormula[index];

      // is it a letter
      while (islower(c) != 0 && index < sizeStr)
      {
        elementName << c;
        index++;
        if (index < sizeStr)
        {
          c = chemicalFormula[index];
        }
        else
        {
          break;
        }
      }

      // is the next char a number
      while (isdigit(c) != 0 && index < sizeStr)
      {
        elementCount << c;
        index++;
        if (index < sizeStr)
        {
          c = chemicalFormula[index];
        }
        else
        {
          break;
        }
      }
    }

    const string& chemicalName = elementName.str();
    const string& chemicalNum = elementCount.str();

    // create the pair
    if (chemicalNum.empty() == true)
    {
      chemicalNumber = 1;
    }
    else
    {
      chemicalNumber = atoi(chemicalNum.c_str());
    }

    chemicalSymbols.push_back(make_pair(chemicalName, chemicalNumber));
  }

}
  /*
  * Constructor
  */
  FbcSpeciesPlugin::FbcSpeciesPlugin (const std::string &uri, 
  const std::string &prefix,
  FbcPkgNamespaces *fbcns)
  : SBasePlugin(uri,prefix, fbcns)
  , mCharge(0)
  , mIsSetCharge(false)
  , mChemicalFormula()
{
}


/*
* Copy constructor. Creates a copy of this FbcSpeciesPlugin object.
*/
FbcSpeciesPlugin::FbcSpeciesPlugin(const FbcSpeciesPlugin& orig)
  : SBasePlugin(orig)
  , mCharge(orig.mCharge)
  , mIsSetCharge(orig.mIsSetCharge)
  , mChemicalFormula(orig.mChemicalFormula)
{
}


/*
* Destroy this object.
*/
FbcSpeciesPlugin::~FbcSpeciesPlugin () {}

/*
* Assignment operator for FbcSpeciesPlugin.
*/
FbcSpeciesPlugin& 
  FbcSpeciesPlugin::operator=(const FbcSpeciesPlugin& orig)
{
  if(&orig!=this)
  {
    this->SBasePlugin::operator =(orig);
    mChemicalFormula = orig.mChemicalFormula;
    mCharge = orig.mCharge;
    mIsSetCharge = orig.mIsSetCharge;
  }    
  return *this;
}


/*
* Creates and returns a deep copy of this FbcSpeciesPlugin object.
* 
* @return a (deep) copy of this FbcSpeciesPlugin object
*/
FbcSpeciesPlugin* 
  FbcSpeciesPlugin::clone () const
{
  return new FbcSpeciesPlugin(*this);  
}


bool 
FbcSpeciesPlugin::isSetCharge() const
{
  return mIsSetCharge;
}

int 
FbcSpeciesPlugin::setCharge(int charge)
{
  mIsSetCharge = true;
  mCharge = charge;
  return LIBSBML_OPERATION_SUCCESS;
}

int FbcSpeciesPlugin::getCharge() const
{
  return mCharge;
}

int 
FbcSpeciesPlugin::unsetCharge()
{
  mIsSetCharge = false;
  return LIBSBML_OPERATION_SUCCESS;
}

bool 
FbcSpeciesPlugin::isSetChemicalFormula() const
{
  return !mChemicalFormula.empty();
}

int 
FbcSpeciesPlugin::setChemicalFormula(const std::string& chemicalFormula)
{
  mChemicalFormula = chemicalFormula;
  return LIBSBML_OPERATION_SUCCESS;
}



const std::string& 
FbcSpeciesPlugin::getChemicalFormula() const
{
  return mChemicalFormula;
}

int 
FbcSpeciesPlugin::unsetChemicalFormula()
{
  mChemicalFormula.erase();
  return LIBSBML_OPERATION_SUCCESS;
}

/** @cond doxygenLibsbmlInternal */
/*
* Sets the parent SBMLDocument of this SBML object.
*
* @param d the SBMLDocument object to use
*/
void 
  FbcSpeciesPlugin::setSBMLDocument (SBMLDocument* d)
{
  SBasePlugin::setSBMLDocument(d);
}
/** @endcond */


/** @cond doxygenLibsbmlInternal */
void
FbcSpeciesPlugin::addExpectedAttributes(ExpectedAttributes& attributes)
{
  //
  // required attribute is not defined for SBML Level 2 or lesser.
  //
  if ( mSBMLExt->getLevel(mURI) > 2 )
  {    
    attributes.add("charge");
    attributes.add("chemicalFormula");
  }
}
/** @endcond */

/** @cond doxygenLibsbmlInternal */
void 
FbcSpeciesPlugin::readAttributes (const XMLAttributes& attributes,
                                             const ExpectedAttributes& expectedAttributes)
{
  // dont call this as all it does it log unknown attributes
//  SBasePlugin::readAttributes(attributes, expectedAttributes);
  for (int i = 0; i < attributes.getLength(); i++)
  {
    std::string name = attributes.getName(i);
    std::string uri  = attributes.getURI(i);

    if (uri != mURI) continue;

    if (!expectedAttributes.hasAttribute(name))
    {  
      getErrorLog()->logPackageError("fbc", FbcSpeciesAllowedL3Attributes,
        getPackageVersion(), getLevel(), getVersion());
    }      
  }
  
  if ( mSBMLExt->getLevel(mURI) > 2 )
  {    
    XMLTriple tripleCharge("charge", mURI, mPrefix);
    unsigned int numErrs = getErrorLog()->getNumErrors();
    mIsSetCharge = attributes.readInto(tripleCharge, mCharge, getErrorLog(), 
      false, getLine(), getColumn());
    if (mIsSetCharge == false)
    {
    if (getErrorLog()->getNumErrors() == numErrs + 1 && 
        getErrorLog()->contains(XMLAttributeTypeMismatch))
    {
      getErrorLog()->remove(XMLAttributeTypeMismatch);
      getErrorLog()->logPackageError("fbc", FbcSpeciesChargeMustBeInteger,
        getPackageVersion(), getLevel(), getVersion());
    }
    }

    XMLTriple tripleChemicalFormula("chemicalFormula", mURI, mPrefix);
    bool assigned = attributes.readInto(tripleChemicalFormula, mChemicalFormula);
    if (assigned == true)
    {
      parseChemicalFormula(mChemicalFormula, *(getErrorLog()), getPackageVersion(),
        getLevel(), getVersion());
    }
    
  }
}
/** @endcond */

/** @cond doxygenLibsbmlInternal */
void 
FbcSpeciesPlugin::writeAttributes (XMLOutputStream& stream) const
{
  //
  // required attribute is not defined for SBML Level 2 .
  //
  if ( mSBMLExt->getLevel(mURI) < 3)
    return;
  
  //cout << "[DEBUG] SBMLDocumentPlugin::writeAttributes() " << endl;
  if ( isSetCharge() ) 
  {
    XMLTriple tripleCharge("charge", mURI, mPrefix);
    stream.writeAttribute(tripleCharge, mCharge);
  }
  if ( isSetChemicalFormula() )
  {
    XMLTriple tripleChemicalFormula("chemicalFormula", mURI, mPrefix);
    stream.writeAttribute(tripleChemicalFormula, mChemicalFormula);
  }
}
/** @endcond */



/** @cond doxygenLibsbmlInternal */
/*
* Sets the parent SBML object of this plugin object to
* this object and child elements (if any).
* (Creates a child-parent relationship by this plugin object)
*/
void
  FbcSpeciesPlugin::connectToParent (SBase* sbase)
{
  SBasePlugin::connectToParent(sbase);

}
/** @endcond */


/** @cond doxygenLibsbmlInternal */
/*
* Enables/Disables the given package with child elements in this plugin
* object (if any).
*/
void
  FbcSpeciesPlugin::enablePackageInternal(const std::string& pkgURI,
  const std::string& pkgPrefix, bool flag)
{
}
/** @endcond */

#endif /* __cplusplus */
/** @cond doxygenIgnored */

LIBSBML_EXTERN
int
FbcSpeciesPlugin_getCharge(SBasePlugin_t * fbc)
{
  return (fbc != NULL) ? static_cast<FbcSpeciesPlugin*>(fbc)->getCharge() 
    : SBML_INT_MAX;
}

LIBSBML_EXTERN
int
FbcSpeciesPlugin_isSetCharge(SBasePlugin_t * fbc)
{
  return (fbc != NULL) ? 
    static_cast<int>(static_cast<FbcSpeciesPlugin*>(fbc)->isSetCharge()) : 0;
}


LIBSBML_EXTERN
int
FbcSpeciesPlugin_setCharge(SBasePlugin_t * fbc, int charge)
{
  return (fbc != NULL) ? static_cast<FbcSpeciesPlugin*>(fbc)->setCharge(charge)
    : LIBSBML_INVALID_OBJECT;
}


LIBSBML_EXTERN
int
FbcSpeciesPlugin_unsetCharge(SBasePlugin_t * fbc)
{
  return (fbc != NULL) ? static_cast<FbcSpeciesPlugin*>(fbc)->unsetCharge()
    : LIBSBML_INVALID_OBJECT;
}


LIBSBML_EXTERN
char *
FbcSpeciesPlugin_getChemicalFormula(SBasePlugin_t * fbc)
{
  if (fbc == NULL) return NULL;

  return static_cast<FbcSpeciesPlugin*>(fbc)->getChemicalFormula().empty() 
    ? safe_strdup("")
    : safe_strdup(static_cast<FbcSpeciesPlugin*>(fbc)->getChemicalFormula().c_str());
}


LIBSBML_EXTERN
int
FbcSpeciesPlugin_isSetChemicalFormula(SBasePlugin_t * fbc)
{
  return (fbc != NULL) 
    ? static_cast<int>
             (static_cast<FbcSpeciesPlugin*>(fbc)->isSetChemicalFormula()) 
    : 0;
}


LIBSBML_EXTERN
int
FbcSpeciesPlugin_setChemicalFormula(SBasePlugin_t * fbc, const char * chemform)
{
  return (fbc != NULL) 
    ? static_cast<FbcSpeciesPlugin*>(fbc)->setChemicalFormula(chemform)
    : LIBSBML_INVALID_OBJECT;
}


LIBSBML_EXTERN
int
FbcSpeciesPlugin_unsetChemicalFormula(SBasePlugin_t * fbc)
{
  return (fbc != NULL) 
    ? static_cast<FbcSpeciesPlugin*>(fbc)->unsetChemicalFormula()
    : LIBSBML_INVALID_OBJECT;
}

/** @endcond */
LIBSBML_CPP_NAMESPACE_END


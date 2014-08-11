/**
 * @file    FluxObjective.cpp
 * @brief   Implementation of FluxObjective, the SBase derived class of the fbc package.
 * @author  Akiya Jouraku
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

#include <iostream>
#include <limits>

#include <sbml/SBMLVisitor.h>
#include <sbml/xml/XMLNode.h>
#include <sbml/xml/XMLToken.h>
#include <sbml/xml/XMLAttributes.h>
#include <sbml/xml/XMLInputStream.h>
#include <sbml/xml/XMLOutputStream.h>

#include <sbml/packages/fbc/sbml/FluxObjective.h>
#include <sbml/packages/fbc/extension/FbcExtension.h>
#include <sbml/packages/fbc/validator/FbcSBMLError.h>

#if defined(WIN32) && !defined(CYGWIN)
#define isnan _isnan
#endif


using namespace std;

LIBSBML_CPP_NAMESPACE_BEGIN
#ifdef __cplusplus

/*
 * Creates a new FluxObjective with the given level, version, and package version.
 */
FluxObjective::FluxObjective (unsigned int level, unsigned int version, unsigned int pkgVersion) 
  : SBase (level,version)
   ,mId("")
   ,mName("")
   ,mReaction("")
   ,mCoefficient(numeric_limits<double>::quiet_NaN())
{
  // set an SBMLNamespaces derived object (FbcPkgNamespaces) of this package.
  setSBMLNamespacesAndOwn(new FbcPkgNamespaces(level,version,pkgVersion));  
}


/*
 * Creates a new FluxObjective with the given FbcPkgNamespaces object.
 */
FluxObjective::FluxObjective(FbcPkgNamespaces* fbcns)
 : SBase(fbcns)
  ,mId("")
  ,mName("")
  ,mReaction("")
  ,mCoefficient(numeric_limits<double>::quiet_NaN())
{
  //
  // set the element namespace of this object
  //
  setElementNamespace(fbcns->getURI());

  // load package extensions bound with this object (if any) 
  loadPlugins(fbcns);
}


/*
 * Copy constructor.
 */
FluxObjective::FluxObjective(const FluxObjective& source) : SBase(source)
{
  this->mId = source.mId;
  this->mName = source.mName;
  this->mReaction=source.mReaction;
  this->mCoefficient=source.mCoefficient;
}

/*
 * Assignment operator.
 */
FluxObjective& FluxObjective::operator=(const FluxObjective& source)
{
  if(&source!=this)
  {
    this->SBase::operator=(source);
    this->mId = source.mId;
    this->mName = source.mName;
    this->mReaction= source.mReaction;
    this->mCoefficient= source.mCoefficient;
  }
  
  return *this;
}


/*
 * Destructor.
 */ 
FluxObjective::~FluxObjective ()
{
}


/*
 * Returns the value of the "id" attribute of this FluxObjective.
 */
const std::string&
FluxObjective::getId () const
{
  return mId;
}


/*
 * Predicate returning @c true or @c false depending on whether this
 * FluxObjective "id" attribute has been set.
 */
bool
FluxObjective::isSetId () const
{
  return (mId.empty() == false);
}

/*
 * Sets the value of the "id" attribute of this FluxObjective.
 */
int
FluxObjective::setId (const std::string& id)
{
  return SyntaxChecker::checkAndSetSId(id,mId);
}


/*
 * Unsets the value of the "id" attribute of this FluxObjective.
 */
int
FluxObjective::unsetId ()
{
  mId.erase();
  if (mId.empty())
  {
    return LIBSBML_OPERATION_SUCCESS;
  }
  else
  {
    return LIBSBML_OPERATION_FAILED;
  }
}



/*
 * Returns the value of the "name" attribute of this FluxObjective.
 */
const std::string&
FluxObjective::getName () const
{
  return mName;
}


/*
 * Predicate returning @c true or @c false depending on whether this
 * FluxObjective "name" attribute has been set.
 */
bool
FluxObjective::isSetName () const
{
  return (mName.empty() == false);
}

/*
 * Sets the value of the "name" attribute of this FluxObjective.
 */
int
FluxObjective::setName (const std::string& name)
{
  mName = name;
  return LIBSBML_OPERATION_SUCCESS;
}


/*
 * Unsets the value of the "name" attribute of this FluxObjective.
 */
int
FluxObjective::unsetName ()
{
  mName.erase();
  if (mName.empty())
  {
    return LIBSBML_OPERATION_SUCCESS;
  }
  else
  {
    return LIBSBML_OPERATION_FAILED;
  }
}



/*
  * Returns the value of the "reaction" attribute of this FluxObjective.
  */
const std::string& 
FluxObjective::getReaction () const
{
  return mReaction;
}


/*
  * Predicate returning @c true or @c false depending on whether this
  * FluxObjective's "reaction" attribute has been set.
  */
bool 
FluxObjective::isSetReaction () const
{
  return (mReaction.empty() == false);
}

/*
  * Sets the value of the "reaction" attribute of this FluxObjective.
  */
int 
FluxObjective::setReaction (const std::string& reaction)
{
  return SyntaxChecker::checkAndSetSId(reaction ,mReaction);
}


/*
  * Unsets the value of the "reaction" attribute of this FluxObjective.
  */
int 
FluxObjective::unsetReaction ()
{
  mReaction.erase();
  if (mReaction.empty())
  {
    return LIBSBML_OPERATION_SUCCESS;
  }
  else
  {
    return LIBSBML_OPERATION_FAILED;
  }
}


/*
  * Returns the value of the "coefficient" attribute of this FluxObjective.
  */
const double 
FluxObjective::getCoefficient () const
{
  return mCoefficient;
}


/*
  * Predicate returning @c true or @c false depending on whether this
  * FluxObjective's "coefficient" attribute has been set.
  */
bool 
FluxObjective::isSetCoefficient () const
{
  return (!isnan(mCoefficient));
}

/*
  * Sets the value of the "coefficient" attribute of this FluxObjective.
  */
int 
FluxObjective::setCoefficient (const double coefficient)
{
  mCoefficient = coefficient;
  return LIBSBML_OPERATION_SUCCESS;
}


/*
  * Unsets the value of the "coefficient" attribute of this FluxObjective.
  */
int 
FluxObjective::unsetCoefficient ()
{
  mCoefficient = numeric_limits<double>::quiet_NaN();
  return LIBSBML_OPERATION_SUCCESS;
}


/*
 * rename attributes that are SIdRefs or instances in math
 */
void
FluxObjective::renameSIdRefs(const std::string& oldid, const std::string& newid)
{
  if (isSetReaction() == true && mReaction == oldid)
  {
    setReaction(newid);
  }

}


/*
 * Returns the XML element name of
 * this SBML object.
 */
const std::string&
FluxObjective::getElementName () const
{
  static const std::string name = "fluxObjective";
  return name;
}


/** @cond doxygenLibsbmlInternal */
SBase*
FluxObjective::createObject (XMLInputStream& stream)
{
  return NULL;
}
/** @endcond */

/** @cond doxygenLibsbmlInternal */
void
FluxObjective::addExpectedAttributes(ExpectedAttributes& attributes)
{
  SBase::addExpectedAttributes(attributes);

  attributes.add("id");
  attributes.add("name");
  attributes.add("reaction");
  attributes.add("coefficient");
}
/** @endcond */

/** @cond doxygenLibsbmlInternal */
void
FluxObjective::readAttributes (const XMLAttributes& attributes,
                        const ExpectedAttributes& expectedAttributes)
{

  const unsigned int sbmlLevel   = getLevel  ();
  const unsigned int sbmlVersion = getVersion();
 
  // look to see whether an unknown attribute error was logged
  // during the read of the listOfFluxBounds - which will have
  // happened immediately prior to this read
  if (getErrorLog() != NULL && 
    static_cast<ListOfFluxObjectives*>(getParentSBMLObject())->size() < 2)
  {
    unsigned int numErrs = getErrorLog()->getNumErrors();
    for (int n = numErrs-1; n >= 0; n--)      
    {
      if (getErrorLog()->getError(n)->getErrorId() == UnknownPackageAttribute)
      {
        const std::string details = 
          getErrorLog()->getError(n)->getMessage();
        getErrorLog()->remove(UnknownPackageAttribute);
        getErrorLog()->logPackageError("fbc", FbcObjectiveLOFluxObjAllowedAttribs,
          getPackageVersion(), sbmlLevel, sbmlVersion, details);
      } 
      else if (getErrorLog()->getError(n)->getErrorId() == UnknownCoreAttribute)
      {
        const std::string details = 
          getErrorLog()->getError(n)->getMessage();
        getErrorLog()->remove(UnknownCoreAttribute);
        getErrorLog()->logPackageError("fbc", FbcObjectiveLOFluxObjAllowedAttribs,
          getPackageVersion(), sbmlLevel, sbmlVersion, details);
      } 
    }
  }

  SBase::readAttributes(attributes,expectedAttributes);

  // look to see whether an unknown attribute error was logged
  if (getErrorLog() != NULL)
  {
    unsigned int numErrs = getErrorLog()->getNumErrors();
    for (int n = numErrs-1; n >= 0; n--)      
    {
      if (getErrorLog()->getError(n)->getErrorId() == UnknownPackageAttribute)
      {
        const std::string details = 
          getErrorLog()->getError(n)->getMessage();
        getErrorLog()->remove(UnknownPackageAttribute);
        getErrorLog()->logPackageError("fbc", FbcFluxObjectRequiredAttributes,
          getPackageVersion(), sbmlLevel, sbmlVersion, details);
      } 
      else if (getErrorLog()->getError(n)->getErrorId() == UnknownCoreAttribute)
      {
        const std::string details = 
          getErrorLog()->getError(n)->getMessage();
        getErrorLog()->remove(UnknownCoreAttribute);
        getErrorLog()->logPackageError("fbc", FbcFluxObjectAllowedL3Attributes,
          getPackageVersion(), sbmlLevel, sbmlVersion, details);
      } 
    }
  }


  //
  // Reads an attribute "id" (optional)
  //
  bool assigned = attributes.readInto("id", mId);

  if (assigned)
  {
    // "id" attribute is set to this fbc element

    if (mId.empty())
    {
      //
      // Logs an error if the "id" attribute is empty.
      //
      logEmptyString(mId, sbmlLevel, sbmlVersion, "<fbc>");
    }
    else if (!SyntaxChecker::isValidSBMLSId(mId)) 
    {
      //
      // Logs an error if the "id" attribute doesn't
      // conform to the SBML type SId.
      //
      getErrorLog()->logPackageError("fbc", FbcSBMLSIdSyntax, 
        getPackageVersion(), sbmlLevel, sbmlVersion);
    }
  }

  attributes.readInto("name", mName);
  
  assigned = attributes.readInto("reaction", mReaction);
  if (assigned == false)
  {
    std::string message = "Fbc attribute 'reaction' is missing.";
    getErrorLog()->logPackageError("fbc", FbcFluxObjectRequiredAttributes, 
      getPackageVersion(), sbmlLevel, sbmlVersion, message);
  }
  else
  {
    if (mReaction.empty())
    {
      //
      // Logs an error if the "id" attribute is empty.
      //
      logEmptyString(mReaction, sbmlLevel, sbmlVersion, "<fbc>");
    }
    else if (!SyntaxChecker::isValidSBMLSId(mReaction)) 
    {
      //
      // Logs an error if the "id" attribute doesn't
      // conform to the SBML type SId.
      //
      getErrorLog()->logPackageError("fbc", FbcFluxObjectReactionMustBeSIdRef, 
        getPackageVersion(), sbmlLevel, sbmlVersion);
    }
  }


  unsigned int numErrs = getErrorLog()->getNumErrors();
  assigned = attributes.readInto("coefficient", mCoefficient, getErrorLog());
  
  if (assigned == false)
  {
    if (getErrorLog()->getNumErrors() == numErrs + 1 && 
        getErrorLog()->contains(XMLAttributeTypeMismatch))
    {
      getErrorLog()->remove(XMLAttributeTypeMismatch);
      getErrorLog()->logPackageError("fbc", FbcFluxObjectCoefficientMustBeDouble,
        getPackageVersion(), sbmlLevel, sbmlVersion);
    }
    else
    {
      std::string message = "Fbc attribute 'coefficient' is missing.";
      getErrorLog()->logPackageError("fbc", FbcFluxObjectRequiredAttributes, 
        getPackageVersion(), sbmlLevel, sbmlVersion, message);
    }
  }
}
/** @endcond */

/** @cond doxygenLibsbmlInternal */
void
FluxObjective::writeAttributes (XMLOutputStream& stream) const
{
  SBase::writeAttributes(stream);

  if (isSetId())
    stream.writeAttribute("id",   getPrefix(), mId);
  if (isSetName())
    stream.writeAttribute("name",   getPrefix(), mName);
  
  stream.writeAttribute("reaction",   getPrefix(), mReaction);
  if (isSetCoefficient())
    stream.writeAttribute("coefficient",   getPrefix(), mCoefficient);

  //
  // (EXTENSION)
  //
  SBase::writeExtensionAttributes(stream);
}
/** @endcond */

/** @cond doxygenLibsbmlInternal */
void
FluxObjective::writeElements (XMLOutputStream& stream) const
{
  SBase::writeElements(stream);

  //
  // (EXTENSION)
  //
  SBase::writeExtensionElements(stream);
}
/** @endcond */


/*
 * @return the typecode (int) of this SBML object or SBML_UNKNOWN
 * (default).
 *
 * @see getElementName()
 */
int
FluxObjective::getTypeCode () const
{
  return SBML_FBC_FLUXOBJECTIVE;
}

FluxObjective*
FluxObjective::clone() const
{
    return new FluxObjective(*this);
}


/*
 * Accepts the given SBMLVisitor.
 */
bool
FluxObjective::accept (SBMLVisitor& v) const
{
  return v.visit(*this);
}


/*
 * Ctor.
 */
ListOfFluxObjectives::ListOfFluxObjectives(FbcPkgNamespaces* fbcns)
 : ListOf(fbcns)
{
  //
  // set the element namespace of this object
  //
  setElementNamespace(fbcns->getURI());
}


/*
 * Ctor.
 */
ListOfFluxObjectives::ListOfFluxObjectives(unsigned int level, unsigned int version, unsigned int pkgVersion)
 : ListOf(level,version)
{
  setSBMLNamespacesAndOwn(new FbcPkgNamespaces(level,version,pkgVersion));
};


/*
 * @return a (deep) copy of this ListOfFluxObjectives.
 */
ListOfFluxObjectives*
ListOfFluxObjectives::clone () const
{
  return new ListOfFluxObjectives(*this);
}


/* return nth item in list */
FluxObjective *
ListOfFluxObjectives::get(unsigned int n)
{
  return static_cast<FluxObjective*>(ListOf::get(n));
}


/* return nth item in list */
const FluxObjective *
ListOfFluxObjectives::get(unsigned int n) const
{
  return static_cast<const FluxObjective*>(ListOf::get(n));
}


/* return item by symbol */
FluxObjective*
ListOfFluxObjectives::get (const std::string& symbol)
{
  return const_cast<FluxObjective*>( 
    static_cast<const ListOfFluxObjectives&>(*this).get(symbol) );
}


/* return item by symbol */
const FluxObjective*
ListOfFluxObjectives::get (const std::string& symbol) const
{
  vector<SBase*>::const_iterator result;

  result = find_if( mItems.begin(), mItems.end(), IdEq<FluxObjective>(symbol) );
  return (result == mItems.end()) ? 0 : static_cast <FluxObjective*> (*result);
}


/* Removes the nth item from this list */
FluxObjective*
ListOfFluxObjectives::remove (unsigned int n)
{
   return static_cast<FluxObjective*>(ListOf::remove(n));
}


/* Removes item in this list by symbol */
FluxObjective*
ListOfFluxObjectives::remove (const std::string& symbol)
{
  SBase* item = 0;
  vector<SBase*>::iterator result;

  result = find_if( mItems.begin(), mItems.end(), IdEq<FluxObjective>(symbol) );

  if (result != mItems.end())
  {
    item = *result;
    mItems.erase(result);
  }

  return static_cast <FluxObjective*> (item);
}


/*
 * @return the typecode (int) of SBML objects contained in this ListOf or
 * SBML_UNKNOWN (default).
 */
int
ListOfFluxObjectives::getItemTypeCode () const
{
  return SBML_FBC_FLUXOBJECTIVE;
}


/*
 * Returns the XML element name of
 * this SBML object.
 */
const std::string&
ListOfFluxObjectives::getElementName () const
{
  static const std::string name = "listOfFluxObjectives";
  return name;
}


/** @cond doxygenLibsbmlInternal */
SBase*
ListOfFluxObjectives::createObject (XMLInputStream& stream)
{
  const std::string& name   = stream.peek().getName();
  SBase*        object = NULL;
  
  if (name == "fluxObjective")
  {
    try
    {
      FBC_CREATE_NS(fbcns, getSBMLNamespaces());
      object = new FluxObjective(fbcns);
      appendAndOwn(object);
      delete fbcns;
      //mItems.push_back(object);
    } 
    catch(...)
    {
      /* 
      * NULL will be returned if the mSBMLNS is invalid (basically this
      * should not happen) or some exception is thrown (e.g. std::bad_alloc)
      *
      * (Maybe this should be changed so that caller can detect what kind 
      *  of error happened in this function.)
      */
    }
  }

  return object;
}
/** @endcond */


#endif /* __cplusplus */
/** @cond doxygenIgnored */

LIBSBML_EXTERN
FluxObjective_t *
FluxObjective_create(unsigned int level, unsigned int version, unsigned int pkgversion)
{
  return new FluxObjective(level, version, pkgversion);
}


LIBSBML_EXTERN
const char *
FluxObjective_getId(FluxObjective_t * fb)
{
  if (fb == NULL)
    return NULL;
  
  return fb->getId().empty() ? "" : safe_strdup(fb->getId().c_str());
}


LIBSBML_EXTERN
int
FluxObjective_isSetId(FluxObjective_t * fb)
{
  return (fb != NULL) ? static_cast<int>(fb->isSetId()) : 0;
}


LIBSBML_EXTERN
int
FluxObjective_setId(FluxObjective_t * fb, const char * id)
{
  return (fb != NULL) ? fb->setId(id) : LIBSBML_INVALID_OBJECT;
}


LIBSBML_EXTERN
int
FluxObjective_unsetId(FluxObjective_t * fb)
{
  return (fb != NULL) ? fb->unsetId() : LIBSBML_INVALID_OBJECT;
}

LIBSBML_EXTERN
const char *
FluxObjective_getName(FluxObjective_t * fb)
{
  if (fb == NULL) return NULL;
  return fb->getName().c_str();
}


LIBSBML_EXTERN
int
FluxObjective_isSetName(FluxObjective_t * fb)
{
  if (fb == NULL) return 0;
  return fb->isSetName();
}


LIBSBML_EXTERN
int
FluxObjective_setName(FluxObjective_t * fb, const char * name)
{
  if (fb!=NULL) return fb->setName(name);
  return LIBSBML_INVALID_OBJECT;
}


LIBSBML_EXTERN
int
FluxObjective_unsetName(FluxObjective_t * fb)
{
  if (fb == NULL) return LIBSBML_INVALID_OBJECT;
  return fb->unsetName();
}




LIBSBML_EXTERN
const char *
FluxObjective_getReaction(FluxObjective_t * flux)
{
  if (flux == NULL)
    return NULL;

  return flux->getReaction().empty() ? "" : safe_strdup(flux->getReaction().c_str());
}


LIBSBML_EXTERN
int
FluxObjective_isSetReaction(FluxObjective_t * flux)
{
  return (flux != NULL) ? static_cast<int>(flux->isSetReaction()) : 0;
}


LIBSBML_EXTERN
int
FluxObjective_setReaction(FluxObjective_t * flux, const char * reaction)
{
  return (flux != NULL) ? flux->setReaction(reaction) : LIBSBML_INVALID_OBJECT;
}


LIBSBML_EXTERN
int
FluxObjective_unsetReaction(FluxObjective_t * flux)
{
  return (flux != NULL) ? flux->unsetReaction() : LIBSBML_INVALID_OBJECT;
}


LIBSBML_EXTERN
double
FluxObjective_getCoefficient(FluxObjective_t * flux)
{
  return (flux != NULL) ? flux->getCoefficient() : numeric_limits<double>::quiet_NaN();
}


LIBSBML_EXTERN
int
FluxObjective_isSetCoefficient(FluxObjective_t * flux)
{
  return (flux != NULL) ? static_cast<int>(flux->isSetCoefficient()) : 0;
}


LIBSBML_EXTERN
int
FluxObjective_setCoefficient(FluxObjective_t * flux, double coeff)
{
  return (flux != NULL) ? flux->setCoefficient(coeff) : LIBSBML_INVALID_OBJECT;
}


LIBSBML_EXTERN
int
FluxObjective_unsetCoefficient(FluxObjective_t * flux)
{
  return (flux != NULL) ? flux->unsetCoefficient() : LIBSBML_INVALID_OBJECT;
}


/** @endcond */
LIBSBML_CPP_NAMESPACE_END


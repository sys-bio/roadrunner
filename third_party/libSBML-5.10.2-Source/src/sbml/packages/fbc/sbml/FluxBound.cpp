/**
 * @file    FluxBound.cpp
 * @brief   Implementation of FluxBound, the SBase derived class of the fbc package.
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

#include <sbml/SBMLVisitor.h>
#include <sbml/xml/XMLNode.h>
#include <sbml/xml/XMLToken.h>
#include <sbml/xml/XMLAttributes.h>
#include <sbml/xml/XMLInputStream.h>
#include <sbml/xml/XMLOutputStream.h>

#include <sbml/packages/fbc/sbml/FluxBound.h>
#include <sbml/packages/fbc/extension//FbcExtension.h>
#include <sbml/packages/fbc/validator/FbcSBMLError.h>

#if defined(WIN32) && !defined(CYGWIN)
#define isnan _isnan
#endif


using namespace std;

LIBSBML_CPP_NAMESPACE_BEGIN
#ifdef __cplusplus

/*
 * Creates a new FluxBound with the given level, version, and package version.
 */
FluxBound::FluxBound (unsigned int level, unsigned int version, unsigned int pkgVersion) 
  : SBase (level,version)
   ,mId("")
   ,mName("")
   ,mReaction("")
   ,mOperation (FLUXBOUND_OPERATION_UNKNOWN)
   ,mOperationString("")
   ,mValue(numeric_limits<double>::quiet_NaN())
{
  // set an SBMLNamespaces derived object (FluxBoundsPkgNamespaces) of this package.
  setSBMLNamespacesAndOwn(new FbcPkgNamespaces(level,version,pkgVersion));  

  // connect child elements to this element.
  connectToChild();
}


/*
 * Creates a new FluxBound with the given FluxBoundsPkgNamespaces object.
 */
FluxBound::FluxBound(FbcPkgNamespaces* fbcns)
 : SBase(fbcns)
   ,mId("")
   ,mName("")
   ,mReaction("")
   ,mOperation (FLUXBOUND_OPERATION_UNKNOWN)
   ,mOperationString("")
   ,mValue(numeric_limits<double>::quiet_NaN())
{
  // set the element namespace of this object
  setElementNamespace(fbcns->getURI());

  // connect child elements to this element.
  connectToChild();

  // load package extensions bound with this object (if any) 
  loadPlugins(fbcns);
}


/*
 * Copy constructor.
 */
FluxBound::FluxBound(const FluxBound& source) : SBase(source)
{
  this->mId=source.mId;
  this->mName=source.mName;
  this->mReaction=source.mReaction;
  this->mOperation=source.mOperation;
  this->mOperationString=source.mOperationString;
  this->mValue=source.mValue;

  // connect child elements to this element.
  connectToChild();
}

/*
 * Assignment operator.
 */
FluxBound& FluxBound::operator=(const FluxBound& source)
{
  if(&source!=this)
  {
    this->SBase::operator=(source);
    this->mId = source.mId;
    this->mName = source.mName;
    this->mReaction=source.mReaction;
    this->mOperation=source.mOperation;
    this->mOperationString=source.mOperationString;
    this->mValue=source.mValue;

    // connect child elements to this element.
    connectToChild();
  }

  return *this;
}


/*
 * Destructor.
 */ 
FluxBound::~FluxBound ()
{
}


/*
  * Returns the value of the "id" attribute of this FluxBound.
  */
const std::string& 
FluxBound::getId () const
{
  return mId;
}


/*
  * Predicate returning @c true or @c false depending on whether this
  * FluxBound's "id" attribute has been set.
  */
bool 
FluxBound::isSetId () const
{
  return (mId.empty() == false);
}

/*
  * Sets the value of the "id" attribute of this FluxBound.
  */
int 
FluxBound::setId (const std::string& id)
{
  return SyntaxChecker::checkAndSetSId(id,mId);
}


/*
  * Unsets the value of the "id" attribute of this FluxBound.
  */
int 
FluxBound::unsetId ()
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
 * Returns the value of the "name" attribute of this FluxBound.
 */
const std::string&
FluxBound::getName () const
{
  return mName;
}


/*
 * Predicate returning @c true or @c false depending on whether this
 * FluxBound's "name" attribute has been set.
 */
bool
FluxBound::isSetName () const
{
  return (mName.empty() == false);
}

/*
 * Sets the value of the "name" attribute of this FluxBound.
 */
int
FluxBound::setName (const std::string& name)
{
  mName = name;
  return LIBSBML_OPERATION_SUCCESS;
}


/*
 * Unsets the value of the "name" attribute of this FluxBound.
 */
int
FluxBound::unsetName ()
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
 * Sets the reaction of this SBML object to a copy of reaction.
 */
int
FluxBound::setReaction (const std::string& reaction)
{
  mReaction = reaction;
  return LIBSBML_OPERATION_SUCCESS;
}


/*
 * @return the reaction of this SBML object.
 */
const string&
FluxBound::getReaction () const
{
  return mReaction;
}


/*
 * @return true if the reaction of this SBML object has been set, false
 * otherwise.
 */
bool
FluxBound::isSetReaction () const
{
  return (mReaction.empty() == false);
}


/*
 * Unsets the reaction of this SBML object.
 */
int
FluxBound::unsetReaction ()
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
 * Sets the operation of this SBML object to a copy of operation.
 */
int
FluxBound::setOperation (FluxBoundOperation_t operation)
{
  if (FluxBoundOperation_isValidFluxBoundOperation(operation) == 0)
  {
    mOperation = FLUXBOUND_OPERATION_UNKNOWN;
    return LIBSBML_INVALID_ATTRIBUTE_VALUE;
  }
  else
  {
    mOperation = operation;
    return LIBSBML_OPERATION_SUCCESS;
  }
}


/*
 * Sets the operation of this SBML object to a copy of operation.
 */
int
FluxBound::setOperation (const std::string& operation)
{
  return setOperation(FluxBoundOperation_fromString(operation.c_str()));
}


/*
 * @return the operation of this SBML object.
 */
const string&
FluxBound::getOperation ()
{
  if (FluxBoundOperation_toString(mOperation) != NULL)
  {
    mOperationString.assign(FluxBoundOperation_toString(mOperation));
  }
  else
  {
    mOperationString.assign("");
  }
  return mOperationString;
}


/*
 * @return the operation of this SBML object.
 */
FluxBoundOperation_t
FluxBound::getFluxBoundOperation () const
{
  return mOperation;
}


/*
 * @return true if the operation of this SBML object has been set, false
 * otherwise.
 */
bool
FluxBound::isSetOperation () const
{
  return (mOperation != FLUXBOUND_OPERATION_UNKNOWN);
}


/*
 * Unsets the operation of this SBML object.
 */
int
FluxBound::unsetOperation ()
{
  mOperation = FLUXBOUND_OPERATION_UNKNOWN;
  return LIBSBML_OPERATION_SUCCESS;
}


/*
 * Sets the value of this SBML object to a copy of value.
 */
int
FluxBound::setValue (const double value)
{
  mValue = value;
  return LIBSBML_OPERATION_SUCCESS;
}


/*
 * @return the value of this SBML object.
 */
const double
FluxBound::getValue () const
{
  return mValue;
}


/*
 * @return true if the value of this SBML object has been set, false
 * otherwise.
 */
bool
FluxBound::isSetValue () const
{
  return (!isnan(mValue));
}


/*
 * Unsets the value of this SBML object.
 */
int
FluxBound::unsetValue ()
{
  mValue = numeric_limits<double>::quiet_NaN();
  return LIBSBML_OPERATION_SUCCESS;
}


/*
 * rename attributes that are SIdRefs or instances in math
 */
void
FluxBound::renameSIdRefs(const std::string& oldid, const std::string& newid)
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
FluxBound::getElementName () const
{
  static const std::string name = "fluxBound";
  return name;
}


/** @cond doxygenLibsbmlInternal */
SBase*
FluxBound::createObject (XMLInputStream& stream)
{
  return NULL;
}
/** @endcond */

/** @cond doxygenLibsbmlInternal */
void
FluxBound::addExpectedAttributes(ExpectedAttributes& attributes)
{
  SBase::addExpectedAttributes(attributes);

  attributes.add("id");
  attributes.add("name");
  attributes.add("reaction");
  attributes.add("operation");
  attributes.add("value");
}
/** @endcond */

/** @cond doxygenLibsbmlInternal */
void
FluxBound::readAttributes (const XMLAttributes& attributes,
                        const ExpectedAttributes& expectedAttributes)
{

  const unsigned int sbmlLevel   = getLevel  ();
  const unsigned int sbmlVersion = getVersion();
 
  // look to see whether an unknown attribute error was logged
  // during the read of the listOfFluxBounds - which will have
  // happened immediately prior to this read
  if (getErrorLog() != NULL && 
    static_cast<ListOfFluxBounds*>(getParentSBMLObject())->size() < 2)
  {
    unsigned int numErrs = getErrorLog()->getNumErrors();
    for (int n = numErrs-1; n >= 0; n--)      
    {
      if (getErrorLog()->getError(n)->getErrorId() == UnknownPackageAttribute)
      {
        const std::string details = 
          getErrorLog()->getError(n)->getMessage();
        getErrorLog()->remove(UnknownPackageAttribute);
        getErrorLog()->logPackageError("fbc", FbcLOFluxBoundsAllowedAttributes,
          getPackageVersion(), sbmlLevel, sbmlVersion, details);
      } 
      else if (getErrorLog()->getError(n)->getErrorId() == UnknownCoreAttribute)
      {
        const std::string details = 
          getErrorLog()->getError(n)->getMessage();
        getErrorLog()->remove(UnknownCoreAttribute);
        getErrorLog()->logPackageError("fbc", FbcLOFluxBoundsAllowedAttributes,
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
        getErrorLog()->logPackageError("fbc", FbcFluxBoundRequiredAttributes,
          getPackageVersion(), sbmlLevel, sbmlVersion, details);
      } 
      else if (getErrorLog()->getError(n)->getErrorId() == UnknownCoreAttribute)
      {
        const std::string details = 
          getErrorLog()->getError(n)->getMessage();
        getErrorLog()->remove(UnknownCoreAttribute);
        getErrorLog()->logPackageError("fbc", FbcFluxBoundAllowedL3Attributes,
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
    getErrorLog()->logPackageError("fbc", FbcFluxBoundRequiredAttributes, 
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
      getErrorLog()->logPackageError("fbc", FbcFluxBoundRectionMustBeSIdRef, 
        getPackageVersion(), sbmlLevel, sbmlVersion);
    }
  }

  //
  // type string   ( use = "required" )
  //
  std::string operation;
  assigned = attributes.readInto("operation", operation);

  if (assigned == true)
  {
    // check string is not empty

    if (operation.empty() == true)
    {
      logEmptyString(operation, sbmlLevel, sbmlVersion, "<Objective>");
    }
    else 
    {
       mOperation = FluxBoundOperation_fromString( operation.c_str() );
       if (FluxBoundOperation_isValidFluxBoundOperation(mOperation) == 0)
       {
          getErrorLog()->logPackageError("fbc", FbcFluxBoundOperationMustBeEnum, 
            getPackageVersion(), sbmlLevel, sbmlVersion);
       }
    }
  }
  else
  {
    std::string message = "Fbc attribute 'operation' is missing.";
    getErrorLog()->logPackageError("fbc", FbcFluxBoundRequiredAttributes, 
      getPackageVersion(), sbmlLevel, sbmlVersion, message);
  }

  unsigned int numErrs = getErrorLog()->getNumErrors();
  assigned = attributes.readInto("value", mValue, getErrorLog());
  
  if (assigned == false)
  {
    if (getErrorLog()->getNumErrors() == numErrs + 1 && 
        getErrorLog()->contains(XMLAttributeTypeMismatch))
    {
      getErrorLog()->remove(XMLAttributeTypeMismatch);
      getErrorLog()->logPackageError("fbc", FbcFluxBoundValueMustBeDouble,
        getPackageVersion(), sbmlLevel, sbmlVersion);
    }
    else
    {
      std::string message = "Fbc attribute 'value' is missing.";
      getErrorLog()->logPackageError("fbc", FbcFluxBoundRequiredAttributes, 
        getPackageVersion(), sbmlLevel, sbmlVersion, message);
    }
  }
}
/** @endcond */

/** @cond doxygenLibsbmlInternal */
void
FluxBound::writeAttributes (XMLOutputStream& stream) const
{
  SBase::writeAttributes(stream);

  if (isSetId())
    stream.writeAttribute("id",   getPrefix(), mId);
  if (isSetName())
    stream.writeAttribute("name",   getPrefix(), mName);
  if (isSetReaction())
    stream.writeAttribute("reaction", getPrefix(), mReaction);
  if (isSetOperation())
    stream.writeAttribute("operation", getPrefix(), 
                     FluxBoundOperation_toString(mOperation));
  if (isSetValue())
    stream.writeAttribute("value", getPrefix(), mValue);

  //
  // (EXTENSION)
  //
  SBase::writeExtensionAttributes(stream);
}
/** @endcond */

/** @cond doxygenLibsbmlInternal */
void
FluxBound::writeElements (XMLOutputStream& stream) const
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
FluxBound::getTypeCode () const
{
  return SBML_FBC_FLUXBOUND;
}

FluxBound*
FluxBound::clone() const
{
    return new FluxBound(*this);
}


/*
 * Accepts the given SBMLVisitor.
 */
bool
FluxBound::accept (SBMLVisitor& v) const
{
  bool visited = false;
  visited = v.visit(*this);
  return  visited;
}


/** @cond doxygenLibsbmlInternal */
/*
 * Sets the parent SBMLDocument of this SBML object.
 */
void
FluxBound::setSBMLDocument (SBMLDocument* d)
{
  SBase::setSBMLDocument(d);
}
/** @endcond */


/** @cond doxygenLibsbmlInternal */
/*
 * Enables/Disables the given package with this element and child
 * elements (if any).
 * (This is an internal implementation for enablePakcage function)
 */
void
FluxBound::enablePackageInternal(const std::string& pkgURI,
                             const std::string& pkgPrefix, bool flag)
{
  SBase::enablePackageInternal(pkgURI,pkgPrefix,flag);

}
/** @endcond */


/** @cond doxygenLibsbmlInternal */
/* default for components that have no required elements */
bool
FluxBound::hasRequiredElements() const
{
  bool allPresent = true;

    
  return allPresent;
}
/** @endcond */


/*
 * Ctor.
 */
ListOfFluxBounds::ListOfFluxBounds(FbcPkgNamespaces* fbcns)
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
ListOfFluxBounds::ListOfFluxBounds(unsigned int level, unsigned int version, unsigned int pkgVersion)
 : ListOf(level,version)
{
  setSBMLNamespacesAndOwn(new FbcPkgNamespaces(level,version,pkgVersion));
};


/*
 * @return a (deep) copy of this ListOfFluxBounds.
 */
ListOfFluxBounds*
ListOfFluxBounds::clone () const
{
  return new ListOfFluxBounds(*this);
}


/* return nth item in list */
FluxBound *
ListOfFluxBounds::get(unsigned int n)
{
  return static_cast<FluxBound*>(ListOf::get(n));
}


/* return nth item in list */
const FluxBound *
ListOfFluxBounds::get(unsigned int n) const
{
  return static_cast<const FluxBound*>(ListOf::get(n));
}


/* return item by id */
FluxBound*
ListOfFluxBounds::get (const std::string& sid)
{
  return const_cast<FluxBound*>( 
    static_cast<const ListOfFluxBounds&>(*this).get(sid) );
}


/* return item by id */
const FluxBound*
ListOfFluxBounds::get (const std::string& sid) const
{
  vector<SBase*>::const_iterator result;

  result = find_if( mItems.begin(), mItems.end(), IdEq<FluxBound>(sid) );
  return (result == mItems.end()) ? 0 : static_cast <FluxBound*> (*result);
}


/* Removes the nth item from this list */
FluxBound*
ListOfFluxBounds::remove (unsigned int n)
{
   return static_cast<FluxBound*>(ListOf::remove(n));
}


/* Removes item in this list by id */
FluxBound*
ListOfFluxBounds::remove (const std::string& sid)
{
  SBase* item = 0;
  vector<SBase*>::iterator result;

  result = find_if( mItems.begin(), mItems.end(), IdEq<FluxBound>(sid) );

  if (result != mItems.end())
  {
    item = *result;
    mItems.erase(result);
  }

  return static_cast <FluxBound*> (item);
}


/*
 * @return the typecode (int) of SBML objects contained in this ListOf or
 * SBML_UNKNOWN (default).
 */
int
ListOfFluxBounds::getItemTypeCode () const
{
  return SBML_FBC_FLUXBOUND;
}


/*
 * Returns the XML element name of
 * this SBML object.
 */
const std::string&
ListOfFluxBounds::getElementName () const
{
  static const std::string name = "listOfFluxBounds";
  return name;
}


/** @cond doxygenLibsbmlInternal */
SBase*
ListOfFluxBounds::createObject (XMLInputStream& stream)
{
  const std::string& name   = stream.peek().getName();
  SBase*        object = 0;

  
  if (name == "fluxBound")
  {
    try
    {
      FBC_CREATE_NS(fbcns, getSBMLNamespaces());
      object = new FluxBound(fbcns);
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

/** @cond doxygenLibsbmlInternal */
void 
ListOfFluxBounds::writeXMLNS (XMLOutputStream& stream) const
{
  XMLNamespaces xmlns;

  std::string prefix = getPrefix();

  if (prefix.empty())
  {
    XMLNamespaces* thisxmlns = getNamespaces();
    if (thisxmlns && thisxmlns->hasURI(FbcExtension::getXmlnsL3V1V1()))
    {
      xmlns.add(FbcExtension::getXmlnsL3V1V1(),prefix);
    }
  }

  stream << xmlns;
}
/** @endcond */



#endif /* __cplusplus */
/** @cond doxygenIgnored */

LIBSBML_EXTERN
FluxBound_t *
FluxBound_create(unsigned int level, unsigned int version, unsigned int pkgversion)
{
  return new FluxBound(level, version, pkgversion);
}


LIBSBML_EXTERN
const char *
FluxBound_getId(FluxBound_t * fb)
{
  if (fb == NULL)
    return NULL;

  return fb->getId().empty() ? "" : safe_strdup(fb->getId().c_str());
}


LIBSBML_EXTERN
int
FluxBound_isSetId(FluxBound_t * fb)
{
  return (fb != NULL) ? static_cast<int>(fb->isSetId()) : 0;
}


LIBSBML_EXTERN
int
FluxBound_setId(FluxBound_t * fb, const char * id)
{
  return (fb != NULL) ? fb->setId(id) : LIBSBML_INVALID_OBJECT;
}


LIBSBML_EXTERN
int
FluxBound_unsetId(FluxBound_t * fb)
{
  return (fb != NULL) ? fb->unsetId() : LIBSBML_INVALID_OBJECT;
}

LIBSBML_EXTERN
const char *
FluxBound_getName(FluxBound_t * fb)
{
  if (fb == NULL) return NULL;
  return fb->getName().c_str();
}


LIBSBML_EXTERN
int
FluxBound_isSetName(FluxBound_t * fb)
{
  if (fb == NULL) return 0;
  return fb->isSetName();
}


LIBSBML_EXTERN
int
FluxBound_setName(FluxBound_t * fb, const char * name)
{
  if (fb!=NULL) return fb->setName(name);
  return LIBSBML_INVALID_OBJECT;
}


LIBSBML_EXTERN
int
FluxBound_unsetName(FluxBound_t * fb)
{
  if (fb == NULL) return LIBSBML_INVALID_OBJECT;
  return fb->unsetName();  
}


LIBSBML_EXTERN
const char *
FluxBound_getReaction(FluxBound_t * fb)
{
  if (fb == NULL)
    return NULL;

  return fb->getReaction().empty() ? "" : safe_strdup(fb->getReaction().c_str());
}


LIBSBML_EXTERN
int
FluxBound_isSetReaction(FluxBound_t * fb)
{
  return (fb != NULL) ? static_cast<int>(fb->isSetReaction()) : 0;
}


LIBSBML_EXTERN
int
FluxBound_setReaction(FluxBound_t * fb, const char * reaction)
{
  return (fb != NULL) ? fb->setReaction(reaction) : LIBSBML_INVALID_OBJECT;
}


LIBSBML_EXTERN
int
FluxBound_unsetReaction(FluxBound_t * fb)
{
  return (fb != NULL) ? fb->unsetReaction() : LIBSBML_INVALID_OBJECT;
}


LIBSBML_EXTERN
const char *
FluxBound_getOperation(FluxBound_t * fb)
{
  if (fb == NULL)
    return NULL;

  return fb->getOperation().empty() ? "" : safe_strdup(fb->getOperation().c_str());
}


LIBSBML_EXTERN
int
FluxBound_isSetOperation(FluxBound_t * fb)
{
  return (fb != NULL) ? static_cast<int>(fb->isSetOperation()) : 0;
}


LIBSBML_EXTERN
int
FluxBound_setOperation(FluxBound_t * fb, const char * operation)
{
  return (fb != NULL) ? fb->setOperation(operation) : LIBSBML_INVALID_OBJECT;
}


LIBSBML_EXTERN
int
FluxBound_unsetOperation(FluxBound_t * fb)
{
  return (fb != NULL) ? fb->unsetOperation() : LIBSBML_INVALID_OBJECT;
}


LIBSBML_EXTERN
double
FluxBound_getValue(FluxBound_t * fb)
{
  return (fb != NULL) ? fb->getValue() : numeric_limits<double>::quiet_NaN();
}


LIBSBML_EXTERN
int
FluxBound_isSetValue(FluxBound_t * fb)
{
  return (fb != NULL) ? static_cast<int>(fb->isSetValue()) : 0;
}


LIBSBML_EXTERN
int
FluxBound_setValue(FluxBound_t * fb, double value)
{
  return (fb != NULL) ? fb->setValue(value) : LIBSBML_INVALID_OBJECT;
}


LIBSBML_EXTERN
int
FluxBound_unsetValue(FluxBound_t * fb)
{
  return (fb != NULL) ? fb->unsetValue() : LIBSBML_INVALID_OBJECT;
}


static
const char* FLUXBOUND_OPERATION_STRINGS[] =
{
    "lessEqual"
  , "greaterEqual"
  , "less"
  , "greater"
  , "equal"
  , "unknown"
};


LIBSBML_EXTERN
const char* 
FluxBoundOperation_toString(FluxBoundOperation_t type)
{
  int max = FLUXBOUND_OPERATION_UNKNOWN;

  if (type < FLUXBOUND_OPERATION_LESS_EQUAL || type >= max)
  {
      return NULL;
  }
  
  if (type == FLUXBOUND_OPERATION_LESS)
    return FLUXBOUND_OPERATION_STRINGS[FLUXBOUND_OPERATION_LESS_EQUAL];
  if (type == FLUXBOUND_OPERATION_GREATER)
    return FLUXBOUND_OPERATION_STRINGS[FLUXBOUND_OPERATION_GREATER_EQUAL];

  return FLUXBOUND_OPERATION_STRINGS[type];
}


LIBSBML_EXTERN
FluxBoundOperation_t 
FluxBoundOperation_fromString(const char* s)
{
  if (s == NULL) 
  {
    return FLUXBOUND_OPERATION_UNKNOWN;
  }

  int max = FLUXBOUND_OPERATION_UNKNOWN;
  for (int i = 0; i < max; i++)
  {
    if (strcmp(FLUXBOUND_OPERATION_STRINGS[i], s) == 0)
    {
      FluxBoundOperation_t current = (FluxBoundOperation_t)i;
      if (current == FLUXBOUND_OPERATION_GREATER)
        return FLUXBOUND_OPERATION_GREATER_EQUAL;
      if (current == FLUXBOUND_OPERATION_LESS)
        return FLUXBOUND_OPERATION_LESS_EQUAL;
      return current;
    }
  }
  return FLUXBOUND_OPERATION_UNKNOWN;
}


LIBSBML_EXTERN
int 
FluxBoundOperation_isValidFluxBoundOperation(FluxBoundOperation_t effect)
{
  int max = FLUXBOUND_OPERATION_UNKNOWN;

  if (effect < FLUXBOUND_OPERATION_LESS_EQUAL || effect >= max)
  {
    return 0;
  }
  else
  {
    return 1;
  }
}


LIBSBML_EXTERN
int 
FluxBoundOperation_isValidFluxBoundOperationString(const char* s)
{
  return FluxBoundOperation_isValidFluxBoundOperation
                                         (FluxBoundOperation_fromString(s));
}




/** @endcond */
LIBSBML_CPP_NAMESPACE_END


/**
 * @file    SBaseRef.cpp
 * @brief   Implementation of SBaseRef, the SBase-derived class of the comp package.
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

#include <iostream>

#include <sbml/SBMLVisitor.h>
#include <sbml/xml/XMLNode.h>
#include <sbml/xml/XMLToken.h>
#include <sbml/xml/XMLAttributes.h>
#include <sbml/xml/XMLInputStream.h>
#include <sbml/xml/XMLOutputStream.h>

#include <sbml/packages/comp/extension/CompExtension.h>
#include <sbml/packages/comp/sbml/SBaseRef.h>
#include <sbml/packages/comp/extension/CompModelPlugin.h>
#include <sbml/packages/comp/validator/CompSBMLErrorTable.h>
#include <sbml/Model.h>

#include <sbml/util/ElementFilter.h>

using namespace std;

LIBSBML_CPP_NAMESPACE_BEGIN
#ifdef __cplusplus

SBaseRef::SBaseRef (unsigned int level, unsigned int version, unsigned int pkgVersion) 
  : CompBase (level,version, pkgVersion)
  , mMetaIdRef("")
  , mPortRef("")
  , mIdRef("")
  , mUnitRef("")
  , mSBaseRef(NULL)
  , mReferencedElement(NULL)
  , mDirectReference(NULL)
{
}


SBaseRef::SBaseRef(CompPkgNamespaces* compns)
  : CompBase(compns)
  , mMetaIdRef("")
  , mPortRef("")
  , mIdRef("")
  , mUnitRef("")
  , mSBaseRef(NULL)
  , mReferencedElement(NULL)
  , mDirectReference(NULL)
{
  loadPlugins(compns);
}


SBaseRef::SBaseRef(const SBaseRef& source) 
  : CompBase (source)
{
  mMetaIdRef=source.mMetaIdRef;
  mPortRef=source.mPortRef;
  mIdRef=source.mIdRef;
  mUnitRef=source.mUnitRef;
  if (source.mSBaseRef!= NULL) {
    mSBaseRef=source.mSBaseRef->clone();
  }
  else {
    mSBaseRef=NULL;
  }
  mReferencedElement = NULL;
  mDirectReference = NULL;
}

SBaseRef& SBaseRef::operator=(const SBaseRef& source)
{
  if(&source!=this)
  {
    CompBase::operator=(source);
    mMetaIdRef=source.mMetaIdRef;
    mPortRef=source.mPortRef;
    mIdRef=source.mIdRef;
    mUnitRef=source.mUnitRef;
    if (source.mSBaseRef!= NULL) {
      mSBaseRef=source.mSBaseRef->clone();
    }
    else {
      mSBaseRef=NULL;
    }
  }
  mReferencedElement = NULL;
  mDirectReference = NULL;

  return *this;
}

SBaseRef*
SBaseRef::clone() const
{
  return new SBaseRef(*this);
}

SBaseRef::~SBaseRef ()
{
  if (mSBaseRef != NULL) delete mSBaseRef;
}


SBase* 
SBaseRef::getElementBySId(const std::string& id)
{
  if (id.empty()) return NULL;
  if (mSBaseRef != NULL) {
    SBase* obj = mSBaseRef->getElementBySId(id);
    if (obj != NULL) return obj;
  }
  return getElementFromPluginsBySId(id);
}


SBase*
SBaseRef::getElementByMetaId(const std::string& metaid)
{
  if (metaid.empty()) return NULL;
  if (mSBaseRef != NULL) {
    if (mSBaseRef->getMetaId() == metaid) return mSBaseRef;

    SBase* obj = mSBaseRef->getElementByMetaId(metaid);
    if (obj != NULL) return obj;
  }
  return getElementFromPluginsByMetaId(metaid);
}


List*
SBaseRef::getAllElements(ElementFilter *filter)
{
  List* ret = new List();
  List* sublist = NULL;

  ADD_FILTERED_POINTER(ret, sublist, mSBaseRef, filter);  
  
  ADD_FILTERED_FROM_PLUGIN(ret, sublist, filter);

  return ret;
}

int
SBaseRef::setMetaIdRef (const std::string& metaIdRef)
{
  //Only set the metaIdRef if we don't refer to anything already, or if we only
  // refer to the metaIdRef.
  if (!(getNumReferents() == 0 || 
        (getNumReferents()==1 && isSetMetaIdRef()))) {
    //LS DEBUG return something else?
    return LIBSBML_OPERATION_FAILED;
  }
  if (!SyntaxChecker::isValidXMLID(metaIdRef)) 
  {
    return LIBSBML_INVALID_ATTRIBUTE_VALUE;
  }
  mMetaIdRef = metaIdRef;
  return LIBSBML_OPERATION_SUCCESS;
}


const string&
SBaseRef::getMetaIdRef () const
{
  return mMetaIdRef;
}


bool
SBaseRef::isSetMetaIdRef () const
{
  return (mMetaIdRef.empty() == false);
}


int
SBaseRef::unsetMetaIdRef ()
{
  mMetaIdRef.erase();

  if (mMetaIdRef.empty())
  {
    return LIBSBML_OPERATION_SUCCESS;
  }
  else
  {
    return LIBSBML_OPERATION_FAILED;
  }
}

/*
 * Sets the portRef of this SBML object to a copy of portRef.
 */
int
SBaseRef::setPortRef (const std::string& portRef)
{
  //Only set the portRef if we don't refer to anything already, or if we only
  // refer to the portRef.
  if (!(getNumReferents() == 0 || 
        (getNumReferents()==1 && isSetPortRef()))) {
    //LS DEBUG return something else?
    return LIBSBML_OPERATION_FAILED;
  }
  if (!SyntaxChecker::isValidSBMLSId(portRef)) 
  {
    return LIBSBML_INVALID_ATTRIBUTE_VALUE;
  }
  mPortRef = portRef;
  return LIBSBML_OPERATION_SUCCESS;
}


/*
 * @return the portRef of this SBML object.
 */
const string&
SBaseRef::getPortRef () const
{
  return mPortRef;
}


/*
 * @return true if the portRef of this SBML object has been set, false
 * otherwise.
 */
bool
SBaseRef::isSetPortRef () const
{
  return (mPortRef.empty() == false);
}


/*
 * Unsets the portRef of this SBML object.
 */
int
SBaseRef::unsetPortRef ()
{
  mPortRef.erase();

  if (mPortRef.empty())
  {
    return LIBSBML_OPERATION_SUCCESS;
  }
  else
  {
    return LIBSBML_OPERATION_FAILED;
  }
}


/*
 * Sets the idRef of this SBML object to a copy of idRef.
 */
int
SBaseRef::setIdRef (const std::string& idRef)
{
  //Only set the idref if we don't refer to anything already, or if we only
  // refer to the idref.
  if (!(getNumReferents() == 0 || 
        (getNumReferents()==1 && isSetIdRef()))) {
    //LS DEBUG return something else?
    return LIBSBML_OPERATION_FAILED;
  }
  if (!SyntaxChecker::isValidSBMLSId(idRef)) 
  {
    //LS DEBUG return something else
    return LIBSBML_INVALID_ATTRIBUTE_VALUE;
  }
  mIdRef = idRef;
  return LIBSBML_OPERATION_SUCCESS;
}


/*
 * @return the idRef of this SBML object.
 */
const string&
SBaseRef::getIdRef () const
{
  return mIdRef;
}


/*
 * @return true if the idRef of this SBML object has been set, false
 * otherwise.
 */
bool
SBaseRef::isSetIdRef () const
{
  return (mIdRef.empty() == false);
}


/*
 * Unsets the idRef of this SBML object.
 */
int
SBaseRef::unsetIdRef ()
{
  mIdRef.erase();

  if (mIdRef.empty())
  {
    return LIBSBML_OPERATION_SUCCESS;
  }
  else
  {
    return LIBSBML_OPERATION_FAILED;
  }
}

/*
 * Sets the unitRef of this SBML object to a copy of unitRef.
 */
int
SBaseRef::setUnitRef (const std::string& unitRef)
{
  //Only set the UnitRef if we don't refer to anything already, or if we only
  // refer to the UnitRef.
  if (!(getNumReferents() == 0 || 
        (getNumReferents()==1 && isSetUnitRef()))) {
    //LS DEBUG return something else?
    return LIBSBML_OPERATION_FAILED;
  }
  if (!SyntaxChecker::isValidSBMLSId(unitRef)) 
  {
    return LIBSBML_INVALID_ATTRIBUTE_VALUE;
  }
  mUnitRef = unitRef;
  return LIBSBML_OPERATION_SUCCESS;
}


/*
 * @return the unitRef of this SBML object.
 */
const string&
SBaseRef::getUnitRef () const
{
  return mUnitRef;
}


/*
 * @return true if the unitRef of this SBML object has been set, false
 * otherwise.
 */
bool
SBaseRef::isSetUnitRef () const
{
  return (mUnitRef.empty() == false);
}


/*
 * Unsets the unitRef of this SBML object.
 */
int
SBaseRef::unsetUnitRef ()
{
  mUnitRef.erase();

  if (mUnitRef.empty())
  {
    return LIBSBML_OPERATION_SUCCESS;
  }
  else
  {
    return LIBSBML_OPERATION_FAILED;
  }
}

/*
 * @return the child SBaseRef of this SBaseRef.
 */
const SBaseRef*
SBaseRef::getSBaseRef () const
{
  return mSBaseRef;
}


/*
 * @return the child SBaseRef of this SBaseRef.
 */
SBaseRef*
SBaseRef::getSBaseRef ()
{
  return mSBaseRef;
}

/*
 * @return true if the child SBaseRef of this SBaseRef is set, false otherwise.
 */
bool
SBaseRef::isSetSBaseRef () const
{
  return (mSBaseRef != NULL);
}


/*
 * Sets the child SBaseRef of this SBaseRef to a copy of the given SBaseRef.
 */
int
SBaseRef::setSBaseRef (const SBaseRef* sBaseRef)
{
  if (mSBaseRef == sBaseRef) 
  {
    return LIBSBML_OPERATION_SUCCESS;
  }
  else if (sBaseRef == NULL)
  {
    if (mSBaseRef != NULL) delete mSBaseRef;
    mSBaseRef = NULL;
    return LIBSBML_OPERATION_SUCCESS;
  }
  else if (getLevel() != sBaseRef->getLevel())
  {
    return LIBSBML_LEVEL_MISMATCH;
  }
  else if (getVersion() != sBaseRef->getVersion())
  {
    return LIBSBML_VERSION_MISMATCH;
  }
  else if (getPackageVersion() != sBaseRef->getPackageVersion())
  {
    return LIBSBML_PKG_VERSION_MISMATCH;
  }
  else
  {
    if (mSBaseRef != NULL) delete mSBaseRef;
    mSBaseRef = sBaseRef->clone();
    if (mSBaseRef == NULL) {
      return LIBSBML_OPERATION_FAILED;
    }
    mSBaseRef->connectToParent(this);
    return LIBSBML_OPERATION_SUCCESS;
  }
}


/*
 * Unsets the sBaseRef of this SBaseRef.
 */
int
SBaseRef::unsetSBaseRef ()
{
  delete mSBaseRef;
  mSBaseRef = NULL;

  if (mSBaseRef == NULL) 
  {
    return LIBSBML_OPERATION_SUCCESS;
  }
  else
  {
    return LIBSBML_OPERATION_FAILED;
  }
}

int 
SBaseRef::getNumReferents() const
{
  int retval = 0;
  if (isSetPortRef()) retval++;
  if (isSetIdRef()) retval++;
  if (isSetUnitRef()) retval++;
  if (isSetMetaIdRef()) retval++;
  return retval;
}

bool 
SBaseRef::hasRequiredAttributes() const
{
  if (!CompBase::hasRequiredAttributes()) return false;
  return (getNumReferents()==1);
}

void
SBaseRef::renameSIdRefs(const std::string& oldid, const std::string& newid)
{
  if (mPortRef==oldid) mPortRef=newid;
  if (mIdRef==oldid) mIdRef=newid;
  if (mUnitRef==oldid) mUnitRef=newid;
  if (mMetaIdRef==oldid) mMetaIdRef=newid;
  SBase::renameSIdRefs(oldid, newid);
}

/*
 * Creates a new SBaseRef, adds it to this SBaseRef
 * and returns it.
 */
SBaseRef*
SBaseRef::createSBaseRef ()
{
  delete mSBaseRef;
  mSBaseRef = NULL;
  
  try
  {
    COMP_CREATE_NS(compns, getSBMLNamespaces());
    mSBaseRef = new SBaseRef(compns);
    delete compns;
  }
  catch (...)
  {
    /* here we do not create a default object as the level/version must
     * match the parent object
     *
     * so do nothing
     */
  }

  if (mSBaseRef)
  {
    mSBaseRef->connectToParent(this);
  }

  return mSBaseRef;
}



const std::string&
SBaseRef::getElementName () const
{
  static const std::string name = "sBaseRef";
  return name;
}

/** @cond doxygenLibsbmlInternal */
void
SBaseRef::addExpectedAttributes(ExpectedAttributes& attributes)
{
  CompBase::addExpectedAttributes(attributes);
  attributes.add("portRef");
  attributes.add("idRef");
  attributes.add("unitRef");
  attributes.add("metaIdRef");
}
/** @endcond */

/** @cond doxygenLibsbmlInternal */
void
SBaseRef::readAttributes (const XMLAttributes& attributes,
                          const ExpectedAttributes& expectedAttributes)
{
  CompBase::readAttributes(attributes,expectedAttributes);

  const unsigned int sbmlLevel   = getLevel  ();
  //const unsigned int sbmlVersion = getVersion();
  if ( sbmlLevel > 2 )
  {
    XMLTriple tripleMetaIdRef("metaIdRef", mURI, getPrefix());
    if (attributes.readInto(tripleMetaIdRef, mMetaIdRef, getErrorLog(), false, getLine(), getColumn())) {
      if (!SyntaxChecker::isValidXMLID(mMetaIdRef)) {
        logInvalidId("comp:metaIdRef", mMetaIdRef);
      }
    }
    XMLTriple triplePort("portRef", mURI, getPrefix());
    if (attributes.readInto(triplePort, mPortRef, getErrorLog(), false, getLine(), getColumn())) {
      if (!SyntaxChecker::isValidSBMLSId(mPortRef)) {
        logInvalidId("comp:portRef", mPortRef);
      }
    }
    XMLTriple tripleIdRef("idRef", mURI, getPrefix());
    if (attributes.readInto(tripleIdRef, mIdRef, getErrorLog(), false, getLine(), getColumn())) {
      if (!SyntaxChecker::isValidSBMLSId(mIdRef)) {
        logInvalidId("comp:idRef", mIdRef);
      }
    }
    XMLTriple tripleUnitRef("unitRef", mURI, getPrefix());
    if (attributes.readInto(tripleUnitRef, mUnitRef, getErrorLog(), false, getLine(), getColumn())) {
      if (!SyntaxChecker::isValidSBMLSId(mUnitRef)) {
        logInvalidId("comp:unitRef", mUnitRef);
      }
    }
  }
  if (getNumReferents() == 0) {
    //LS DEBUG Set the error log here
  }
  if (getNumReferents() > 1) {
    //LS DEBUG Set the error log here 
  }
}
/** @endcond */

/** @cond doxygenLibsbmlInternal */
SBase* 
SBaseRef::createObject (XMLInputStream& stream)
{
  SBase*        object = NULL;

  const std::string&   name   = stream.peek().getName();
  const XMLNamespaces& xmlns  = stream.peek().getNamespaces();
  const std::string&   prefix = stream.peek().getPrefix();

  const std::string& targetPrefix = (xmlns.hasURI(mURI)) ? xmlns.getPrefix(mURI) : getPrefix();
  
  if (prefix == targetPrefix)
  {
    SBMLErrorLog* errlog = getErrorLog();
    if (mSBaseRef != NULL && (name =="sBaseRef" || name=="sbaseRef")) {
      if (errlog != NULL) {
          errlog->logPackageError(getPackageName(), CompOneSBaseRefOnly, 
            getPackageVersion(), getLevel(), getVersion());
      }
      object = mSBaseRef;
    }
    else if ( name == "sBaseRef" ) {
      COMP_CREATE_NS(compns, getSBMLNamespaces());
      mSBaseRef = new SBaseRef(compns);
      object = mSBaseRef;
      object->connectToParent(this);
      delete compns;
    }
    else if ( name == "sbaseRef" ) {
      if (errlog != NULL) {
          errlog->logPackageError(getPackageName(), CompDeprecatedSBaseRefSpelling, 
            getPackageVersion(), getLevel(), getVersion());
      }
      COMP_CREATE_NS(compns, getSBMLNamespaces());
      mSBaseRef = new SBaseRef(compns);
      object = mSBaseRef;
      object->connectToParent(this);
      delete compns;
    }
  }
  return object;
}
/** @endcond */

/** @cond doxygenLibsbmlInternal */
void
SBaseRef::writeAttributes (XMLOutputStream& stream) const
{
  CompBase::writeAttributes(stream);

  if (isSetMetaIdRef()) {
    stream.writeAttribute("metaIdRef", getPrefix(), mMetaIdRef);
  }
  if (isSetPortRef()) {
    stream.writeAttribute("portRef", getPrefix(), mPortRef);
  }
  if (isSetIdRef()) {
    stream.writeAttribute("idRef", getPrefix(), mIdRef);
  }
  if (isSetUnitRef()) {
    stream.writeAttribute("unitRef", getPrefix(), mUnitRef);
  }
}
/** @endcond */

/** @cond doxygenLibsbmlInternal */
void
SBaseRef::writeElements (XMLOutputStream& stream) const
{
  CompBase::writeElements(stream);

  if (isSetSBaseRef()) {
    mSBaseRef->write(stream);
  }
  SBaseRef::writeExtensionElements(stream);
}
/** @endcond */


int
SBaseRef::getTypeCode () const
{
  return SBML_COMP_SBASEREF;
}


/*
 * Accepts the given SBMLVisitor.
 */
bool
SBaseRef::accept (SBMLVisitor& v) const
{
  if (isSetSBaseRef() == true)
  {
    mSBaseRef->accept(v);
  }
  return v.visit(*this);
}


/** @cond doxygenLibsbmlInternal */
/*
 * Sets the parent SBMLDocument of this SBML object.
 */
void
SBaseRef::setSBMLDocument (SBMLDocument* d)
{
  CompBase::setSBMLDocument(d);
  if (isSetSBaseRef()) {
    mSBaseRef->setSBMLDocument(d);
  }
}
/** @endcond */


/** @cond doxygenLibsbmlInternal */
/*
 * Sets this SBML object to child SBML objects (if any).
 * (Creates a child-parent relationship by the parent)
 */
void
SBaseRef::connectToChild()
{
  CompBase::connectToChild();
  if (isSetSBaseRef()) {
    mSBaseRef->connectToParent(this);
  }
}
/** @endcond */


SBase* 
SBaseRef::getReferencedElementFrom(Model* model)
{
  SBMLDocument* doc = getSBMLDocument();
  if (!hasRequiredAttributes()) {
    if (doc) {
      string error = "In SBaseRef::getReferencedElementFrom, unable to find referenced element from <" + getElementName() + "> ";
      if (isSetId()) {
        error += "with ID '" + getId() + "' ";
      }
      error += "as it does not have the required attributes.";
      int en = CompSBaseRefMustReferenceObject;
      switch(getTypeCode()) {
      case SBML_COMP_REPLACEDBY:
        en = CompReplacedByAllowedAttributes;
        break;
      case SBML_COMP_REPLACEDELEMENT:
        en = CompReplacedElementAllowedAttributes;
        break;
      case SBML_COMP_PORT:
        en = CompPortAllowedAttributes;
        break;
      case SBML_COMP_DELETION:
        en = CompDeletionAllowedAttributes;
      }
      doc->getErrorLog()->logPackageError("comp", en, getPackageVersion(), getLevel(), getVersion(), error, getLine(), getColumn());
    }
    return NULL;
  }
  SBase* referent = NULL;
  if (isSetPortRef()) {
    CompModelPlugin* mplugin = static_cast<CompModelPlugin*>(model->getPlugin(getPrefix()));
    Port* port = mplugin->getPort(getPortRef());
    if (port==NULL) {
      if (doc) {
        string error = "In SBaseRef::getReferencedElementFrom, unable to find referenced element from SBase reference ";
        if (isSetId()) {
          error += "'" + getId() + "' ";
        }
        error += "as the port it references ('" + getPortRef() +"') could not be found.";
        doc->getErrorLog()->logPackageError("comp", CompPortRefMustReferencePort, getPackageVersion(), getLevel(), getVersion(), error, getLine(), getColumn());
      }
      return NULL;
    }
    mDirectReference = port;
    referent = port->getReferencedElementFrom(model);
  }
  else if (isSetIdRef()) {
    referent = model->getElementBySId(getIdRef());
    if (referent == NULL && doc) {
      string error = "In SBaseRef::getReferencedElementFrom, unable to find referenced element: no such SId in the model: '" + getIdRef() + "'.";
      if (doc->getErrorLog()->contains(UnrequiredPackagePresent) 
        || doc->getErrorLog()->contains(RequiredPackagePresent))
      {
        doc->getErrorLog()->logPackageError("comp", 
          CompIdRefMayReferenceUnknownPackage, getPackageVersion(), 
          getLevel(), getVersion(), error, getLine(), getColumn());
      }
      else
      {
        doc->getErrorLog()->logPackageError("comp", 
          CompIdRefMustReferenceObject, getPackageVersion(), 
          getLevel(), getVersion(), error, getLine(), getColumn());
      }
    }
  }
  else if (isSetUnitRef()) {
    referent = model->getUnitDefinition(getUnitRef());
    if (referent == NULL && doc) {
      string error = "In SBaseRef::getReferencedElementFrom, unable to find referenced element: no such Unit in the model: '" + getUnitRef() + "'.";
      doc->getErrorLog()->logPackageError("comp", CompUnitRefMustReferenceUnitDef, getPackageVersion(), getLevel(), getVersion(), error, getLine(), getColumn());
    }
  }
  else if (isSetMetaIdRef()) {
    referent = model->getElementByMetaId(getMetaIdRef());
    if (referent == NULL && doc) {
      string error = "In SBaseRef::getReferencedElementFrom, unable to find referenced element: no such metaid in the model: '" + getMetaIdRef() + "'.";
      if (doc->getErrorLog()->contains(UnrequiredPackagePresent) 
        || doc->getErrorLog()->contains(RequiredPackagePresent))
      {
        doc->getErrorLog()->logPackageError("comp", 
          CompIdRefMayReferenceUnknownPackage, getPackageVersion(), 
          getLevel(), getVersion(), error, getLine(), getColumn());
      }
      else
      {
        doc->getErrorLog()->logPackageError("comp", 
          CompMetaIdRefMustReferenceObject, getPackageVersion(), 
          getLevel(), getVersion(), error, getLine(), getColumn());
      }
    }
  }
  else {
    //This is actually possible if the subclass overrides getNumReferents() (which some do).  In that case, we just return NULL and let the overriding function find the referent instead.
    return NULL;
  }
  if (referent == NULL) {
    //No need to set an error message--one was already set above.
    return NULL;
  }
  if (isSetSBaseRef()) {
    //We're drilling into the submodels here, so our referent must be a submodel.
    if (referent->getTypeCode() != SBML_COMP_SUBMODEL) {
      if (doc) {
        string error = "In SBaseRef::getReferencedElementFrom, unable to find referenced element: the element ";
        if (referent->isSetId()) {
          error += "'" + referent->getId() + "'";
        }
        else if (referent->isSetMetaId()) {
          error += "with the metaid '" + referent->getMetaId() + "'";
        }
        error += " is not a submodel, and therefore has no subobjects for the child <sBaseRef> to refer to.";
        doc->getErrorLog()->logPackageError("comp", CompParentOfSBRefChildMustBeSubmodel, getPackageVersion(), getLevel(), getVersion(), error, getLine(), getColumn());
      }
      return NULL;
    }
    Submodel* subm = static_cast<Submodel*>(referent);
    if (subm==NULL) {
      //Note:  should be impossible.
      if (doc) {
        string error = "In SBaseRef::getReferencedElementFrom, unable to find referenced element: the element ";
        if (referent->isSetId()) {
          error += "'" + referent->getId() + "' ";
        }
        else if (referent->isSetMetaId()) {
          error += "with the metaid '" + referent->getMetaId() + "' ";
        }
        error += "claims to be a Submodel, but could not be programmatically turned into one.";
        doc->getErrorLog()->logPackageError("comp", CompParentOfSBRefChildMustBeSubmodel, getPackageVersion(), getLevel(), getVersion(), error, getLine(), getColumn());
      }
      return NULL;
    }
    Model* inst = subm->getInstantiation();
    if (inst==NULL) {
      //No need to set an additional error, as 'getInstantiation' will set one itself.
      return NULL;
    }
    //Recursive, so will set its own error messages:
    referent = getSBaseRef()->getReferencedElementFrom(inst);
    mDirectReference = getSBaseRef()->getDirectReference();
  }
  return referent;
}

int SBaseRef::saveReferencedElement()
{
  //The only thing that knows what Model we should point to is the parent of this object.  Since it will also be of the class SBaseRef, just call this recursively.
  SBMLDocument* doc = getSBMLDocument();
  SBase* parent = getParentSBMLObject();
  if (parent==NULL) {
    if (doc) {
      string error = "In SBaseRef::saveReferencedElement, unable to find referenced element: no parent could be found for the given <sBaseRef> element.";
      doc->getErrorLog()->logPackageError("comp", CompModelFlatteningFailed, getPackageVersion(), getLevel(), getVersion(), error, getLine(), getColumn());
    }
    return LIBSBML_OPERATION_FAILED;
  }
  SBaseRef* parentref = static_cast<SBaseRef*>(parent);
  if (parentref==NULL || (parent->getTypeCode() != SBML_COMP_SBASEREF &&
                          parent->getTypeCode() != SBML_COMP_PORT &&
                          parent->getTypeCode() != SBML_COMP_DELETION &&
                          parent->getTypeCode() != SBML_COMP_REPLACEDBY &&
                          parent->getTypeCode() != SBML_COMP_REPLACEDELEMENT)) {
    if (doc) {
      string error = "In SBaseRef::saveReferencedElement, unable to find referenced element: the parent of the given <sBaseRef> element was not the correct type.";
      doc->getErrorLog()->logPackageError("comp", CompModelFlatteningFailed, getPackageVersion(), getLevel(), getVersion(), error, getLine(), getColumn());
    }
    return LIBSBML_OPERATION_FAILED;
  }
  if (parentref->saveReferencedElement() != LIBSBML_OPERATION_SUCCESS) {
    //saveReferencedElement will set its own error messages.
    return LIBSBML_OPERATION_FAILED;
  }
  mReferencedElement = parentref->getReferencedElement();
  mDirectReference = parentref->getDirectReference();
  if (mReferencedElement==NULL) {
    //getReferencedElement will set its own error messages.
    return LIBSBML_OPERATION_FAILED;
  }
  return LIBSBML_OPERATION_SUCCESS;
}

SBase* SBaseRef::getReferencedElement()
{
  if (mReferencedElement==NULL) {
    saveReferencedElement();
  }
  return mReferencedElement;
}

void SBaseRef::clearReferencedElement()
{
  mReferencedElement = NULL;
}

int SBaseRef::collectDeletions(set<SBase*>* removed, set<SBase*>* toremove)
{
  SBase* todelete = getReferencedElement();
  if (todelete==NULL) {
    return LIBSBML_INVALID_OBJECT;
  }
  if (removed) {
    if (removed->find(todelete) != removed->end()) {
      //Already deleted or replaced.
      return LIBSBML_OPERATION_SUCCESS;
    }
  }

  if (toremove) {
    toremove->insert(todelete);
  }

  CompSBasePlugin* todplug = static_cast<CompSBasePlugin*>(todelete->getPlugin(getPrefix()));
  if (todplug != NULL) {
    for (unsigned long re=0; re<todplug->getNumReplacedElements(); re++) {
      todplug->getReplacedElement((unsigned int)re)->collectDeletions(removed, toremove);
    }
    if (todplug->isSetReplacedBy()) {
      todplug->getReplacedBy()->collectDeletions(removed, toremove);
    }
  }
  return LIBSBML_OPERATION_SUCCESS;
}

//Deprecated function
int SBaseRef::performDeletion()
{
  set<SBase*> toremove;
  set<SBase*>* removed=NULL;
  CompModelPlugin* cmp = NULL;
  SBase* parent = getParentSBMLObject();
  while (parent != NULL && parent->getTypeCode() != SBML_DOCUMENT) {
    if (parent->getTypeCode() == SBML_COMP_MODELDEFINITION ||
        parent->getTypeCode() == SBML_MODEL) {
          cmp = static_cast<CompModelPlugin*>(parent->getPlugin("comp"));
          if (cmp != NULL) {
            removed = cmp->getRemovedSet();
          }
    }
    parent = parent->getParentSBMLObject();
  }
  int ret = removed != NULL ? collectDeletions(removed, &toremove) : LIBSBML_INVALID_OBJECT ;
  if (ret != LIBSBML_OPERATION_SUCCESS) {
    return ret;
  }
  if (cmp == NULL) {
    return LIBSBML_INVALID_OBJECT;
  }
  return cmp->removeCollectedElements(removed, &toremove);
}

int SBaseRef::removeFromParentAndDelete()
{
  SBase* parent = getParentSBMLObject();
  if (parent==NULL) return LIBSBML_OPERATION_FAILED;
  SBaseRef* parentSBR;
  switch(parent->getTypeCode()) {
  case SBML_LIST_OF:
    //This will be the case if we are a deletion, replaced element, replaced by, or port.
    return SBase::removeFromParentAndDelete();
  case SBML_COMP_DELETION:
  case SBML_COMP_SBASEREF:
  case SBML_COMP_REPLACEDELEMENT:
  case SBML_COMP_REPLACEDBY:
  case SBML_COMP_PORT:
    //This will be the case if we are a 'raw' SBaseRef object.
    parentSBR = static_cast<SBaseRef*>(parent);
    if (parentSBR==NULL) return LIBSBML_OPERATION_FAILED;
    return parentSBR->unsetSBaseRef();
  default:
    assert(false); //Nothing else should ever be a parent of an SBaseRef object.
    return LIBSBML_OPERATION_FAILED;
  }
}

/** @cond doxygenLibsbmlInternal */
SBase* SBaseRef::getDirectReference()
{
  return mDirectReference;
}
/** @endcond */

#endif /* __cplusplus */
/** @cond doxygenIgnored */

LIBSBML_EXTERN
SBaseRef_t *
SBaseRef_create(unsigned int level, unsigned int version,
                unsigned int pkgVersion)
{
  return new SBaseRef(level, version, pkgVersion);
}


LIBSBML_EXTERN
void
SBaseRef_free(SBaseRef_t * sbr)
{
  if (sbr != NULL)
    delete sbr;
}


LIBSBML_EXTERN
SBaseRef_t *
SBaseRef_clone(SBaseRef_t * sbr)
{
  if (sbr != NULL)
  {
    return static_cast<SBaseRef_t*>(sbr->clone());
  }
  else
  {
    return NULL;
  }
}


LIBSBML_EXTERN
char *
SBaseRef_getPortRef(SBaseRef_t * sbr)
{
  if (sbr == NULL)
    return NULL;

  return sbr->getPortRef().empty() ? NULL : safe_strdup(sbr->getPortRef().c_str());
}


LIBSBML_EXTERN
char *
SBaseRef_getIdRef(SBaseRef_t * sbr)
{
  if (sbr == NULL)
    return NULL;

  return sbr->getIdRef().empty() ? NULL : safe_strdup(sbr->getIdRef().c_str());
}


LIBSBML_EXTERN
char *
SBaseRef_getUnitRef(SBaseRef_t * sbr)
{
  if (sbr == NULL)
    return NULL;

  return sbr->getUnitRef().empty() ? NULL : safe_strdup(sbr->getUnitRef().c_str());
}


LIBSBML_EXTERN
char *
SBaseRef_getMetaIdRef(SBaseRef_t * sbr)
{
  if (sbr == NULL)
    return NULL;

  return sbr->getMetaIdRef().empty() ? NULL : safe_strdup(sbr->getMetaIdRef().c_str());
}


LIBSBML_EXTERN
SBaseRef_t*
SBaseRef_getSBaseRef(SBaseRef_t * sbr)
{
  if (sbr == NULL)
    return NULL;

  return sbr->getSBaseRef();
}


LIBSBML_EXTERN
int
SBaseRef_isSetPortRef(SBaseRef_t * sbr)
{
  return (sbr != NULL) ? static_cast<int>(sbr->isSetPortRef()) : 0;
}


LIBSBML_EXTERN
int
SBaseRef_isSetIdRef(SBaseRef_t * sbr)
{
  return (sbr != NULL) ? static_cast<int>(sbr->isSetIdRef()) : 0;
}


LIBSBML_EXTERN
int
SBaseRef_isSetUnitRef(SBaseRef_t * sbr)
{
  return (sbr != NULL) ? static_cast<int>(sbr->isSetUnitRef()) : 0;
}


LIBSBML_EXTERN
int
SBaseRef_isSetMetaIdRef(SBaseRef_t * sbr)
{
  return (sbr != NULL) ? static_cast<int>(sbr->isSetMetaIdRef()) : 0;
}


LIBSBML_EXTERN
int
SBaseRef_isSetSBaseRef(SBaseRef_t * sbr)
{
  return (sbr != NULL) ? static_cast<int>(sbr->isSetSBaseRef()) : 0;
}


LIBSBML_EXTERN
int
SBaseRef_setPortRef(SBaseRef_t * sbr, const char * portRef)
{
  return (sbr != NULL) ? sbr->setPortRef(portRef) : LIBSBML_INVALID_OBJECT;
}


LIBSBML_EXTERN
int
SBaseRef_setIdRef(SBaseRef_t * sbr, const char * idRef)
{
  return (sbr != NULL) ? sbr->setIdRef(idRef) : LIBSBML_INVALID_OBJECT;
}


LIBSBML_EXTERN
int
SBaseRef_setUnitRef(SBaseRef_t * sbr, const char * unitRef)
{
  return (sbr != NULL) ? sbr->setUnitRef(unitRef) : LIBSBML_INVALID_OBJECT;
}


LIBSBML_EXTERN
int
SBaseRef_setMetaIdRef(SBaseRef_t * sbr, const char * metaIdRef)
{
  return (sbr != NULL) ? sbr->setMetaIdRef(metaIdRef) : LIBSBML_INVALID_OBJECT;
}


LIBSBML_EXTERN
int
SBaseRef_setSBaseRef(SBaseRef_t * sbr, SBaseRef_t * sBaseRef)
{
  return (sbr != NULL) ? sbr->setSBaseRef(sBaseRef) : LIBSBML_INVALID_OBJECT;
}


LIBSBML_EXTERN
int
SBaseRef_unsetPortRef(SBaseRef_t * sbr)
{
  return (sbr != NULL) ? sbr->unsetPortRef() : LIBSBML_INVALID_OBJECT;
}


LIBSBML_EXTERN
int
SBaseRef_unsetIdRef(SBaseRef_t * sbr)
{
  return (sbr != NULL) ? sbr->unsetIdRef() : LIBSBML_INVALID_OBJECT;
}


LIBSBML_EXTERN
int
SBaseRef_unsetUnitRef(SBaseRef_t * sbr)
{
  return (sbr != NULL) ? sbr->unsetUnitRef() : LIBSBML_INVALID_OBJECT;
}


LIBSBML_EXTERN
int
SBaseRef_unsetMetaIdRef(SBaseRef_t * sbr)
{
  return (sbr != NULL) ? sbr->unsetMetaIdRef() : LIBSBML_INVALID_OBJECT;
}


LIBSBML_EXTERN
int
SBaseRef_unsetSBaseRef(SBaseRef_t * sbr)
{
  return (sbr != NULL) ? sbr->unsetSBaseRef() : LIBSBML_INVALID_OBJECT;
}


LIBSBML_EXTERN
int
SBaseRef_hasRequiredAttributes(SBaseRef_t * sbr)
{
  return (sbr != NULL) ? static_cast<int>(sbr->hasRequiredAttributes()) : 0;
}


/** @endcond */
LIBSBML_CPP_NAMESPACE_END

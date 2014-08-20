/**
 * @file    Deletion.cpp
 * @brief   Implementation of Deletion, the SBaseRef derived class of deletions package.
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

#include <sbml/xml/XMLNode.h>
#include <sbml/xml/XMLToken.h>
#include <sbml/xml/XMLAttributes.h>
#include <sbml/xml/XMLInputStream.h>
#include <sbml/xml/XMLOutputStream.h>

#include <sbml/packages/comp/extension/CompExtension.h>
#include <sbml/packages/comp/sbml/Deletion.h>
#include <sbml/packages/comp/sbml/Port.h>
#include <sbml/packages/comp/sbml/Submodel.h>
#include <sbml/packages/comp/validator/CompSBMLError.h>

using namespace std;

LIBSBML_CPP_NAMESPACE_BEGIN
#ifdef __cplusplus

Deletion::Deletion (unsigned int level, unsigned int version, unsigned int pkgVersion) 
  : SBaseRef (level,version, pkgVersion)
  ,mId("")
  ,mName("")
{
}


Deletion::Deletion(CompPkgNamespaces* compns)
  : SBaseRef(compns)
  ,mId("")
  ,mName("")
{
  loadPlugins(compns);
}


Deletion::Deletion(const Deletion& source) : SBaseRef(source)
{
  mId=source.mId;
  mName=source.mName;
}


Deletion& Deletion::operator=(const Deletion& source)
{
  if(&source!=this)
  {
    SBaseRef::operator=(source);
    mId = source.mId;
    mName = source.mName;
  }
  return *this;
}


Deletion*
Deletion::clone() const
{
  return new Deletion(*this);
}


Deletion::~Deletion ()
{
}


const string&
Deletion::getId() const
{
  return mId;
}


bool 
Deletion::isSetId() const
{
  return (mId.empty()==false);
}


int 
Deletion::setId(const string& id)
{
  if (!SyntaxChecker::isValidSBMLSId(id)) {
    return LIBSBML_INVALID_ATTRIBUTE_VALUE;
  }
  mId = id;
  return LIBSBML_OPERATION_SUCCESS;
}


int 
Deletion::unsetId()
{
  mId = "";
  return LIBSBML_OPERATION_SUCCESS;
}



const string&
Deletion::getName() const
{
  return mName;
}


bool 
Deletion::isSetName() const
{
  return (mName.empty()==false);
}

int 
Deletion::setName(const string& name)
{
  if (name.empty()) {
    return LIBSBML_INVALID_ATTRIBUTE_VALUE;
  }
  mName = name;
  return LIBSBML_OPERATION_SUCCESS;
}


int 
Deletion::unsetName()
{
  mName = "";
  return LIBSBML_OPERATION_SUCCESS;
}



const std::string&
Deletion::getElementName () const
{
  static const std::string name = "deletion";
  return name;
}


/** @cond doxygenLibsbmlInternal */
void
Deletion::addExpectedAttributes(ExpectedAttributes& attributes)
{
  SBaseRef::addExpectedAttributes(attributes);
  attributes.add("id");
  attributes.add("name");
}
/** @endcond */

/** @cond doxygenLibsbmlInternal */
void
Deletion::readAttributes (const XMLAttributes& attributes,
                          const ExpectedAttributes& expectedAttributes)
{
  const unsigned int sbmlLevel   = getLevel  ();
  const unsigned int sbmlVersion = getVersion();

  // look to see whether an unknown attribute error was logged
  // during the read of the ListOfDeletions - which will have
  // happened immediately prior to this read
  if (getErrorLog() != NULL && 
    static_cast<ListOfDeletions*>(getParentSBMLObject())->size() < 2)
  {
    unsigned int numErrs = getErrorLog()->getNumErrors();
    for (int n = numErrs-1; n >= 0; n--)      
    {
      if (getErrorLog()->getError(n)->getErrorId() == UnknownPackageAttribute)
      {
        const std::string details = 
          getErrorLog()->getError(n)->getMessage();
        getErrorLog()->remove(UnknownPackageAttribute);
        getErrorLog()->logPackageError("comp", CompLODeletionAllowedAttributes,
          getPackageVersion(), sbmlLevel, sbmlVersion, details);
      } 
      else if (getErrorLog()->getError(n)->getErrorId() == UnknownCoreAttribute)
      {
        const std::string details = 
          getErrorLog()->getError(n)->getMessage();
        getErrorLog()->remove(UnknownCoreAttribute);
        getErrorLog()->logPackageError("comp", CompLODeletionAllowedAttributes,
          getPackageVersion(), sbmlLevel, sbmlVersion, details);
      } 
    }
  }
  SBaseRef::readAttributes(attributes,expectedAttributes);

  //const unsigned int sbmlLevel   = getLevel  ();
  //const unsigned int sbmlVersion = getVersion();

  XMLTriple tripleId("id", mURI, getPrefix());
  if (attributes.readInto(tripleId, mId, getErrorLog(), 
                          false, getLine(), getColumn()))
  {
    if (mId.size() == 0)
    {
      logEmptyString("id", "<Deletion>");
    }
    else 
    {
      if (!SyntaxChecker::isValidSBMLSId(mId)) 
      {
        logInvalidId("comp:id", mId);
      }
    }
  }

  XMLTriple tripleName("name", mURI, getPrefix());
  if (attributes.readInto(tripleName, mName, getErrorLog(), false, getLine(), getColumn())) {
    if (mName.empty()) {
      logInvalidId("comp:name", mName);
    }
  }
}
/** @endcond */

/** @cond doxygenLibsbmlInternal */
void
Deletion::writeAttributes (XMLOutputStream& stream) const
{
  SBaseRef::writeAttributes(stream);

  if (isSetId())
    stream.writeAttribute("id", getPrefix(), mId);
  if (isSetName())
    stream.writeAttribute("name", getPrefix(), mName);
  //
  // (EXTENSION)
  //
  SBaseRef::writeExtensionAttributes(stream);
}
/** @endcond */

int
Deletion::getTypeCode () const
{
  return SBML_COMP_DELETION;
}

int 
Deletion::saveReferencedElement()
{
  SBMLDocument* doc = getSBMLDocument();
  SBase* listodels = getParentSBMLObject();
  ListOf* listodelslist = static_cast<ListOf*>(listodels);
  if (listodels==NULL || listodels->getTypeCode() != SBML_LIST_OF || listodelslist->getItemTypeCode() != SBML_COMP_DELETION ) {
    if (doc) {
      string error = "Unable to find referenced element in Deletion::saveReferencedElement: the deletion ";
      if (isSetId()) {
        error += "'" + getId() + "' ";
      }
      error += "has no parent list of deletions.";
      doc->getErrorLog()->logPackageError("comp", CompModelFlatteningFailed, getPackageVersion(), getLevel(), getVersion(), error, getLine(), getColumn());
    }
    return LIBSBML_OPERATION_FAILED;
  }
  SBase* submodsb = listodels->getParentSBMLObject();
  Submodel* submod = static_cast<Submodel*>(submodsb);
  if (submodsb==NULL || submod->getTypeCode() != SBML_COMP_SUBMODEL) {
    if (doc) {
      string error = "Unable to find referenced element in Deletion::saveReferencedElement: the deletion ";
      if (isSetId()) {
        error += "'" + getId() + "' ";
      }
      error += "has no parent submodel.";
      doc->getErrorLog()->logPackageError("comp", CompModelFlatteningFailed, getPackageVersion(), getLevel(), getVersion(), error, getLine(), getColumn());
    }
    return LIBSBML_OPERATION_FAILED;
  }
  Model* referencedmod = submod->getInstantiation();
  mReferencedElement = getReferencedElementFrom(referencedmod);
  if (mDirectReference==NULL) {
    mDirectReference = mReferencedElement;
  }
  if (mReferencedElement==NULL) {
    //getReferencedElementFrom will provide its own error messages.
    return LIBSBML_OPERATION_FAILED;
  }
  if (mReferencedElement->getTypeCode()==SBML_COMP_PORT) {
    mReferencedElement = static_cast<Port*>(mReferencedElement)->getReferencedElement();
  }
  if (mReferencedElement==NULL) {
    //getReferencedElementFrom will provide its own error messages.
    return LIBSBML_OPERATION_FAILED;
  }
  return LIBSBML_OPERATION_SUCCESS;
}

/** @cond doxygenLibsbmlInternal */

bool
Deletion::accept (SBMLVisitor& v) const
{
  v.visit(*this);

  if (isSetSBaseRef() == true)
  {
    getSBaseRef()->accept(v);
  }

  return true;
}

/** @endcond */


#endif /* __cplusplus */
/** @cond doxygenIgnored */
LIBSBML_EXTERN
Deletion_t *
Deletion_create(unsigned int level, unsigned int version,
                unsigned int pkgVersion)
{
  return new Deletion(level, version, pkgVersion);
}


LIBSBML_EXTERN
void
Deletion_free(Deletion_t * d)
{
  if (d != NULL)
    delete d;
}


LIBSBML_EXTERN
Deletion_t *
Deletion_clone(Deletion_t * d)
{
  if (d != NULL)
  {
    return static_cast<Deletion_t*>(d->clone());
  }
  else
  {
    return NULL;
  }
}


LIBSBML_EXTERN
char *
Deletion_getId(Deletion_t * d)
{
  if (d == NULL)
    return NULL;

  return d->getId().empty() ? NULL : safe_strdup(d->getId().c_str());
}


LIBSBML_EXTERN
char *
Deletion_getName(Deletion_t * d)
{
  if (d == NULL)
    return NULL;

  return d->getName().empty() ? NULL : safe_strdup(d->getName().c_str());
}


LIBSBML_EXTERN
int
Deletion_isSetId(Deletion_t * d)
{
  return (d != NULL) ? static_cast<int>(d->isSetId()) : 0;
}


LIBSBML_EXTERN
int
Deletion_isSetName(Deletion_t * d)
{
  return (d != NULL) ? static_cast<int>(d->isSetName()) : 0;
}


LIBSBML_EXTERN
int
Deletion_setId(Deletion_t * d, const char * id)
{
  return (d != NULL) ? d->setId(id) : LIBSBML_INVALID_OBJECT;
}


LIBSBML_EXTERN
int
Deletion_setName(Deletion_t * d, const char * name)
{
  return (d != NULL) ? d->setName(name) : LIBSBML_INVALID_OBJECT;
}


LIBSBML_EXTERN
int
Deletion_unsetId(Deletion_t * d)
{
  return (d != NULL) ? d->unsetId() : LIBSBML_INVALID_OBJECT;
}


LIBSBML_EXTERN
int
Deletion_unsetName(Deletion_t * d)
{
  return (d != NULL) ? d->unsetName() : LIBSBML_INVALID_OBJECT;
}


LIBSBML_EXTERN
int
Deletion_hasRequiredAttributes(Deletion_t * d)
{
  return (d != NULL) ? static_cast<int>(d->hasRequiredAttributes()) : 0;
}


LIBSBML_EXTERN
Deletion_t *
ListOfDeletions_getById(ListOf_t * lo, const char * sid)
{
  if (lo == NULL)
    return NULL;

  return (sid != NULL) ? static_cast <ListOfDeletions *>(lo)->get(sid) : NULL;
}


LIBSBML_EXTERN
Deletion_t *
ListOfDeletions_removeById(ListOf_t * lo, const char * sid)
{
  if (lo == NULL)
    return NULL;

  return (sid != NULL) ? static_cast <ListOfDeletions *>(lo)->remove(sid) : NULL;
}



/** @endcond */

LIBSBML_CPP_NAMESPACE_END


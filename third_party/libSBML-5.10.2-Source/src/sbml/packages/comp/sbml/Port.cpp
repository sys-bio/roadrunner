/**
 * @file    Port.cpp
 * @brief   Implementation of Port, the SBase-derived class of the comp package.
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
#include <sbml/packages/comp/sbml/Port.h>
#include <sbml/packages/comp/sbml/ListOfPorts.h>
#include <sbml/packages/comp/validator/CompSBMLError.h>

using namespace std;

LIBSBML_CPP_NAMESPACE_BEGIN
#ifdef __cplusplus

Port::Port (unsigned int level, unsigned int version, unsigned int pkgVersion) 
  : SBaseRef (level,version, pkgVersion)
  , mId("")
  , mName("")
{
}


Port::Port(CompPkgNamespaces* compns)
  : SBaseRef(compns)
  , mId("")
  , mName("")
{
  loadPlugins(compns);
}


Port::Port(const Port& source) 
  : SBaseRef (source)
{
  mId=source.mId;
  mName=source.mName;
}

Port& Port::operator=(const Port& source)
{
  if(&source!=this)
  {
    CompBase::operator=(source);
    mId=source.mId;
    mName=source.mName;
  }

  return *this;
}

Port*
Port::clone() const
{
  return new Port(*this);
}

Port::~Port ()
{
}


int
Port::setId (const std::string& id)
{
  if (!SyntaxChecker::isValidSBMLSId(id)) 
  {
    //LS DEBUG return something else
    return LIBSBML_INVALID_ATTRIBUTE_VALUE;
  }
  mId = id;
  return LIBSBML_OPERATION_SUCCESS;
}


const string&
Port::getId () const
{
  return mId;
}


bool
Port::isSetId () const
{
  return (mId.empty() == false);
}


int
Port::unsetId ()
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

int
Port::setName (const std::string& name)
{
  if (name.empty()) 
  {
    return LIBSBML_INVALID_ATTRIBUTE_VALUE;
  }
  mName = name;
  return LIBSBML_OPERATION_SUCCESS;
}


const string&
Port::getName () const
{
  return mName;
}


bool
Port::isSetName () const
{
  return (mName.empty() == false);
}


int
Port::unsetName ()
{
  mName = "";
  return LIBSBML_OPERATION_SUCCESS;
}


int 
Port::setPortRef (const std::string& id)
{
  return LIBSBML_OPERATION_FAILED;
}

bool 
Port::hasRequiredAttributes() const
{
  if (!SBaseRef::hasRequiredAttributes()) return false;
  return (isSetId());
}

const std::string&
Port::getElementName () const
{
  static const std::string name = "port";
  return name;
}


/** @cond doxygenLibsbmlInternal */
void
Port::addExpectedAttributes(ExpectedAttributes& attributes)
{
  SBaseRef::addExpectedAttributes(attributes);
  attributes.add("id");
  attributes.add("name");
}
/** @endcond */

/** @cond doxygenLibsbmlInternal */
void
Port::readAttributes (const XMLAttributes& attributes,
                      const ExpectedAttributes& expectedAttributes)
{
  const unsigned int sbmlLevel   = getLevel  ();
  const unsigned int sbmlVersion = getVersion();

  // look to see whether an unknown attribute error was logged
  // during the read of the ListOfPorts - which will have
  // happened immediately prior to this read
  if (getErrorLog() != NULL && 
    static_cast<ListOfPorts*>(getParentSBMLObject())->size() < 2)
  {
    unsigned int numErrs = getErrorLog()->getNumErrors();
    for (int n = numErrs-1; n >= 0; n--)      
    {
      if (getErrorLog()->getError(n)->getErrorId() == UnknownPackageAttribute)
      {
        const std::string details = 
          getErrorLog()->getError(n)->getMessage();
        getErrorLog()->remove(UnknownPackageAttribute);
        getErrorLog()->logPackageError("comp", CompLOPortsAllowedAttributes,
          getPackageVersion(), sbmlLevel, sbmlVersion, details);
      } 
      else if (getErrorLog()->getError(n)->getErrorId() == UnknownCoreAttribute)
      {
        const std::string details = 
          getErrorLog()->getError(n)->getMessage();
        getErrorLog()->remove(UnknownCoreAttribute);
        getErrorLog()->logPackageError("comp", CompLOPortsAllowedAttributes,
          getPackageVersion(), sbmlLevel, sbmlVersion, details);
      } 
    }
  }


  SBaseRef::readAttributes(attributes,expectedAttributes);

  if ( sbmlLevel > 2 )
  {
    XMLTriple tripleId("id", mURI, getPrefix());
    bool assigned = attributes.readInto(tripleId, mId, getErrorLog(), 
                                        false, getLine(), getColumn());
    if (assigned == false)
    {
      logMissingAttribute("id", "<Port>");
    }
    else if (assigned == true && mId.size() == 0)
    {
      logEmptyString("id", "<Port>");
    }
    else 
    {
      if (!SyntaxChecker::isValidSBMLSId(mId)) 
      {
        logInvalidId("comp:id", mId);
      }
    }

    XMLTriple tripleName("name", mURI, getPrefix());
    assigned = attributes.readInto(tripleName, mName, getErrorLog(), 
                                   false, getLine(), getColumn());
    
    if (assigned == true)
    {
      if (mName.empty()) 
      {
        logEmptyString("name", "<Port>");
      }
    }
    //LS DEBUG:  add an error if the 'port' attribute was used.
  }
}
/** @endcond */

/** @cond doxygenLibsbmlInternal */
void
Port::writeAttributes (XMLOutputStream& stream) const
{
  SBaseRef::writeAttributes(stream);

  if (isSetId()) {
    stream.writeAttribute("id", getPrefix(), mId);
  }
  if (isSetName()) {
    stream.writeAttribute("name", getPrefix(), mName);
  }
}
/** @endcond */

/** @cond doxygenLibsbmlInternal */
void
Port::writeElements (XMLOutputStream& stream) const
{
  SBaseRef::writeElements(stream);

  Port::writeExtensionElements(stream);
}
/** @endcond */


int
Port::getTypeCode () const
{
  return SBML_COMP_PORT;
}

int 
Port::saveReferencedElement()
{
  SBMLDocument* doc = getSBMLDocument();
  Model* model = CompBase::getParentModel(this);
  if (model==NULL) {
    if (doc) {
      string error = "Unable to discover referenced element: no model could be found for the given <port> element";
      if (isSetId()) {
        error += " '" + getId() + "'.";
      }
      doc->getErrorLog()->logPackageError("comp", CompModelFlatteningFailed, getPackageVersion(), getLevel(), getVersion(), error, getLine(), getColumn());
    }
    return LIBSBML_OPERATION_FAILED;
  }
  //No other error messages need to be set--'getReferencedElement*' will set them.
  mReferencedElement = getReferencedElementFrom(model);
  if (mDirectReference == NULL) {
    mDirectReference = mReferencedElement;
  }
  if (mReferencedElement==NULL) {
    return LIBSBML_OPERATION_FAILED;
  }
  if (mReferencedElement->getTypeCode()==SBML_COMP_PORT) {
    mReferencedElement = static_cast<Port*>(mReferencedElement)->getReferencedElement();
  }
  if (mReferencedElement==NULL) {
    return LIBSBML_OPERATION_FAILED;
  }
  return LIBSBML_OPERATION_SUCCESS;
}

void
Port::renameSIdRefs(const std::string& oldid, const std::string& newid)
{
  if (mIdRef==oldid) mIdRef=newid;
  SBaseRef::renameSIdRefs(oldid, newid);
}


void
Port::renameUnitSIdRefs(const std::string& oldid, const std::string& newid)
{
  if (mUnitRef==oldid) mUnitRef=newid;
  SBaseRef::renameUnitSIdRefs(oldid, newid);
}


void
Port::renameMetaIdRefs(const std::string& oldid, const std::string& newid)
{
  if (mMetaIdRef==oldid) mMetaIdRef=newid;
  SBaseRef::renameMetaIdRefs(oldid, newid);
}


/** @cond doxygenLibsbmlInternal */

bool
Port::accept (SBMLVisitor& v) const
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
Port_t *
Port_create(unsigned int level, unsigned int version,
            unsigned int pkgVersion)
{
  return new Port(level, version, pkgVersion);
}


LIBSBML_EXTERN
void
Port_free(Port_t * p)
{
  if (p != NULL)
    delete p;
}


LIBSBML_EXTERN
Port_t *
Port_clone(Port_t * p)
{
  if (p != NULL)
  {
    return static_cast<Port_t*>(p->clone());
  }
  else
  {
    return NULL;
  }
}


LIBSBML_EXTERN
char *
Port_getId(Port_t * p)
{
  if (p == NULL)
    return NULL;

  return p->getId().empty() ? NULL : safe_strdup(p->getId().c_str());
}


LIBSBML_EXTERN
char *
Port_getName(Port_t * p)
{
  if (p == NULL)
    return NULL;

  return p->getName().empty() ? NULL : safe_strdup(p->getName().c_str());
}


LIBSBML_EXTERN
int
Port_isSetId(Port_t * p)
{
  return (p != NULL) ? static_cast<int>(p->isSetId()) : 0;
}


LIBSBML_EXTERN
int
Port_isSetName(Port_t * p)
{
  return (p != NULL) ? static_cast<int>(p->isSetName()) : 0;
}


LIBSBML_EXTERN
int
Port_setId(Port_t * p, const char * id)
{
  return (p != NULL) ? p->setId(id) : LIBSBML_INVALID_OBJECT;
}


LIBSBML_EXTERN
int
Port_setName(Port_t * p, const char * name)
{
  return (p != NULL) ? p->setName(name) : LIBSBML_INVALID_OBJECT;
}


LIBSBML_EXTERN
int
Port_unsetId(Port_t * p)
{
  return (p != NULL) ? p->unsetId() : LIBSBML_INVALID_OBJECT;
}


LIBSBML_EXTERN
int
Port_unsetName(Port_t * p)
{
  return (p != NULL) ? p->unsetName() : LIBSBML_INVALID_OBJECT;
}


LIBSBML_EXTERN
int
Port_hasRequiredAttributes(Port_t * p)
{
  return (p != NULL) ? static_cast<int>(p->hasRequiredAttributes()) : 0;
}


LIBSBML_EXTERN
Port_t *
ListOfPorts_getById(ListOf_t * lo, const char * sid)
{
  if (lo == NULL)
    return NULL;

  return (sid != NULL) ? static_cast <ListOfPorts *>(lo)->get(sid) : NULL;
}


LIBSBML_EXTERN
Port_t *
ListOfPorts_removeById(ListOf_t * lo, const char * sid)
{
  if (lo == NULL)
    return NULL;

  return (sid != NULL) ? static_cast <ListOfPorts *>(lo)->remove(sid) : NULL;
}


/** @endcond */
LIBSBML_CPP_NAMESPACE_END


/**
 * @file    CompSBasePlugin.cpp
 * @brief   Implementation of CompSBasePlugin, the plugin class of
 *          comp package for the SBase element.
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

#include <ostream>

#include <sbml/common/libsbml-version.h>
#include <sbml/packages/comp/common/compfwd.h>
#include <sbml/packages/comp/extension/CompSBasePlugin.h>
#include <sbml/packages/comp/validator/CompSBMLError.h>
#include <sbml/util/ElementFilter.h>

#include <iostream>

#ifdef __cplusplus

//using namespace std;

LIBSBML_CPP_NAMESPACE_BEGIN

CompSBasePlugin::CompSBasePlugin (const std::string &uri, const std::string &prefix, CompPkgNamespaces *compns)
  : SBasePlugin(uri,prefix, compns)
  , mListOfReplacedElements(NULL)
  , mReplacedBy(NULL)
{
  connectToChild();
}


CompSBasePlugin::CompSBasePlugin(const CompSBasePlugin& orig)
  : SBasePlugin(orig)
  , mListOfReplacedElements(NULL)
  , mReplacedBy(NULL)
{
  if (orig.isSetReplacedBy()) {
    mReplacedBy = orig.mReplacedBy->clone();
    mReplacedBy->connectToParent(getParentSBMLObject());
  }
  if (orig.getNumReplacedElements() > 0) {
    createListOfReplacedElements();
    for (unsigned int re=0; re<orig.getNumReplacedElements(); re++) {
      mListOfReplacedElements->append(orig.getReplacedElement(re)->clone());
    }
  }
  connectToChild();
}


CompSBasePlugin::~CompSBasePlugin () 
{
  delete mListOfReplacedElements;
  if (isSetReplacedBy()) {
    delete mReplacedBy;
  }
}

CompSBasePlugin& 
CompSBasePlugin::operator=(const CompSBasePlugin& orig)
{
  if(&orig!=this)
  {
    SBasePlugin::operator =(orig);
    if (orig.getNumReplacedElements() > 0) {
      createListOfReplacedElements();
      for (unsigned int re=0; re<orig.getNumReplacedElements(); re++) {
        mListOfReplacedElements->append(orig.getReplacedElement(re)->clone());
      }
    }
    if (orig.mReplacedBy != NULL) {
      mReplacedBy=orig.mReplacedBy->clone();
      mReplacedBy->connectToParent(getParentSBMLObject());
    }
  }    
  return *this;
}


CompSBasePlugin* 
CompSBasePlugin::clone () const
{
  return new CompSBasePlugin(*this);  
}

/** @cond doxygenLibsbmlInternal */
SBase*
CompSBasePlugin::createObject(XMLInputStream& stream)
{
  SBase*        object = 0;

  const std::string&   name   = stream.peek().getName();
  const XMLNamespaces& xmlns  = stream.peek().getNamespaces();
  const std::string&   prefix = stream.peek().getPrefix();

  const std::string& targetPrefix = (xmlns.hasURI(mURI)) ? xmlns.getPrefix(mURI) : mPrefix;
  
  if (prefix == targetPrefix)
  {
    if ( name == "listOfReplacedElements" ) 
    {
      if (mListOfReplacedElements != NULL)
      {
        getErrorLog()->logPackageError("comp", CompOneListOfReplacedElements, 
          getPackageVersion(), getLevel(), getVersion());
      }
      createListOfReplacedElements();
      object = mListOfReplacedElements;
    
      if (targetPrefix.empty())
      {
        //
        // (NOTE)
        //
        // A top-level element (listOfReplacedElements) of the comp extension is located 
        // in a default namespace, and thus xmlns=".." attribute must be added to 
        // the element.
        // This is done by invoking SBMLDocument::enableDefaultNS() function with 
        // the two arguments (the URI of this package and true value).
        //
        mListOfReplacedElements->getSBMLDocument()->enableDefaultNS(mURI,true);
      }
    }
    if ( name == "replacedBy" ) {
      if (mReplacedBy != NULL)
      {
        if (mSBML != NULL && getErrorLog() != NULL)
        {
          getErrorLog()->logPackageError("comp", CompOneReplacedByElement, 
                          getPackageVersion(), getLevel(), getVersion());
        }
      }
      delete mReplacedBy;

      COMP_CREATE_NS(compns, getSBMLNamespaces());
      mReplacedBy = new ReplacedBy(compns);
      object = mReplacedBy;
      object->connectToParent(getParentSBMLObject());
      delete compns;
    }
  }
  return object;
}
/** @endcond */

/** @cond doxygenLibsbmlInternal */
void
CompSBasePlugin::writeElements (XMLOutputStream& stream) const
{
  if (getNumReplacedElements() > 0) {
    mListOfReplacedElements->write(stream);
  }
  if (isSetReplacedBy()) {
    mReplacedBy->write(stream);
  }
}
/** @endcond */

SBase* 
CompSBasePlugin::getElementBySId(const std::string& id)
{
  if (id.empty()) return NULL;
  SBase* obj = NULL;
  if (mListOfReplacedElements != NULL) {
    obj = mListOfReplacedElements->getElementBySId(id);
    if (obj != NULL) return obj;
  }
  if (isSetReplacedBy()) {
    obj = mReplacedBy->getElementBySId(id);
    if (obj != NULL) return obj;
  }
  return NULL;
}


SBase*
CompSBasePlugin::getElementByMetaId(const std::string& metaid)
{
  if (metaid.empty()) return NULL;
  SBase* obj = NULL;
  if (mListOfReplacedElements != NULL) {
    if (mListOfReplacedElements->getMetaId() == metaid) return mListOfReplacedElements;
    obj = mListOfReplacedElements->getElementByMetaId(metaid);
    if (obj != NULL) return obj;
  }
  if (isSetReplacedBy()) {
    if (mReplacedBy->getMetaId() == metaid) return mReplacedBy;
    obj = mReplacedBy->getElementByMetaId(metaid);
    if (obj != NULL) return obj;
  }
  return NULL;
}


List*
CompSBasePlugin::getAllElements(ElementFilter *filter)
{
  List* ret = new List();
  List* sublist = NULL;

  ADD_FILTERED_PLIST(ret, sublist,mListOfReplacedElements, filter);

  ADD_FILTERED_POINTER(ret, sublist, mReplacedBy, filter);

  return ret;
}


/** @cond doxygenLibsbmlInternal */
void
CompSBasePlugin::addExpectedAttributes(ExpectedAttributes& attributes)
{
  SBasePlugin::addExpectedAttributes(attributes);
}
/** @endcond */

/** @cond doxygenLibsbmlInternal */
void 
CompSBasePlugin::readAttributes (const XMLAttributes& attributes,
                                 const ExpectedAttributes& expectedAttributes)
{
  SBasePlugin::readAttributes(attributes, expectedAttributes);
}
/** @endcond */

/** @cond doxygenLibsbmlInternal */
void 
CompSBasePlugin::writeAttributes (XMLOutputStream& stream) const
{
  SBasePlugin::writeAttributes(stream);
}
/** @endcond */

const ListOfReplacedElements*
CompSBasePlugin::getListOfReplacedElements () const
{
  return mListOfReplacedElements;
}


/*
 * Remove the replacedElement with the given index.
 * A pointer to the removed replacedElement is returned.
 * If no replacedElement has been removed, @c NULL is returned.
 */
ReplacedElement*
CompSBasePlugin::removeReplacedElement(unsigned int index)
{
  if (mListOfReplacedElements==NULL) return NULL;
  return mListOfReplacedElements->remove(index);
}


/*
 * Returns the replacedElement with the given index.
 * If the index is invalid, @c NULL is returned.
 */ 
ReplacedElement* 
CompSBasePlugin::getReplacedElement (unsigned int index)
{
  if (mListOfReplacedElements==NULL) return NULL;
  return mListOfReplacedElements->get(index);
}

/*
 * Returns the replacedElement with the given index.
 * If the index is invalid, @c NULL is returned.
 */ 
const ReplacedElement* 
CompSBasePlugin::getReplacedElement (unsigned int index) const
{
  if (mListOfReplacedElements==NULL) return NULL;
  return mListOfReplacedElements->get(index);
}


int
CompSBasePlugin::addReplacedElement (const ReplacedElement* replacedElement)
{
  if (replacedElement == NULL)
  {
    return LIBSBML_INVALID_OBJECT;
  }
  else if (!(replacedElement->hasRequiredAttributes()) || !(replacedElement->hasRequiredElements()))
  {
    return LIBSBML_INVALID_OBJECT;
  }
  else if (getLevel() != replacedElement->getLevel())
  {
    return LIBSBML_LEVEL_MISMATCH;
  }
  else if (getVersion() != replacedElement->getVersion())
  {
    return LIBSBML_VERSION_MISMATCH;
  }
  else if (getPackageVersion() != replacedElement->getPackageVersion())
  {
    return LIBSBML_PKG_VERSION_MISMATCH;
  }
  else
  {
    createListOfReplacedElements();
    return mListOfReplacedElements->append(replacedElement);
  }
}


unsigned int
CompSBasePlugin::getNumReplacedElements () const
{
  if (mListOfReplacedElements==NULL) return 0;
  return mListOfReplacedElements->size();
}

void 
CompSBasePlugin::clearReplacedElements()
{
  if (mListOfReplacedElements!=NULL) {
    mListOfReplacedElements->clear();
  }
}
  
/*
 * Creates a ReplacedElement object, adds it to the end of the replacedElement
 * objects list and returns a reference to the newly created object.
 */
ReplacedElement*
CompSBasePlugin::createReplacedElement ()
{
  createListOfReplacedElements();
  COMP_CREATE_NS(compns, getSBMLNamespaces());
  ReplacedElement* m = new ReplacedElement(compns);
  mListOfReplacedElements->appendAndOwn(m);
  delete compns;
  return m;
}


const ReplacedBy*
CompSBasePlugin::getReplacedBy () const
{
  return mReplacedBy;
}


ReplacedBy*
CompSBasePlugin::getReplacedBy ()
{
  return mReplacedBy;
}

bool
CompSBasePlugin::isSetReplacedBy () const
{
  return (mReplacedBy != NULL);
}


int
CompSBasePlugin::setReplacedBy (const ReplacedBy* replacedBy)
{
  if (mReplacedBy == replacedBy) 
  {
    return LIBSBML_OPERATION_SUCCESS;
  }
  else if (replacedBy == NULL)
  {
    if (mReplacedBy != NULL) delete mReplacedBy;
    mReplacedBy = NULL;
    return LIBSBML_OPERATION_SUCCESS;
  }
  else if (!replacedBy->hasRequiredAttributes() || !replacedBy->hasRequiredElements()) {
    return LIBSBML_INVALID_OBJECT;
  }
  else if (getLevel() != replacedBy->getLevel())
  {
    return LIBSBML_LEVEL_MISMATCH;
  }
  else if (getVersion() != replacedBy->getVersion())
  {
    return LIBSBML_VERSION_MISMATCH;
  }
  else if (getPackageVersion() != replacedBy->getPackageVersion())
  {
    return LIBSBML_PKG_VERSION_MISMATCH;
  }
  else
  {
    if (mReplacedBy != NULL) delete mReplacedBy;
    mReplacedBy = replacedBy->clone();
    if (mReplacedBy == NULL) {
      return LIBSBML_OPERATION_FAILED;
    }
    mReplacedBy->connectToParent(getParentSBMLObject());
    return LIBSBML_OPERATION_SUCCESS;
  }
}

ReplacedBy*
CompSBasePlugin::createReplacedBy()
{
  if (mReplacedBy!=NULL) {
    delete mReplacedBy;
  }
  COMP_CREATE_NS(compns, getSBMLNamespaces());
  mReplacedBy = new ReplacedBy(compns);
  mReplacedBy->connectToParent(getParentSBMLObject());
  delete compns;
  return mReplacedBy;
}

int
CompSBasePlugin::unsetReplacedBy ()
{
  delete mReplacedBy;
  mReplacedBy = NULL;

  if (mReplacedBy == NULL) 
  {
    return LIBSBML_OPERATION_SUCCESS;
  }
  else
  {
    return LIBSBML_OPERATION_FAILED;
  }
}

void 
CompSBasePlugin::logInvalidId(const std::string& attribute,
                              const std::string& wrongattribute)
{

  if (&attribute == NULL || &wrongattribute == NULL) return;
  bool knownelement = (getParentSBMLObject() == NULL);
  std::ostringstream msg;

  msg << "Setting the attribute '" << attribute << "' ";
  if (knownelement) {
    msg << "of a <" << getParentSBMLObject()->getElementName() << "> ";
  }
  msg << "in the " << getPackageName() 
      << " package (version " << getPackageVersion() << ") to '" << wrongattribute
      << "' is illegal:  the string is not a well-formed SId.";

  SBMLErrorLog* errlog = getErrorLog();
  if (errlog)
  {
    errlog->logError(NotSchemaConformant, getLevel(), getVersion(), msg.str());
  }
}

/** @cond doxygenLibsbmlInternal */
void 
CompSBasePlugin::setSBMLDocument (SBMLDocument* d)
{
  SBasePlugin::setSBMLDocument(d);

  if (mListOfReplacedElements != NULL) {
    mListOfReplacedElements->setSBMLDocument(d);  
  }
  if (isSetReplacedBy()) {
    mReplacedBy->setSBMLDocument(d);
  }
}
/** @endcond */


/** @cond doxygenLibsbmlInternal */
void
CompSBasePlugin::connectToChild()
{
//  SBasePlugin::connectToChild();
  connectToParent(this->getParentSBMLObject());
}
/** @endcond */


/** @cond doxygenLibsbmlInternal */
void
CompSBasePlugin::connectToParent(SBase* sbase)
{
  SBasePlugin::connectToParent(sbase);
  if (mListOfReplacedElements!=NULL) {
    mListOfReplacedElements->connectToParent(sbase);
  }
  if (isSetReplacedBy()) {
    mReplacedBy->connectToParent(sbase);
  }
}
/** @endcond */



/** @cond doxygenLibsbmlInternal */
void
CompSBasePlugin::enablePackageInternal(const std::string& pkgURI,
                                       const std::string& pkgPrefix, bool flag)
{
  if (mListOfReplacedElements != NULL) {
    mListOfReplacedElements->enablePackageInternal(pkgURI,pkgPrefix,flag);
  }
  if (isSetReplacedBy()) {
    mReplacedBy->enablePackageInternal(pkgURI, pkgPrefix, flag);
  }
}
/** @endcond */


void
CompSBasePlugin::createListOfReplacedElements()
{
  if (mListOfReplacedElements==NULL) {
    COMP_CREATE_NS(compns, getSBMLNamespaces());
    mListOfReplacedElements = new ListOfReplacedElements(compns);
    mListOfReplacedElements->connectToParent(getParentSBMLObject());
    delete compns;
  }
}

/** @cond doxygenLibsbmlInternal */

bool 
CompSBasePlugin::accept(SBMLVisitor& v) const
{
  for (unsigned int i = 0; i < getNumReplacedElements(); i++)
  {
    getReplacedElement(i)->accept(v);
  }

  if (getReplacedBy() != NULL)
  {
    getReplacedBy()->accept(v);
  }

  return true;
}

/** @endcond */




LIBSBML_CPP_NAMESPACE_END

#endif  /* __cplusplus */

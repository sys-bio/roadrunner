/**
 * @file    CompSBMLDocumentPlugin.cpp
 * @brief   Implementation of CompSBMLDocumentPlugin, the plugin class of
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
#include <sbml/packages/comp/util/SBMLResolverRegistry.h>
#include <sbml/packages/comp/util/SBMLUri.h>
#include <sbml/packages/comp/extension/CompSBMLDocumentPlugin.h>
#include <sbml/packages/comp/extension/CompModelPlugin.h>
#include <sbml/packages/comp/validator/CompUnitConsistencyValidator.h>
#include <sbml/packages/comp/validator/CompIdentifierConsistencyValidator.h>
#include <sbml/packages/comp/validator/CompConsistencyValidator.h>
#include <sbml/packages/comp/validator/CompValidator.h>
#include <sbml/packages/comp/validator/CompSBMLError.h>
#include <sbml/packages/comp/util/SBMLResolverRegistry.h>
#include <sbml/conversion/SBMLConverterRegistry.h>
#include <sbml/packages/comp/util/SBMLUri.h>

#include <sbml/util/ElementFilter.h>

#include <iostream>


using namespace std;

LIBSBML_CPP_NAMESPACE_BEGIN
#ifdef __cplusplus

CompSBMLDocumentPlugin::CompSBMLDocumentPlugin (const string &uri, const string &prefix, CompPkgNamespaces *compns)
  : SBMLDocumentPlugin(uri,prefix, compns)
  , mListOfModelDefinitions(compns)
  , mListOfExternalModelDefinitions(compns)
  , mURIToDocumentMap()
  , mCheckingDummyDoc (false)
  , mFlattenAndCheck (true)
  , mOverrideCompFlattening (false)
{
  connectToChild();
}


CompSBMLDocumentPlugin::CompSBMLDocumentPlugin(const CompSBMLDocumentPlugin& orig)
  : SBMLDocumentPlugin(orig)
  , mListOfModelDefinitions(orig.mListOfModelDefinitions)
  , mListOfExternalModelDefinitions(orig.mListOfExternalModelDefinitions)
  , mURIToDocumentMap() //The documents are owning pointers, so don't copy them.
  , mCheckingDummyDoc (orig.mCheckingDummyDoc)
  , mFlattenAndCheck (orig.mFlattenAndCheck)
  , mOverrideCompFlattening (orig.mOverrideCompFlattening)
{
  connectToChild();
}


CompSBMLDocumentPlugin& 
CompSBMLDocumentPlugin::operator=(const CompSBMLDocumentPlugin& orig)
{
  if(&orig!=this)
  {
    SBMLDocumentPlugin::operator =(orig);
    mListOfModelDefinitions = orig.mListOfModelDefinitions;
    mListOfExternalModelDefinitions = orig.mListOfExternalModelDefinitions;
    mURIToDocumentMap.clear(); //Don't copy the pointers to this object, as they are owning pointers

    mCheckingDummyDoc = orig.mCheckingDummyDoc;
    mFlattenAndCheck = orig.mFlattenAndCheck;
    mOverrideCompFlattening = orig.mOverrideCompFlattening;
    connectToChild();
  }    
  return *this;
}


/*
 * Creates and returns a deep copy of this CompSBMLDocumentPlugin object.
 */
CompSBMLDocumentPlugin* 
CompSBMLDocumentPlugin::clone () const
{
  return new CompSBMLDocumentPlugin(*this);  
}

/*
 * Destroy this object.
 */
CompSBMLDocumentPlugin::~CompSBMLDocumentPlugin () 
{
  clearStoredURIDocuments();
}


/** @cond doxygenLibsbmlInternal */
SBase*
CompSBMLDocumentPlugin::createObject(XMLInputStream& stream)
{
  SBase*        object = 0;

  const string&   name   = stream.peek().getName();
  const XMLNamespaces& xmlns  = stream.peek().getNamespaces();
  const string&   prefix = stream.peek().getPrefix();

  const string& targetPrefix = (xmlns.hasURI(mURI)) ? xmlns.getPrefix(mURI) : mPrefix;
  
  if (prefix == targetPrefix)
  {
    if ( name == "listOfModelDefinitions" ) 
    {
      if (mListOfModelDefinitions.size() != 0)
      {
        getErrorLog()->logPackageError("comp", CompOneListOfModelDefinitions, 
          getPackageVersion(), getLevel(), getVersion());
      }
     
      object = &mListOfModelDefinitions;
    
      if (targetPrefix.empty())
      {
        //
        // (NOTE)
        //
        // A top-level element (listOfModelDefinitions) of the comp extension is located 
        // in a default namespace, and thus xmlns=".." attribute must be added to 
        // the element.
        // This is done by invoking SBMLDocument::enableDefaultNS() function with 
        // the two arguments (the URI of this package and true value).
        //
        mListOfModelDefinitions.getSBMLDocument()->enableDefaultNS(mURI,true);
      }
    }          
    if ( name == "listOfExternalModelDefinitions" ) 
    {
      if (mListOfExternalModelDefinitions.size() != 0)
      {
        getErrorLog()->logPackageError("comp", CompOneListOfExtModelDefinitions, 
          getPackageVersion(), getLevel(), getVersion());
      }
     
      object = &mListOfExternalModelDefinitions;
    
      if (targetPrefix.empty())
      {
        //
        // (NOTE)
        //
        // A top-level element (listOfExternalModelDefinitions) of the comp extension is located 
        // in a default namespace, and thus xmlns=".." attribute must be added to 
        // the element.
        // This is done by invoking SBMLDocument::enableDefaultNS() function with 
        // the two arguments (the URI of this package and true value).
        //
        mListOfExternalModelDefinitions.getSBMLDocument()->enableDefaultNS(mURI,true);
      }
    }          
  }
  return object;
}
/** @endcond */

/** @cond doxygenLibsbmlInternal */
void
CompSBMLDocumentPlugin::writeElements (XMLOutputStream& stream) const
{
  if (getNumModelDefinitions() > 0)
  {
    mListOfModelDefinitions.write(stream);
  }    
  if (getNumExternalModelDefinitions() > 0)
  {
    mListOfExternalModelDefinitions.write(stream);
  }    
}
/** @endcond */

SBase* 
CompSBMLDocumentPlugin::getElementBySId(const std::string& id)
{
  if (id.empty()) return NULL;
  SBase* obj = mListOfModelDefinitions.getElementBySId(id);
  if (obj != NULL) return obj;
  obj = mListOfExternalModelDefinitions.getElementBySId(id);
  if (obj != NULL) return obj;
  return NULL;
}


SBase*
CompSBMLDocumentPlugin::getElementByMetaId(const std::string& metaid)
{
  if (metaid.empty()) return NULL;
  if (mListOfModelDefinitions.getMetaId() == metaid) return &mListOfModelDefinitions;
  if (mListOfExternalModelDefinitions.getMetaId() == metaid) return &mListOfExternalModelDefinitions;

  SBase* obj = mListOfModelDefinitions.getElementByMetaId(metaid);
  if (obj != NULL) return obj;
  obj = mListOfExternalModelDefinitions.getElementByMetaId(metaid);
  if (obj != NULL) return obj;
  return NULL;
}


List*
CompSBMLDocumentPlugin::getAllElements(ElementFilter *filter)
{
  List* ret = new List();
  List* sublist = NULL;

  ADD_FILTERED_LIST(ret, sublist, mListOfModelDefinitions, filter);
  ADD_FILTERED_LIST(ret, sublist, mListOfExternalModelDefinitions, filter);

  return ret;
}


/** @cond doxygenLibsbmlInternal */
void
CompSBMLDocumentPlugin::addExpectedAttributes(ExpectedAttributes& attributes)
{
  //
  // required attribute is defined for SBML Level 3 (and higher?).
  //
  if ( mSBMLExt->getLevel(mURI) > 2 )
  {    
    attributes.add("required");
  }
}
/** @endcond */

/** @cond doxygenLibsbmlInternal */
void 
CompSBMLDocumentPlugin::readAttributes (const XMLAttributes& attributes,
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
      getErrorLog()->logPackageError("comp", CompAttributeRequiredMustBeBoolean,
        getPackageVersion(), getLevel(), getVersion());
    }
    else
    {
      getErrorLog()->logPackageError("comp", CompAttributeRequiredMissing,
        getPackageVersion(), getLevel(), getVersion());
    }
  }
  else
  {
    mIsSetRequired = true;
    if (mRequired == false) 
    {
      getErrorLog()->logPackageError("comp", CompAttributeRequiredMustBeTrue,
        getPackageVersion(), getLevel(), getVersion());
    }
  }
}
/** @endcond */


const ListOfModelDefinitions*
CompSBMLDocumentPlugin::getListOfModelDefinitions () const
{
  return &mListOfModelDefinitions;
}


/*
 * Returns the modelDefinition with the given index.
 * If the index is invalid, @c NULL is returned.
 */ 
ModelDefinition* 
CompSBMLDocumentPlugin::getModelDefinition (unsigned int index)
{
  return mListOfModelDefinitions.get(index);
}

/*
 * Returns the modelDefinition with the given index.
 * If the index is invalid, @c NULL is returned.
 */ 
const ModelDefinition* 
CompSBMLDocumentPlugin::getModelDefinition (unsigned int index) const
{
  return mListOfModelDefinitions.get(index);
}

/*
 * Returns the ModelDefinition object based on its identifier.
 */
ModelDefinition*
CompSBMLDocumentPlugin::getModelDefinition(const string& sid)
{
  return static_cast<ModelDefinition*>( mListOfModelDefinitions.get(sid) );
}


/*
 * Returns the ModelDefinition object based on its identifier.
 */
const ModelDefinition*
CompSBMLDocumentPlugin::getModelDefinition (const string& sid) const
{
  return static_cast<const ModelDefinition*>( mListOfModelDefinitions.get(sid) );
}


/*
 * Adds a modelDefinition element
 */
int
CompSBMLDocumentPlugin::addModelDefinition (const ModelDefinition* modelDefinition)
{
  if (modelDefinition == NULL)
  {
    return LIBSBML_INVALID_OBJECT;
  }
  else if (!(modelDefinition->hasRequiredAttributes()) || !(modelDefinition->hasRequiredElements()))
  {
    return LIBSBML_INVALID_OBJECT;
  }
  else if (getLevel() != modelDefinition->getLevel())
  {
    return LIBSBML_LEVEL_MISMATCH;
  }
  else if (getVersion() != modelDefinition->getVersion())
  {
    return LIBSBML_VERSION_MISMATCH;
  }
  else if (getPackageVersion() != modelDefinition->getPackageVersion())
  {
    return LIBSBML_PKG_VERSION_MISMATCH;
  }
  else
  {
    return mListOfModelDefinitions.append(modelDefinition);
  }
}


unsigned int
CompSBMLDocumentPlugin::getNumModelDefinitions () const
{
  return mListOfModelDefinitions.size();
}


/*
 * Creates a ModelDefinition object, adds it to the end of the modelDefinition
 * objects list and returns a reference to the newly created object.
 */
ModelDefinition*
CompSBMLDocumentPlugin::createModelDefinition ()
{
  COMP_CREATE_NS(compns, getSBMLNamespaces());
  ModelDefinition* m = new ModelDefinition(compns);
  mListOfModelDefinitions.appendAndOwn(m);
  delete compns;
  return m;
}


/*
 * Remove the modelDefinition with the given index.
 * A pointer to the removed modelDefinition is returned.
 * If no modelDefinition has been removed, @c NULL is returned.
 */
ModelDefinition*
CompSBMLDocumentPlugin::removeModelDefinition(unsigned int index)
{
  return mListOfModelDefinitions.remove(index);
}

/*
 * Removes the modelDefinition with the given @p id.
 * A pointer to the modelDefinition that was removed is returned.
 * If no modelDefinition has been removed, @c NULL is returned.
 */
ModelDefinition*
CompSBMLDocumentPlugin::removeModelDefinition(string id)
{
  return mListOfModelDefinitions.remove( id);
}


/*
 * Returns the externalModelDefinition object that holds all externalModelDefinitions.
 */ 
const ListOfExternalModelDefinitions*
CompSBMLDocumentPlugin::getListOfExternalModelDefinitions () const
{
  return &mListOfExternalModelDefinitions;
}


/*
 * Returns the externalModelDefinition with the given index.
 * If the index is invalid, @c NULL is returned.
 */ 
ExternalModelDefinition* 
CompSBMLDocumentPlugin::getExternalModelDefinition (unsigned int index)
{
  return mListOfExternalModelDefinitions.get(index);
}

/*
 * Returns the externalModelDefinition with the given index.
 * If the index is invalid, @c NULL is returned.
 */ 
const ExternalModelDefinition* 
CompSBMLDocumentPlugin::getExternalModelDefinition (unsigned int index) const
{
  return mListOfExternalModelDefinitions.get(index);
}

/*
 * Returns the ExternalModelDefinition object based on its identifier.
 */
ExternalModelDefinition*
CompSBMLDocumentPlugin::getExternalModelDefinition(const string& sid)
{
  return mListOfExternalModelDefinitions.get(sid);
}


/*
 * Returns the ExternalModelDefinition object based on its identifier.
 */
const ExternalModelDefinition*
CompSBMLDocumentPlugin::getExternalModelDefinition (const string& sid) const
{
  return mListOfExternalModelDefinitions.get(sid);
}


/*
 * Adds a externalModelDefinition element
 */
int
CompSBMLDocumentPlugin::addExternalModelDefinition (const ExternalModelDefinition* externalModelDefinition)
{
  if (externalModelDefinition == NULL)
  {
    return LIBSBML_INVALID_OBJECT;
  }
  else if (!(externalModelDefinition->hasRequiredAttributes()) || !(externalModelDefinition->hasRequiredElements()))
  {
    return LIBSBML_INVALID_OBJECT;
  }
  else if (getLevel() != externalModelDefinition->getLevel())
  {
    return LIBSBML_LEVEL_MISMATCH;
  }
  else if (getVersion() != externalModelDefinition->getVersion())
  {
    return LIBSBML_VERSION_MISMATCH;
  }
  else if (getPackageVersion() != externalModelDefinition->getPackageVersion())
  {
    return LIBSBML_PKG_VERSION_MISMATCH;
  }
  else
  {
    return mListOfExternalModelDefinitions.append(externalModelDefinition);
  }
}


/*
 * Returns the number of externalModelDefinitions.
 */
unsigned int
CompSBMLDocumentPlugin::getNumExternalModelDefinitions () const
{
  return mListOfExternalModelDefinitions.size();
}


/*
 * Creates a ExternalModelDefinition object, adds it to the end of the externalModelDefinition
 * objects list and returns a reference to the newly created object.
 */
ExternalModelDefinition*
CompSBMLDocumentPlugin::createExternalModelDefinition ()
{
  COMP_CREATE_NS(compns, getSBMLNamespaces());
  ExternalModelDefinition* m = new ExternalModelDefinition(compns);
  mListOfExternalModelDefinitions.appendAndOwn(m);
  delete compns;
  return m;
}


/*
 * Removes the externalModelDefinition with the given @p id.
 * A pointer to the externalModelDefinition that was removed is returned.
 * If no externalModelDefinition has been removed, @c NULL is returned.
 */
ExternalModelDefinition*
CompSBMLDocumentPlugin::removeExternalModelDefinition(string id)
{
  return mListOfExternalModelDefinitions.remove(id);
}

/*
 * Remove the externalModelDefinition with the given index.
 * A pointer to the removed externalModelDefinition is returned.
 * If no externalModelDefinition has been removed, @c NULL is returned.
 */
ExternalModelDefinition*
CompSBMLDocumentPlugin::removeExternalModelDefinition(unsigned int index)
{
  return mListOfExternalModelDefinitions.remove(index);
}


/*
 * Gets the model with the given sid, whether from <model>, <listOfModelDefinitions>, or <externalModelDefinitions>
 */
const SBase* CompSBMLDocumentPlugin::getModel(const string& sid) const
{
  const SBase* ret = getSBMLDocument()->getModel();
  if (ret->getId() == sid) return ret;
  ret = getModelDefinition(sid);
  if (ret!=NULL) return ret;
  ret = getExternalModelDefinition(sid);
  return ret;
}


/*
 * Gets the model with the given sid, whether from <model>, <listOfModelDefinitions>, or <externalModelDefinitions>
 */
SBase* CompSBMLDocumentPlugin::getModel(const string& sid)
{
  SBMLDocument* doc = getSBMLDocument();
  if (doc==NULL) return NULL;
  SBase* ret = getSBMLDocument()->getModel();
  if (ret != NULL && ret->getId() == sid) return ret;
  ret = getModelDefinition(sid);
  if (ret!=NULL) return ret;
  ret = getExternalModelDefinition(sid);
  return ret;
}
  
  
int
CompSBMLDocumentPlugin::setRequired(bool required)
{
  //
  // required attribute is not defined for SBML Level 2 .
  //
  if ( getLevel()  < 3) {
    return LIBSBML_UNEXPECTED_ATTRIBUTE;
  }

  if (required==false) {
    return LIBSBML_INVALID_ATTRIBUTE_VALUE;
  }

  mRequired = required;
  mIsSetRequired = true;
  return LIBSBML_OPERATION_SUCCESS;
}


/** @cond doxygenLibsbmlInternal */
void 
CompSBMLDocumentPlugin::setSBMLDocument (SBMLDocument* d)
{
  SBMLDocumentPlugin::setSBMLDocument(d);

  mListOfModelDefinitions.setSBMLDocument(d);  
  mListOfExternalModelDefinitions.setSBMLDocument(d);  
}
/** @endcond */


/** @cond doxygenLibsbmlInternal */
void
CompSBMLDocumentPlugin::connectToChild()
{
  connectToParent(this->getParentSBMLObject());
}
/** @endcond */


/** @cond doxygenLibsbmlInternal */
void
CompSBMLDocumentPlugin::connectToParent (SBase* sbase)
{
  SBMLDocumentPlugin::connectToParent(sbase);
  mListOfModelDefinitions.connectToParent(sbase);
  mListOfExternalModelDefinitions.connectToParent(sbase);
}
/** @endcond */


/** @cond doxygenLibsbmlInternal */
/*
 * Enables/Disables the given package with child elements in this plugin
 * object (if any).
 */
void
CompSBMLDocumentPlugin::enablePackageInternal(const string& pkgURI,
                                              const string& pkgPrefix, bool flag)
{
  mListOfModelDefinitions.enablePackageInternal(pkgURI,pkgPrefix,flag);
  mListOfExternalModelDefinitions.enablePackageInternal(pkgURI,pkgPrefix,flag);
}
/** @endcond */


SBMLDocument* 
CompSBMLDocumentPlugin::getSBMLDocumentFromURI(const std::string& uri)
{
  const SBMLResolverRegistry& registry = SBMLResolverRegistry::getInstance();

  SBMLUri* resolved = registry.resolveUri(uri, mSBML->getLocationURI());
  if (resolved == NULL )
    return NULL;

  string resolvedURI = resolved->getUri();
  delete resolved;

  map<string, SBMLDocument*>::iterator stored = mURIToDocumentMap.find(resolvedURI);
  if (stored == mURIToDocumentMap.end()) 
  {
    SBMLDocument* doc = registry.resolve(uri, mSBML->getLocationURI());

    if (doc==NULL) 
      return NULL;

    mURIToDocumentMap.insert(make_pair(resolvedURI, doc));
    doc->setLocationURI(resolvedURI);

    return doc;
  }
  return stored->second;
}

std::string 
  CompSBMLDocumentPlugin::getResolvedURI(const string& sUri)
{
  std::string baseUri; 
  if (mSBML != NULL)
    baseUri = mSBML->getLocationURI();

  SBMLUri* uri = SBMLResolverRegistry::getInstance().resolveUri(sUri, baseUri);
  const std::string resolvedUri = (uri == NULL ? "" : uri->getUri());
  delete uri;
  
  return resolvedUri;
}

void 
CompSBMLDocumentPlugin::clearStoredURIDocuments()
{
  for (map<string, SBMLDocument*>::iterator mi = mURIToDocumentMap.begin(); mi != mURIToDocumentMap.end(); mi++) {
    delete mi->second;
  }
  mURIToDocumentMap.clear();
}

/** @cond doxygenLibsbmlInternal */
bool
CompSBMLDocumentPlugin::isCompFlatteningImplemented() const
{
  return true;
}
/** @endcond */


/** @cond doxygenLibsbmlInternal */
unsigned int 
CompSBMLDocumentPlugin::checkConsistency()
{
  unsigned int nerrors = 0;
  unsigned int total_errors = 0;

  SBMLDocument* doc = static_cast<SBMLDocument *>(this->getParentSBMLObject());

  /* just in case */
  if (doc == NULL)
  {
    return total_errors;
  }

  /* 
   * note number of errors before we do anything here 
   * so we get the right number after
   * total_errors here means the total number logged by this function
   */

  unsigned int errorsB4 = doc->getErrorLog()->getNumErrors();

  SBMLErrorLog *log = doc->getErrorLog();
  total_errors = log->getNumErrors() - errorsB4;

  /* log a message to say not to trust line numbers 
   * but only do this if we are actually logging errors
   * use boolean to ensure we only log it once
   */
  bool lineNumMessageLogged = false;

  unsigned char applicableValidators = doc->getApplicableValidators();

  /* determine which validators to run */
  bool id    = ((applicableValidators & 0x01) == 0x01);
  bool sbml  = ((applicableValidators & 0x02) == 0x02);
  bool units = ((applicableValidators & 0x10) == 0x10);

  CompIdentifierConsistencyValidator id_validator;
  CompConsistencyValidator validator;
  CompUnitConsistencyValidator unit_validator;

  if (id)
  {
    id_validator.init();
    nerrors = id_validator.validate(*doc);
    total_errors += nerrors;
    if (nerrors > 0) 
    {
      /* log a message to say not to trust line numbers 
       * but only do this if we are actually logging errors
       * and only do it once
       */
      if (lineNumMessageLogged == false
          && log->contains(CompLineNumbersUnreliable) == false)
      {
        log->logPackageError("comp", CompLineNumbersUnreliable, 
          getPackageVersion(), getLevel(), getVersion());
        total_errors++;
        lineNumMessageLogged = true;
      }

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
      /* log a message to say not to trust line numbers 
       * but only do this if we are actually logging errors
       * and only do it once
       */
      if (lineNumMessageLogged == false
          && log->contains(CompLineNumbersUnreliable) == false)
      {
        log->logPackageError("comp", CompLineNumbersUnreliable, 
          getPackageVersion(), getLevel(), getVersion());
        total_errors++;
        lineNumMessageLogged = true;
      }

      log->add( validator.getFailures() );

      /* only want to bail if errors not warnings */
      if (log->getNumFailsWithSeverity(LIBSBML_SEV_ERROR) > 0)
      {
        return total_errors;
      }
    }
  }

  if (units)
  {
    unit_validator.init();
    nerrors = unit_validator.validate(*doc);
    total_errors += nerrors;
    if (nerrors > 0) 
    {
      /* log a message to say not to trust line numbers 
       * but only do this if we are actually logging errors
       * and only do it once
       */
      if (lineNumMessageLogged == false
          && log->contains(CompLineNumbersUnreliable) == false)
      {
        log->logPackageError("comp", CompLineNumbersUnreliable, 
          getPackageVersion(), getLevel(), getVersion());
        total_errors++;
        lineNumMessageLogged = true;
      }

      log->add( unit_validator.getFailures() );

      /* only want to bail if errors not warnings */
      if (log->getNumFailsWithSeverity(LIBSBML_SEV_ERROR) > 0)
      {
        return total_errors;
      }
    }
  }

  if (mCheckingDummyDoc == false)
  {
    unsigned int numMD = getNumModelDefinitions();

    /* create dummy documents for the modelDefinitions */
    for (unsigned int i = 0; i < numMD; i++)
    {
      mCheckingDummyDoc = true;
      mFlattenAndCheck = false;
      SBMLDocument * dummyDoc = doc->clone();

      /* a document clone does not clone the error log as this was deemed
       * to be a situation where you wanted an empty log
       *
       * BUT for some of the comp rules they actually need to know 
       * whether there are unrecognised packages present
       * so add these errors if they exist in the original
       */
      if (doc->getErrorLog()->contains(UnrequiredPackagePresent) == true)
      {
        dummyDoc->getErrorLog()->logError(UnrequiredPackagePresent, 
          doc->getLevel(), doc->getVersion());
      }
      if (doc->getErrorLog()->contains(RequiredPackagePresent) == true)
      {
        dummyDoc->getErrorLog()->logError(RequiredPackagePresent, 
          doc->getLevel(), doc->getVersion());
      }
      
      const Model * dummyModel = doc->getModel();
      
      CompSBMLDocumentPlugin * dummyPlugin = 
        static_cast<CompSBMLDocumentPlugin*>(dummyDoc->getPlugin("comp"));

      /* now swap the existing model with the modeldefinition */
      ModelDefinition dummyDef(*dummyModel);
      dummyPlugin->addModelDefinition(&dummyDef);

      dummyDoc->setModel(getModelDefinition(i));

      dummyPlugin->removeModelDefinition(i);


      nerrors = dummyDoc->checkConsistency();

      /* remove the unknown package errors as these will just get relogged
       */
      if (dummyDoc->getErrorLog()->contains(UnrequiredPackagePresent) == true)
      {
        dummyDoc->getErrorLog()->remove(UnrequiredPackagePresent);
      }
      if (dummyDoc->getErrorLog()->contains(RequiredPackagePresent) == true)
      {
        dummyDoc->getErrorLog()->remove(RequiredPackagePresent);
      }


      total_errors += nerrors;
      if (nerrors > 0) 
      {
        /* log a message to say not to trust line numbers 
         * but only do this if we are actually logging errors
         * and only do it once
         */
        if (lineNumMessageLogged == false 
          && log->contains(CompLineNumbersUnreliable) == false)
        {
          log->logPackageError("comp", CompLineNumbersUnreliable, 
            getPackageVersion(), getLevel(), getVersion());
          total_errors++;
          lineNumMessageLogged = true;
        }

        for (unsigned int n = 0; n < nerrors; n++)
        {
          if (dummyDoc->getErrorLog()->getError(n)->getErrorId() 
            != CompLineNumbersUnreliable)
          {
            log->add( *(dummyDoc->getErrorLog()->getError(n)) );
          }
        }

        /* only want to bail if errors not warnings */
        if (log->getNumFailsWithSeverity(LIBSBML_SEV_ERROR) > 0)
        {
          delete dummyDoc;
          return total_errors;
        }
      }

      delete dummyDoc;
      mFlattenAndCheck = true;
    }


  }



  if (mFlattenAndCheck == true && mOverrideCompFlattening == false)
  {
    SBMLDocument * dummyDoc = doc->clone();
    ConversionProperties props;
    
    props.addOption("flatten comp");
    props.addOption("performValidation", false);

    SBMLConverter* converter = 
               SBMLConverterRegistry::getInstance().getConverterFor(props);
    

    converter->setDocument(dummyDoc);
    
    int result = converter->convert();

    if (result == LIBSBML_OPERATION_SUCCESS)
    {
      nerrors = dummyDoc->checkConsistency();
      if (dummyDoc->getErrorLog()->
                              getNumFailsWithSeverity(LIBSBML_SEV_ERROR) > 0)
      {
        /* log a message to say not to trust line numbers 
         * but only do this if we are actually logging errors
         * and only do it once
         */
        if (lineNumMessageLogged == false
          && log->contains(CompLineNumbersUnreliable) == false)
        {
          log->logPackageError("comp", CompLineNumbersUnreliable, 
            getPackageVersion(), getLevel(), getVersion());
          total_errors++;
          lineNumMessageLogged = true;
        }

        std::string message = "Errors that follow relate to the flattened ";
        message += "document produced using the CompFlatteningConverter.";
        log->logPackageError("comp", CompFlatModelNotValid, 
          getPackageVersion(), getLevel(), getVersion(), message);
      }

      total_errors += nerrors;
      if (nerrors > 0) 
      {
         /* log a message to say not to trust line numbers 
         * but only do this if we are actually logging errors
         * and only do it once
         */
        if (lineNumMessageLogged == false
          && log->contains(CompLineNumbersUnreliable) == false)
        {
          log->logPackageError("comp", CompLineNumbersUnreliable, 
            getPackageVersion(), getLevel(), getVersion());
          total_errors++;
          //lineNumMessageLogged = true;
        }

        for (unsigned int n = 0; n < nerrors; n++)
        {
          if (dummyDoc->getErrorLog()->getError(n)->getErrorId() 
            != CompLineNumbersUnreliable)
          {
            log->add( *(dummyDoc->getErrorLog()->getError(n)) );
          }
        }
      }
    }
    else
    {
      nerrors = dummyDoc->getNumErrors();
      total_errors += nerrors;
      if (nerrors > 0) 
      {
        /* log a message to say not to trust line numbers 
         * but only do this if we are actually logging errors
         * and only do it once
         */
        if (lineNumMessageLogged == false
          && log->contains(CompLineNumbersUnreliable) == false)
        {
          log->logPackageError("comp", CompLineNumbersUnreliable, 
            getPackageVersion(), getLevel(), getVersion());
          total_errors++;
          //lineNumMessageLogged = true;
        }

        for (unsigned int n = 0; n < nerrors; n++)
        {
          if (dummyDoc->getErrorLog()->getError(n)->getErrorId() 
            != CompLineNumbersUnreliable)
          {
            log->add( *(dummyDoc->getErrorLog()->getError(n)) );
          }
        }
      }
    }
      
    delete dummyDoc;
  }
  return total_errors;  
}
/** @endcond */

/** @cond doxygenLibsbmlInternal */

bool 
CompSBMLDocumentPlugin::accept(SBMLVisitor& v) const
{
  const SBMLDocument *doc = static_cast<const SBMLDocument *>(this->getParentSBMLObject());
  v.visit(*doc);

  //The list of external model definitions must be checked here.
  mListOfExternalModelDefinitions.accept(v);

  v.leave(*doc);

  return true;
}

/** @endcond */


/** @cond doxygenLibsbmlInternal */

bool
CompSBMLDocumentPlugin::getOverrideCompFlattening() const
{
  return mOverrideCompFlattening;
}

/** @endcond */


/** @cond doxygenLibsbmlInternal */

void
CompSBMLDocumentPlugin::setOverrideCompFlattening(bool overrideCompFlattening)
{
  mOverrideCompFlattening = overrideCompFlattening;
}

/** @endcond */


#endif /* __cplusplus */
/** @cond doxygenIgnored */

LIBSBML_EXTERN
ModelDefinition_t *
CompSBMLDocumentPlugin_createModelDefinition(CompSBMLDocumentPlugin_t * docPlug)
{
  return docPlug->createModelDefinition();
}

/** @endcond */

LIBSBML_CPP_NAMESPACE_END


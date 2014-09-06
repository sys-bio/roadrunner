/**
 * @file    ExternalModelDefinition.cpp
 * @brief   Implementation of ExternalModelDefinition, the SBase-derived class of the comp package.
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
#include <sbml/packages/comp/validator/CompSBMLErrorTable.h>
#include <sbml/packages/comp/extension/CompSBMLDocumentPlugin.h>
#include <sbml/packages/comp/sbml/ExternalModelDefinition.h>

using namespace std;

LIBSBML_CPP_NAMESPACE_BEGIN
#ifdef __cplusplus

ExternalModelDefinition::ExternalModelDefinition (unsigned int level, unsigned int version, unsigned int pkgVersion) 
  : CompBase (level,version, pkgVersion)
  , mId("")
  , mName("")
  , mSource("")
  , mModelRef("")
  , mMd5("")
{
  // set an SBMLNamespaces derived object (CompPkgNamespaces) of this package.
  setSBMLNamespacesAndOwn(new CompPkgNamespaces(level,version,pkgVersion));  
  //getSBMLDocument()->enableDefaultNS(mXlinkURI,true);
}


ExternalModelDefinition::ExternalModelDefinition(CompPkgNamespaces* compns)
  : CompBase(compns)
  , mId("")
  , mName("")
  , mSource("")
  , mModelRef("")
  , mMd5("")
{
  // set the element namespace of this object
  setElementNamespace(compns->getURI());

  // connect child elements to this element.
  connectToChild();

  // load package extensions bound with this object (if any) 
  loadPlugins(compns);
}


ExternalModelDefinition::ExternalModelDefinition(const ExternalModelDefinition& source) : CompBase (source)
{
  mId=source.mId;
  mName=source.mName;
  mSource=source.mSource;
  mModelRef=source.mModelRef;
  mMd5=source.mMd5;

  // connect child elements to this element.
  connectToChild();
}

ExternalModelDefinition& ExternalModelDefinition::operator=(const ExternalModelDefinition& source)
{
  if(&source!=this)
  {
    ExternalModelDefinition::operator=(source);
    mId=source.mId;
    mName=source.mName;
    mSource=source.mSource;
    mModelRef=source.mModelRef;
    mMd5=source.mMd5;

    // connect child elements to this element.
    connectToChild();
  }

  return *this;
}


ExternalModelDefinition*
ExternalModelDefinition::clone() const
{
  return new ExternalModelDefinition(*this);
}


ExternalModelDefinition::~ExternalModelDefinition ()
{
}


/*
 * Returns the value of the "id" attribute of this ExternalModelDefinition.
 */
const string&
ExternalModelDefinition::getId() const
{
  return mId;
}


/*
 * Predicate returning @c true or @c false depending on whether this
 * ExternalModelDefinition's "id" attribute has been set.
 */
bool 
ExternalModelDefinition::isSetId() const
{
  return (mId.empty()==false);
}

/*
 * Sets the value of the "id" attribute of this ExternalModelDefinition.
 */
int 
ExternalModelDefinition::setId(const string& id)
{
  if (!SyntaxChecker::isValidSBMLSId(id)) {
    return LIBSBML_INVALID_ATTRIBUTE_VALUE;
  }
  mId = id;
  return LIBSBML_OPERATION_SUCCESS;
}


/*
 * Unsets the value of the "id" attribute of this ExternalModelDefinition.
 */
int 
ExternalModelDefinition::unsetId()
{
  mId = "";
  return LIBSBML_OPERATION_SUCCESS;
}



/*
 * Returns the value of the "name" attribute of this ExternalModelDefinition.
 */
const string&
ExternalModelDefinition::getName() const
{
  return mName;
}


/*
 * Predicate returning @c true or @c false depending on whether this
 * ExternalModelDefinition's "name" attribute has been set.
 */
bool 
ExternalModelDefinition::isSetName() const
{
  return (mName.empty()==false);
}

/*
 * Sets the value of the "name" attribute of this ExternalModelDefinition.
 */
int 
ExternalModelDefinition::setName(const string& name)
{
  if (name.empty()) {
    return LIBSBML_INVALID_ATTRIBUTE_VALUE;
  }
  mName = name;
  return LIBSBML_OPERATION_SUCCESS;
}


/*
 * Unsets the value of the "name" attribute of this ExternalModelDefinition.
 */
int 
ExternalModelDefinition::unsetName()
{
  mName = "";
  return LIBSBML_OPERATION_SUCCESS;
}



/*
 * Sets the modelRef of this SBML object to a copy of modelRef.
 */
int
ExternalModelDefinition::setModelRef (const std::string& modelRef)
{
  if (!SyntaxChecker::isValidSBMLSId(modelRef)) 
  {
    return LIBSBML_INVALID_ATTRIBUTE_VALUE;
  }
  mModelRef = modelRef;
  return LIBSBML_OPERATION_SUCCESS;
}


/*
 * @return the modelRef of this SBML object.
 */
const string&
ExternalModelDefinition::getModelRef () const
{
  return mModelRef;
}


/*
 * @return true if the modelRef of this SBML object has been set, false
 * otherwise.
 */
bool
ExternalModelDefinition::isSetModelRef () const
{
  return (mModelRef.empty() == false);
}


/*
 * Unsets the modelRef of this SBML object.
 */
int
ExternalModelDefinition::unsetModelRef ()
{
  mModelRef.erase();

  if (mModelRef.empty())
  {
    return LIBSBML_OPERATION_SUCCESS;
  }
  else
  {
    return LIBSBML_OPERATION_FAILED;
  }
}


/*
 * Sets the md5 of this SBML object to a copy of md5.
 */
int
ExternalModelDefinition::setMd5 (const std::string& md5)
{
  //LS DEBUG:  Check format of md5 to ensure validity?
  mMd5 = md5;
  return LIBSBML_OPERATION_SUCCESS;
}


/*
 * @return the md5 of this SBML object.
 */
const string&
ExternalModelDefinition::getMd5 () const
{
  return mMd5;
}


/*
 * @return true if the md5 of this SBML object has been set, false
 * otherwise.
 */
bool
ExternalModelDefinition::isSetMd5 () const
{
  return (mMd5.empty() == false);
}


/*
 * Unsets the md5 of this SBML object.
 */
int
ExternalModelDefinition::unsetMd5 ()
{
  mMd5.erase();

  if (mMd5.empty())
  {
    return LIBSBML_OPERATION_SUCCESS;
  }
  else
  {
    return LIBSBML_OPERATION_FAILED;
  }
}

/*
 * Sets the source of this SBML object to a copy of source.
 */
int
ExternalModelDefinition::setSource (const std::string& source)
{
  //LS DEBUG:  check syntax of source.  Somehow.  Maybe at least so that it's not ""?
  mSource = source;
  return LIBSBML_OPERATION_SUCCESS;
}


/*
 * @return the source of this SBML object.
 */
const string&
ExternalModelDefinition::getSource () const
{
  return mSource;
}


/*
 * @return true if the source of this SBML object has been set, false
 * otherwise.
 */
bool
ExternalModelDefinition::isSetSource () const
{
  return (mSource.empty() == false);
}


/*
 * Unsets the source of this SBML object.
 */
int
ExternalModelDefinition::unsetSource ()
{
  mSource.erase();

  if (mSource.empty())
  {
    return LIBSBML_OPERATION_SUCCESS;
  }
  else
  {
    return LIBSBML_OPERATION_FAILED;
  }
}

bool 
ExternalModelDefinition::hasRequiredAttributes() const
{
  if (!isSetSource()) return false;
  if (!isSetId()) return false;
  return CompBase::hasRequiredAttributes();
}

const std::string&
ExternalModelDefinition::getElementName () const
{
  static const std::string name = "externalModelDefinition";
  return name;
}


/** @cond doxygenLibsbmlInternal */
void
ExternalModelDefinition::addExpectedAttributes(ExpectedAttributes& attributes)
{
  CompBase::addExpectedAttributes(attributes);
  attributes.add("id");
  attributes.add("name");
  attributes.add("source");
  attributes.add("modelRef");
  attributes.add("md5");
}
/** @endcond */

/** @cond doxygenLibsbmlInternal */
void
ExternalModelDefinition::readAttributes (const XMLAttributes& attributes,
                          const ExpectedAttributes& expectedAttributes)
{
  const unsigned int sbmlLevel   = getLevel  ();
  const unsigned int sbmlVersion = getVersion();

  // look to see whether an unknown attribute error was logged
  // during the read of the ListOfExternalModelDefinitions - which will have
  // happened immediately prior to this read
  if (getErrorLog() != NULL && 
    static_cast<ListOfExternalModelDefinitions*>(getParentSBMLObject())->size() < 2)
  {
    unsigned int numErrs = getErrorLog()->getNumErrors();
    for (int n = numErrs-1; n >= 0; n--)      
    {
      if (getErrorLog()->getError(n)->getErrorId() == UnknownPackageAttribute)
      {
        const std::string details = 
          getErrorLog()->getError(n)->getMessage();
        getErrorLog()->remove(UnknownPackageAttribute);
        getErrorLog()->logPackageError("comp", CompLOExtModDefsAllowedAttributes,
          getPackageVersion(), sbmlLevel, sbmlVersion, details);
      } 
      else if (getErrorLog()->getError(n)->getErrorId() == UnknownCoreAttribute)
      {
        const std::string details = 
          getErrorLog()->getError(n)->getMessage();
        getErrorLog()->remove(UnknownCoreAttribute);
        getErrorLog()->logPackageError("comp", CompLOExtModDefsAllowedAttributes,
          getPackageVersion(), sbmlLevel, sbmlVersion, details);
      } 
    }
  }


  CompBase::readAttributes(attributes,expectedAttributes);

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
        getErrorLog()->logPackageError("comp", CompExtModDefAllowedAttributes,
          getPackageVersion(), sbmlLevel, sbmlVersion, details);
      } 
      else if (getErrorLog()->getError(n)->getErrorId() == UnknownCoreAttribute)
      {
        const std::string details = 
          getErrorLog()->getError(n)->getMessage();
        getErrorLog()->remove(UnknownCoreAttribute);
        getErrorLog()->logPackageError("comp", CompExtModDefAllowedCoreAttributes,
          getPackageVersion(), sbmlLevel, sbmlVersion, details);
      } 
    }
  }


  if ( sbmlLevel > 2 )
  {
    XMLTriple tripleId("id", mURI, getPrefix());
    bool assigned = false;
    assigned = attributes.readInto(tripleId, mId, getErrorLog(), 
      false, getLine(), getColumn()); 
    
    if (assigned == false)
    {
      logMissingAttribute("id", "<ExternalModelDefinition>");
    }
    else if (assigned == true && mId.size() == 0)
    {
      logEmptyString("id", "<ExternalModelDefinition>");
    }
    else 
    {
      if (!SyntaxChecker::isValidSBMLSId(mId)) {
        logInvalidId("comp:id", mId);
      }
    }

 
    XMLTriple tripleName("name", mURI, getPrefix());
    assigned = attributes.readInto(tripleName, mName);
    
    if (assigned == true)
    {
      if (mName.empty()) 
      {
        logEmptyString("comp:name", mName);
      }
    }

    XMLTriple tripleSource("source", mURI, getPrefix());
    assigned = attributes.readInto(tripleSource, mSource);

    if (assigned == false)
    {
      logMissingAttribute("source", "<ExternalModelDefinition>");
    }
    else
    {
      if (!SyntaxChecker::isValidXMLanyURI(mSource)) 
      {
        getErrorLog()->logPackageError("comp", CompInvalidSourceSyntax,
          getPackageVersion(), getLevel(), getVersion());
      }
    }
    
    XMLTriple tripleModel("modelRef", mURI, getPrefix());
    assigned = attributes.readInto(tripleModel, mModelRef);
    
    if (assigned == true)
    {
      if (!SyntaxChecker::isValidSBMLSId(mModelRef)) 
      {
        logInvalidId("comp:modelRef", mModelRef);
      }
    }
    XMLTriple tripleMd5("md5", mURI, getPrefix());
    assigned = attributes.readInto(tripleMd5, mMd5);

    if (assigned == true)
    {
      //TODO:  md5 syntax checking here.
    }
  }
}
/** @endcond */

/** @cond doxygenLibsbmlInternal */
void
ExternalModelDefinition::writeAttributes (XMLOutputStream& stream) const
{
  CompBase::writeAttributes(stream);

  if (isSetId()) {
    stream.writeAttribute("id", getPrefix(), mId);
  }
  if (isSetName()) {
    stream.writeAttribute("name", getPrefix(), mName);
  }
  if (isSetSource()) {
    stream.writeAttribute("source", getPrefix(), mSource);
  }
  if (isSetModelRef()) {
    stream.writeAttribute("modelRef", getPrefix(), mModelRef);
  }
  if (isSetMd5()) {
    stream.writeAttribute("md5", getPrefix(), mMd5);
  }

  ExternalModelDefinition::writeExtensionAttributes(stream);
}
/** @endcond */

/** @cond doxygenLibsbmlInternal */
void
ExternalModelDefinition::writeElements (XMLOutputStream& stream) const
{
  CompBase::writeElements(stream);

  ExternalModelDefinition::writeExtensionElements(stream);
}
/** @endcond */


int
ExternalModelDefinition::getTypeCode () const
{
  return SBML_COMP_EXTERNALMODELDEFINITION;
}

/** @cond doxygenLibsbmlInternal */

bool
ExternalModelDefinition::accept (SBMLVisitor& v) const
{
  return v.visit(*this);
}

/** @endcond */

/*
 * Resolves the referenced document.
 * 
 * Note: the document does not need to be freed, as it will be once the document plugin is deleteacceptComp 
 * 
 * @return the resolved model or NULL.
 */
Model*
ExternalModelDefinition::getReferencedModel()
{
  set<pair<string, string> > parents;
  return getReferencedModel(NULL, parents);
}

/* Internal getReferencedModel call sets the errordoc and parents
 */
Model*
ExternalModelDefinition::getReferencedModel(SBMLDocument* errordoc, set<pair<string, string> > parents)
{
  SBMLDocument* origdoc = getSBMLDocument();
  if (errordoc == NULL) {
    errordoc = origdoc;
  }

  CompSBMLDocumentPlugin* csdp = static_cast<CompSBMLDocumentPlugin*>(origdoc->getPlugin(getPrefix()));
  if (csdp == NULL) 
  {
    if (errordoc) {
      string error = "In ExternalModelDefinition::getReferencedModel, unable to resolve the external model definition '" + getId() + "': no 'comp' plugin found.";
      errordoc->getErrorLog()->logPackageError("comp", CompUnresolvedReference, getPackageVersion(), getLevel(), getVersion(), error, getLine(), getColumn());
    }
    return NULL;
  }

  if (!isSetSource()) 
  {
    if (errordoc) {
      string error = "In ExternalModelDefinition::getReferencedModel, unable to resolve the external model definition '" + getId() + "': the 'source' attribute was not set.";
      errordoc->getErrorLog()->logPackageError("comp", CompExtModDefAllowedAttributes, getPackageVersion(), getLevel(), getVersion(), error, getLine(), getColumn());
    }
    return NULL;
  }

  SBMLDocument* doc = csdp->getSBMLDocumentFromURI(getSource());
  if (doc == NULL) 
  {
    // resolving the document failed, add to the error log a notice so that 
    // other operations are informed, and for example flattening will fail
    // with error
    if (errordoc) {
      string error = "In ExternalModelDefinition::getReferencedModel, unable to resolve the external model definition '" + getId() + "': could not the resolve 'source' attribute '" + getSource() + "' as a valid SBML Document.";
      errordoc->getErrorLog()->logPackageError("comp", CompUnresolvedReference, getPackageVersion(), getLevel(), getVersion(), error, getLine(), getColumn());
    }
    return NULL;
  }

  if (!(doc->getLevel() == 3  && doc->getVersion() == 1))
  {
    // comp v1 ONLY allows L3v1 models. All other levels and versions are not supported. 
    // 
    if (errordoc) {
      string error = "In ExternalModelDefinition::getReferencedModel, unable to resolve the external model definition '" + getId() + "': the SBML document found at source '" + getSource() + "' was not SBML Level 3 Version 1.";
      errordoc->getErrorLog()->logPackageError("comp", CompReferenceMustBeL3, getPackageVersion(), getLevel(), getVersion(), error, getLine(), getColumn());
    }
    return NULL;
  }

  Model* model = doc->getModel();
  if (isSetModelRef()) 
  {
    csdp = static_cast<CompSBMLDocumentPlugin*>(doc->getPlugin(getPrefix()));
    if (csdp == NULL || (csdp->getNumExternalModelDefinitions()==0 && csdp->getNumModelDefinitions()==0)) 
    {
      if (model==NULL) {
        if (errordoc) {
          string error = "In ExternalModelDefinition::getReferencedModel, unable to resolve the external model definition '" + getId() + "': the SBML document found at source '" + getSource() + "' had no resolvable models at all.";
          errordoc->getErrorLog()->logPackageError("comp", CompNoModelInReference, getPackageVersion(), getLevel(), getVersion(), error, getLine(), getColumn());
        }
        return NULL;
      }
      else if (model->getId() != getModelRef()) {
        if (errordoc) {
          string error = "In ExternalModelDefinition::getReferencedModel, unable to resolve the external model definition '" + getId() + "': the SBML document found at source '" + getSource() + "' had no 'comp' model definitions, and the single model's ID ('" + model->getId() + "') did not match the modelRef '" + getModelRef() + "'.";
          errordoc->getErrorLog()->logPackageError("comp", CompModReferenceMustIdOfModel, getPackageVersion(), getLevel(), getVersion(), error, getLine(), getColumn());
        }
        return NULL;
      }
      //else we can return the main 'model' object of the document.
    }
    else 
    {
      SBase* referencedmod = csdp->getModel(getModelRef());
      if (referencedmod == NULL) 
      {
        if (errordoc) {
          string error = "In ExternalModelDefinition::getReferencedModel, unable to resolve the external model definition '" + getId() + "': the SBML document found at source '" + getSource() + "' did not have any model with the id '" + getModelRef() + "'.";
          errordoc->getErrorLog()->logPackageError("comp", CompModReferenceMustIdOfModel, getPackageVersion(), getLevel(), getVersion(), error, getLine(), getColumn());
        }
        return NULL;
      }

      ExternalModelDefinition* subextmod;
      pair<string, string> child;

      switch(referencedmod->getTypeCode()) 
      {
      case SBML_MODEL:
      case SBML_COMP_MODELDEFINITION:
        return static_cast<Model*>(referencedmod);

      case SBML_COMP_EXTERNALMODELDEFINITION:
        subextmod = static_cast<ExternalModelDefinition*>(referencedmod);
        //Before we call this recursively, make sure we aren't in a loop:
        child = make_pair(subextmod->getSource(), subextmod->getModelRef());
        if (parents.insert(child).second == false) {
          if (errordoc) {
            string error = "In ExternalModelDefinition::getReferencedModel, unable to resolve the external model definition '" + getId() + "': the external model definition it referenced ('" + getModelRef() + "', from the document at '" + getSource() + "') was already referenced in this external model definition chain.";
            errordoc->getErrorLog()->logPackageError("comp", CompCircularExternalModelReference, getPackageVersion(), getLevel(), getVersion(), error, getLine(), getColumn());
          }
          return NULL;
        }
        model = subextmod->getReferencedModel(errordoc, parents);
        if (model==NULL) {
          //The recursive call will set its own error message.
          return NULL;
        }
        return model;

      default:
        {
          // TODO: this case will be hit whenever another package
          //       derives from model, or external model definition.
          //       It might be safer not to just rely on the package
          //       type code and instead go for a dynamic_cast, but 
          //       again this can wait for the next release, as there
          //       are currently no packages that plan to do this.
          //
          // assert(false); 

          if (errordoc) {
            string error = "In ExternalModelDefinition::getReferencedModel, unable to resolve the external model definition '" + getId() + "': the model object discovered at in the SBML document found at source '" + getSource() + "' was not of the type 'model' 'modelDefinition', or 'externalModelDefinition'.  The most likely cause of this situation is if some other package extended one of those three types, but the external model definition code was not updated.";
            errordoc->getErrorLog()->logPackageError("comp", CompUnresolvedReference, getPackageVersion(), getLevel(), getVersion(), error, getLine(), getColumn());
          }
          return NULL;
        }
      }
    }
  }
  if (model==NULL) 
  {
    if (errordoc) {
      string error = "In ExternalModelDefinition::getReferencedModel, unable to resolve the external model definition '" + getId() + "': the SBML document found at source '" + getSource() + "' did not have a model child of the SBML Document, and no 'modelRef' attribute was set to discover any other model in that document.";
      errordoc->getErrorLog()->logPackageError("comp", CompNoModelInReference, getPackageVersion(), getLevel(), getVersion(), error, getLine(), getColumn());
    }
  }

  return model;
}

#endif /* __cplusplus */
/** @cond doxygenIgnored */
LIBSBML_EXTERN
ExternalModelDefinition_t *
ExternalModelDefinition_create(unsigned int level, unsigned int version,
                               unsigned int pkgVersion)
{
  return new ExternalModelDefinition(level, version, pkgVersion);
}


LIBSBML_EXTERN
void
ExternalModelDefinition_free(ExternalModelDefinition_t * emd)
{
  if (emd != NULL)
    delete emd;
}


LIBSBML_EXTERN
ExternalModelDefinition_t *
ExternalModelDefinition_clone(ExternalModelDefinition_t * emd)
{
  if (emd != NULL)
  {
    return static_cast<ExternalModelDefinition_t*>(emd->clone());
  }
  else
  {
    return NULL;
  }
}


LIBSBML_EXTERN
char *
ExternalModelDefinition_getId(ExternalModelDefinition_t * emd)
{
  if (emd == NULL)
    return NULL;

  return emd->getId().empty() ? NULL : safe_strdup(emd->getId().c_str());
}


LIBSBML_EXTERN
char *
ExternalModelDefinition_getSource(ExternalModelDefinition_t * emd)
{
  if (emd == NULL)
    return NULL;

  return emd->getSource().empty() ? NULL : safe_strdup(emd->getSource().c_str());
}


LIBSBML_EXTERN
char *
ExternalModelDefinition_getName(ExternalModelDefinition_t * emd)
{
  if (emd == NULL)
    return NULL;

  return emd->getName().empty() ? NULL : safe_strdup(emd->getName().c_str());
}


LIBSBML_EXTERN
char *
ExternalModelDefinition_getModelRef(ExternalModelDefinition_t * emd)
{
  if (emd == NULL)
    return NULL;

  return emd->getModelRef().empty() ? NULL : safe_strdup(emd->getModelRef().c_str());
}


LIBSBML_EXTERN
int
ExternalModelDefinition_isSetId(ExternalModelDefinition_t * emd)
{
  return (emd != NULL) ? static_cast<int>(emd->isSetId()) : 0;
}


LIBSBML_EXTERN
int
ExternalModelDefinition_isSetSource(ExternalModelDefinition_t * emd)
{
  return (emd != NULL) ? static_cast<int>(emd->isSetSource()) : 0;
}


LIBSBML_EXTERN
int
ExternalModelDefinition_isSetName(ExternalModelDefinition_t * emd)
{
  return (emd != NULL) ? static_cast<int>(emd->isSetName()) : 0;
}


LIBSBML_EXTERN
int
ExternalModelDefinition_isSetModelRef(ExternalModelDefinition_t * emd)
{
  return (emd != NULL) ? static_cast<int>(emd->isSetModelRef()) : 0;
}


LIBSBML_EXTERN
int
ExternalModelDefinition_setId(ExternalModelDefinition_t * emd, const char * id)
{
  return (emd != NULL) ? emd->setId(id) : LIBSBML_INVALID_OBJECT;
}


LIBSBML_EXTERN
int
ExternalModelDefinition_setSource(ExternalModelDefinition_t * emd, const char * source)
{
  return (emd != NULL) ? emd->setSource(source) : LIBSBML_INVALID_OBJECT;
}


LIBSBML_EXTERN
int
ExternalModelDefinition_setName(ExternalModelDefinition_t * emd, const char * name)
{
  return (emd != NULL) ? emd->setName(name) : LIBSBML_INVALID_OBJECT;
}


LIBSBML_EXTERN
int
ExternalModelDefinition_setModelRef(ExternalModelDefinition_t * emd, const char * modelRef)
{
  return (emd != NULL) ? emd->setModelRef(modelRef) : LIBSBML_INVALID_OBJECT;
}


LIBSBML_EXTERN
int
ExternalModelDefinition_unsetId(ExternalModelDefinition_t * emd)
{
  return (emd != NULL) ? emd->unsetId() : LIBSBML_INVALID_OBJECT;
}


LIBSBML_EXTERN
int
ExternalModelDefinition_unsetSource(ExternalModelDefinition_t * emd)
{
  return (emd != NULL) ? emd->unsetSource() : LIBSBML_INVALID_OBJECT;
}


LIBSBML_EXTERN
int
ExternalModelDefinition_unsetName(ExternalModelDefinition_t * emd)
{
  return (emd != NULL) ? emd->unsetName() : LIBSBML_INVALID_OBJECT;
}


LIBSBML_EXTERN
int
ExternalModelDefinition_unsetModelRef(ExternalModelDefinition_t * emd)
{
  return (emd != NULL) ? emd->unsetModelRef() : LIBSBML_INVALID_OBJECT;
}


LIBSBML_EXTERN
int
ExternalModelDefinition_hasRequiredAttributes(ExternalModelDefinition_t * emd)
{
  return (emd != NULL) ? static_cast<int>(emd->hasRequiredAttributes()) : 0;
}


LIBSBML_EXTERN
ExternalModelDefinition_t *
ListOfExternalModelDefinitions_getById(ListOf_t * lo, const char * sid)
{
  if (lo == NULL)
    return NULL;

  return (sid != NULL) ? static_cast <ListOfExternalModelDefinitions *>(lo)->get(sid) : NULL;
}


LIBSBML_EXTERN
ExternalModelDefinition_t *
ListOfExternalModelDefinitions_removeById(ListOf_t * lo, const char * sid)
{
  if (lo == NULL)
    return NULL;

  return (sid != NULL) ? static_cast <ListOfExternalModelDefinitions *>(lo)->remove(sid) : NULL;
}



/** @endcond */

LIBSBML_CPP_NAMESPACE_END


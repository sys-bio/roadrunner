/**
 * @file    Submodel.cpp
 * @brief   Implementation of Submodel, the SBase-derived class of the comp package.
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
#include <set>

#include <sbml/SBMLVisitor.h>
#include <sbml/RateRule.h>
#include <sbml/math/L3Parser.h>
#include <sbml/xml/XMLNode.h>
#include <sbml/xml/XMLToken.h>
#include <sbml/xml/XMLAttributes.h>
#include <sbml/xml/XMLInputStream.h>
#include <sbml/xml/XMLOutputStream.h>

#include <sbml/packages/comp/extension/CompExtension.h>
#include <sbml/packages/comp/extension/CompSBMLDocumentPlugin.h>
#include <sbml/packages/comp/extension/CompModelPlugin.h>
#include <sbml/packages/comp/sbml/Submodel.h>
#include <sbml/packages/comp/validator/CompSBMLError.h>

#include <sbml/util/ElementFilter.h>

using namespace std;

LIBSBML_CPP_NAMESPACE_BEGIN
#ifdef __cplusplus

Submodel::Submodel (unsigned int level, unsigned int version, unsigned int pkgVersion) 
  : CompBase (level,version, pkgVersion)
  , mId("")
  , mName("")
  , mModelRef("")
  , mTimeConversionFactor("")
  , mExtentConversionFactor("")
  , mListOfDeletions()
  , mInstantiatedModel(NULL)
  , mInstantiationOriginalURI("")
{
  mListOfDeletions.connectToParent(this);
}


Submodel::Submodel(CompPkgNamespaces* compns)
  : CompBase(compns)
  , mId("")
  , mName("")
  , mModelRef("")
  , mTimeConversionFactor("")
  , mExtentConversionFactor("")
  , mListOfDeletions()
  , mInstantiatedModel(NULL)
  , mInstantiationOriginalURI("")
{
  loadPlugins(compns);
  mListOfDeletions.connectToParent(this);
}


Submodel::Submodel(const Submodel& source) 
  : CompBase (source)
  , mId(source.mId)
  , mName(source.mName)
  , mModelRef(source.mModelRef)
  , mTimeConversionFactor(source.mTimeConversionFactor)
  , mExtentConversionFactor(source.mExtentConversionFactor)
  , mListOfDeletions(source.mListOfDeletions)
  , mInstantiatedModel(NULL) //Must call 'instantiate()' again if you want a copy.
  , mInstantiationOriginalURI("")
{
  mListOfDeletions.connectToParent(this);
}

Submodel& Submodel::operator=(const Submodel& source)
{
  if(&source!=this)
  {
    CompBase::operator=(source);
    mId = source.mId;
    mName = source.mName;
    mModelRef = source.mModelRef;
    mTimeConversionFactor = source.mTimeConversionFactor;
    mExtentConversionFactor = source.mExtentConversionFactor;
    mListOfDeletions = source.mListOfDeletions;
    mInstantiatedModel = NULL; //Must call 'instantiate()' again if you want a copy.
    mInstantiationOriginalURI = "";
  }

  return *this;
}

Submodel*
Submodel::clone() const
{
  return new Submodel(*this);
}

Submodel::~Submodel ()
{
  if (mInstantiatedModel != NULL) delete mInstantiatedModel;
}


SBase* 
Submodel::getElementBySId(const std::string& id)
{
  if (id.empty()) return NULL;
  SBase* obj = mListOfDeletions.getElementBySId(id);
  if (obj != NULL) return obj;

  return getElementFromPluginsBySId(id);
}


SBase*
Submodel::getElementByMetaId(const std::string& metaid)
{
  if (metaid.empty()) return NULL;
  if (mListOfDeletions.getMetaId()==metaid) return &mListOfDeletions;
  SBase* obj = mListOfDeletions.getElementByMetaId(metaid);
  if (obj != NULL) return obj;

  return getElementFromPluginsByMetaId(metaid);
}


List*
Submodel::getAllElements(ElementFilter *filter)
{
  List* ret = new List();
  List* sublist = NULL;

  ADD_FILTERED_LIST(ret, sublist, mListOfDeletions, filter);  

  ADD_FILTERED_FROM_PLUGIN(ret, sublist, filter);

  return ret;
}

int
Submodel::setId (const std::string& id)
{
  if (!SyntaxChecker::isValidSBMLSId(id)) 
  {
    return LIBSBML_INVALID_ATTRIBUTE_VALUE;
  }
  mId = id;
  return LIBSBML_OPERATION_SUCCESS;
}


const string&
Submodel::getId () const
{
  return mId;
}


bool
Submodel::isSetId () const
{
  return (mId.empty() == false);
}


int
Submodel::unsetId ()
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
Submodel::setName (const std::string& name)
{
  if (name.empty()) 
  {
    return LIBSBML_INVALID_ATTRIBUTE_VALUE;
  }
  mName = name;
  return LIBSBML_OPERATION_SUCCESS;
}


const string&
Submodel::getName () const
{
  return mName;
}


bool
Submodel::isSetName () const
{
  return (mName.empty() == false);
}


int
Submodel::unsetName ()
{
  mName = "";
  return LIBSBML_OPERATION_SUCCESS;
}


int
Submodel::setModelRef (const std::string& modelRef)
{
  if (!SyntaxChecker::isValidSBMLSId(modelRef)) 
  {
    return LIBSBML_INVALID_ATTRIBUTE_VALUE;
  }
  mModelRef = modelRef;
  return LIBSBML_OPERATION_SUCCESS;
}


const string&
Submodel::getModelRef () const
{
  return mModelRef;
}


bool
Submodel::isSetModelRef () const
{
  return (mModelRef.empty() == false);
}


int
Submodel::unsetModelRef ()
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



int
Submodel::setSubstanceConversionFactor (const std::string& substanceConversionFactor)
{
  return LIBSBML_INVALID_ATTRIBUTE_VALUE;
}


const string&
Submodel::getSubstanceConversionFactor () const
{
  static string blank("");
  return blank;
}


bool
Submodel::isSetSubstanceConversionFactor () const
{
  return false;
}


int
Submodel::unsetSubstanceConversionFactor ()
{
  return LIBSBML_OPERATION_FAILED;
}


int
Submodel::setTimeConversionFactor (const std::string& timeConversionFactor)
{
  if (!SyntaxChecker::isValidSBMLSId(timeConversionFactor)) 
  {
    return LIBSBML_INVALID_ATTRIBUTE_VALUE;
  }
  mTimeConversionFactor = timeConversionFactor;
  return LIBSBML_OPERATION_SUCCESS;
}


const string&
Submodel::getTimeConversionFactor () const
{
  return mTimeConversionFactor;
}


bool
Submodel::isSetTimeConversionFactor () const
{
  return (mTimeConversionFactor.empty() == false);
}


int
Submodel::unsetTimeConversionFactor ()
{
  mTimeConversionFactor.erase();

  if (mTimeConversionFactor.empty())
  {
    return LIBSBML_OPERATION_SUCCESS;
  }
  else
  {
    return LIBSBML_OPERATION_FAILED;
  }
}


int
Submodel::setExtentConversionFactor (const std::string& extentConversionFactor)
{
  if (!SyntaxChecker::isValidSBMLSId(extentConversionFactor)) 
  {
    return LIBSBML_INVALID_ATTRIBUTE_VALUE;
  }
  mExtentConversionFactor = extentConversionFactor;
  return LIBSBML_OPERATION_SUCCESS;
}


const string&
Submodel::getExtentConversionFactor () const
{
  return mExtentConversionFactor;
}


bool
Submodel::isSetExtentConversionFactor () const
{
  return (mExtentConversionFactor.empty() == false);
}


int
Submodel::unsetExtentConversionFactor ()
{
  mExtentConversionFactor.erase();

  if (mExtentConversionFactor.empty())
  {
    return LIBSBML_OPERATION_SUCCESS;
  }
  else
  {
    return LIBSBML_OPERATION_FAILED;
  }
}


const ListOfDeletions*
Submodel::getListOfDeletions () const
{
  return &mListOfDeletions;
}


ListOfDeletions*
Submodel::getListOfDeletions ()
{
  return &mListOfDeletions;
}


Deletion*
Submodel::removeDeletion(unsigned int index)
{
  return mListOfDeletions.remove(index);
}


Deletion*
Submodel::removeDeletion(const std::string& sid)
{
  return mListOfDeletions.remove(sid);
}


Deletion* 
Submodel::getDeletion (unsigned int index)
{
  return mListOfDeletions.get(index);
}

const Deletion* 
Submodel::getDeletion (unsigned int index) const
{
  return mListOfDeletions.get(index);
}

Deletion* 
Submodel::getDeletion (std::string id)
{
  return mListOfDeletions.get(id);
}

const Deletion* 
Submodel::getDeletion (std::string id) const
{
  return mListOfDeletions.get(id);
}


int
Submodel::addDeletion (const Deletion* deletion)
{
  if (deletion == NULL)
  {
    return LIBSBML_INVALID_OBJECT;
  }
  else if (!(deletion->hasRequiredAttributes()) || !(deletion->hasRequiredElements()))
  {
    return LIBSBML_INVALID_OBJECT;
  }
  else if (getLevel() != deletion->getLevel())
  {
    return LIBSBML_LEVEL_MISMATCH;
  }
  else if (getVersion() != deletion->getVersion())
  {
    return LIBSBML_VERSION_MISMATCH;
  }
  else if (getPackageVersion() != deletion->getPackageVersion())
  {
    return LIBSBML_PKG_VERSION_MISMATCH;
  }
  else
  {
    return mListOfDeletions.append(deletion);
  }
}


unsigned int
Submodel::getNumDeletions () const
{
  return mListOfDeletions.size();
}


Deletion*
Submodel::createDeletion ()
{
  COMP_CREATE_NS(compns, getSBMLNamespaces());
  Deletion* m = new Deletion(compns);
  mListOfDeletions.appendAndOwn(m);
  delete compns;
  return m;
}



bool 
Submodel::hasRequiredAttributes() const
{
  if(!CompBase::hasRequiredAttributes()) return false;
  if(!isSetId()) return false;
  return (isSetModelRef());
}

/*
 * This object's XML name.
 */
const std::string&
Submodel::getElementName () const
{
  static const std::string name = "submodel";
  return name;
}


/** @cond doxygenLibsbmlInternal */
void
Submodel::addExpectedAttributes(ExpectedAttributes& attributes)
{
  CompBase::addExpectedAttributes(attributes);
  attributes.add("id");
  attributes.add("name");
  attributes.add("modelRef");
  attributes.add("timeConversionFactor");
  attributes.add("extentConversionFactor");
}
/** @endcond */

/** @cond doxygenLibsbmlInternal */
void
Submodel::readAttributes (const XMLAttributes& attributes,
                          const ExpectedAttributes& expectedAttributes)
{
  const unsigned int sbmlLevel   = getLevel  ();
  const unsigned int sbmlVersion = getVersion();

  // look to see whether an unknown attribute error was logged
  // during the read of the ListOfSubmodels - which will have
  // happened immediately prior to this read
  if (getErrorLog() != NULL && 
    static_cast<ListOfSubmodels*>(getParentSBMLObject())->size() < 2)
  {
    unsigned int numErrs = getErrorLog()->getNumErrors();
    for (int n = numErrs-1; n >= 0; n--)      
    {
      if (getErrorLog()->getError(n)->getErrorId() == UnknownPackageAttribute)
      {
        const std::string details = 
          getErrorLog()->getError(n)->getMessage();
        getErrorLog()->remove(UnknownPackageAttribute);
        getErrorLog()->logPackageError("comp", CompLOSubmodelsAllowedAttributes,
          getPackageVersion(), sbmlLevel, sbmlVersion, details);
      } 
      else if (getErrorLog()->getError(n)->getErrorId() == UnknownCoreAttribute)
      {
        const std::string details = 
          getErrorLog()->getError(n)->getMessage();
        getErrorLog()->remove(UnknownCoreAttribute);
        getErrorLog()->logPackageError("comp", CompLOSubmodelsAllowedAttributes,
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
        getErrorLog()->logPackageError("fbc", CompSubmodelAllowedAttributes,
          getPackageVersion(), sbmlLevel, sbmlVersion, details);
      } 
      else if (getErrorLog()->getError(n)->getErrorId() == UnknownCoreAttribute)
      {
        const std::string details = 
          getErrorLog()->getError(n)->getMessage();
        getErrorLog()->remove(UnknownCoreAttribute);
        getErrorLog()->logPackageError("fbc", CompSubmodelAllowedCoreAttributes,
          getPackageVersion(), sbmlLevel, sbmlVersion, details);
      } 
    }
  }


  if ( sbmlLevel > 2 )
  {
    XMLTriple tripleId("id", mURI, getPrefix());
    bool assigned = attributes.readInto(tripleId, mId);

    if (assigned == false)
    {
      std::string message = "Comp attribute 'id' is missing.";
      getErrorLog()->logPackageError("comp", CompSubmodelAllowedAttributes, 
        getPackageVersion(), sbmlLevel, sbmlVersion, message);
    }
    else
    {
      if (!SyntaxChecker::isValidSBMLSId(mId)) {
        logInvalidId("comp:id", mId);
      }
    }
    XMLTriple tripleName("name", mURI, getPrefix());
    if (attributes.readInto(tripleName, mName, getErrorLog(), false, getLine(), getColumn())) {
      if (mName.empty()) {
        logInvalidId("comp:name", mName);
      }
    }
    XMLTriple tripleModelRef("modelRef", mURI, getPrefix());
    assigned = attributes.readInto(tripleModelRef, mModelRef);
    if (assigned == false)
    {
      std::string message = "Comp attribute 'modelRef' is missing.";
      getErrorLog()->logPackageError("comp", CompSubmodelAllowedAttributes, 
        getPackageVersion(), sbmlLevel, sbmlVersion, message);
    }
    else
    {
      if (!SyntaxChecker::isValidSBMLSId(mModelRef)) 
      {
        logInvalidId("comp:modelRef", mModelRef, "Submodel");
      }
    }
    XMLTriple tripletcf("timeConversionFactor", mURI, getPrefix());
    if (attributes.readInto(tripletcf, mTimeConversionFactor, getErrorLog(), false, getLine(), getColumn())) {
      if (!SyntaxChecker::isValidSBMLSId(mTimeConversionFactor)) {
        logInvalidId("comp:timeConversionFactor", mTimeConversionFactor);
      }
    }
    XMLTriple triplexcf("extentConversionFactor", mURI, getPrefix());
    if (attributes.readInto(triplexcf, mExtentConversionFactor, getErrorLog(), false, getLine(), getColumn())) {
      if (!SyntaxChecker::isValidSBMLSId(mExtentConversionFactor)) {
        logInvalidId("comp:extentConversionFactor", mExtentConversionFactor);
      }
    }
  }
}
/** @endcond */

/** @cond doxygenLibsbmlInternal */
void
Submodel::writeAttributes (XMLOutputStream& stream) const
{
  CompBase::writeAttributes(stream);

  if (isSetId()) {
    stream.writeAttribute("id", getPrefix(), mId);
  }
  if (isSetName()) {
    stream.writeAttribute("name", getPrefix(), mName);
  }
  if (isSetModelRef()) {
    stream.writeAttribute("modelRef", getPrefix(), mModelRef);
  }
  if (isSetTimeConversionFactor()) {
    stream.writeAttribute("timeConversionFactor", getPrefix(), mTimeConversionFactor);
  }
  if (isSetExtentConversionFactor()) {
    stream.writeAttribute("extentConversionFactor", getPrefix(), mExtentConversionFactor);
  }
  Submodel::writeExtensionAttributes(stream);
}
/** @endcond */

/** @cond doxygenLibsbmlInternal */
void
Submodel::writeElements (XMLOutputStream& stream) const
{
  CompBase::writeElements(stream);
  if (getNumDeletions() > 0)
  {
    mListOfDeletions.write(stream);
  }    

  Submodel::writeExtensionElements(stream);
}
/** @endcond */

/** @cond doxygenLibsbmlInternal */
SBase*
Submodel::createObject(XMLInputStream& stream)
{
  SBase*        object = 0;

  const std::string&   name   = stream.peek().getName();
  const XMLNamespaces& xmlns  = stream.peek().getNamespaces();
  const std::string&   prefix = stream.peek().getPrefix();

  const std::string& targetPrefix = (xmlns.hasURI(mURI)) ? xmlns.getPrefix(mURI) : getPrefix();
  
  if (prefix == targetPrefix) {
    if ( name == "listOfDeletions" )  
    {
      if (mListOfDeletions.size() != 0)
      {
        getErrorLog()->logPackageError("comp", CompOneListOfDeletionOnSubmodel, 
          getPackageVersion(), getLevel(), getVersion());
      }

      object = &mListOfDeletions;
   
      if (targetPrefix.empty()) {
        mListOfDeletions.getSBMLDocument()->enableDefaultNS(mURI,true);
      }
    }
  }    

  return object;
}
/** @endcond */


void
Submodel::renameSIdRefs(const std::string& oldid, const std::string& newid)
{
  if (mTimeConversionFactor==oldid) mTimeConversionFactor=newid;
  if (mExtentConversionFactor==oldid) mExtentConversionFactor=newid;
  CompBase::renameSIdRefs(oldid, newid);
}


int
Submodel::getTypeCode () const
{
  return SBML_COMP_SUBMODEL;
}

/** @cond doxygenLibsbmlInternal */

bool
Submodel::accept (SBMLVisitor& v) const
{
  v.visit(*this);

  for (unsigned int i = 0; i < getNumDeletions(); i++)
  {
    getDeletion(i)->accept(v);
  }

  v.leave(*this);

  return true;
}

/** @endcond */


/** @cond doxygenLibsbmlInternal */
/*
 * Sets the parent SBMLDocument of this SBML object.
 */
void
Submodel::setSBMLDocument (SBMLDocument* d)
{
  CompBase::setSBMLDocument(d);
  mListOfDeletions.setSBMLDocument(d);  
  if (mInstantiatedModel != NULL) {
    mInstantiatedModel->setSBMLDocument(d);
  }
}
/** @endcond */


/** @cond doxygenLibsbmlInternal */
/*
 * Sets this SBML object to child SBML objects (if any).
 * (Creates a child-parent relationship by the parent)
 */
void
Submodel::connectToChild()
{
  CompBase::connectToChild();
  mListOfDeletions.connectToParent(this);
  if (mInstantiatedModel != NULL) {
    mInstantiatedModel->connectToParent(this);
  }
}
/** @endcond */


int 
Submodel::instantiate()
{
  SBMLDocument* doc = getSBMLDocument();
  SBMLDocument* rootdoc = doc;
  if (doc==NULL) 
  {
    return LIBSBML_OPERATION_FAILED;
  }

  CompSBMLDocumentPlugin* docplugin = 
    static_cast<CompSBMLDocumentPlugin*>(doc->getPlugin(getPrefix()));
  if (docplugin==NULL)
  {
    return LIBSBML_OPERATION_FAILED;
  }

  SBase* parent  = getParentSBMLObject();
  string parentmodelname = "";
  string parentURI = "";
  set<string> uniqueModels;
  while (parent != NULL && parent->getTypeCode() != SBML_DOCUMENT) {
    if (parent->getTypeCode() == SBML_COMP_SUBMODEL) {
      const Submodel* parentsub = static_cast<const Submodel*>(parent);
      uniqueModels.insert(parentsub->mInstantiationOriginalURI + "::" + parentsub->getModelRef());
      if (parentURI=="") {
        parentURI=parentsub->mInstantiationOriginalURI;
      }
    }
    if (parent->getTypeCode() == SBML_MODEL ||
      parent->getTypeCode() == SBML_COMP_MODELDEFINITION)
    {
      if (parentmodelname == "") {
        parentmodelname = parent->getId();
      }
    }
    rootdoc = parent->getSBMLDocument();
    parent = parent->getParentSBMLObject();
  }

  if (mInstantiatedModel != NULL) 
  {
    delete mInstantiatedModel;
    mInstantiatedModel = NULL;
    mInstantiationOriginalURI.clear();
  }

  if (!hasRequiredAttributes()) {
    string error = "Instantiation error in Submodel::instantiate:  ";
    if (!isSetId()) {
      error += "A submodel in model '" + getParentModel(this)->getId() + "' does not have an 'id' attribute.";
    }
    else if (!isSetModelRef()) {
      error += "The submodel '" + getId() + "' does not have a 'modelRef' attribute.";
    }
    rootdoc->getErrorLog()->logPackageError("comp", CompSubmodelAllowedAttributes, getPackageVersion(), getLevel(), getVersion(), error, getLine(), getColumn());
    return LIBSBML_INVALID_OBJECT;
  }

  SBase* origmodel = docplugin->getModel(getModelRef());
  
  if (origmodel==NULL) {
    string error = "In Submodel::instantiate, unable to instantiate submodel '" + getId() + "' because the referenced model ('" + getModelRef() +"') does not exist.";
    rootdoc->getErrorLog()->logPackageError("comp", CompSubmodelMustReferenceModel, getPackageVersion(), getLevel(), getVersion(), error, getLine(), getColumn());
    return LIBSBML_INVALID_OBJECT;
  }
  ExternalModelDefinition* extmod;
  SBMLDocument* origdoc = NULL;
  string newmodel = parentURI + "::" + getModelRef();
  
  set<pair<string, string> > parents;
  switch(origmodel->getTypeCode()) 
  {
  case SBML_MODEL:
  case SBML_COMP_MODELDEFINITION:
    origdoc = origmodel->getSBMLDocument();
    mInstantiatedModel = static_cast<Model*>(origmodel)->clone();
    if (uniqueModels.insert(newmodel).second == false) {
      //Can't instantiate this model, because we are already a child of it.
      string error = "Error in Submodel::instantiate:  cannot instantiate submodel '" + getId() + "' in model '" + parentmodelname + "' because it references the model '" + getModelRef() + "', which is already an ancestor of the submodel.";
      rootdoc->getErrorLog()->logPackageError("comp", CompSubmodelCannotReferenceSelf, getPackageVersion(), getLevel(), getVersion(), error, getLine(), getColumn());
      return LIBSBML_OPERATION_FAILED;
    }
    mInstantiationOriginalURI = parentURI;
    break;
  case SBML_COMP_EXTERNALMODELDEFINITION:
    extmod = static_cast<ExternalModelDefinition*>(origmodel);
    if (extmod==NULL) 
    {
      //No error message:  it should be impossible, if origmodel has the type code 'external model definition', for it to not be castable to an external model definition.
      mInstantiatedModel = NULL;
      mInstantiationOriginalURI = "";
      return LIBSBML_OPERATION_FAILED;
    }
    mInstantiatedModel = extmod->getReferencedModel(rootdoc, parents);
    if (mInstantiatedModel == NULL) 
    {
      string error = "In Submodel::instantiate, unable to instantiate submodel '" + getId() + "' because the external model definition it referenced (model '" + getModelRef() +"') could not be resolved.";
      rootdoc->getErrorLog()->logPackageError("comp", CompSubmodelMustReferenceModel, getPackageVersion(), getLevel(), getVersion(), error, getLine(), getColumn());
      mInstantiationOriginalURI = "";
      return LIBSBML_OPERATION_FAILED;
    }
    mInstantiationOriginalURI = extmod->getSource();
    origdoc = mInstantiatedModel->getSBMLDocument();
    newmodel = extmod->getSource() + "::" + getModelRef();
    if (uniqueModels.insert(newmodel).second == false) {
      //Can't instantiate this model, because we are already a child of it.
      string error = "Error in Submodel::instantiate:  cannot instantiate submodel '" + getId() + "' in model '" + parentmodelname + "' because it references the model '" + getModelRef() + "', which is already an ancestor of the submodel.";
      rootdoc->getErrorLog()->logPackageError("comp", CompSubmodelCannotReferenceSelf, getPackageVersion(), getLevel(), getVersion(), error, getLine(), getColumn());
      mInstantiatedModel = NULL;
      mInstantiationOriginalURI = "";
      return LIBSBML_OPERATION_FAILED;
    }
    mInstantiatedModel = mInstantiatedModel->clone();
    mInstantiationOriginalURI = extmod->getSource();
    break;
  default:
    //Should always be one of the above, unless someone extends one of the above and doesn't tell us.
    string error = "Instantiation error in Submodel::instantiate:  unable to parse the model '" + origmodel->getId() + "', as it was not of the type 'model' 'modelDefinition', or 'externalModelDefinition'.  The most likely cause of this situation is if some other package extended one of those three types, but the submodel code was not updated.";
    rootdoc->getErrorLog()->logPackageError("comp", CompUnresolvedReference, getPackageVersion(), getLevel(), getVersion(), error, getLine(), getColumn());
    mInstantiatedModel = NULL;
    mInstantiationOriginalURI = "";
    return LIBSBML_OPERATION_FAILED;
  }
  
  if (mInstantiatedModel==NULL) 
  {
    string error = "Instantiation error in Submodel::instantiate:  unable to create a valid copy of model '" + getModelRef() + "'.";
    rootdoc->getErrorLog()->logPackageError("comp", CompModelFlatteningFailed, getPackageVersion(), getLevel(), getVersion(), error, getLine(), getColumn());
    mInstantiationOriginalURI = "";
    return LIBSBML_OPERATION_FAILED;
  }

  mInstantiatedModel->connectToParent(this);
  mInstantiatedModel->setSBMLDocument(origdoc);
  mInstantiatedModel->enablePackage(getPackageURI(), getPrefix(), true);

  // enable any packages that were enabled on the doc
  //for (unsigned int n = 0; n < origdoc->getNumPlugins(); n++)
  //{
  //  SBasePlugin * plugin = origdoc->getPlugin(n);
  //  
  //  mInstantiatedModel->enablePackage(plugin->getURI(),
  //                                    plugin->getPrefix(), true);
  //}
  
  CompModelPlugin* instmodplug = 
    static_cast<CompModelPlugin*>(mInstantiatedModel->getPlugin(getPrefix()));
  
  for (unsigned int sub=0; sub<instmodplug->getNumSubmodels(); sub++) 
  {
    Submodel* instsub = instmodplug->getSubmodel(sub);
    int ret = instsub->instantiate();
    if (ret != LIBSBML_OPERATION_SUCCESS) {
      //'instantiate' already sets its own error messages.
      delete mInstantiatedModel;
      mInstantiatedModel = NULL;
      mInstantiationOriginalURI = "";
      return ret;
    }
  }

  return LIBSBML_OPERATION_SUCCESS;
}

int Submodel::performDeletions()
{
  if (mInstantiatedModel == NULL) return LIBSBML_INVALID_OBJECT;

  /* since we realised that we need a means of checking whether 
   * an object has already been deleted before deleting it
   * this function has become unstable
   *
   * it is being left in since it was part of the publiuc API
   * but needs addressing
   */

  //for (unsigned int i = 0; i < getNumDeletions(); i++)
  //{
  //  SBase* todelete = getDeletion(i)->getReferencedElementFrom(mInstantiatedModel);
  //  if (todelete == NULL) continue;
  //  CompBase::removeFromParentAndPorts(todelete);
  //}
  return LIBSBML_OPERATION_FAILED;
}

int 
Submodel::replaceElement(SBase* toReplace, SBase* replacement)
{
  if (mInstantiatedModel == NULL) return LIBSBML_INVALID_OBJECT; //Must call 'instantiate' first (and probably rename your objects, too!).
  string oldSId = toReplace->getId();
  string oldMetaId = toReplace->getMetaId();

  List* allelements = mInstantiatedModel->getAllElements();
  for (unsigned int el=0; el<allelements->getSize(); el++) {
    SBase* element = static_cast<SBase*>(allelements->get(el));
    assert(element != NULL);
    if (element == NULL) continue;
    if (toReplace->isSetId()) {
      if (replacement->getTypeCode() == SBML_UNIT_DEFINITION) {
        element->renameUnitSIdRefs(toReplace->getId(), replacement->getId());
      }
      else {
        element->renameSIdRefs(toReplace->getId(), replacement->getId());
      }
    }
    if (toReplace->isSetMetaId()) {
      element->renameMetaIdRefs(toReplace->getMetaId(), replacement->getMetaId());
    }
  }

  delete allelements;
  return LIBSBML_OPERATION_FAILED;
}

Model* 
Submodel::getInstantiation()
{
  if (mInstantiatedModel == NULL) {
    instantiate();
  }
  return mInstantiatedModel;
}

const Model* 
Submodel::getInstantiation() const
{
  return mInstantiatedModel;
}

void 
Submodel::clearInstantiation()
{
  if (mInstantiatedModel != NULL) {
    delete mInstantiatedModel;
  }
  mInstantiatedModel = NULL;
}
  
List* 
Submodel::getAllInstantiatedElements()
{
  Model* inst = getInstantiation();
  if (inst==NULL) return NULL;
  List* allElements = inst->getAllElements();
  vector<List*> sublists;
  CompModelPlugin* instp = static_cast<CompModelPlugin*>(inst->getPlugin(getPrefix()));
  for (unsigned int sm=0; sm<instp->getNumSubmodels(); sm++) {
    Submodel* subm=instp->getSubmodel(sm);
    if (subm==NULL) return NULL;
    List* sublist = subm->getAllInstantiatedElements();
    sublists.push_back(sublist);
  }
  for (size_t l=0; l<sublists.size(); l++) {
    allElements->transferFrom(sublists[l]);
    delete sublists[l];
  }  
  return allElements;
}


int Submodel::convertTimeAndExtent()
{
  int ret=LIBSBML_OPERATION_SUCCESS;
  string tcf = "";
  ASTNode* tcf_ast = NULL;
  if (isSetTimeConversionFactor()) {
    tcf = getTimeConversionFactor();
    tcf_ast = new ASTNode(AST_NAME);
    tcf_ast->setName(tcf.c_str());
  }
  string xcf = "";
  ASTNode* xcf_ast = NULL;
  if (isSetExtentConversionFactor()) {
    xcf = getExtentConversionFactor();
    xcf_ast = new ASTNode(AST_NAME);
    xcf_ast->setName(xcf.c_str());
  }

  ASTNode* klmod = NULL;
  if (xcf_ast != NULL) {
    klmod = xcf_ast;
  }
  if (tcf_ast != NULL) {
    if (klmod==NULL) {
      klmod = new ASTNode(AST_INTEGER);
      klmod->setValue(1);
    }
    ASTNode* divide = new ASTNode(AST_DIVIDE);
    divide->addChild(klmod);
    divide->addChild(tcf_ast);
    klmod = divide;
  }

  ret = convertTimeAndExtentWith(tcf_ast, xcf_ast, klmod);
  delete klmod;
  return ret;
}

int Submodel::convertTimeAndExtentWith(const ASTNode* tcf, const ASTNode* xcf, const ASTNode* klmod)
{
  if (tcf==NULL && xcf==NULL) return LIBSBML_OPERATION_SUCCESS;
  Model* model = getInstantiation();
  if (model==NULL) {
    //getInstantiation sets its own error messages.
    return LIBSBML_OPERATION_FAILED;
  }
  ASTNode* tcftimes = NULL;
  ASTNode* tcfdiv = NULL;
  if (tcf != NULL) {
    tcftimes = new ASTNode(AST_TIMES);
    tcftimes->addChild(tcf->deepCopy());
    tcfdiv = new ASTNode(AST_DIVIDE);
    tcfdiv->addChild(tcf->deepCopy());
  }
  ASTNode* rxndivide = NULL;
  if (klmod != NULL) {
    rxndivide = new ASTNode(AST_DIVIDE);
    ASTNode rxnref(AST_NAME);
    rxndivide->addChild(rxnref.deepCopy());
    rxndivide->addChild(klmod->deepCopy());
  }
  List* allelements = model->getAllElements();
  for (unsigned int el=0; el<allelements->getSize(); el++) {
    SBase* element = static_cast<SBase*>(allelements->get(el));
    assert(element != NULL);
    ASTNode* ast1 = NULL;
    ASTNode* ast2 = NULL;
    Constraint* constraint = NULL;
    Delay* delay = NULL;
    EventAssignment* ea = NULL;
    InitialAssignment* ia = NULL;
    KineticLaw* kl = NULL;
    Priority* priority = NULL;
    RateRule* rrule = NULL;
    Rule* rule = NULL;
    Submodel* submodel = NULL;
    Trigger* trigger = NULL;
    string cf = "";
    //Reaction math will be converted below, in the bits with the kinetic law.  But because of that, we need to handle references *to* the reaction:  even if it has no kinetic law, the units have changed, and this needs to be reflected by the flattening routine.
    if (rxndivide != NULL && element->getTypeCode()==SBML_REACTION && element->isSetId()) {
      rxndivide->getChild(0)->setName(element->getId().c_str());
      for (unsigned int sube=0; sube<allelements->getSize(); sube++) {
        SBase* subelement = static_cast<SBase*>(allelements->get(sube));
        subelement->replaceSIDWithFunction(element->getId(), rxndivide);
      }
    }

    //Submodels need their timeConversionFactor and extentConversionFactor attributes converted.  We're moving top-down, so all we need to do here is fix the conversion factor attributes themselves, pointing them to new parameters if need be.
    if ((tcf !=NULL || xcf != NULL) && element->getTypeCode()==SBML_COMP_SUBMODEL) {
      submodel = static_cast<Submodel*>(element);
      if (tcf != NULL) {
        if (submodel->isSetTimeConversionFactor()) {
          createNewConversionFactor(cf, tcf, submodel->getTimeConversionFactor(), model);
          submodel->setTimeConversionFactor(cf);
        }
        else {
          submodel->setTimeConversionFactor(tcf->getName());
        }
      }
      if (xcf != NULL) {
        if (submodel->isSetExtentConversionFactor()) {
          createNewConversionFactor(cf, xcf, submodel->getExtentConversionFactor(), model);
          submodel->setExtentConversionFactor(cf);
        }
        else {
          submodel->setExtentConversionFactor(xcf->getName());
        }
      }
    }
    if (tcf==NULL) {
      if (klmod !=NULL && element->getTypeCode()==SBML_KINETIC_LAW) {
        kl = static_cast<KineticLaw*>(element);
        if (kl->isSetMath()) {
          ast1 = new ASTNode(AST_TIMES);
          ast1->addChild(klmod->deepCopy());
          ast1->addChild(kl->getMath()->deepCopy());
          kl->setMath(ast1);
          delete ast1;
        }
      }
    }
    else {
      // All math 'time' and 'delay' csymbols must still be converted.
      // Also, several constructs are modified directly.
      switch(element->getTypeCode()) {
        //This would be a WHOLE LOT SIMPLER if there was a 'hasMath' class in libsbml.  But even so, it would have to
        // handle the kinetic laws, rate rules, and delays separately.
      case SBML_KINETIC_LAW:
        //Kinetic laws are multiplied by 'klmod'.
        kl = static_cast<KineticLaw*>(element);
        ast1 = kl->getMath()->deepCopy();
        convertCSymbols(ast1, tcfdiv, tcftimes);
        if (klmod !=NULL) {
          kl = static_cast<KineticLaw*>(element);
          if (kl->isSetMath()) {
            ast2 = new ASTNode(AST_TIMES);
            ast2->addChild(klmod->deepCopy());
            ast2->addChild(ast1);
            kl->setMath(ast2);
            delete ast2;
          }
        }
        else {
          kl->setMath(ast1);
          delete ast1;
        }
        break;
      case SBML_DELAY:
        //Delays are multiplied by the time conversion factor.
        delay = static_cast<Delay*>(element);
        if (delay->isSetMath()) {
          ast1 = delay->getMath()->deepCopy();
          convertCSymbols(ast1, tcfdiv, tcftimes);
          tcftimes->addChild(ast1);
          delay->setMath(tcftimes);
          tcftimes->removeChild(1);
        }
        break;
      case SBML_RATE_RULE:
        //Rate rules are divided by the time conversion factor.
        rrule = static_cast<RateRule*>(element);
        if (rrule->isSetMath()) {
          //ast1 = rrule->getMath()->deepCopy();
          tcfdiv->insertChild(0, rrule->getMath()->deepCopy());
          rrule->setMath(tcfdiv);
          tcfdiv->removeChild(0);
        }
        //Fall through to:
      case SBML_ASSIGNMENT_RULE:
      case SBML_ALGEBRAIC_RULE:
        //Rules in general need csymbols converted.
        rule = static_cast<Rule*>(element);
        if (rule->isSetMath()) {
          ast1 = rule->getMath()->deepCopy();
          convertCSymbols(ast1, tcfdiv, tcftimes);
          rule->setMath(ast1);
          delete ast1;
        }
        break;
      case SBML_EVENT_ASSIGNMENT:
        //Event assignments need csymbols converted.
        ea = static_cast<EventAssignment*>(element);
        if (ea->isSetMath()) {
          ast1 = ea->getMath()->deepCopy();
          convertCSymbols(ast1, tcfdiv, tcftimes);
          ea->setMath(ast1);
          delete ast1;
        }
        break;
      case SBML_INITIAL_ASSIGNMENT:
        //Initial assignments need csymbols converted.
        ia = static_cast<InitialAssignment*>(element);
        if (ia->isSetMath()) {
          ast1 = ia->getMath()->deepCopy();
          convertCSymbols(ast1, tcfdiv, tcftimes);
          ia->setMath(ast1);
          delete ast1;
        }
        break;
      case SBML_CONSTRAINT:
        //Constraints need csymbols converted.
        constraint = static_cast<Constraint*>(element);
        if (constraint->isSetMath()) {
          ast1 = constraint->getMath()->deepCopy();
          convertCSymbols(ast1, tcfdiv, tcftimes);
          constraint->setMath(ast1);
          delete ast1;
        }
        break;
      case SBML_PRIORITY:
        //Priorities need csymbols converted.
        priority = static_cast<Priority*>(element);
        if (priority->isSetMath()) {
          ast1 = priority->getMath()->deepCopy();
          convertCSymbols(ast1, tcfdiv, tcftimes);
          priority->setMath(ast1);
          delete ast1;
        }
        break;
      case SBML_TRIGGER:
        //Triggers need csymbols converted.
        trigger = static_cast<Trigger*>(element);
        if (trigger->isSetMath()) {
          ast1 = trigger->getMath()->deepCopy();
          convertCSymbols(ast1, tcfdiv, tcftimes);
          trigger->setMath(ast1);
          delete ast1;
        }
        break;
      default:
        //Do nothing!  If we wanted to call a plugin routine, this would be the place.  The only other alternative is to #ifdef some code in here that deals with the math-containing package objects explicitly.  Which might be the best option, all told.
        break;
      }
    }
  }

  delete allelements;

  return LIBSBML_OPERATION_SUCCESS;
}


void Submodel::convertCSymbols(ASTNode*& math, const ASTNode* tcfdiv, const ASTNode* tcftimes)
{
  if (tcfdiv==NULL) return;
  if (math->getType()==AST_NAME_TIME) {
    ASTNode* div = tcfdiv->deepCopy();
    div->insertChild(0, math);
    math = div;
    return;
  }
  for (unsigned int ch=0; ch<math->getNumChildren(); ch++) {
    ASTNode* child = math->getChild(ch);
    convertCSymbols(child, tcfdiv, tcftimes);
    if (child != math->getChild(ch)) {
      math->removeChild(ch);
      math->insertChild(ch, child);
    }
  }
  if (math->getType()==AST_FUNCTION_DELAY) {
    if (math->getNumChildren() != 2) return;
    ASTNode* timechild = math->getChild(1);
    ASTNode* newtime = tcftimes->deepCopy();
    newtime->addChild(timechild);
    math->removeChild(1);
    math->addChild(newtime);
  }
}

void Submodel::createNewConversionFactor(string& cf, const ASTNode* newcf, string oldcf, Model* model)
{
  stringstream npID;
  npID << oldcf << "_times_" << newcf->getName();
  int i=0;
  while (model->getElementBySId(npID.str()) != NULL) {
    i++;
    npID.clear();
    npID << oldcf << "_times_" << newcf->getName() << "_" << i;
  }
  cf = npID.str();

  Parameter* newparam = model->createParameter();
  newparam->setId(cf);
  newparam->setConstant(true);
  InitialAssignment* ia = model->createInitialAssignment();
  ia->setSymbol(cf);
  string math = oldcf + " * " + newcf->getName();
  ia->setMath(SBML_parseL3Formula(math.c_str()));
}


#endif /* __cplusplus */
/** @cond doxygenIgnored */

LIBSBML_EXTERN
Submodel_t *
Submodel_create(unsigned int level, unsigned int version,
                unsigned int pkgVersion)
{
  return new Submodel(level, version, pkgVersion);
}


LIBSBML_EXTERN
void
Submodel_free(Submodel_t * s)
{
  if (s != NULL)
    delete s;
}


LIBSBML_EXTERN
Submodel_t *
Submodel_clone(Submodel_t * s)
{
  if (s != NULL)
  {
    return static_cast<Submodel_t*>(s->clone());
  }
  else
  {
    return NULL;
  }
}


LIBSBML_EXTERN
char *
Submodel_getId(Submodel_t * s)
{
  if (s == NULL)
    return NULL;

  return s->getId().empty() ? NULL : safe_strdup(s->getId().c_str());
}


LIBSBML_EXTERN
char *
Submodel_getName(Submodel_t * s)
{
  if (s == NULL)
    return NULL;

  return s->getName().empty() ? NULL : safe_strdup(s->getName().c_str());
}


LIBSBML_EXTERN
char *
Submodel_getModelRef(Submodel_t * s)
{
  if (s == NULL)
    return NULL;

  return s->getModelRef().empty() ? NULL : safe_strdup(s->getModelRef().c_str());
}


LIBSBML_EXTERN
char *
Submodel_getSubstanceConversionFactor(Submodel_t * s)
{
  return NULL;
}


LIBSBML_EXTERN
char *
Submodel_getTimeConversionFactor(Submodel_t * s)
{
  if (s == NULL)
    return NULL;

  return s->getTimeConversionFactor().empty() ? NULL : safe_strdup(s->getTimeConversionFactor().c_str());
}


LIBSBML_EXTERN
char *
Submodel_getExtentConversionFactor(Submodel_t * s)
{
  if (s == NULL)
    return NULL;

  return s->getExtentConversionFactor().empty() ? NULL : safe_strdup(s->getExtentConversionFactor().c_str());
}


LIBSBML_EXTERN
int
Submodel_isSetId(Submodel_t * s)
{
  return (s != NULL) ? static_cast<int>(s->isSetId()) : 0;
}


LIBSBML_EXTERN
int
Submodel_isSetName(Submodel_t * s)
{
  return (s != NULL) ? static_cast<int>(s->isSetName()) : 0;
}


LIBSBML_EXTERN
int
Submodel_isSetModelRef(Submodel_t * s)
{
  return (s != NULL) ? static_cast<int>(s->isSetModelRef()) : 0;
}


LIBSBML_EXTERN
int
Submodel_isSetSubstanceConversionFactor(Submodel_t * s)
{
  return 0;
}


LIBSBML_EXTERN
int
Submodel_isSetTimeConversionFactor(Submodel_t * s)
{
  return (s != NULL) ? static_cast<int>(s->isSetTimeConversionFactor()) : 0;
}


LIBSBML_EXTERN
int
Submodel_isSetExtentConversionFactor(Submodel_t * s)
{
  return (s != NULL) ? static_cast<int>(s->isSetExtentConversionFactor()) : 0;
}


LIBSBML_EXTERN
int
Submodel_setId(Submodel_t * s, const char * id)
{
  return (s != NULL) ? s->setId(id) : LIBSBML_INVALID_OBJECT;
}


LIBSBML_EXTERN
int
Submodel_setName(Submodel_t * s, const char * name)
{
  return (s != NULL) ? s->setName(name) : LIBSBML_INVALID_OBJECT;
}


LIBSBML_EXTERN
int
Submodel_setModelRef(Submodel_t * s, const char * modelRef)
{
  return (s != NULL) ? s->setModelRef(modelRef) : LIBSBML_INVALID_OBJECT;
}


LIBSBML_EXTERN
int
Submodel_setSubstanceConversionFactor(Submodel_t * s, const char * substanceConversionFactor)
{
  return LIBSBML_INVALID_ATTRIBUTE_VALUE;
}


LIBSBML_EXTERN
int
Submodel_setTimeConversionFactor(Submodel_t * s, const char * timeConversionFactor)
{
  return (s != NULL) ? s->setTimeConversionFactor(timeConversionFactor) : LIBSBML_INVALID_OBJECT;
}


LIBSBML_EXTERN
int
Submodel_setExtentConversionFactor(Submodel_t * s, const char * extentConversionFactor)
{
  return (s != NULL) ? s->setExtentConversionFactor(extentConversionFactor) : LIBSBML_INVALID_OBJECT;
}


LIBSBML_EXTERN
int
Submodel_unsetId(Submodel_t * s)
{
  return (s != NULL) ? s->unsetId() : LIBSBML_INVALID_OBJECT;
}


LIBSBML_EXTERN
int
Submodel_unsetName(Submodel_t * s)
{
  return (s != NULL) ? s->unsetName() : LIBSBML_INVALID_OBJECT;
}


LIBSBML_EXTERN
int
Submodel_unsetModelRef(Submodel_t * s)
{
  return (s != NULL) ? s->unsetModelRef() : LIBSBML_INVALID_OBJECT;
}


LIBSBML_EXTERN
int
Submodel_unsetSubstanceConversionFactor(Submodel_t * s)
{
  return LIBSBML_OPERATION_FAILED;
}


LIBSBML_EXTERN
int
Submodel_unsetTimeConversionFactor(Submodel_t * s)
{
  return (s != NULL) ? s->unsetTimeConversionFactor() : LIBSBML_INVALID_OBJECT;
}


LIBSBML_EXTERN
int
Submodel_unsetExtentConversionFactor(Submodel_t * s)
{
  return (s != NULL) ? s->unsetExtentConversionFactor() : LIBSBML_INVALID_OBJECT;
}


LIBSBML_EXTERN
int
Submodel_addDeletion(Submodel_t * s, Deletion_t * d)
{
  return (s != NULL) ? s->addDeletion(d) : LIBSBML_INVALID_OBJECT;
}


LIBSBML_EXTERN
Deletion_t *
Submodel_createDeletion(Submodel_t * s)
{
  return s->createDeletion();
}


LIBSBML_EXTERN
ListOf_t *
Submodel_getListOfDeletions(Submodel_t * s)
{
  return (s != NULL) ? s->getListOfDeletions() : NULL;
}


LIBSBML_EXTERN
Deletion_t *
Submodel_getDeletion(Submodel_t * s, unsigned int n)
{
  return s->getDeletion(n);
}


LIBSBML_EXTERN
Deletion_t *
Submodel_getDeletionById(Submodel_t * s, const char * sid)
{
  return s->getDeletion(sid);
}


LIBSBML_EXTERN
unsigned int
Submodel_getNumDeletions(Submodel_t * s)
{
  return s->getNumDeletions();
}


LIBSBML_EXTERN
Deletion_t *
Submodel_removeDeletion(Submodel_t * s, unsigned int n)
{
  return s->removeDeletion(n);
}


LIBSBML_EXTERN
Deletion_t *
Submodel_removeDeletionById(Submodel_t * s, const char * sid)
{
  return s->removeDeletion(sid);
}


LIBSBML_EXTERN
int
Submodel_hasRequiredAttributes(Submodel_t * s)
{
  return (s != NULL) ? static_cast<int>(s->hasRequiredAttributes()) : 0;
}


LIBSBML_EXTERN
Submodel_t *
ListOfSubmodels_getById(ListOf_t * lo, const char * sid)
{
  if (lo == NULL)
    return NULL;

  return (sid != NULL) ? static_cast <ListOfSubmodels *>(lo)->get(sid) : NULL;
}


LIBSBML_EXTERN
Submodel_t *
ListOfSubmodels_removeById(ListOf_t * lo, const char * sid)
{
  if (lo == NULL)
    return NULL;

  return (sid != NULL) ? static_cast <ListOfSubmodels *>(lo)->remove(sid) : NULL;
}



/** @endcond */
LIBSBML_CPP_NAMESPACE_END


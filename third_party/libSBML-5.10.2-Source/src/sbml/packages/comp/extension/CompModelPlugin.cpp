/**
 * @file    CompModelPlugin.cpp
 * @brief   Implementation of CompModelPlugin, the plugin class of
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
#include <iostream>
#include <vector>
#include <set>

#include <sbml/common/libsbml-version.h>
#include <sbml/packages/comp/common/compfwd.h>
#include <sbml/packages/comp/extension/CompModelPlugin.h>
#include <sbml/packages/comp/extension/CompExtension.h>
#include <sbml/packages/comp/validator/CompSBMLError.h>
#include <sbml/Model.h>

#include <sbml/util/ElementFilter.h>
#include <sbml/util/PrefixTransformer.h>

using namespace std;

LIBSBML_CPP_NAMESPACE_BEGIN
#ifdef __cplusplus

CompModelPlugin::CompModelPlugin (const std::string &uri, const std::string &prefix, CompPkgNamespaces *compns)
  : CompSBasePlugin(uri,prefix, compns)
  , mListOfSubmodels(compns)
  , mListOfPorts(compns)
  , mDivider("__")
  , mRemoved()
  , mTransformer(NULL)
{
  connectToChild();
}


CompModelPlugin::CompModelPlugin(const CompModelPlugin& orig)
  : CompSBasePlugin(orig)
  , mListOfSubmodels(orig.mListOfSubmodels)
  , mListOfPorts(orig.mListOfPorts)
  , mDivider("__")
  , mRemoved() //If we're making a copy, the list of things we've removed is new.
  , mTransformer(orig.mTransformer)
{
  connectToChild();
}


CompModelPlugin::~CompModelPlugin () 
{
}

CompModelPlugin& 
CompModelPlugin::operator=(const CompModelPlugin& orig)
{
  if(&orig!=this)
  {
    CompSBasePlugin::operator =(orig);
    mListOfSubmodels = orig.mListOfSubmodels;
    mListOfPorts = orig.mListOfPorts;
    mDivider = orig.mDivider;
    mRemoved.clear(); //If we're making a copy, the list of things we've removed is new.
    mTransformer = orig.mTransformer;
    connectToChild();
  }
  return *this;
}


CompModelPlugin* 
CompModelPlugin::clone () const
{
  return new CompModelPlugin(*this);  
}


/** @cond doxygenLibsbmlInternal */
SBase*
CompModelPlugin::createObject(XMLInputStream& stream)
{
  SBase*        object = 0;

  const std::string&   name   = stream.peek().getName();
  const XMLNamespaces& xmlns  = stream.peek().getNamespaces();
  const std::string&   prefix = stream.peek().getPrefix();

  const std::string& targetPrefix = (xmlns.hasURI(mURI)) ? xmlns.getPrefix(mURI) : mPrefix;
  
  if (prefix == targetPrefix)
  {
    if ( name == "listOfSubmodels" ) 
    {
      if (mListOfSubmodels.size() != 0)
      {
        getErrorLog()->logPackageError("comp", CompOneListOfOnModel, 
          getPackageVersion(), getLevel(), getVersion());
      }

      object = &mListOfSubmodels;
    
      if (targetPrefix.empty())
      {
        //
        // (NOTE)
        //
        // A top-level element (listOfSubmodels) of the comp extension is located 
        // in a default namespace, and thus xmlns=".." attribute must be added to 
        // the element.
        // This is done by invoking SBMLDocument::enableDefaultNS() function with 
        // the two arguments (the uri of this package and true value).
        //
        mListOfSubmodels.getSBMLDocument()->enableDefaultNS(mURI,true);
      }
    }
    if ( name == "listOfPorts" ) 
    {
      if (mListOfPorts.size() != 0)
      {
        getErrorLog()->logPackageError("comp", CompOneListOfOnModel, 
          getPackageVersion(), getLevel(), getVersion());
      }

      object = &mListOfPorts;
    
      if (targetPrefix.empty())
      {
        mListOfPorts.getSBMLDocument()->enableDefaultNS(mURI,true);
      }
    }
  }    

  return object;
}
/** @endcond */

/** @cond doxygenLibsbmlInternal */
void
CompModelPlugin::writeElements (XMLOutputStream& stream) const
{
  if (getNumSubmodels() > 0)
  {
    mListOfSubmodels.write(stream);
  }
  if (getNumPorts() > 0)
  {
    mListOfPorts.write(stream);
  }    
}
/** @endcond */

SBase* 
CompModelPlugin::getElementBySId(const std::string& id)
{
  if (id.empty()) return NULL;
  SBase* obj = mListOfSubmodels.getElementBySId(id);
  if (obj != NULL) return obj;
  obj = mListOfPorts.getElementBySId(id);
  if (obj != NULL) return obj;
  return NULL;
}


SBase*
CompModelPlugin::getElementByMetaId(const std::string& metaid)
{
  if (metaid.empty()) return NULL;
  if (mListOfSubmodels.getMetaId() == metaid) return &mListOfSubmodels;
  if (mListOfPorts.getMetaId() == metaid) return &mListOfPorts;

  SBase* obj = mListOfSubmodels.getElementByMetaId(metaid);
  if (obj != NULL) return obj;
  obj = mListOfPorts.getElementByMetaId(metaid);
  if (obj != NULL) return obj;
  return NULL;
}

List*
  CompModelPlugin::getAllElements(ElementFilter *filter)
{
  List* ret = new List();
  List* sublist = NULL;

  ADD_FILTERED_LIST(ret, sublist, mListOfSubmodels, filter);
  ADD_FILTERED_LIST(ret, sublist, mListOfPorts, filter);

  return ret;
}


const ListOfSubmodels*
CompModelPlugin::getListOfSubmodels () const
{
  return &mListOfSubmodels;
}


Submodel*
CompModelPlugin::removeSubmodel(unsigned int index)
{
  return mListOfSubmodels.remove(index);
}


Submodel* 
CompModelPlugin::getSubmodel (unsigned int index)
{
  return mListOfSubmodels.get(index);
}

const Submodel* 
CompModelPlugin::getSubmodel (unsigned int index) const
{
  return mListOfSubmodels.get(index);
}


Submodel* 
CompModelPlugin::getSubmodel (std::string id)
{
  return mListOfSubmodels.get(id);
}

const Submodel* 
CompModelPlugin::getSubmodel (std::string id) const
{
  return mListOfSubmodels.get(id);
}


int
CompModelPlugin::addSubmodel (const Submodel* submodel)
{
  if (submodel == NULL)
  {
    return LIBSBML_INVALID_OBJECT;
  }
  else if (!(submodel->hasRequiredAttributes()) || !(submodel->hasRequiredElements()))
  {
    return LIBSBML_INVALID_OBJECT;
  }
  else if (getLevel() != submodel->getLevel())
  {
    return LIBSBML_LEVEL_MISMATCH;
  }
  else if (getVersion() != submodel->getVersion())
  {
    return LIBSBML_VERSION_MISMATCH;
  }
  else if (getPackageVersion() != submodel->getPackageVersion())
  {
    return LIBSBML_PKG_VERSION_MISMATCH;
  }
  else
  {
    return mListOfSubmodels.append(submodel);
  }
}


unsigned int
CompModelPlugin::getNumSubmodels () const
{
  return mListOfSubmodels.size();
}


/*
 * Creates a Submodel object, adds it to the end of the submodel
 * objects list and returns a reference to the newly created object.
 */
Submodel*
CompModelPlugin::createSubmodel ()
{
  COMP_CREATE_NS(compns, getSBMLNamespaces());
  Submodel* m = new Submodel(compns);
  mListOfSubmodels.appendAndOwn(m);
  delete compns;
  return m;
}


/*
 * Returns the listofports object that holds all ports.
 */ 
const ListOfPorts*
CompModelPlugin::getListOfPorts () const
{
  return &mListOfPorts;
}


/*
 * Remove the portRef with the given index.
 * A pointer to the removed port is returned.
 * If no portRef has been removed, @c NULL is returned.
 */
Port*
CompModelPlugin::removePort(unsigned int index)
{
  return mListOfPorts.remove(index);
}


int 
CompModelPlugin::setDivider(const std::string& divider)
{
  if (divider.empty()) return LIBSBML_OPERATION_FAILED;
  if (!SyntaxChecker::isValidSBMLSId("a" + divider + "a")) return LIBSBML_OPERATION_FAILED;
  mDivider = divider;
  return LIBSBML_OPERATION_SUCCESS;
}


string 
CompModelPlugin::getDivider()
{
  return mDivider;
}

/*
 * Returns the port with the given index.
 * If the index is invalid, @c NULL is returned.
 */ 
Port* 
CompModelPlugin::getPort (unsigned int index)
{
  return mListOfPorts.get(index);
}

/*
 * Returns the port with the given index.
 * If the index is invalid, @c NULL is returned.
 */ 
const Port* 
CompModelPlugin::getPort (unsigned int index) const
{
  return mListOfPorts.get(index);
}

/*
 * Returns the port with the given @p id.
 * If the id is invalid, @c NULL is returned.
 */ 
Port* 
CompModelPlugin::getPort (std::string id)
{
  return mListOfPorts.get(id);
}

/*
 * Returns the port with the given @p id.
 * If the id is invalid, @c NULL is returned.
 */ 
const Port* 
CompModelPlugin::getPort (std::string id) const
{
  return mListOfPorts.get(id);
}


int
CompModelPlugin::addPort (const Port* port)
{
  if (port == NULL)
  {
    return LIBSBML_INVALID_OBJECT;
  }
  else if (!(port->hasRequiredAttributes()) || !(port->hasRequiredElements()))
  {
    return LIBSBML_INVALID_OBJECT;
  }
  else if (getLevel() != port->getLevel())
  {
    return LIBSBML_LEVEL_MISMATCH;
  }
  else if (getVersion() != port->getVersion())
  {
    return LIBSBML_VERSION_MISMATCH;
  }
  else if (getPackageVersion() != port->getPackageVersion())
  {
    return LIBSBML_PKG_VERSION_MISMATCH;
  }
  else
  {
    return mListOfPorts.append(port);
  }
}


unsigned int
CompModelPlugin::getNumPorts () const
{
  return mListOfPorts.size();
}


/*
 * Creates a Port object, adds it to the end of the portRef
 * objects list and returns a reference to the newly created object.
 */
Port*
CompModelPlugin::createPort ()
{
  COMP_CREATE_NS(compns, getSBMLNamespaces());
  Port* m = new Port(compns);
  mListOfPorts.appendAndOwn(m);
  delete compns;
  return m;
}


/** @cond doxygenLibsbmlInternal */
/*
 * Sets the parent SBMLDocument of this SBML object.
 */
void 
CompModelPlugin::setSBMLDocument (SBMLDocument* d)
{
  SBasePlugin::setSBMLDocument(d);

  mListOfSubmodels.setSBMLDocument(d);  
  mListOfPorts.setSBMLDocument(d);  
}
/** @endcond */


/** @cond doxygenLibsbmlInternal */
void
CompModelPlugin::connectToChild()
{
  CompSBasePlugin::connectToChild();
  connectToParent(getParentSBMLObject());
}
/** @endcond */


/** @cond doxygenLibsbmlInternal */
void
CompModelPlugin::connectToParent (SBase* sbase)
{
  CompSBasePlugin::connectToParent(sbase);
  mListOfSubmodels.connectToParent(sbase);
  mListOfPorts.connectToParent(sbase);
}
/** @endcond */


/** @cond doxygenLibsbmlInternal */
/*
 * Enables/Disables the given package with child elements in this plugin
 * object (if any).
 */
void
CompModelPlugin::enablePackageInternal(const std::string& pkgURI,
                                       const std::string& pkgPrefix, bool flag)
{
  mListOfSubmodels.enablePackageInternal(pkgURI,pkgPrefix,flag);
  mListOfPorts.enablePackageInternal(pkgURI,pkgPrefix,flag);
}
/** @endcond */


Model* CompModelPlugin::flattenModel() const
{
  //First make a copy of our parent (the model to be flattened):
  const Model* parent = static_cast<const Model*>(getParentSBMLObject());
  
  if (parent==NULL) {
    return NULL;
  }
  //doc needs to be non-const so that the error messages can be updated.  Otherwise, nothing changes.
  SBMLDocument* doc = const_cast<SBMLDocument*>(getSBMLDocument());
  if (doc==NULL) {
    return NULL;
  }

  //Set the original document so that it can find the model definitions 
  //and external model definitions while we flatten.
  Model* flat = parent->clone();
  flat->setSBMLDocument(doc);
  CompModelPlugin* flatplug = 
    static_cast<CompModelPlugin*>(flat->getPlugin(getPrefix()));

  // Now instantiate its submodels and 
  // follow all renaming/deletion/replacement rules.
  vector<const Model*> submods;
  int success = flatplug->instantiateSubmodels();

  if (success != LIBSBML_OPERATION_SUCCESS) {
    //instantiateSubmodels sets its own error messages.
    delete flat;
    return NULL;
  }

  //Now start the aggregation process.  
  //This goes from the bottom up, calling 'appendFrom' iteratively 
  //(from the plugin).
  for (unsigned int sm=0; sm<flatplug->getNumSubmodels(); sm++) 
  {
    Model* submodel = flatplug->getSubmodel(sm)->getInstantiation();
    if (submodel==NULL) {
      //getInstantiation should be calling a cached value by now, but if not, it will set its own error messages.
      delete flat;
      return NULL;
    }
    CompModelPlugin* submodplug = 
      static_cast<CompModelPlugin*>(submodel->getPlugin(getPrefix()));

    if (submodplug != NULL) {
      //Strip the ports from the submodel, as we no longer need them.
      while (submodplug->getNumPorts() > 0) 
      {
        submodplug->removePort(0);
      }
    }
    success = flat->appendFrom(submodel);
    if (success != LIBSBML_OPERATION_SUCCESS) {
      string error = "Unable to flatten model in CompModelPlugin::flattenModel: appending elements from the submodel '" + submodel->getId() + "' to the elements of the parent model failed.";
      doc->getErrorLog()->logPackageError("comp", CompModelFlatteningFailed, getPackageVersion(), getLevel(), getVersion(), error);
      delete flat;
      return NULL;
    }
  }

  // Now we clear the saved referenced elements in the local Port objects, 
  // but point them to the new object if necessary.
  flatplug->resetPorts();

  // Next, strip the package info from 'flat'.  
  // We're going to remove everything but the Ports:
  flatplug->mListOfSubmodels.clear();
  flatplug->clearReplacedElements();
  flatplug->unsetReplacedBy();
  
  List* allelements = flat->getAllElements();
  
  vector<SBase*> nonReplacedElements;
  
  for (unsigned int el=0; el<allelements->getSize(); el++) 
  {
    SBase* element = static_cast<SBase*>(allelements->get(el));
    int type = element->getTypeCode();
    if (!(type==SBML_COMP_REPLACEDBY ||
          type==SBML_COMP_REPLACEDELEMENT ||
          type==SBML_COMP_SBASEREF)) 
    {
            nonReplacedElements.push_back(element);
    }
  }

  // delete the list
  delete allelements;

  for (unsigned int el=0; el<nonReplacedElements.size(); el++) 
  {
    SBase* element = nonReplacedElements[el];
    CompSBasePlugin* elplug = 
      static_cast<CompSBasePlugin*>(element->getPlugin(getPrefix()));
    if (elplug != NULL) 
    {
      elplug->clearReplacedElements();
      elplug->unsetReplacedBy();
    }
  }



  //Finally, unset the document again.
  flat->setSBMLDocument(NULL);

  return flat;
}

/** @cond doxygenLibsbmlInternal */
void CompModelPlugin::resetPorts()
{
  for (unsigned int p=0; p<getNumPorts(); p++) {
    Port* port = getPort(p);
    SBase* referenced = port->getReferencedElement();
    if (port->isSetSBaseRef()) {
      port->unsetSBaseRef();
      port->unsetIdRef();
      port->unsetMetaIdRef();
      port->unsetUnitRef();
      int type = referenced->getTypeCode();
      if (referenced->isSetId() && 
          type != SBML_INITIAL_ASSIGNMENT &&
          type != SBML_ASSIGNMENT_RULE &&
          type != SBML_RATE_RULE &&
          type != SBML_EVENT_ASSIGNMENT) {
        if (type==SBML_UNIT_DEFINITION) {
          port->setUnitRef(referenced->getId());
        }
        else {
          port->setIdRef(referenced->getId());
        }
      }
      else if (referenced->isSetMetaId()) {
        port->setMetaIdRef(referenced->getMetaId());
      }
      else {
        stringstream newname;
        newname << "auto_port_" << p;
        referenced->setMetaId(newname.str());
        port->setMetaIdRef(newname.str());
      }
    }
    port->clearReferencedElement();
  }
}
/** @endcond */


int 
CompModelPlugin::appendFrom(const Model* model)
{
  if (model==NULL) 
    return LIBSBML_INVALID_OBJECT;

  const CompModelPlugin* modplug = 
    static_cast<const CompModelPlugin*>(model->getPlugin(getPrefix()));
  
  if (modplug==NULL) 
    return LIBSBML_INVALID_OBJECT;

  Model* parent = static_cast<Model*>(getParentSBMLObject());

  if (parent==NULL) 
    return LIBSBML_INVALID_OBJECT;
  
  for (unsigned int sm = 0; sm<modplug->getNumSubmodels(); sm++) 
  {
    const Submodel* sub = modplug->getSubmodel(sm);
    int ret = parent->appendFrom(sub->getInstantiation());
    
    if (ret != LIBSBML_OPERATION_SUCCESS) 
      return ret;
  }
  return LIBSBML_OPERATION_SUCCESS;
}

/*
 * Loop through all Submodels in this Model, instantiate all of them, 
 * perform all deletions, and synchronize all replacements.  
 * The resulting models are stored in the Submodel objects, 
 * and available from 'Submodel::getInstantiation()'
 */
int 
CompModelPlugin::instantiateSubmodels()
{
  Model* model = static_cast<Model*>(getParentSBMLObject());
  
  if (model==NULL) 
    return LIBSBML_INVALID_OBJECT;
  
  int ret;

  // First we instantiate all the submodels.  
  // This acts recursively downward through the stack.
  for (unsigned int sub=0; sub<mListOfSubmodels.size(); sub++) 
  {
    Submodel* submodel = mListOfSubmodels.get(sub);
    // Instead of 'instantiate', since we might have already 
    // been instantiated ourselves from above.
    Model* submodinst = submodel->getInstantiation(); 
    
    if (submodinst == NULL ) {
      //'getInstantiation' already sets any errors that might have occurred.
      return LIBSBML_OPERATION_FAILED;
    }
    
    // if we have a transformer specified, then we need to propagate it, so it can
    // be used
    if (isSetTransformer())
    {
      CompModelPlugin* other = dynamic_cast<CompModelPlugin*>(submodinst->getPlugin("comp"));
      if (other != NULL)
        other->setTransformer(getTransformer());
    }
  }

  // Next, recursively find all the targets of SBaseRef elements 
  // and save them, since we're about to rename everything and 
  // we won't be able to find things by name any more.
  ret = saveAllReferencedElements(); 
  if (ret != LIBSBML_OPERATION_SUCCESS) {
    //saveAllReferencedElements sets any errors.
    return ret;
  }

  mRemoved.clear();
  set<SBase*> toremove;

  // Collect deletions (top-down):  
  // need to do this before renaming in case we delete a local parameter.
  ret = collectDeletionsAndDeleteSome(&mRemoved, &toremove);

  if (ret != LIBSBML_OPERATION_SUCCESS) {
    return ret;
  }

  //Next, we rename *all* the elements so everything is unique.
  ret = renameAllIDsAndPrepend("");
  if (ret != LIBSBML_OPERATION_SUCCESS) {
    return ret;
  }

  //Perform replacements and conversions (top-down) and collect them.
  ret = collectRenameAndConvertReplacements(&mRemoved, &toremove);

  if (ret != LIBSBML_OPERATION_SUCCESS) {
    return ret;
  }

  //Finally, actually remove the collected elements from the model--they are
  // all now redundant.  Have to wait until now to do this, because of the
  // possibility of nested constructs:  replacing the child of a replaced
  // element, for example, or even replacing the child of a deleted
  // element.
  removeCollectedElements(&mRemoved, &toremove);

  mRemoved.clear();

  return LIBSBML_OPERATION_SUCCESS;
}

int CompModelPlugin::saveAllReferencedElements()
{
  set<SBase*> norefs;
  return saveAllReferencedElements(norefs, norefs);
}

int CompModelPlugin::saveAllReferencedElements(set<SBase*> uniqueRefs, set<SBase*> replacedBys)
{
  SBMLDocument* doc = getSBMLDocument();
  Model* model = static_cast<Model*>(getParentSBMLObject());
  if (model==NULL) {
    if (doc) {
      string error = "Unable to discover any referenced elements in CompModelPlugin::saveAllReferencedElements: no Model parent of the 'comp' model plugin.";
      doc->getErrorLog()->logPackageError("comp", CompModelFlatteningFailed, getPackageVersion(), getLevel(), getVersion(), error);
    }
    return LIBSBML_OPERATION_FAILED;
  }
  int ret = LIBSBML_OPERATION_SUCCESS;

  //Get a list of everything, pull out anything that's a deletion, replacement, or port, and save what they're pointing to.
  //At the same time, make sure that no two things point to the same thing.
  set<SBase*> RE_deletions = set<SBase*>(); //Deletions only point to things in the same model.
  List* allElements = model->getAllElements();
  string modname = "the main model in the document";
  if (model->isSetId()) {
    modname = "the model '" + model->getId() + "'";
  }
  for (unsigned int el=0; el<allElements->getSize(); el++) {
    SBase* element = static_cast<SBase*>(allElements->get(el));
    int type = element->getTypeCode();
    if (type==SBML_COMP_DELETION ||
        type==SBML_COMP_REPLACEDBY ||
        type==SBML_COMP_REPLACEDELEMENT ||
        type==SBML_COMP_PORT) {
          //Don't worry about SBML_COMP_SBASEREF because they're all children of one of the above types.
          SBaseRef* reference = static_cast<SBaseRef*>(element);
          ReplacedElement* re = static_cast<ReplacedElement*>(element);
          ret = reference->saveReferencedElement();
          if (ret != LIBSBML_OPERATION_SUCCESS) 
          {
            if (type != SBML_COMP_REPLACEDBY && doc) 
            {
              SBMLErrorLog* errlog = doc->getErrorLog();
              SBMLError* lasterr = const_cast<SBMLError*>
                (doc->getErrorLog()->getError(doc->getNumErrors()-1));
              if ( (errlog->contains(UnrequiredPackagePresent) || 
                    errlog->contains(RequiredPackagePresent))) 
              {
                if ( lasterr->getErrorId() == CompIdRefMustReferenceObject)
                {
                   //Change the error into a warning
                   string fullmsg = lasterr->getMessage() 
                     + "  However, this may be because of the unrecognized "
                     + "package present in this document:  ignoring this "
                     + "element and flattening anyway.";
                   errlog->remove(lasterr->getErrorId());
                   errlog->logPackageError("comp", 
                     CompIdRefMayReferenceUnknownPackage, getPackageVersion(), 
                     getLevel(), getVersion(), fullmsg, element->getLine(), 
                     element->getColumn(), LIBSBML_SEV_WARNING);
                    element->removeFromParentAndDelete();
                    continue;
                }
                else if ( lasterr->getErrorId() == CompMetaIdRefMustReferenceObject)
                {
                   //Change the error into a warning
                   string fullmsg = lasterr->getMessage() 
                     + "  However, this may be because of the unrecognized "
                     + "package present in this document:  ignoring this "
                     + "element and flattening anyway.";
                   errlog->remove(lasterr->getErrorId());
                   errlog->logPackageError("comp", 
                     CompMetaIdRefMayReferenceUnknownPkg, getPackageVersion(), 
                     getLevel(), getVersion(), fullmsg, element->getLine(), 
                     element->getColumn(), LIBSBML_SEV_WARNING);
                    element->removeFromParentAndDelete();
                    continue;
                }
                else if (lasterr->getErrorId() == 
                                  CompIdRefMayReferenceUnknownPackage)
                {
                  element->removeFromParentAndDelete();
                  continue;
                }
                else if (lasterr->getErrorId() == 
                                  CompMetaIdRefMayReferenceUnknownPkg)
                {
                  element->removeFromParentAndDelete();
                  continue;
                }
              }
              else 
              {
                delete allElements;
                return ret;
              }
            }
            else {
              delete allElements;
              return ret;
            }
          }
          SBase* direct = reference->getDirectReference();
          bool adddirect = true;
          if (type == SBML_COMP_REPLACEDBY) {
            SBase* rbParent = reference->getParentSBMLObject();
            if (uniqueRefs.insert(rbParent).second == false) {
              if (doc) {
                string error = "Error discovered in CompModelPlugin::saveAllReferencedElements when checking " + modname + ": a <" + rbParent->getElementName() + "> ";
                if (direct->isSetId()) {
                  error += "with the id '" + rbParent->getId() + "'";
                  if (rbParent->isSetMetaId()) {
                    error += ", and the metaid '" + rbParent->getMetaId() + "'";
                  }
                }
                else if (rbParent->isSetMetaId()) {
                  error += "with the metaId '" + rbParent->getMetaId() + "'";
                }
                error += " has a <replacedBy> child and is also pointed to by a <port>, <deletion>, <replacedElement>, or one or more <replacedBy> objects.";
                doc->getErrorLog()->logPackageError("comp", CompNoMultipleReferences, getPackageVersion(), getLevel(), getVersion(), error);
              }
              delete allElements;
              return LIBSBML_OPERATION_FAILED;
            }
            adddirect = replacedBys.insert(direct).second;
          }
          if (type==SBML_COMP_REPLACEDELEMENT && re->isSetDeletion()) {
            adddirect = RE_deletions.insert(direct).second;
          }
          if (adddirect) {
            if (uniqueRefs.insert(direct).second == false) {
              if (doc) {
                string error = "Error discovered in CompModelPlugin::saveAllReferencedElements when checking " + modname + ": ";
                if (replacedBys.find(direct) != replacedBys.end()) {
                  error += "one or more <replacedBy> elements, plus a <deletion>, <replacedElement>, or <port> element";
                }
                else if (RE_deletions.find(direct) != RE_deletions.end()) {
                  error += "one or more <replacedElement> elements using a 'deletion' attribute, plus a <deletion>, <replacedElement>, or <port> element";
                }
                else {
                  error += "multiple <deletion>, <replacedElement>, and/or <port> elements";
                }
                error += " point directly to the <" + direct->getElementName() + "> ";
                if (direct->isSetId()) {
                  error += "with the id '" + direct->getId() + "'";
                  if (direct->isSetMetaId()) {
                    error += ", and the metaid '" + direct->getMetaId() + "'";
                  }
                  error += ".";
                }
                else if (direct->isSetMetaId()) {
                  error += "with the metaId '" + direct->getMetaId() + "'.";
                }
                doc->getErrorLog()->logPackageError("comp", CompNoMultipleReferences, getPackageVersion(), getLevel(), getVersion(), error);
              }
              delete allElements;
              return LIBSBML_OPERATION_FAILED;
            }
          }
    }
  }

  delete allElements;

  //Now call saveAllReferencedElements for all instantiated submodels.
  for (unsigned long sm=0; sm<getNumSubmodels(); ++sm) {
    Model* sub = getSubmodel((unsigned int)sm)->getInstantiation();
    if (sub==NULL) {
      return LIBSBML_OPERATION_FAILED;
    }
    CompModelPlugin* subplug = static_cast<CompModelPlugin*>(sub->getPlugin(getPrefix()));
    if (subplug==NULL) {
      return LIBSBML_OPERATION_FAILED;
    }
    ret = subplug->saveAllReferencedElements(uniqueRefs, replacedBys);
    if (ret != LIBSBML_OPERATION_SUCCESS) {
      return ret;
    }
  }

  return LIBSBML_OPERATION_SUCCESS;
}

int 
CompModelPlugin::renameAllIDsAndPrepend(const std::string& prefix)
{
  SBMLDocument* doc = getSBMLDocument();
  Model* model = static_cast<Model*>(getParentSBMLObject());
  if (model==NULL) {
    if (doc) {
      string error = "Unable to rename elements in CompModelPlugin::renameAllIDsAndPrepend: no parent model could be found for the given 'comp' model plugin element.";
      doc->getErrorLog()->logPackageError("comp", CompModelFlatteningFailed, getPackageVersion(), getLevel(), getVersion(), error);
    }
    return LIBSBML_INVALID_OBJECT;
  }

  //First rename the elements in all instantiated submodels.
  vector<string> submodids;

  for (unsigned int sm=0; sm<getNumSubmodels(); sm++) {
    Submodel* subm=getSubmodel(sm);
    if (subm==NULL) {
      if (doc) {
        stringstream error;
        error << "Unable to rename elements in CompModelPlugin::renameAllIDsAndPrepend: no valid submodel number " << sm << "for model " << model->getId();
        doc->getErrorLog()->logPackageError("comp", CompModelFlatteningFailed,
          getPackageVersion(), getLevel(), getVersion(), error.str());
      }
      return LIBSBML_OPERATION_FAILED;
    }
    if (!subm->isSetId()) {
      if (doc) {
        stringstream error;
        error << "Unable to rename elements in CompModelPlugin::renameAllIDsAndPrepend: submodel number " << sm << "for model " << model->getId() << " is invalid: it has no 'id' attribute set.";
        doc->getErrorLog()->logPackageError("comp", CompSubmodelAllowedAttributes,
          getPackageVersion(), getLevel(), getVersion(), error.str());
      }
      return LIBSBML_INVALID_OBJECT;
    }
    submodids.push_back(subm->getId());
  }

  //Check to see if any of the various submodel ids are used as a prefix 
  List* allElements = model->getAllElements();
  findUniqueSubmodPrefixes(submodids, allElements);

  //Now that we've found valid prefixes for all our submodels, call this function recursively on them.
  for (unsigned int sm=0; sm<getNumSubmodels(); sm++) {
    Submodel* subm=getSubmodel(sm); //already checked this above.
    Model* inst = subm->getInstantiation();
    if (inst==NULL) {
      //'getInstantiation' will set its own error messages.
      delete allElements;
      return LIBSBML_OPERATION_FAILED;
    }
    CompModelPlugin* instp = static_cast<CompModelPlugin*>(inst->getPlugin(getPrefix()));
    if (instp==NULL) {
      if (doc) {
        //Shouldn't happen:  'getInstantiation' turns on the comp plugin.
        string error = "Unable to rename elements in CompModelPlugin::renameAllIDsAndPrepend: no valid 'comp' plugin for the model instantiated from submodel " + subm->getId();
        doc->getErrorLog()->logPackageError("comp", CompModelFlatteningFailed, getPackageVersion(), getLevel(), getVersion(), error);
      }
      delete allElements;
      return LIBSBML_OPERATION_FAILED;
    }
    int ret = instp->renameAllIDsAndPrepend(prefix + submodids[sm]);
    if (ret != LIBSBML_OPERATION_SUCCESS) {
      //'renameAllIds..' will set its own error messages.
      delete allElements;
      return ret;
    }
  }

  //Finally, actually rename the elements in *this* model with the prefix.
  if (prefix.empty()) 
  {
    delete allElements;
    return LIBSBML_OPERATION_SUCCESS; //Nothing to add
  }

  //Rename the SIds, UnitSIds, and MetaIDs, and references to them.
  renameIDs(allElements, prefix);
  delete allElements;

  return LIBSBML_OPERATION_SUCCESS;
}


/** @cond doxygenLibsbmlInternal */
void CompModelPlugin::findUniqueSubmodPrefixes(vector<string>& submodids, List* allElements)
{
  vector<int> suffixes(submodids.size(), 0);
  bool done=false;
  while (!done) {
    done = true;
    for (size_t str=0; str<submodids.size(); str++) {
      stringstream fullprefix;
      fullprefix << submodids[str];
      if (suffixes[str] > 0) {
        fullprefix << suffixes[str];
      }
      fullprefix << getDivider();
      for (unsigned long el=0; el<allElements->getSize(); ++el) {
        SBase* element = static_cast<SBase*>(allElements->get((unsigned int)el));
        if (element==NULL) {
          assert(false);
          continue;
        }
        if (element->isSetId() && element->getId().find(fullprefix.str())==0) {
          done = false;
          continue;
        }
        else if (element->isSetMetaId() && element->getMetaId().find(fullprefix.str())==0) {
          done = false;
          continue;
        }
        else if (element->hasNonstandardIdentifierBeginningWith(fullprefix.str())) {
          done = false;
          continue;
        }
        else {
          for (unsigned int p=0; p<element->getNumPlugins(); p++) {
            if (element->getPlugin(p)->hasIdentifierBeginningWith(fullprefix.str())) {
              done = false;
              continue;
            }
          }
        }
      }
      if (!done) {
        suffixes[str]++;
        continue; //Start over from the first ID; otherwise we end up checking a lot of things twice.
      }
    }
  }
  //Now change the submodid's:
  for (size_t str=0; str<submodids.size(); str++) {
    stringstream fullprefix;
    fullprefix << submodids[str];
    if (suffixes[str] > 0) {
      fullprefix << suffixes[str];
    }
    fullprefix << getDivider();
    submodids[str] = fullprefix.str();
  }
}
/** @endcond */

/** @cond doxygenLibsbmlInternal */
void CompModelPlugin::renameIDs(List* allElements, const string& prefix)
{
  if (prefix=="") return; //Nothing to prepend.
  vector<pair<string, string> > renamedSIds;
  vector<pair<string, string> > renamedUnitSIds;
  vector<pair<string, string> > renamedMetaIds;
  
  // if a custom prefix transformer was specified, then set the 
  // current prefix
  if (isSetTransformer())
    mTransformer->setPrefix(prefix);
  
  for (unsigned long el=0; el < allElements->getSize(); ++el) 
  {
    SBase* element = static_cast<SBase*>(allElements->get((unsigned int)el));
    string id = element->getId();
    string metaid = element->getMetaId();
    
    // if a custom prefix transformer was specified, use it, other wise
    // default to the sbase method. 
    if (isSetTransformer())
      element->transformIdentifiers(mTransformer);
    else 
      element->prependStringToAllIdentifiers(prefix);

    if (element->getTypeCode() == SBML_LOCAL_PARAMETER) {
      element->setId(id); //Change it back.  This would perhaps be better served by overriding 'prependStringToAllIdentifiers' but hey.
    }
    string newid = element->getId();
    string newmetaid = element->getMetaId();
    if (id != newid) {
      int type = element->getTypeCode();
      if (type==SBML_UNIT_DEFINITION) {
        renamedUnitSIds.push_back(make_pair(id, newid));
      }
      else if (type==SBML_COMP_PORT) {
        //Do nothing--these can only be referenced from outside the Model, so they need to be handled specially.
        // (In the default case, we throw them away).
      }
      else {
        //This is a little dangerous, but hey!  What's a little danger between friends!
        //(What we are assuming is that any attribute you can get with 'getId' is of the type 'SId')
        renamedSIds.push_back(make_pair(id, newid));
      }
    }
    if (metaid != newmetaid) {
      renamedMetaIds.push_back(make_pair(metaid, newmetaid));
    }
  }

  for (unsigned long el=0; el<allElements->getSize(); el++) {
    SBase* element = static_cast<SBase*>(allElements->get((unsigned int)el));
    for (size_t id=0; id<renamedSIds.size(); id++) 
    {
      element->renameSIdRefs(renamedSIds[id].first, renamedSIds[id].second);
    }
    for (size_t uid=0; uid<renamedUnitSIds.size(); uid++) 
    {
      element->renameUnitSIdRefs(renamedUnitSIds[uid].first, renamedUnitSIds[uid].second);
    }
    for (size_t mid=0; mid<renamedMetaIds.size(); mid++) 
    {
      element->renameMetaIdRefs(renamedMetaIds[mid].first, renamedMetaIds[mid].second);
    }
  }
}
/** @endcond */

int CompModelPlugin::collectDeletionsAndDeleteSome(set<SBase*>* removed, set<SBase*>* toremove)
{
  int ret = LIBSBML_OPERATION_SUCCESS;
  SBMLDocument* doc = getSBMLDocument();
  Model* model = static_cast<Model*>(getParentSBMLObject());
  if (model==NULL) {
    if (doc) {
      string error = "Unable to attempt to perform deletions in CompModelPlugin::collectDeletionsAndDeleteSome: no parent model could be found for the given 'comp' model plugin element.";
      doc->getErrorLog()->logPackageError("comp", CompModelFlatteningFailed, getPackageVersion(), getLevel(), getVersion(), error);
    }
    return LIBSBML_OPERATION_FAILED;
  }

  //Since deletions only exist in submodels, loop through the submodels.
  for (unsigned int sub=0; sub<getNumSubmodels(); sub++) {
    Submodel* submodel = getSubmodel(sub);
    //First perform any deletions
    for (unsigned int d=0; d<submodel->getNumDeletions(); d++) {
      Deletion* deletion = submodel->getDeletion(d);
      SBase* todel = deletion->getReferencedElement();
      if (todel && (todel->getTypeCode() == SBML_COMP_DELETION ||
                    todel->getTypeCode() == SBML_COMP_REPLACEDBY ||
                    todel->getTypeCode() == SBML_COMP_REPLACEDELEMENT ||
                    todel->getTypeCode() == SBML_LOCAL_PARAMETER) )
      {
        //Go ahead and delete it!
        set<SBase*> newToRemove;
        newToRemove.insert(todel);
        removeCollectedElements(removed, &newToRemove);
      }
      else {
        //Otherwise, just collect it.
        ret = deletion->collectDeletions(removed, toremove);
        if (ret!=LIBSBML_OPERATION_SUCCESS) {
          return ret;
        }
      }
    }
    //Next collect any deletions in that instantiated submodel (any that weren't just deleted)
    Model* mod = submodel->getInstantiation();
    if (mod==NULL) {
      //getInstantiation sets its own error messages.
      return LIBSBML_OPERATION_FAILED;
    }
    CompModelPlugin* modplug = static_cast<CompModelPlugin*>(mod->getPlugin(getPrefix()));
    if (modplug==NULL) {
      if (doc) {
        //Shouldn't happen:  'getInstantiation' turns on the comp plugin.
        string error = "Unable to rename elements in CompModelPlugin::collectDeletionsAndDeleteSome: no valid 'comp' plugin for the model instantiated from submodel " + submodel->getId();
        doc->getErrorLog()->logPackageError("comp", CompModelFlatteningFailed, getPackageVersion(), getLevel(), getVersion(), error);
      }
      return LIBSBML_OPERATION_FAILED;
    }
    modplug->collectDeletionsAndDeleteSome(removed, toremove);
  }
  return ret;
}


/** @cond doxygenLibsbmlInternal */
int CompModelPlugin::performDeletions()
{
  SBMLDocument* doc = getSBMLDocument();
  if (doc) {
    doc->getErrorLog()->logPackageError("comp", CompDeprecatedDeleteFunction, getPackageVersion(), getLevel(), getVersion());
  }

  set<SBase*> toremove;
  //We have to assume that mRemoved has been set properly, though there's no guarantee that this has happened.
  int ret = collectDeletionsAndDeleteSome(&mRemoved, &toremove);
  if (ret != LIBSBML_OPERATION_SUCCESS) {
    return ret;
  }
  return removeCollectedElements(&mRemoved, &toremove);
}
/** @endcond */

int CompModelPlugin::collectRenameAndConvertReplacements(set<SBase*>* removed, set<SBase*>* toremove)
{
  int ret = LIBSBML_OPERATION_SUCCESS;
  SBMLDocument* doc = getSBMLDocument();
  Model* model = static_cast<Model*>(getParentSBMLObject());
  if (model==NULL) {
    if (doc) {
      string error = "Unable to perform replacements in CompModelPlugin::collectRenameAndConvertReplacements: no parent model could be found for the given 'comp' model plugin element.";
      doc->getErrorLog()->logPackageError("comp", CompModelFlatteningFailed, getPackageVersion(), getLevel(), getVersion(), error);
    }
    return LIBSBML_OPERATION_FAILED;
  }
  List* allElements = model->getAllElements();
  vector<ReplacedElement*> res;
  vector<ReplacedBy*> rbs;
  //Collect replaced elements and replaced by's.
  for (unsigned int e=0; e<allElements->getSize(); e++) {
    SBase* element = static_cast<SBase*>(allElements->get(e));
    int type = element->getTypeCode();
    if (type==SBML_COMP_REPLACEDELEMENT) {
      ReplacedElement* reference = static_cast<ReplacedElement*>(element);
      res.push_back(reference);
    }
    if (type==SBML_COMP_REPLACEDBY) {
      ReplacedBy* reference = static_cast<ReplacedBy*>(element);
      rbs.push_back(reference);
    }
  }
  delete allElements;

  //ReplacedElement replacements
  for (size_t re=0; re<res.size(); re++) {
    ret = res[re]->performReplacementAndCollect(removed, toremove);
    if (ret != LIBSBML_OPERATION_SUCCESS) {
      return ret;
    }
  }

  //Now do the same thing for anything left over in the submodels
  for (unsigned int sub=0; sub<getNumSubmodels(); sub++) {
    Submodel* submodel = getSubmodel(sub);
    Model* mod = submodel->getInstantiation();
    if (mod==NULL) return LIBSBML_OPERATION_FAILED;
    CompModelPlugin* modplug = static_cast<CompModelPlugin*>(mod->getPlugin(getPrefix()));
    if (modplug==NULL) return LIBSBML_OPERATION_FAILED;
    //'left behind' converions (not LaHaye-style)
    ret = submodel->convertTimeAndExtent();
    if (ret != LIBSBML_OPERATION_SUCCESS) return ret;
    ret = modplug->collectRenameAndConvertReplacements(removed, toremove);
    if (ret != LIBSBML_OPERATION_SUCCESS) return ret;
  }

  //Perform ReplacedBy replacements *after* the submodels are done, so that the topmost-level names take precedence.
  for (size_t rb=0; rb<rbs.size(); rb++) {
    ret = rbs[rb]->performReplacementAndCollect(removed, toremove);
    if (ret != LIBSBML_OPERATION_SUCCESS) {
      return ret;
    }
  }

  return ret;
}

/** @cond doxygenLibsbmlInternal */
//Deprecated function
int CompModelPlugin::performReplacementsAndConversions()
{
  SBMLDocument* doc = getSBMLDocument();
  if (doc) {
    doc->getErrorLog()->logPackageError("comp", CompDeprecatedReplaceFunction, getPackageVersion(), getLevel(), getVersion());
  }

  set<SBase*> toremove;
  //We have to assume that mRemoved has been set properly, though there's no guarantee that this has happened.
  int ret = collectRenameAndConvertReplacements(&mRemoved, &toremove);
  if (ret != LIBSBML_OPERATION_SUCCESS) {
    return ret;
  }
  return removeCollectedElements(&mRemoved, &toremove);
}
/** @endcond */

int CompModelPlugin::removeCollectedElements(set<SBase*>* removed, set<SBase*>* toremove)
{
  while (toremove->size() > 0) {
    SBase* removeme = *(toremove->begin());
    if (removed->insert(removeme).second == true) {
      //Need to remove the element.
      List* children = removeme->getAllElements();
      for (unsigned int el=0; el<children->getSize(); el++) {
        SBase* element = static_cast<SBase*>(children->get(el));
        removed->insert(element);
      }
      delete children;
      CompBase::removeFromParentAndPorts(removeme, removed);
    }
    toremove->erase(removeme);
  }
  return LIBSBML_OPERATION_SUCCESS;
}

  
  /** @cond doxygenLibsbmlInternal */

bool 
CompModelPlugin::accept(SBMLVisitor& v) const
{
  const Model * model = static_cast<const Model * >(this->getParentSBMLObject()); 
  
  v.visit(*model);
  v.leave(*model);

  for (unsigned int i = 0; i < getNumSubmodels(); i++)
  {
    getSubmodel(i)->accept(v);
  }
  for (unsigned int i = 0; i < getNumPorts(); i++)
  {
    getPort(i)->accept(v);
  }

  return true;
}

/** @endcond */

void 
CompModelPlugin::setTransformer(PrefixTransformer* transformer)
{
  mTransformer = transformer;
}

PrefixTransformer* 
CompModelPlugin::getTransformer() const
{
  return mTransformer;
}

bool 
CompModelPlugin::isSetTransformer() const
{
  return mTransformer != NULL;
}


void 
CompModelPlugin::unsetTransformer()
{
  mTransformer = NULL;
}



/** @cond doxygenLibsbmlInternal */
std::set<SBase*>* 
CompModelPlugin::getRemovedSet() 
{ 
  return &mRemoved; 
}
/** @endcond */

#endif /* __cplusplus */
/** @cond doxygenIgnored */
LIBSBML_EXTERN
Submodel_t *
CompModelPlugin_createSubmodel(CompModelPlugin_t * modelPlug)
{
  return modelPlug->createSubmodel();
}
/** @endcond */


LIBSBML_CPP_NAMESPACE_END


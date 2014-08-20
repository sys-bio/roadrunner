/**
 * @file    Replacing.cpp
 * @brief   Implementation of Replacing, the SBaseRef derived class of Replacings package.
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
#include <sbml/packages/comp/extension/CompModelPlugin.h>
#include <sbml/packages/comp/sbml/Replacing.h>
#include <sbml/packages/comp/validator/CompSBMLError.h>
#include <sbml/Model.h>

using namespace std;

LIBSBML_CPP_NAMESPACE_BEGIN
#ifdef __cplusplus

Replacing::Replacing (unsigned int level, unsigned int version, unsigned int pkgVersion) 
  : SBaseRef (level,version, pkgVersion)
  , mSubmodelRef("")
  , mConversionFactor("")
{
  setSBMLNamespacesAndOwn(new CompPkgNamespaces(level,version,pkgVersion));  
}


Replacing::Replacing(CompPkgNamespaces* compns)
  : SBaseRef(compns)
  , mSubmodelRef("")
  , mConversionFactor("")
{
  loadPlugins(compns);
}


Replacing::Replacing(const Replacing& source) 
  : SBaseRef(source)
{
  mSubmodelRef=source.mSubmodelRef;
  mConversionFactor=source.mConversionFactor;
}

Replacing& Replacing::operator=(const Replacing& source)
{
  if(&source!=this)
  {
    SBaseRef::operator=(source);
    mSubmodelRef=source.mSubmodelRef;
    mConversionFactor=source.mConversionFactor;
  }
  return *this;
}

Replacing::~Replacing ()
{
}


/*
 * Sets the submodelRef of this SBML object to a copy of submodelRef.
 */
int
Replacing::setSubmodelRef (const std::string& submodelRef)
{
  if (!SyntaxChecker::isValidSBMLSId(submodelRef)) 
  {
    return LIBSBML_INVALID_ATTRIBUTE_VALUE;
  }
  mSubmodelRef = submodelRef;
  return LIBSBML_OPERATION_SUCCESS;
}


/*
 * @return the submodelRef of this SBML object.
 */
const string&
Replacing::getSubmodelRef () const
{
  return mSubmodelRef;
}


/*
 * @return true if the submodelRef of this SBML object has been set, false
 * otherwise.
 */
bool
Replacing::isSetSubmodelRef () const
{
  return (mSubmodelRef.empty() == false);
}


/*
 * Unsets the submodelRef of this SBML object.
 */
int
Replacing::unsetSubmodelRef ()
{
  mSubmodelRef.erase();

  if (mSubmodelRef.empty())
  {
    return LIBSBML_OPERATION_SUCCESS;
  }
  else
  {
    return LIBSBML_OPERATION_FAILED;
  }
}

bool 
Replacing::hasRequiredAttributes() const
{
  if (!SBaseRef::hasRequiredAttributes()) return false;
  return (isSetSubmodelRef());
}


int 
Replacing::saveReferencedElement()
{
  SBMLDocument* doc = getSBMLDocument();
  if (!isSetSubmodelRef()) {
    if (doc) {
      string error = "Unable to find referenced element in Replacing::saveReferencedElement: the given <" + getElementName() + "> element";
      if (isSetId()) {
        error += " '" + getId() + "'";
      }
      error += " has no 'submodelRef' attribute.";
      doc->getErrorLog()->logPackageError("comp", CompReplacedElementAllowedAttributes, getPackageVersion(), getLevel(), getVersion(), error, getLine(), getColumn());
    }
    return LIBSBML_INVALID_OBJECT;
  }
  Model* model = getParentModel(this);
  if (model==NULL) {
    if (doc) {
      string error = "Unable to find referenced element in Replacing::saveReferencedElement: no parent model could be found for the given <" + getElementName() + "> element";
      if (isSetId()) {
        error += " '" + getId() + "'.";
      }
      doc->getErrorLog()->logPackageError("comp", CompModelFlatteningFailed, getPackageVersion(), getLevel(), getVersion(), error, getLine(), getColumn());
    }
    return LIBSBML_OPERATION_FAILED;
  }
  CompModelPlugin* cmp = static_cast<CompModelPlugin*>(model->getPlugin(getPrefix()));
  if (cmp==NULL) {
    if (doc) {
      string error = "Unable to find referenced element in Replacing::saveReferencedElement: no 'comp' plugin for the parent model could be found for the given <" + getElementName() + "> element";
      if (isSetId()) {
        error += " '" + getId() + "'.";
      }
      doc->getErrorLog()->logPackageError("comp", CompModelFlatteningFailed, getPackageVersion(), getLevel(), getVersion(), error, getLine(), getColumn());
    }
    return LIBSBML_OPERATION_FAILED;
  }
  Submodel* submod = cmp->getSubmodel(getSubmodelRef());
  if (submod==NULL) {
    if (doc) {
      string error = "Unable to find referenced element for the given <" + getElementName() + "> element";
      if (isSetId()) {
        error += " '" + getId() + "'";
      }
      error += " in Replacing::saveReferencedElement: the submodelRef '" + getSubmodelRef() + "' could not be found in the model.";
      int errnumber = CompReplacedElementSubModelRef;
      if (getTypeCode() == SBML_COMP_REPLACEDBY) {
        errnumber = CompReplacedBySubModelRef;
      }
      doc->getErrorLog()->logPackageError("comp", errnumber, getPackageVersion(), getLevel(), getVersion(), error, getLine(), getColumn());
    }
    return LIBSBML_INVALID_ATTRIBUTE_VALUE;
  }
  Model* inst = submod->getInstantiation();
  if (inst==NULL) {
    //getInstantiation sets it own error messages.
    return LIBSBML_OPERATION_FAILED;
  }
  mReferencedElement = getReferencedElementFrom(inst);
  if (mDirectReference==NULL) {
    mDirectReference = mReferencedElement;
  }
  //getReferencedElement* set their own error messages:
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
Replacing::renameSIdRefs(const std::string& oldid, const std::string& newid)
{
  if (mSubmodelRef==oldid) mSubmodelRef=newid;
  if (mConversionFactor==oldid) mConversionFactor=newid;
  SBaseRef::renameSIdRefs(oldid, newid);
}

/** @cond doxygenLibsbmlInternal */
void
Replacing::addExpectedAttributes(ExpectedAttributes& attributes)
{
  SBaseRef::addExpectedAttributes(attributes);
  attributes.add("submodelRef");
  attributes.add("conversionFactor");
}
/** @endcond */

/** @cond doxygenLibsbmlInternal */
void
Replacing::readAttributes (const XMLAttributes& attributes,
                                 const ExpectedAttributes& expectedAttributes)
{
  const unsigned int sbmlLevel   = getLevel  ();
  const unsigned int sbmlVersion = getVersion();

  XMLTriple tripleSubmodelRef("submodelRef", mURI, getPrefix());
  bool assigned = attributes.readInto(tripleSubmodelRef, mSubmodelRef);
  if (assigned == false)
  {
    if (getElementName() == "replacedElement")
    {
      std::string message = "Comp attribute 'submodelRef' is missing.";
      getErrorLog()->logPackageError("comp", CompReplacedElementAllowedAttributes, 
        getPackageVersion(), sbmlLevel, sbmlVersion, message);
    }
    else
    {
      std::string message = "Comp attribute 'submodelRef' is missing.";
      getErrorLog()->logPackageError("comp", CompReplacedByAllowedAttributes, 
        getPackageVersion(), sbmlLevel, sbmlVersion, message);
    }
  }
  else
  {
     if (!SyntaxChecker::isValidSBMLSId(mSubmodelRef)) 
     {
      logInvalidId("comp:submodelRef", mSubmodelRef);
     }
  }
  //We call the base class version here because of the error checking for having set exactly one of the mutually-exclusive attributes, and one of them (deletion) only exists for Replacings, not SBaseRef.
  SBaseRef::readAttributes(attributes,expectedAttributes);
}
/** @endcond */

/** @cond doxygenLibsbmlInternal */
void
Replacing::writeAttributes (XMLOutputStream& stream) const
{
  SBaseRef::writeAttributes(stream);

  if (isSetSubmodelRef()) {
    stream.writeAttribute("submodelRef", getPrefix(), mSubmodelRef);
  }
  SBaseRef::writeExtensionAttributes(stream);
}
/** @endcond */

/** @cond doxygenLibsbmlInternal */
void
Replacing::writeElements (XMLOutputStream& stream) const
{
  SBaseRef::writeElements(stream);

  SBaseRef::writeExtensionElements(stream);
}
/** @endcond */


/*
 * Accepts the given SBMLVisitor.
 */
bool
Replacing::accept (SBMLVisitor& v) const
{
  return false;
}


/** @cond doxygenLibsbmlInternal */
/*
 * Sets the parent SBMLDocument of this SBML object.
 */
void
Replacing::setSBMLDocument (SBMLDocument* d)
{
  SBaseRef::setSBMLDocument(d);
}
/** @endcond */


/** @cond doxygenLibsbmlInternal */
int 
Replacing::replaceWithAndMaybeDelete(SBase* replacement, bool deleteme, ASTNode* conversionFactor)
{
  //All of the following function calls that could result in errors set their own error messages.
  SBase* replaced = getReferencedElement();
  if (replaced==NULL) return LIBSBML_INVALID_OBJECT;

  //Rename things
  int ret = updateIDs(replaced, replacement);
  if (ret != LIBSBML_OPERATION_SUCCESS) return ret;

  //Perform any conversions on references in the submodel.
  ret = performConversions(replacement, &conversionFactor);
  if (ret != LIBSBML_OPERATION_SUCCESS) return ret;

  //Finally, recurse down if there are things the replaced element itself replaced (or used to be replaced by)
  CompSBasePlugin* replacedplug = static_cast<CompSBasePlugin*>(replaced->getPlugin(getPrefix()));
  if (replacedplug==NULL) {
    //assert(false); //Not sure when this situation would come up, so I would like to see an example.
    //(sadly, I cannot set an assert in production code.)
    return LIBSBML_OPERATION_SUCCESS; //I guess?  LS DEBUG
  }
  for (unsigned int re=0; re<replacedplug->getNumReplacedElements(); re++) {
    ret = replacedplug->getReplacedElement(re)->replaceWithAndMaybeDelete(replacement, true, conversionFactor);
    if (ret != LIBSBML_OPERATION_SUCCESS) return ret;
  }
  if (replacedplug->isSetReplacedBy()) {
    ret = replacedplug->getReplacedBy()->replaceWithAndMaybeDelete(replacement, deleteme, conversionFactor);
    if (ret != LIBSBML_OPERATION_SUCCESS) return ret;
  }
  return LIBSBML_OPERATION_SUCCESS;
}
/** @endcond */


/** @cond doxygenLibsbmlInternal */
int 
Replacing::updateIDs(SBase* oldnames, SBase* newnames)
{
  int ret = LIBSBML_OPERATION_SUCCESS;
  SBMLDocument* doc = getSBMLDocument();
  if (oldnames->isSetId() && !newnames->isSetId()) {
    if (doc) {
      string error = "Unable to transform IDs in Replacing::updateIDs during replacement:  the '" + oldnames->getId() + "' element's replacement does not have an ID set.";
      doc->getErrorLog()->logPackageError("comp", CompMustReplaceIDs, getPackageVersion(), getLevel(), getVersion(), error, getLine(), getColumn());
    }
    return LIBSBML_INVALID_OBJECT;
  }
  if (oldnames->isSetMetaId() && !newnames->isSetMetaId()) {
    if (doc) {
      string error = "Unable to transform IDs in Replacing::updateIDs during replacement:  the replacement of the element with metaid '" + oldnames->getMetaId() + "' does not have a metaid.";
      doc->getErrorLog()->logPackageError("comp", CompMustReplaceMetaIDs, getPackageVersion(), getLevel(), getVersion(), error, getLine(), getColumn());
    }
    return LIBSBML_INVALID_OBJECT;
  }
  //LS DEBUG Somehow we need to check identifiers from other packages here (like spatial id's).  How, exactly, is anyone's guess.
  Model* replacedmod = const_cast<Model*>(CompBase::getParentModel(oldnames));
  KineticLaw* replacedkl;
  ASTNode newkl;
  if (replacedmod==NULL) {
    if (doc) {
      string error = "Unable to transform IDs in Replacing::updateIDs during replacement:  the replacement of '" + oldnames->getId() + "' does not have a valid model.";
      doc->getErrorLog()->logPackageError("comp", CompModelFlatteningFailed, getPackageVersion(), getLevel(), getVersion(), error, getLine(), getColumn());
    }
    return LIBSBML_INVALID_OBJECT;
  }
  List* allElements = replacedmod->getAllElements();
  string oldid = oldnames->getId();
  string newid = newnames->getId();
  if (!oldid.empty()) {
    switch(oldnames->getTypeCode()) {
    case SBML_UNIT_DEFINITION:
      replacedmod->renameUnitSIdRefs(oldid, newid);
      for (unsigned int e=0; e<allElements->getSize(); e++) {
        SBase* element = static_cast<SBase*>(allElements->get(e));
        element->renameUnitSIdRefs(oldid, newid);
      }
      break;
    case SBML_LOCAL_PARAMETER:
      replacedkl = static_cast<KineticLaw*>(oldnames->getAncestorOfType(SBML_KINETIC_LAW));
      if (replacedkl->isSetMath()) {
        newkl = *replacedkl->getMath();
        newkl.renameSIdRefs(oldid, newid);
        replacedkl->setMath(&newkl);
      }
      break;
    case SBML_COMP_PORT:
      break;
      //LS DEBUG And here is where we would need some sort of way to check if the id wasn't an SId for some objects.
    default:
      replacedmod->renameSIdRefs(oldnames->getId(), newnames->getId());
      for (unsigned int e=0; e<allElements->getSize(); e++) {
        SBase* element = static_cast<SBase*>(allElements->get(e));
        element->renameSIdRefs(oldid, newid);
      }
    }
  }
  string oldmetaid = oldnames->getMetaId();
  string newmetaid = newnames->getMetaId();
  if (oldnames->isSetMetaId()) {
    replacedmod->renameMetaIdRefs(oldmetaid, newmetaid);
    for (unsigned int e=0; e<allElements->getSize(); e++) {
      SBase* element = static_cast<SBase*>(allElements->get(e));
      element->renameMetaIdRefs(oldmetaid, newmetaid);
    }
  }
  //LS DEBUG And here is where we would need some sort of way to check for ids that were not 'id' or 'metaid'.
  delete allElements;
  return ret;
}
/** @endcond */


/** @cond doxygenLibsbmlInternal */
int Replacing::performConversions(SBase* replacement, ASTNode** conversionFactor)
{
  SBMLDocument* doc = getSBMLDocument();
  int ret = convertConversionFactor(conversionFactor);
  if (ret != LIBSBML_OPERATION_SUCCESS) {
    //convertConversionFactor sets its own error messages.
    return ret;
  }
  if (*conversionFactor==NULL) {
    return ret;
  }
  if (replacement==NULL) {
    if (doc) {
      string error = "Internal error in Replacing::performConversions:  cannot perform a conversion of NULL.";
      doc->getErrorLog()->logPackageError("comp", CompModelFlatteningFailed, getPackageVersion(), getLevel(), getVersion(), error, getLine(), getColumn());
    }
    return LIBSBML_OPERATION_FAILED;
  }

  SBase* replaced = getReferencedElement();
  if (replaced==NULL) {
    //getReferencedElement sets its own error messages.
    return LIBSBML_INVALID_OBJECT;
  }
  Model* replacedmod = const_cast<Model*>(CompBase::getParentModel(replaced));
  if (replacedmod==NULL) {
    if (doc) {
      string error = "Unable to perform conversion of replacement in Replacing::performConversions:  No model parent could be found for replacement";
      if (replacement->isSetId()) {
        error += replacement->getId() + ".";
      }
      doc->getErrorLog()->logPackageError("comp", CompModelFlatteningFailed, getPackageVersion(), getLevel(), getVersion(), error, getLine(), getColumn());
    }
    return LIBSBML_INVALID_OBJECT;
  }

  if (!replacement->isSetId()) {
    //If the replacement has no ID, it won't need to be converted anywhere.  Only theoretically possible for non-core SBase elements.
    return LIBSBML_OPERATION_SUCCESS;
  }
  string id = replacement->getId();
  ASTNode replacementAST(AST_NAME);
  replacementAST.setName(id.c_str());
  ASTNode divide(AST_DIVIDE);
  divide.addChild(replacementAST.deepCopy());
  divide.addChild((*conversionFactor)->deepCopy());
  List* allElements = replacedmod->getAllElements();
  for (unsigned int e=0; e<allElements->getSize(); e++) {
    SBase* element = static_cast<SBase*>(allElements->get(e));
    element->replaceSIDWithFunction(id, &divide);
    element->multiplyAssignmentsToSIdByFunction(id, *conversionFactor);
  }
  delete allElements;
  return ret;
}
/** @endcond */


/** @cond doxygenLibsbmlInternal */
int Replacing::convertConversionFactor(ASTNode** conversionFactor)
{
  int ret = LIBSBML_OPERATION_SUCCESS;
  ASTNode* newCF = NULL;
  if (mConversionFactor=="") {
    //newCF = *conversionFactor;
  }
  else {
    ASTNode factor(AST_NAME);
    factor.setName(mConversionFactor.c_str());
    if (*conversionFactor==NULL) {
      newCF = new ASTNode(factor);
      *conversionFactor = newCF;
    }
    else if ((*conversionFactor)->getType()==AST_NAME) {
      newCF = new ASTNode(AST_TIMES);
      newCF->addChild(*conversionFactor);
      newCF->addChild(factor.deepCopy());
      *conversionFactor = newCF;
    }
    else if ((*conversionFactor)->getType()==AST_TIMES) {
      (*conversionFactor)->addChild(factor.deepCopy());
      //newCF = *conversionFactor;
    }
    else {
      SBMLDocument* doc = getSBMLDocument();
      if (doc) {
        string error = "Internal error in Replacing::convertConversionFactor:  unknown conversion factor form.";
        doc->getErrorLog()->logPackageError("comp", CompModelFlatteningFailed, getPackageVersion(), getLevel(), getVersion(), error, getLine(), getColumn());
      }
      return LIBSBML_OPERATION_FAILED;
    }
  }
  return ret;
}
/** @endcond */


//Deprecated function
int Replacing::performReplacement()
{
  set<SBase*> toremove;
  set<SBase*>* removed = NULL;
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
  int ret = performReplacementAndCollect(removed, &toremove);
  if (ret != LIBSBML_OPERATION_SUCCESS) {
    return ret;
  }
  if (cmp == NULL) {
    return LIBSBML_INVALID_OBJECT;
  }
  return cmp->removeCollectedElements(removed, &toremove);
}
 
#endif /* __cplusplus */

LIBSBML_CPP_NAMESPACE_END


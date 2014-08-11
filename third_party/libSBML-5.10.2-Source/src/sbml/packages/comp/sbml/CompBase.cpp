/**
 * @file    CompBase.cpp
 * @brief   Implementation of CompBase, the base class of extension 
 *          entities plugged in SBase derived classes in the SBML Core package.
 * @author  Lucian Smith
 *
 * <!--------------------------------------------------------------------------
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
 * ------------------------------------------------------------------------ -->
 */

#include <sbml/packages/comp/sbml/CompBase.h>
#include <sbml/packages/comp/extension/CompModelPlugin.h>
#include <sbml/packages/comp/sbml/Port.h>
#include <sbml/Model.h>
#include <sbml/extension/SBMLExtensionRegistry.h>
#include <sbml/packages/comp/validator/CompSBMLError.h>

#ifdef __cplusplus

#include <sstream>
#include <iostream>

using namespace std;

LIBSBML_CPP_NAMESPACE_BEGIN
#ifdef __cplusplus

CompBase::CompBase (unsigned int level, unsigned int version, unsigned int pkgVersion) 
  : SBase (level,version)
{
  // set an SBMLNamespaces derived object (CompPkgNamespaces) of this package.
  CompPkgNamespaces* cpn = new CompPkgNamespaces(level,version,pkgVersion);
  setSBMLNamespacesAndOwn(cpn);

  // connect child elements to this element.
  connectToChild();

  mSBMLExt = SBMLExtensionRegistry::getInstance().getExtension(mSBMLNamespaces->getURI());
}


CompBase::CompBase(CompPkgNamespaces* compns)
  : SBase(compns)
  , mSBMLExt(SBMLExtensionRegistry::getInstance().getExtension(compns->getURI()))
{
  // set the element namespace of this object
  setElementNamespace(compns->getURI());

  // connect child elements to this element.
  connectToChild();
}


CompBase::CompBase(const CompBase& source) 
  : SBase (source)
  , mSBMLExt(NULL)
{
  if (source.mSBMLExt != NULL) {
    mSBMLExt = source.mSBMLExt->clone();
  }
  // connect child elements to this element.
  connectToChild();

  // load package extensions bound with this object (if any) 
  loadPlugins(mSBMLNamespaces);
}

CompBase& CompBase::operator=(const CompBase& source)
{
  if(&source!=this)
  {
    SBase::operator=(source);
    if (source.mSBMLExt != NULL) {
      mSBMLExt = source.mSBMLExt->clone();
    }

    // connect child elements to this element.
    connectToChild();

    // load package extensions bound with this object (if any) 
    loadPlugins(mSBMLNamespaces);
  }

  return *this;
}


CompBase::~CompBase ()
{
  delete mSBMLExt;
}


/*
 * Returns the XML namespace (URI) of the package extension
 * of this object.
 */
const std::string& 
CompBase::getPackageURI() const
{
  return mURI;
}

/*
 * Returns the package name of this object.
 */
const std::string& 
CompBase::getPackageName() const
{
  return mSBMLExt->getName();
}

/*
 * Returns the package version of this plugin object.
 */
unsigned int 
CompBase::getPackageVersion() const
{
  return mSBMLExt->getPackageVersion(mURI);
}

Model* 
CompBase::getParentModel(SBase* child)
{
  SBase* parent = child->getParentSBMLObject();
  while (parent != NULL && parent->getTypeCode() != SBML_DOCUMENT) {
    if ((parent->getTypeCode() == SBML_MODEL) || 
        (parent->getTypeCode() == SBML_COMP_MODELDEFINITION)) {
          return static_cast<Model*>(parent);
    }
    if (parent->getTypeCode() == SBML_COMP_EXTERNALMODELDEFINITION) return NULL;
    parent = parent->getParentSBMLObject();
  }
  return NULL;
}

/** @cond doxygenLibsbmlInternal */
void
CompBase::readAttributes (const XMLAttributes& attributes,
                          const ExpectedAttributes& expectedAttributes)
{
  SBase::readAttributes(attributes,expectedAttributes);
  if (&attributes == NULL || &expectedAttributes == NULL ) return;

  //const unsigned int sbmlLevel   = getLevel  ();
  //const unsigned int sbmlVersion = getVersion();
  //const unsigned int pkgVersion  = getPackageVersion();

  std::string element = getElementName();

  //
  // check that all attributes of this plugin object are expected
  //
  for (int i = 0; i < attributes.getLength(); i++)
  {
    std::string name = attributes.getName(i);
    std::string uri  = attributes.getURI(i);

    if (uri != mURI) continue;

    if (!expectedAttributes.hasAttribute(name))
    {    
      logUnknownAttribute(name, element);
    }      
  }
}
/** @endcond */

/** @cond doxygenLibsbmlInternal */
void
CompBase::writeAttributes (XMLOutputStream& stream) const
{
  SBase::writeAttributes(stream);
  //
  // If some other package extends this one, we'll need to write its attributes
  //
  writeExtensionAttributes(stream);
}
/** @endcond */

/** @cond doxygenLibsbmlInternal */
/*
 * Helper to log a common type of error.
 */
void 
CompBase::logUnknownElement(const std::string &element)
{
  if(&element == NULL) return;
  
  std::ostringstream msg;

  msg << "Element '"   << element << "' is not part of the definition of "
      << "SBML Level " << getLevel() << " Version " << getVersion() 
      << " Package \""   << getPrefix() << "\" Version "
      << getPackageVersion() << ".";

  SBMLErrorLog* errlog = getErrorLog();
  if (errlog)
  {
    errlog->logError(UnrecognizedElement, getLevel(), getVersion(), msg.str());
  }
}
/** @endcond */


/** @cond doxygenLibsbmlInternal */
/*
 * Helper to log a common type of error.
 */
void 
CompBase::logUnknownAttribute(const std::string &attribute,
                              const std::string& element)
{
  if (&attribute == NULL || &element == NULL) return;
  
  std::ostringstream msg;

  msg << "Attribute '" << attribute << "' is not part of the "
      << "definition of an SBML Level " << getLevel()
      << " Version " << getVersion() << " Package \"" 
      << getPrefix() << "\" Version " << getPackageVersion() 
      << " on " << element << " element.";

  SBMLErrorLog* errlog = getErrorLog();
  if (errlog != NULL)
  {
    if (element == "port")
    {
      errlog->logPackageError(getPackageName(), CompPortAllowedAttributes, 
        getPackageVersion(), getLevel(), getVersion(), msg.str());
    }
    else
    {
      errlog->logError(NotSchemaConformant, getLevel(), getVersion(), msg.str());
    }
  }
}
/** @endcond */


/** @cond doxygenLibsbmlInternal */
/*
 * Helper to log a common type of error.
 */
void 
CompBase::logEmptyString(const std::string &attribute, 
                         const std::string& element)
{

  if (&attribute == NULL || &element == NULL) return;
  
  std::ostringstream msg;

  msg << "Attribute '" << attribute << "' on an "
      << element << " of package \"" << getPrefix()
      << "\" version " << getPackageVersion() << " must not be an empty string.";

  SBMLErrorLog* errlog = getErrorLog();
  if (errlog)
  {
    errlog->logError(NotSchemaConformant, getLevel(), getVersion(), msg.str());
  }
}
/** @endcond */


/** @cond doxygenLibsbmlInternal */
void 
CompBase::logInvalidId(const std::string& attribute,
                       const std::string& wrongattribute,
                       const std::string& object)
{

  if (&attribute == NULL || &wrongattribute == NULL) return;
  
  std::ostringstream msg;

  if (attribute == "comp:metaIdRef")
  {
    msg << "Setting the attribute '" << attribute << "' of a <"
        << getElementName() << "> in the " << getPackageName() 
        << " package (version " << getPackageVersion() << ") to '" << wrongattribute
        << "' is illegal:  the string is not a well-formed XML ID.";
  }
  else
  {
    msg << "Setting the attribute '" << attribute << "' of a <"
        << getElementName() << "> in the " << getPackageName() 
        << " package (version " << getPackageVersion() << ") to '" << wrongattribute
        << "' is illegal:  the string is not a well-formed SId.";
  }

  SBMLErrorLog* errlog = getErrorLog();
  
  if (errlog != NULL)
  {
    // so here we have different errors depending on 
    // what the attribute actually is
    if (attribute == "comp:deletion")
    {
      errlog->logPackageError(getPackageName(), CompInvalidDeletionSyntax, 
        getPackageVersion(), getLevel(), getVersion(), msg.str());
    }
    else if (attribute == "comp:conversionFactor")
    {
      errlog->logPackageError(getPackageName(), CompInvalidConversionFactorSyntax, 
        getPackageVersion(), getLevel(), getVersion(), msg.str());
    }
    else if (attribute == "comp:submodelRef")
    {
      errlog->logPackageError(getPackageName(), CompInvalidSubmodelRefSyntax, 
        getPackageVersion(), getLevel(), getVersion(), msg.str());
    }
    else if (attribute == "comp:modelRef")
    {
      if (object == "Submodel")
      {
        errlog->logPackageError(getPackageName(), CompModReferenceSyntax, 
          getPackageVersion(), getLevel(), getVersion(), msg.str());
      }
      else
      {
        errlog->logPackageError(getPackageName(), CompInvalidModelRefSyntax, 
          getPackageVersion(), getLevel(), getVersion(), msg.str());
      }
    }
    else if (attribute == "comp:metaIdRef")
    {
      errlog->logPackageError(getPackageName(), CompInvalidMetaIdRefSyntax, 
        getPackageVersion(), getLevel(), getVersion(), msg.str());
    }
    else if (attribute == "comp:idRef")
    {
      errlog->logPackageError(getPackageName(), CompInvalidIdRefSyntax, 
        getPackageVersion(), getLevel(), getVersion(), msg.str());
    }
    else if (attribute == "comp:portRef")
    {
      errlog->logPackageError(getPackageName(), CompInvalidPortRefSyntax, 
        getPackageVersion(), getLevel(), getVersion(), msg.str());
    }
    else if (attribute == "comp:unitRef")
    {
      errlog->logPackageError(getPackageName(), CompInvalidUnitRefSyntax, 
        getPackageVersion(), getLevel(), getVersion(), msg.str());
    }
    else if (attribute == "comp:timeConversionFactor")
    {
      errlog->logPackageError(getPackageName(), CompInvalidTimeConvFactorSyntax, 
        getPackageVersion(), getLevel(), getVersion(), msg.str());
    }
    else if (attribute == "comp:extentConversionFactor")
    {
      errlog->logPackageError(getPackageName(), CompInvalidExtentConvFactorSyntax, 
        getPackageVersion(), getLevel(), getVersion(), msg.str());
    }
    else
    {
      errlog->logPackageError(getPackageName(), CompInvalidSIdSyntax, 
        getPackageVersion(), getLevel(), getVersion(), msg.str());
    }
  }
}
/** @endcond */

/** @cond doxygenLibsbmlInternal */
void 
CompBase::logMissingAttribute(const std::string& attribute,
                              const std::string& element)
{

  if (&attribute == NULL || &element == NULL) return;
  
  std::ostringstream msg;

  msg << "The required attribute '" << attribute << "' of a <"
      << getElementName() << "> in the " << getPackageName() 
      << " package (version " << getPackageVersion() << ") is missing.";

  SBMLErrorLog* errlog = getErrorLog();
  
  if (errlog != NULL)
  {
    // so here we have different errors depending on 
    // what the attribute actually is
    if (element == "<Port>")
    {
      errlog->logPackageError(getPackageName(), CompPortAllowedAttributes, 
        getPackageVersion(), getLevel(), getVersion(), msg.str());
    }
    else if (element == "<ExternalModelDefinition>")
    {
      errlog->logPackageError(getPackageName(), CompExtModDefAllowedAttributes, 
        getPackageVersion(), getLevel(), getVersion(), msg.str());
    }
    else if (element == "<Deletion>")
    {
      errlog->logPackageError(getPackageName(), CompDeletionAllowedAttributes, 
        getPackageVersion(), getLevel(), getVersion(), msg.str());
    }
  }
}
/** @endcond */


/** @cond doxygenLibsbmlInternal */
bool 
CompBase::hasValidLevelVersionNamespaceCombination()
{  
  //Here is where we could combine all namespace checking into one place, like is done for SBase.
  // But for now, we just need to check to see if the single legal URI is present.

  XMLNamespaces *xmlns = getNamespaces();  
  if (xmlns == NULL) return false;
  if (xmlns->hasURI("http://www.sbml.org/sbml/level3/version1/comp/version1")) return true;
  return false;
}
/** @endcond */


/** @cond doxygenLibsbmlInternal */
int CompBase::removeFromParentAndPorts(SBase* todelete, set<SBase*>* removed)
{
  //First remove from ports:
  Model* parent = static_cast<Model*>(todelete->getAncestorOfType(SBML_COMP_MODELDEFINITION, "comp"));
  if (parent==NULL) {
    parent = static_cast<Model*>(todelete->getAncestorOfType(SBML_MODEL));
  }
  while (parent != NULL) {
    CompModelPlugin* cmp = static_cast<CompModelPlugin*>(parent->getPlugin("comp"));
    if (cmp==NULL) {
      parent = NULL;
      continue;
    }
    for (unsigned long p=0; p<cmp->getNumPorts();) {
      Port* port = cmp->getPort((unsigned int)p);
      if (port->getReferencedElement() == todelete) {
        if (removed) {
          removed->insert(port);
        }
        port->removeFromParentAndDelete();
      }
      else {
        p++;
      }
    }
    Model* tempparent = static_cast<Model*>(parent->getAncestorOfType(SBML_COMP_MODELDEFINITION, "comp"));
    if (tempparent==NULL) {
      parent = static_cast<Model*>(parent->getAncestorOfType(SBML_MODEL));
    }
    else parent = tempparent;
  }
  //And secondly, remove from parent
  if (removed) {
    removed->insert(todelete);
  }
  return todelete->removeFromParentAndDelete();
}
/** @endcond */


/** @cond doxygenLibsbmlInternal */
//Deprecated function
int CompBase::removeFromParentAndPorts(SBase* todelete)
{
  //First remove from ports:
  Model* parent = static_cast<Model*>(todelete->getAncestorOfType(SBML_COMP_MODELDEFINITION, "comp"));
  if (parent==NULL) {
    parent = static_cast<Model*>(todelete->getAncestorOfType(SBML_MODEL));
  }
  while (parent != NULL) {
    CompModelPlugin* cmp = static_cast<CompModelPlugin*>(parent->getPlugin("comp"));
    if (cmp==NULL) {
      parent = NULL;
      continue;
    }
    CompModelPlugin* basecmp = cmp;
    SBase* base = parent->getParentSBMLObject();
    while (base != NULL && base->getTypeCode() != SBML_DOCUMENT) {
      if (base->getTypeCode() == SBML_COMP_MODELDEFINITION ||
          base->getTypeCode() == SBML_MODEL) 
      {
        CompModelPlugin* testcmp = static_cast<CompModelPlugin*>(base->getPlugin("comp"));
        if (testcmp != NULL) {
          basecmp = testcmp;
        }
      }
      base = base->getParentSBMLObject();
    }
    for (unsigned long p=0; p<cmp->getNumPorts();) {
      Port* port = cmp->getPort((unsigned int)p);
      if (port->getReferencedElement() == todelete) {
        set<SBase*>* removed = basecmp->getRemovedSet();
        set<SBase*>  toremove;
        toremove.insert(port);
        basecmp->removeCollectedElements(removed, &toremove);
      }
      else {
        p++;
      }
    }
    Model* tempparent = static_cast<Model*>(parent->getAncestorOfType(SBML_COMP_MODELDEFINITION, "comp"));
    if (tempparent==NULL) {
      parent = static_cast<Model*>(parent->getAncestorOfType(SBML_MODEL));
    }
    else parent = tempparent;
  }
  //And secondly, remove from parent
  return todelete->removeFromParentAndDelete();
}
/** @endcond */

#endif  /* __cplusplus */

LIBSBML_CPP_NAMESPACE_END

#endif  /* __cplusplus */


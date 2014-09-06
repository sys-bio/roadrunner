/**
 * @file    ReplacedBy.cpp
 * @brief   Implementation of ReplacedBy, the SBaseRef derived class of the comp package.
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

#include <sbml/packages/comp/sbml/ReplacedBy.h>
#include <sbml/packages/comp/extension/CompExtension.h>
#include <sbml/packages/comp/extension/CompSBasePlugin.h>
#include <sbml/packages/comp/extension/CompModelPlugin.h>
#include <sbml/packages/comp/validator/CompSBMLError.h>

using namespace std;

LIBSBML_CPP_NAMESPACE_BEGIN
#ifdef __cplusplus

ReplacedBy::ReplacedBy (unsigned int level, unsigned int version, unsigned int pkgVersion) 
  : Replacing(level,version, pkgVersion)
{
}


ReplacedBy::ReplacedBy(CompPkgNamespaces* compns)
  : Replacing(compns)
{
  loadPlugins(compns);
}


ReplacedBy::ReplacedBy(const ReplacedBy& source) 
  : Replacing(source)
{
}

ReplacedBy& ReplacedBy::operator=(const ReplacedBy& source)
{
  if(&source!=this)
  {
    Replacing::operator=(source);
  }
  return *this;
}

ReplacedBy*
ReplacedBy::clone() const
{
  return new ReplacedBy(*this);
}

ReplacedBy::~ReplacedBy ()
{
}


const std::string&
ReplacedBy::getElementName () const
{
  static const std::string name = "replacedBy";
  return name;
}


int
ReplacedBy::getTypeCode () const
{
  return SBML_COMP_REPLACEDBY;
}


int 
ReplacedBy::removeFromParentAndDelete()
{
  SBase* parent = getParentSBMLObject();
  if (parent==NULL) return LIBSBML_OPERATION_FAILED;
  CompSBasePlugin* comp_parent = static_cast<CompSBasePlugin*>(parent->getPlugin(getPrefix()));
  if (comp_parent==NULL) return LIBSBML_OPERATION_FAILED;
  return comp_parent->unsetReplacedBy();
}

int ReplacedBy::performReplacementAndCollect(set<SBase*>* removed, set<SBase*>* toremove)
{
  SBMLDocument* doc = getSBMLDocument();
  //Find the various objects and plugin objects we need for this to work.
  SBase* parent = getParentSBMLObject();
  if (parent==NULL) {
    if (doc) {
      string error = "Unable to perform replacement in ReplacedBy::performReplacement: no parent object for this <replacedBy> could be found.";
      doc->getErrorLog()->logPackageError("comp", CompModelFlatteningFailed, getPackageVersion(), getLevel(), getVersion(), error, getLine(), getColumn());
    }
    return LIBSBML_INVALID_OBJECT;
  }
  SBase* ref = getReferencedElement();
  if (ref==NULL) {
    //getReferencedElement sets its own error messages.
    return LIBSBML_INVALID_OBJECT;
  }

  //Update the IDs. (Will set its own error messages.)
  int ret = updateIDs(ref, parent);
  
  //ReplacedBy elements do get the name of the top-level element, assuming it has one:
  if (parent->isSetId()) {
    ref->setId(parent->getId());
  }
  if (parent->isSetMetaId()) {
    ref->setMetaId(parent->getMetaId());
  }
  if (ret != LIBSBML_OPERATION_SUCCESS) return ret;

  //And finally, get ready to delete the parent object.
  if (toremove) {
    toremove->insert(parent);
  }
  return LIBSBML_OPERATION_SUCCESS;
}

/** @cond doxygenLibsbmlInternal */
int 
ReplacedBy::updateIDs(SBase* oldnames, SBase* newnames)
{
  //The trick here is that 'oldnames' is actually replacing 'newnames' so we need to get the error messages correct.
  SBMLDocument* doc = getSBMLDocument();
  if (!oldnames->isSetId() && newnames->isSetId()) {
    if (doc) {
      string error = "Unable to transform IDs in ReplacedBy::updateIDs during replacement:  the '" + newnames->getId() + "' element's replacement does not have an ID set.";
      doc->getErrorLog()->logPackageError("comp", CompMustReplaceIDs, getPackageVersion(), getLevel(), getVersion(), error, getLine(), getColumn());
    }
    return LIBSBML_INVALID_OBJECT;
  }
  if (!oldnames->isSetMetaId() && newnames->isSetMetaId()) {
    if (doc) {
      string error = "Unable to transform IDs in ReplacedBy::updateIDs during replacement:  the replacement of the element with metaid '" + newnames->getMetaId() + "' does not have a metaid.";
      doc->getErrorLog()->logPackageError("comp", CompMustReplaceMetaIDs, getPackageVersion(), getLevel(), getVersion(), error, getLine(), getColumn());
    }
    return LIBSBML_INVALID_OBJECT;
  }
  //LS DEBUG Somehow we need to check identifiers from other packages here (like spatial id's).  How, exactly, is anyone's guess.

  //Now update the IDs of 'newnames', if something wasn't set.  (This avoids errors in Replacing::updateIDs)
  if (oldnames->isSetId() && !newnames->isSetId()) {
    newnames->setId(oldnames->getId());
  }
  if (oldnames->isSetMetaId() && !newnames->isSetMetaId()) {
    newnames->setMetaId(oldnames->getMetaId());
  }
  //LS DEBUG We also need to update the other package IDs.
  return Replacing::updateIDs(oldnames, newnames);
}
/** @endcond */


/** @cond doxygenLibsbmlInternal */

bool
ReplacedBy::accept (SBMLVisitor& v) const
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
ReplacedBy_t *
ReplacedBy_create(unsigned int level, unsigned int version,
                  unsigned int pkgVersion)
{
  return new ReplacedBy(level, version, pkgVersion);
}


LIBSBML_EXTERN
void
ReplacedBy_free(ReplacedBy_t * rb)
{
  if (rb != NULL)
    delete rb;
}


LIBSBML_EXTERN
ReplacedBy_t *
ReplacedBy_clone(ReplacedBy_t * rb)
{
  if (rb != NULL)
  {
    return static_cast<ReplacedBy_t*>(rb->clone());
  }
  else
  {
    return NULL;
  }
}


LIBSBML_EXTERN
char *
ReplacedBy_getSubmodelRef(ReplacedBy_t * rb)
{
  if (rb == NULL)
    return NULL;

  return rb->getSubmodelRef().empty() ? NULL : safe_strdup(rb->getSubmodelRef().c_str());
}


LIBSBML_EXTERN
int
ReplacedBy_isSetSubmodelRef(ReplacedBy_t * rb)
{
  return (rb != NULL) ? static_cast<int>(rb->isSetSubmodelRef()) : 0;
}


LIBSBML_EXTERN
int
ReplacedBy_setSubmodelRef(ReplacedBy_t * rb, const char * submodelRef)
{
  return (rb != NULL) ? rb->setSubmodelRef(submodelRef) : LIBSBML_INVALID_OBJECT;
}


LIBSBML_EXTERN
int
ReplacedBy_unsetSubmodelRef(ReplacedBy_t * rb)
{
  return (rb != NULL) ? rb->unsetSubmodelRef() : LIBSBML_INVALID_OBJECT;
}


LIBSBML_EXTERN
int
ReplacedBy_hasRequiredAttributes(ReplacedBy_t * rb)
{
  return (rb != NULL) ? static_cast<int>(rb->hasRequiredAttributes()) : 0;
}



/** @endcond */
LIBSBML_CPP_NAMESPACE_END


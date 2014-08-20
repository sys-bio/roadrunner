/**
 * @file    Objective.cpp
 * @brief   Implementation of Objective, the SBase derived class of the fbc package.
 * @author  Akiya Jouraku
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
 * Copyright (C) 2009-2013 jointly by the following organizations: 
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
#include <limits>

#include <sbml/SBMLVisitor.h>
#include <sbml/xml/XMLNode.h>
#include <sbml/xml/XMLToken.h>
#include <sbml/xml/XMLAttributes.h>
#include <sbml/xml/XMLInputStream.h>
#include <sbml/xml/XMLOutputStream.h>

#include <sbml/packages/fbc/sbml/Objective.h>
#include <sbml/packages/fbc/extension/FbcExtension.h>
#include <sbml/packages/fbc/validator/FbcSBMLError.h>

#include <sbml/util/ElementFilter.h>

using namespace std;

LIBSBML_CPP_NAMESPACE_BEGIN
#ifdef __cplusplus

/*
 * Creates a new Objective with the given level, version, and package version.
 */
Objective::Objective (unsigned int level, unsigned int version, unsigned int pkgVersion) 
  : SBase (level,version)
  ,mId("")
  ,mName("")
  ,mType(OBJECTIVE_TYPE_UNKNOWN)
  ,mTypeString("")
  ,mFluxes()
  ,mIsSetListOfFluxObjectives(false)
{
  // set an SBMLNamespaces derived object (FbcPkgNamespaces) of this package.
  setSBMLNamespacesAndOwn(new FbcPkgNamespaces(level,version,pkgVersion));  

  // connect child elements to this element.
  connectToChild();
}


/*
 * Creates a new Objective with the given FbcPkgNamespaces object.
 */
Objective::Objective(FbcPkgNamespaces* fbcns)
 : SBase(fbcns)
  ,mId("")
  ,mName("")
  ,mType(OBJECTIVE_TYPE_UNKNOWN)
  ,mTypeString("")
  ,mFluxes()
  ,mIsSetListOfFluxObjectives(false)
{
  //
  // set the element namespace of this object
  //
  setElementNamespace(fbcns->getURI());

  // load package extensions bound with this object (if any) 
  loadPlugins(fbcns);

  // connect child elements to this element.
  connectToChild();
}


/*
 * Copy constructor.
 */
Objective::Objective(const Objective& source) : SBase(source)
{
  this->mId=source.mId;
  this->mName=source.mName;
  this->mType=source.mType;
  this->mFluxes=source.mFluxes;
  this->mTypeString = source.mTypeString;
  this->mIsSetListOfFluxObjectives = source.mIsSetListOfFluxObjectives;

  // connect child elements to this element.
  connectToChild();
}

/*
 * Assignment operator.
 */
Objective& Objective::operator=(const Objective& source)
{
  if(&source!=this)
  {
    this->SBase::operator=(source);
    this->mId= source.mId;
    this->mName= source.mName;
    this->mType= source.mType;
    this->mFluxes= source.mFluxes;
    this->mTypeString = source.mTypeString;
    this->mIsSetListOfFluxObjectives = source.mIsSetListOfFluxObjectives;

    // connect child elements to this element.
    connectToChild();
  }
  
  return *this;
}


/*
 * Destructor.
 */ 
Objective::~Objective ()
{
}


SBase* 
Objective::getElementBySId(const std::string& id)
{
  if (id.empty()) return NULL;
  SBase* obj = mFluxes.getElementBySId(id);
  return obj;
}


SBase*
Objective::getElementByMetaId(const std::string& metaid)
{
  if (metaid.empty()) return NULL;
  if (mFluxes.getMetaId() == metaid) return &mFluxes;
  SBase* obj = mFluxes.getElementByMetaId(metaid);
  return obj;
}


List*
Objective::getAllElements(ElementFilter *filter)
{
  List* ret = new List();
  List* sublist = NULL;

  ADD_FILTERED_LIST(ret, sublist, mFluxes, filter);
    
  ADD_FILTERED_FROM_PLUGIN(ret, sublist, filter);

  return ret;
}


/*
  * Returns the value of the "id" attribute of this Objective.
  */
const std::string& 
Objective::getId () const
{
  return mId;
}


/*
  * Predicate returning @c true or @c false depending on whether this
  * Objective's "id" attribute has been set.
  */
bool 
Objective::isSetId () const
{
  return (mId.empty() == false);
}

/*
  * Sets the value of the "id" attribute of this Objective.
  */
int 
Objective::setId (const std::string& id)
{
  return SyntaxChecker::checkAndSetSId(id ,mId);
}


/*
  * Unsets the value of the "id" attribute of this Objective.
  */
int 
Objective::unsetId ()
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




/*
 * Returns the value of the "name" attribute of this Objective.
 */
const std::string&
Objective::getName () const
{
  return mName;
}


/*
 * Predicate returning @c true or @c false depending on whether this
 * Objective's "name" attribute has been set.
 */
bool
Objective::isSetName () const
{
  return (mName.empty() == false);
}

/*
 * Sets the value of the "name" attribute of this Objective.
 */
int
Objective::setName (const std::string& name)
{
  mName = name;
  return LIBSBML_OPERATION_SUCCESS;
}


/*
 * Unsets the value of the "name" attribute of this Objective.
 */
int
Objective::unsetName ()
{
  mName.erase();
  if (mName.empty())
  {
    return LIBSBML_OPERATION_SUCCESS;
  }
  else
  {
    return LIBSBML_OPERATION_FAILED;
  }
}




/*
  * Returns the value of the "type" attribute of this Objective.
  */
const std::string& 
Objective::getType ()
{
  if (ObjectiveType_toString(mType) != NULL)
  {
    mTypeString.assign(ObjectiveType_toString(mType));
  }
  else
  {
    mTypeString.assign("");
  }
  return mTypeString;
}


/*
  * Returns the value of the "type" attribute of this Objective.
  */
ObjectiveType_t 
Objective::getObjectiveType () const
{
  return mType;
}


/*
  * Predicate returning @c true or @c false depending on whether this
  * Objective's "type" attribute has been set.
  */
bool 
Objective::isSetType () const
{
  return (mType != OBJECTIVE_TYPE_UNKNOWN);
}

/*
  * Sets the value of the "type" attribute of this Objective.
  */
int 
Objective::setType (const std::string& type)
{
  return setType(ObjectiveType_fromString(type.c_str()));
}


/*
  * Sets the value of the "type" attribute of this Objective.
  */
int 
Objective::setType (ObjectiveType_t type)
{
  if (ObjectiveType_isValidObjectiveType(type) == 0)
  {
    mType = OBJECTIVE_TYPE_UNKNOWN;
    return LIBSBML_INVALID_ATTRIBUTE_VALUE;
  }
  else
  {
    mType = type;
    return LIBSBML_OPERATION_SUCCESS;
  }
}


/*
  * Unsets the value of the "type" attribute of this Objective.
  */
int 
Objective::unsetType ()
{
  mType = OBJECTIVE_TYPE_UNKNOWN;
  return LIBSBML_OPERATION_SUCCESS;
}


/*
 * Returns the fluxObjective object that holds all fluxObjectives.
 */ 
const ListOfFluxObjectives*
Objective::getListOfFluxObjectives () const
{
  return &this->mFluxes;
}


/*
 * Remove the fluxObjective with the given @p id.
 * A pointer to the removed fluxObjective is returned.
 * If no fluxObjective has been removed, @c NULL is returned.
 */
FluxObjective*
Objective::removeFluxObjective(const std::string& symbol)
{
  return mFluxes.remove(symbol);
}


/*
 * Remove the fluxObjective with the given index.
 * A pointer to the removed fluxObjective is returned.
 * If no fluxObjective has been removed, @c NULL is returned.
 */
FluxObjective*
Objective::removeFluxObjective(unsigned int index)
{
  return mFluxes.remove(index);
}


/*
 * Returns the fluxObjective with the given index.
 * If the index is invalid, @c NULL is returned.
 */ 
FluxObjective* 
Objective::getFluxObjective (unsigned int index)
{
  return mFluxes.get(index);
}

/*
 * Returns the fluxObjective with the given index.
 * If the index is invalid, @c NULL is returned.
 */ 
const FluxObjective* 
Objective::getFluxObjective (unsigned int index) const
{
  return mFluxes.get(index);
}


/*
 * Returns the fluxObjective that has the given @p id, or @c NULL if no
 * fluxObjective has the id.
 */
FluxObjective*
Objective::getFluxObjective (const std::string& symbol) 
{
  return mFluxes.get(symbol);
}


/*
 * Returns the fluxObjective that has the given @p id, or @c NULL if no
 * fluxObjective has the id.
 */
const FluxObjective*
Objective::getFluxObjective (const std::string& symbol) const
{
  return mFluxes.get(symbol);
}


/*
 * Adds a fluxObjective element
 */
int
Objective::addFluxObjective (const FluxObjective* fluxObjective)
{
  if (fluxObjective == NULL)
  {
    return LIBSBML_OPERATION_FAILED;
  }
  else if (getLevel() != fluxObjective->getLevel())
  {
    return LIBSBML_LEVEL_MISMATCH;
  }
  else if (getVersion() != fluxObjective->getVersion())
  {
    return LIBSBML_VERSION_MISMATCH;
  }
  else if (getFluxObjective(fluxObjective->getId()) != NULL)
  {
    // an object with this id already exists
    return LIBSBML_DUPLICATE_OBJECT_ID;
  }
  else if (getPackageVersion() != fluxObjective->getPackageVersion())
  {
    return LIBSBML_PKG_VERSION_MISMATCH;
  }
  else
  {
    mFluxes.append(fluxObjective);

    return LIBSBML_OPERATION_SUCCESS;
  }

}


/*
 * Returns the number of fluxObjectives for the objective.
 */
unsigned int
Objective::getNumFluxObjectives () const
{
  return this->mFluxes.size();
}


/*
 * Creates a FluxObjective object, adds it to the end of the fluxObjective
 * objects list and returns a reference to the newly created object.
 */
FluxObjective*
Objective::createFluxObjective ()
{
  FluxObjective* result = NULL;

  try
  {
    FBC_CREATE_NS(fbcns, getSBMLNamespaces());
    result = new FluxObjective(fbcns);
    this->mFluxes.appendAndOwn(result);
    delete fbcns;
  } 
  catch(...)
  {
    /* 
    * NULL will be returned if the mSBMLNS is invalid (basically this
    * should not happen) or some exception is thrown (e.g. std::bad_alloc)
    *
    * (Maybe this should be changed so that caller can detect what kind 
    *  of error happened in this function.)
    */
  }

  return result;
}

/*
 * Returns the XML element name of
 * this SBML object.
 */
const std::string&
Objective::getElementName () const
{
  static const std::string name = "objective";
  return name;
}


/** @cond doxygenLibsbmlInternal */
SBase*
Objective::createObject (XMLInputStream& stream)
{
  const std::string& name   = stream.peek().getName();
  SBase*        object = NULL;

  if (name == "listOfFluxes" || name == "listOfFluxObjectives")
  {
    if (mFluxes.size() != 0)
    {
      getErrorLog()->logPackageError("fbc", FbcObjectiveOneListOfObjectives, 
        getPackageVersion(), getLevel(), getVersion());
    }

    object = &mFluxes;

    mIsSetListOfFluxObjectives = true;
  }

  return object;
}
/** @endcond */

/** @cond doxygenLibsbmlInternal */
void
Objective::addExpectedAttributes(ExpectedAttributes& attributes)
{
  SBase::addExpectedAttributes(attributes);

  attributes.add("id");
  attributes.add("name");
  attributes.add("type");
}
/** @endcond */

/** @cond doxygenLibsbmlInternal */
void
Objective::readAttributes (const XMLAttributes& attributes,
                        const ExpectedAttributes& expectedAttributes)
{

  const unsigned int sbmlLevel   = getLevel  ();
  const unsigned int sbmlVersion = getVersion();
 
  // look to see whether an unknown attribute error was logged
  // during the read of the listOfFluxBounds - which will have
  // happened immediately prior to this read
  if (getErrorLog() != NULL && 
    static_cast<ListOfObjectives*>(getParentSBMLObject())->size() < 2)
  {
    unsigned int numErrs = getErrorLog()->getNumErrors();
    for (int n = numErrs-1; n >= 0; n--)      
    {
      if (getErrorLog()->getError(n)->getErrorId() == UnknownPackageAttribute)
      {
        const std::string details = 
          getErrorLog()->getError(n)->getMessage();
        getErrorLog()->remove(UnknownPackageAttribute);
        getErrorLog()->logPackageError("fbc", FbcLOObjectivesAllowedAttributes,
          getPackageVersion(), sbmlLevel, sbmlVersion, details);
      } 
      else if (getErrorLog()->getError(n)->getErrorId() == UnknownCoreAttribute)
      {
        const std::string details = 
          getErrorLog()->getError(n)->getMessage();
        getErrorLog()->remove(UnknownCoreAttribute);
        getErrorLog()->logPackageError("fbc", FbcLOObjectivesAllowedAttributes,
          getPackageVersion(), sbmlLevel, sbmlVersion, details);
      } 
    }
  }

  SBase::readAttributes(attributes,expectedAttributes);

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
        getErrorLog()->logPackageError("fbc", FbcObjectiveRequiredAttributes,
          getPackageVersion(), sbmlLevel, sbmlVersion, details);
      } 
      else if (getErrorLog()->getError(n)->getErrorId() == UnknownCoreAttribute)
      {
        const std::string details = 
          getErrorLog()->getError(n)->getMessage();
        getErrorLog()->remove(UnknownCoreAttribute);
        getErrorLog()->logPackageError("fbc", FbcObjectiveAllowedL3Attributes,
          getPackageVersion(), sbmlLevel, sbmlVersion, details);
      } 
    }
  }


  //
  // Reads an attribute "id" (optional)
  //
  bool assigned = attributes.readInto("id", mId);

  if (assigned)
  {
    // "id" attribute is set to this fbc element

    if (mId.empty())
    {
      //
      // Logs an error if the "id" attribute is empty.
      //
      logEmptyString(mId, sbmlLevel, sbmlVersion, "<fbc>");
    }
    else if (!SyntaxChecker::isValidSBMLSId(mId)) 
    {
      //
      // Logs an error if the "id" attribute doesn't
      // conform to the SBML type SId.
      //
      getErrorLog()->logPackageError("fbc", FbcSBMLSIdSyntax, 
        getPackageVersion(), sbmlLevel, sbmlVersion);
    }
  }
  else
  {
    std::string message = "Fbc attribute 'id' is missing.";
    getErrorLog()->logPackageError("fbc", FbcObjectiveRequiredAttributes, 
      getPackageVersion(), sbmlLevel, sbmlVersion, message);
  }

  attributes.readInto("name", mName);
  
  //
  // type string   ( use = "required" )
  //
  std::string type;
  assigned = attributes.readInto("type", type);

  if (assigned == true)
  {
    // check string is not empty

    if (type.empty() == true)
    {
      logEmptyString(type, sbmlLevel, sbmlVersion, "<Objective>");
    }
    else 
    {
       mType = ObjectiveType_fromString( type.c_str() );
       if (ObjectiveType_isValidObjectiveType(mType) == 0)
       {
          getErrorLog()->logPackageError("fbc", FbcObjectiveTypeMustBeEnum, 
            getPackageVersion(), sbmlLevel, sbmlVersion);
       }
    }
  }
  else
  {
    std::string message = "Fbc attribute 'type' is missing.";
    getErrorLog()->logPackageError("fbc", FbcObjectiveRequiredAttributes, 
      getPackageVersion(), sbmlLevel, sbmlVersion, message);
  }

}
/** @endcond */

/** @cond doxygenLibsbmlInternal */
void
Objective::writeAttributes (XMLOutputStream& stream) const
{
  SBase::writeAttributes(stream);

  stream.writeAttribute("id",   getPrefix(), mId);
  
  if(isSetName())
    stream.writeAttribute("name",   getPrefix(), mName);
  
  if (isSetType() == true)
    stream.writeAttribute("type", getPrefix(), 
                     ObjectiveType_toString(mType));

  //
  // (EXTENSION)
  //
  SBase::writeExtensionAttributes(stream);
}
/** @endcond */

/** @cond doxygenLibsbmlInternal */
void
Objective::writeElements (XMLOutputStream& stream) const
{
  SBase::writeElements(stream);

  if (getNumFluxObjectives() > 0)
    mFluxes.write(stream);

  //
  // (EXTENSION)
  //
  SBase::writeExtensionElements(stream);
}
/** @endcond */


/*
 * @return the typecode (int) of this SBML object or SBML_UNKNOWN
 * (default).
 *
 * @see getElementName()
 */
int
Objective::getTypeCode () const
{
  return SBML_FBC_OBJECTIVE;
}

Objective*
Objective::clone() const
{
    return new Objective(*this);
}


/*
 * Accepts the given SBMLVisitor.
 */
bool
Objective::accept (SBMLVisitor& v) const
{
  v.visit(*this);
  for (unsigned int n = 0; n < getNumFluxObjectives(); n++)
  {
    getFluxObjective(n)->accept(v);
  }
  v.leave(*this);

  return true;
}


/** @cond doxygenLibsbmlInternal */
/*
 * Sets the parent SBMLDocument of this SBML object.
 */
void
Objective::setSBMLDocument (SBMLDocument* d)
{
  SBase::setSBMLDocument(d);

  mFluxes.setSBMLDocument(d);
}
/** @endcond */


/** @cond doxygenLibsbmlInternal */
/*
 * Sets this SBML object to child SBML objects (if any).
 * (Creates a child-parent relationship by the parent)
  */
void
Objective::connectToChild()
{
  SBase::connectToChild();
  mFluxes.connectToParent(this);
}
/** @endcond */


/** @cond doxygenLibsbmlInternal */
/*
 * Enables/Disables the given package with this element and child
 * elements (if any).
 * (This is an internal implementation for enablePakcage function)
 */
void
Objective::enablePackageInternal(const std::string& pkgURI,
                             const std::string& pkgPrefix, bool flag)
{
  SBase::enablePackageInternal(pkgURI,pkgPrefix,flag);

  mFluxes.enablePackageInternal(pkgURI,pkgPrefix,flag);
}
/** @endcond */


/** @cond doxygenLibsbmlInternal */
/* default for components that have no required elements */
bool
Objective::hasRequiredElements() const
{
  bool allPresent = true;

  
  if (mFluxes.size() < 1)
  {
    allPresent = false;
  }
  
  return allPresent;
}
/** @endcond */


/** @cond doxygenLibsbmlInternal */
bool 
Objective::getIsSetListOfFluxObjectives() const
{
  return mIsSetListOfFluxObjectives;
}
/** @endcond */


/** @cond doxygenLibsbmlInternal */
/*
 * Ctor.
 */
ListOfObjectives::ListOfObjectives(FbcPkgNamespaces* fbcns)
 : ListOf(fbcns)
 , mActiveObjective()
{
  //
  // set the element namespace of this object
  //
  setElementNamespace(fbcns->getURI());
}
/** @endcond */


/** @cond doxygenLibsbmlInternal */
/*
 * Ctor.
 */
ListOfObjectives::ListOfObjectives(unsigned int level, unsigned int version, unsigned int pkgVersion)
 : ListOf(level,version)
  , mActiveObjective()
{
  setSBMLNamespacesAndOwn(new FbcPkgNamespaces(level,version,pkgVersion));
};
/** @endcond */


/*
 * @return a (deep) copy of this ListOfObjectives.
 */
ListOfObjectives*
ListOfObjectives::clone () const
{
  return new ListOfObjectives(*this);
}


/* return nth item in list */
Objective *
ListOfObjectives::get(unsigned int n)
{
  return static_cast<Objective*>(ListOf::get(n));
}


/* return nth item in list */
const Objective *
ListOfObjectives::get(unsigned int n) const
{
  return static_cast<const Objective*>(ListOf::get(n));
}


/* return item by symbol */
Objective*
ListOfObjectives::get (const std::string& symbol)
{
  return const_cast<Objective*>( 
    static_cast<const ListOfObjectives&>(*this).get(symbol) );
}


/* return item by symbol */
const Objective*
ListOfObjectives::get (const std::string& symbol) const
{
  vector<SBase*>::const_iterator result;

  result = find_if( mItems.begin(), mItems.end(), IdEq<Objective>(symbol) );
  return (result == mItems.end()) ? 0 : static_cast <Objective*> (*result);
}


/* Removes the nth item from this list */
Objective*
ListOfObjectives::remove (unsigned int n)
{
   return static_cast<Objective*>(ListOf::remove(n));
}


/* Removes item in this list by symbol */
Objective*
ListOfObjectives::remove (const std::string& symbol)
{
  SBase* item = 0;
  vector<SBase*>::iterator result;

  result = find_if( mItems.begin(), mItems.end(), IdEq<Objective>(symbol) );

  if (result != mItems.end())
  {
    item = *result;
    mItems.erase(result);
  }

  return static_cast <Objective*> (item);
}


/*
 * @return the typecode (int) of SBML objects contained in this ListOf or
 * SBML_UNKNOWN (default).
 */
int
ListOfObjectives::getItemTypeCode () const
{
  return SBML_FBC_OBJECTIVE;
}


/*
 * Returns the XML element name of
 * this SBML object.
 */
const std::string&
ListOfObjectives::getElementName () const
{
  static const std::string name = "listOfObjectives";
  return name;
}


int ListOfObjectives::appendFrom(const ListOf* list)
{
  int ret = ListOf::appendFrom(list);
  if (ret != LIBSBML_OPERATION_SUCCESS) return ret;

  const ListOfObjectives* objectives = static_cast<const ListOfObjectives*>(list);
  if (objectives==NULL) return LIBSBML_INVALID_OBJECT;

  if (!isSetActiveObjective()) {
    setActiveObjective(objectives->getActiveObjective());
  }
  return ret;
}

void
ListOfObjectives::renameSIdRefs(const std::string& oldid, const std::string& newid)
{
  if (mActiveObjective==oldid) mActiveObjective=newid;
  ListOf::renameSIdRefs(oldid, newid);
}

/** @cond doxygenLibsbmlInternal */
SBase*
ListOfObjectives::createObject (XMLInputStream& stream)
{
  const std::string& name   = stream.peek().getName();
  SBase*        object = 0;


  if (name == "objective")
  {
    try
    {
      FBC_CREATE_NS(fbcns, getSBMLNamespaces());
      object = new Objective(fbcns);
      appendAndOwn(object);
      delete fbcns;
      //mItems.push_back(object);
    }
    catch(...)
    {
      /* 
      * NULL will be returned if the mSBMLNS is invalid (basically this
      * should not happen) or some exception is thrown (e.g. std::bad_alloc)
      *
      * (Maybe this should be changed so that caller can detect what kind 
      *  of error happened in this function.)
      */
    }

  }

  return object;
}
/** @endcond */

/** @cond doxygenLibsbmlInternal */
void
ListOfObjectives::addExpectedAttributes(ExpectedAttributes& attributes)
{
  SBase::addExpectedAttributes(attributes);
  //
  // required attribute is not defined for SBML Level 2 or lesser.
  //
  if ( getLevel() > 2 )
  {    
    attributes.add("activeObjective");
  }
}
/** @endcond */

/** @cond doxygenLibsbmlInternal */
void 
ListOfObjectives::readAttributes (const XMLAttributes& attributes,
                                  const ExpectedAttributes& expectedAttributes)
{
  SBase::readAttributes(attributes, expectedAttributes);
  
  if ( getLevel() > 2 )
  {    
    bool assigned = attributes.readInto("activeObjective", mActiveObjective, 
      getErrorLog(), false, getLine(), getColumn());
    if (assigned && mActiveObjective.empty())
    {
      logEmptyString(mActiveObjective, getLevel(), getVersion(), 
        "<listOfObjectives>");
    }
    if (!SyntaxChecker::isValidSBMLSId(mActiveObjective)) 
    {
      getErrorLog()->logPackageError("fbc", FbcActiveObjectiveSyntax, 
        getPackageVersion(), getLevel(), getVersion());
    }
    
  }
}
/** @endcond */

/** @cond doxygenLibsbmlInternal */
void 
ListOfObjectives::writeAttributes (XMLOutputStream& stream) const
{
  //
  // required attribute is not defined for SBML Level 2 .
  //
  if ( getLevel() < 3)
    return;
  
  //cout << "[DEBUG] SBMLDocumentPlugin::writeAttributes() " << endl;
  if ( isSetActiveObjective() ) 
  {
    stream.writeAttribute("activeObjective", getPrefix(), mActiveObjective);
  }
}
/** @endcond */



bool 
ListOfObjectives::isSetActiveObjective() const
{
  return !mActiveObjective.empty();
}

int 
ListOfObjectives::setActiveObjective(const std::string &activeObjective)
{
  if (!SyntaxChecker::isValidSBMLSId(activeObjective)) 
  {
    return LIBSBML_INVALID_ATTRIBUTE_VALUE;
  }
  mActiveObjective = activeObjective;
  return LIBSBML_OPERATION_SUCCESS;
}

const std::string &
ListOfObjectives::getActiveObjective() const
{
  return mActiveObjective;
}

int 
ListOfObjectives::unsetActiveObjective()
{
  mActiveObjective.erase();
  if (mActiveObjective.empty())
    return LIBSBML_OPERATION_SUCCESS;
  else
    return LIBSBML_OPERATION_FAILED;
}


/** @cond doxygenLibsbmlInternal */

bool 
ListOfObjectives::accept(SBMLVisitor& v) const
{
  return v.visit(*this);
}

/** @endcond */



#endif /* __cplusplus */
/** @cond doxygenIgnored */

LIBSBML_EXTERN
Objective_t *
Objective_create(unsigned int level, unsigned int version, unsigned int pkgversion)
{
  return new Objective(level, version, pkgversion);
}


LIBSBML_EXTERN
const char *
Objective_getId(Objective_t * obj)
{
  if (obj == NULL)
    return NULL;

  return obj->getId().empty() ? "" : safe_strdup(obj->getId().c_str());
}


LIBSBML_EXTERN
int
Objective_isSetId(Objective_t * obj)
{
  return (obj != NULL) ? static_cast<int>(obj->isSetId()) : 0;
}


LIBSBML_EXTERN
int
Objective_setId(Objective_t * obj, const char * id)
{
  return (obj != NULL) ? obj->setId(id) : LIBSBML_INVALID_OBJECT;
}


LIBSBML_EXTERN
int
Objective_unsetId(Objective_t * obj)
{
  return (obj != NULL) ? obj->unsetId() : LIBSBML_INVALID_OBJECT;
}


LIBSBML_EXTERN
const char *
Objective_getName(Objective_t * obj)
{
  if (obj == NULL)
    return NULL;
  
  return obj->getName().c_str();
}


LIBSBML_EXTERN
int
Objective_isSetName(Objective_t * obj)
{
  return (obj != NULL) ? static_cast<int>(obj->isSetName()) : 0;
}


LIBSBML_EXTERN
int
Objective_setName(Objective_t * obj, const char * name)
{
  return (obj != NULL) ? obj->setName(name) : LIBSBML_INVALID_OBJECT;
}


LIBSBML_EXTERN
int
Objective_unsetName(Objective_t * obj)
{
  return (obj != NULL) ? obj->unsetName() : LIBSBML_INVALID_OBJECT;
}



LIBSBML_EXTERN
const char *
Objective_getType(Objective_t * obj)
{
  if (obj == NULL)
    return NULL;

  return obj->getType().empty() ? "" : safe_strdup(obj->getType().c_str());
}


LIBSBML_EXTERN
int
Objective_isSetType(Objective_t * obj)
{
  return (obj != NULL) ? static_cast<int>(obj->isSetType()) : 0;
}


LIBSBML_EXTERN
int
Objective_setType(Objective_t * obj, const char * type)
{
  return (obj != NULL) ? obj->setType(type) : LIBSBML_INVALID_OBJECT;
}


LIBSBML_EXTERN
int
Objective_unsetType(Objective_t * obj)
{
  return (obj != NULL) ? obj->unsetType() : LIBSBML_INVALID_OBJECT;
}


LIBSBML_EXTERN
int
Objective_addFluxObjective(Objective_t * obj, FluxObjective_t * flux)
{
  return (obj != NULL) ? obj->addFluxObjective(flux) : LIBSBML_INVALID_OBJECT;
}


LIBSBML_EXTERN
FluxObjective_t *
Objective_getFluxObjective(Objective_t * obj, unsigned int n)
{
  return (obj != NULL) ? obj->getFluxObjective(n) : NULL;
}


LIBSBML_EXTERN
unsigned int
Objective_getNumFluxObjectives(Objective_t * obj)
{
  return (obj != NULL) ? obj->getNumFluxObjectives() : SBML_INT_MAX;
}


LIBSBML_EXTERN
const char *
ListOfObjectives_getActiveObjective(ListOf_t * lo)
{
  if (lo == NULL)
    return NULL;

  return static_cast<ListOfObjectives *>(lo)->getActiveObjective().empty() ? safe_strdup("") 
                                    : safe_strdup(static_cast<ListOfObjectives *>(lo)->getActiveObjective().c_str());
}


LIBSBML_EXTERN
int
ListOfObjectives_isSetActiveObjective(ListOf_t * lo)
{
  return (lo != NULL) ? static_cast<int>(static_cast<ListOfObjectives *>(lo)->isSetActiveObjective()) : 0;
}


LIBSBML_EXTERN
int
ListOfObjectives_setActiveObjective(ListOf_t * lo, const char * obj)
{
  return (lo != NULL) ? static_cast<ListOfObjectives *>(lo)->setActiveObjective(obj) : LIBSBML_INVALID_OBJECT;
}


LIBSBML_EXTERN
int
ListOfObjectives_unsetActiveObjective(ListOf_t * lo)
{
  return (lo != NULL) ? static_cast<ListOfObjectives *>(lo)->unsetActiveObjective() : LIBSBML_INVALID_OBJECT;
}


static
const char* OBJECTIVE_TYPE_STRINGS[] =
{
    "maximize"
  , "minimize"
  , "unknown"
};


LIBSBML_EXTERN
const char* 
ObjectiveType_toString(ObjectiveType_t type)
{
  int max = OBJECTIVE_TYPE_UNKNOWN;

  if (type < OBJECTIVE_TYPE_MAXIMIZE || type >= max)
  {
      return NULL;
  }

  return OBJECTIVE_TYPE_STRINGS[type];
}


LIBSBML_EXTERN
ObjectiveType_t 
ObjectiveType_fromString(const char* s)
{
  if (s == NULL) 
  {
    return OBJECTIVE_TYPE_UNKNOWN;
  }

  int max = OBJECTIVE_TYPE_UNKNOWN;
  for (int i = 0; i < max; i++)
  {
    if (strcmp(OBJECTIVE_TYPE_STRINGS[i], s) == 0)
      return (ObjectiveType_t)i;
  }
  return OBJECTIVE_TYPE_UNKNOWN;
}


LIBSBML_EXTERN
int 
ObjectiveType_isValidObjectiveType(ObjectiveType_t effect)
{
  int max = OBJECTIVE_TYPE_UNKNOWN;

  if (effect < OBJECTIVE_TYPE_MAXIMIZE || effect >= max)
  {
    return 0;
  }
  else
  {
    return 1;
  }
}


LIBSBML_EXTERN
int 
ObjectiveType_isValidObjectiveTypeString(const char* s)
{
  return ObjectiveType_isValidObjectiveType
                                         (ObjectiveType_fromString(s));
}



/** @endcond */
LIBSBML_CPP_NAMESPACE_END


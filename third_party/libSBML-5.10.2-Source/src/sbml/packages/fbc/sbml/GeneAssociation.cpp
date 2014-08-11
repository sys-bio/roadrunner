/**
 * @file    GeneAssociation.cpp
 * @brief   Implementation of GeneAssociation, the SBase derived class of the fbc package.
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

#include <sbml/packages/fbc/sbml/GeneAssociation.h>
#include <sbml/packages/fbc/extension/FbcExtension.h>

using namespace std;

LIBSBML_CPP_NAMESPACE_BEGIN
#ifdef __cplusplus

/*
 * Creates a new GeneAssociation with the given level, version, and package version.
 */
GeneAssociation::GeneAssociation (unsigned int level, unsigned int version, unsigned int pkgVersion) 
  : SBase (level,version)
   ,mId("")
   ,mReaction("")
   ,mAssociation(NULL)
{
  // set an SBMLNamespaces derived object (FbcPkgNamespaces) of this package.
  setSBMLNamespacesAndOwn(new FbcPkgNamespaces(level,version,pkgVersion));  
}


/*
 * Creates a new GeneAssociation with the given FbcPkgNamespaces object.
 */
GeneAssociation::GeneAssociation(FbcPkgNamespaces* fbcns)
 : SBase(fbcns)
  ,mId("")
  ,mReaction("")
  ,mAssociation(NULL)
{
  //
  // set the element namespace of this object
  //
  setElementNamespace(fbcns->getURI());

  // load package extensions bound with this object (if any) 
  loadPlugins(fbcns);
}


/*
 * Copy constructor.
 */
GeneAssociation::GeneAssociation(const GeneAssociation& source)
  : SBase(source)
  , mId(source.mId)
  , mReaction(source.mReaction)
  , mAssociation(NULL)
{
    if (source.mAssociation != NULL)
      this->mAssociation = new Association(*source.mAssociation);
    else
      this->mAssociation = NULL;
}



/*
 * Creates a new GeneAssociation from the given XMLNode
 */
GeneAssociation::GeneAssociation(const XMLNode& node, FbcPkgNamespaces* fbcns)
: SBase(fbcns)
,mId("")
,mReaction("")
,mAssociation(NULL)
{
  //
  // set the element namespace of this object
  //
  setElementNamespace(fbcns->getURI());
  
  // load package extensions bound with this object (if any)
  loadPlugins(fbcns);
  
  const XMLAttributes& attributes=node.getAttributes();
  const XMLNode* child;
  
  //ExpectedAttributes ea(getElementName());
  ExpectedAttributes ea;
  addExpectedAttributes(ea);
  this->readAttributes(attributes,ea);
  unsigned int n=0,nMax = node.getNumChildren();
  while(n<nMax)
  {
    child=&node.getChild(n);
    const std::string& childName=child->getName();
    
    if(childName=="gene" || childName=="or" || childName == "and")
    {
      mAssociation = new Association(*child, new FbcPkgNamespaces(*fbcns));
    }
    else if(childName=="annotation")
    {
      this->mAnnotation=new XMLNode(*child);
    }
    else if(childName=="notes")
    {
      this->mNotes=new XMLNode(*child);
    }
    else
    {
      //throw;
    }
    ++n;
  }
  setSBMLNamespacesAndOwn(fbcns);
  connectToChild();
}


/*
 * Creates an XMLNode object from this.
 */
XMLNode GeneAssociation::toXML() const
{
  XMLNamespaces xmlns = XMLNamespaces();
  XMLTriple triple = XMLTriple(getElementName(), "", "");
  XMLAttributes att = XMLAttributes();

  if (isSetId())
  {
    att.add("id", mId);
  }
  if (isSetReaction())
  {
    att.add("reaction", mReaction);
  }

  XMLToken token = XMLToken(triple, att, xmlns);
  XMLNode node(token);
  // add the notes and annotations
  if(mNotes) node.addChild(*this->mNotes);
  if(mAnnotation) node.addChild(*this->mAnnotation);
  
  if (isSetAssociation())
  {
    node.addChild(mAssociation->toXML());
  }
  return node;
}


/*
 * Assignment operator.
 */
GeneAssociation& GeneAssociation::operator=(const GeneAssociation& source)
{
  if(&source!=this)
  {
    this->SBase::operator=(source);
    this->mId = source.mId;
    this->mReaction = source.mReaction;
    if (mAssociation != NULL)
      delete mAssociation;
    if (source.mAssociation != NULL)
      this->mAssociation = new Association(*source.mAssociation);
    else
      this->mAssociation = NULL;
  }
  
  return *this;
}


/*
 * Destructor.
 */ 
GeneAssociation::~GeneAssociation ()
{
  if (isSetAssociation())
  {
    delete mAssociation;
    mAssociation = NULL;
  }
}


/*
  * Returns the value of the "id" attribute of this GeneAssociation.
  */
const std::string& 
GeneAssociation::getId () const
{
  return mId;
}


/*
  * Predicate returning @c true or @c false depending on whether this
  * GeneAssociation's "id" attribute has been set.
  */
bool 
GeneAssociation::isSetId () const
{
  return (mId.empty() == false);
}

/*
  * Sets the value of the "id" attribute of this GeneAssociation.
  */
int 
GeneAssociation::setId (const std::string& id)
{
  return SyntaxChecker::checkAndSetSId(id ,mId);
}


/*
  * Unsets the value of the "id" attribute of this GeneAssociation.
  */
int 
GeneAssociation::unsetId ()
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
  * Returns the value of the "reaction" attribute of this GeneAssociation.
  */
const std::string& 
GeneAssociation::getReaction () const
{
  return mReaction;
}


/*
  * Predicate returning @c true or @c false depending on whether this
  * GeneAssociation's "reaction" attribute has been set.
  */
bool 
GeneAssociation::isSetReaction () const
{
  return (mReaction.empty() == false);
}

/*
  * Sets the value of the "reaction" attribute of this GeneAssociation.
  */
int 
GeneAssociation::setReaction (const std::string& reaction)
{
  return SyntaxChecker::checkAndSetSId(reaction ,mReaction);
}


/*
  * Unsets the value of the "reaction" attribute of this GeneAssociation.
  */
int 
GeneAssociation::unsetReaction ()
{
  mReaction.erase();
  if (mReaction.empty())
  {
    return LIBSBML_OPERATION_SUCCESS;
  }
  else
  {
    return LIBSBML_OPERATION_FAILED;
  }
}


/*
  * Returns the value of the "association" attribute of this GeneAssociation.
  */
const Association* 
GeneAssociation::getAssociation () const
{
  return mAssociation;
}

/*
  * Returns the value of the "association" attribute of this GeneAssociation.
  */
Association* 
GeneAssociation::getAssociation ()
{
  return mAssociation;
}

/*
  * Predicate returning @c true or @c false depending on whether this
  * GeneAssociation's "association" attribute has been set.
  */
bool 
GeneAssociation::isSetAssociation () const
{
  return (mAssociation != NULL);
}


Association* GeneAssociation::createAssociation()
{
  Association* result = NULL;
  try 
  {
    FBC_CREATE_NS(fbcns, getSBMLNamespaces());
    result = new Association(fbcns);
    unsetAssociation();
    mAssociation = result;
    mAssociation->connectToParent(this);
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
  * Sets the value of the "association" attribute of this GeneAssociation.
  */
int 
  GeneAssociation::setAssociation (const Association* association)
{
 if (mAssociation == association)
  {
    return LIBSBML_OPERATION_SUCCESS;
  }
  else if (association == NULL)
  {
    return unsetAssociation();
  }
  else if (getLevel() != association->getLevel())
  {
    return LIBSBML_LEVEL_MISMATCH;
  }
  else if (getVersion() != association->getVersion())
  {
    return LIBSBML_VERSION_MISMATCH;
  }
  else
  {
    if (mAssociation != NULL)
      delete mAssociation;
    
    mAssociation = (association != NULL) ? 
      static_cast<Association*>( association->clone() ) : NULL;

    if (mAssociation != NULL) 
      mAssociation->connectToParent(this);
    
    return LIBSBML_OPERATION_SUCCESS;
  }
}


/*
  * Unsets the value of the "association" attribute of this GeneAssociation.
  */
int 
GeneAssociation::unsetAssociation ()
{
  if (mAssociation != NULL)
  delete mAssociation;
  mAssociation=NULL;
  return LIBSBML_OPERATION_SUCCESS;
}


/*
 * Returns the XML element name of
 * this SBML object.
 */
const std::string&
GeneAssociation::getElementName () const
{
  static const std::string name = "geneAssociation";
  return name;
}


/** @cond doxygenLibsbmlInternal */
SBase*
GeneAssociation::createObject (XMLInputStream& stream)
{
  SBase* object = NULL;

  const string& name = stream.peek().getName();
  if (name == "gene" || name == "or" || name == "and")
  {
    if (mAssociation != NULL)
    {
      logError(NotSchemaConformant, getLevel(), getVersion(),
         "Only one <association> element is permitted "
         "in a single <geneAssociation> element.");
    }    

      mAssociation = new Association(getLevel(), getVersion());
      if (name == "gene")
        mAssociation->setType(GENE_ASSOCIATION);
      else if (name == "and")
        mAssociation->setType(AND_ASSOCIATION);
      else if (name == "or")
        mAssociation->setType(OR_ASSOCIATION);
      object = mAssociation;
  }
  return object;
}
/** @endcond */

/** @cond doxygenLibsbmlInternal */
void
GeneAssociation::addExpectedAttributes(ExpectedAttributes& attributes)
{
  SBase::addExpectedAttributes(attributes);

  attributes.add("id");
  attributes.add("reaction");
}
/** @endcond */

/** @cond doxygenLibsbmlInternal */
void
GeneAssociation::readAttributes (const XMLAttributes& attributes,
                        const ExpectedAttributes& expectedAttributes)
{
  SBase::readAttributes(attributes,expectedAttributes);

  const unsigned int sbmlLevel   = getLevel  ();
  const unsigned int sbmlVersion = getVersion();

  bool assigned = attributes.readInto("id", mId, getErrorLog(), true, getLine(), getColumn());
  if (assigned && mId.empty())
  {
    logEmptyString(mId, sbmlLevel, sbmlVersion, "<geneAssociation>");
  }
  if (!SyntaxChecker::isValidSBMLSId(mId)) logError(InvalidIdSyntax);

  assigned = attributes.readInto("reaction", mReaction, getErrorLog(), true, getLine(), getColumn());
  if (assigned && mReaction.empty())
  {
    logEmptyString(mReaction, sbmlLevel, sbmlVersion, "<geneAssociation>");
  }
  if (!SyntaxChecker::isValidSBMLSId(mReaction)) 
    logError(InvalidIdSyntax, getLevel(), getVersion(), 
    "The syntax of the attribute reaction='" + mReaction + "' does not conform.");
}
/** @endcond */

/** @cond doxygenLibsbmlInternal */
void
GeneAssociation::writeAttributes (XMLOutputStream& stream) const
{
  SBase::writeAttributes(stream);

  stream.writeAttribute("id",   getPrefix(), mId);
  stream.writeAttribute("reaction",   getPrefix(), mReaction);

  //
  // (EXTENSION)
  //
  SBase::writeExtensionAttributes(stream);
}
/** @endcond */

/** @cond doxygenLibsbmlInternal */
void
GeneAssociation::writeElements (XMLOutputStream& stream) const
{
  SBase::writeElements(stream);

  if (isSetAssociation())
    mAssociation ->write(stream);


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
GeneAssociation::getTypeCode () const
{
  return SBML_FBC_GENEASSOCIATION;
}

GeneAssociation*
GeneAssociation::clone() const
{
    return new GeneAssociation(*this);
}


/*
 * Accepts the given SBMLVisitor.
 */
bool
GeneAssociation::accept (SBMLVisitor& v) const
{
  return false;
}


/*
 * Ctor.
 */
ListOfGeneAssociations::ListOfGeneAssociations(FbcPkgNamespaces* fbcns)
 : ListOf(fbcns)
{
  //
  // set the element namespace of this object
  //
  setElementNamespace(fbcns->getURI());
}


/*
 * Ctor.
 */
ListOfGeneAssociations::ListOfGeneAssociations(unsigned int level, unsigned int version, unsigned int pkgVersion)
 : ListOf(level,version)
{
  setSBMLNamespacesAndOwn(new FbcPkgNamespaces(level,version,pkgVersion));
};


/*
 * @return a (deep) copy of this ListOfGeneAssociations.
 */
ListOfGeneAssociations*
ListOfGeneAssociations::clone () const
{
  return new ListOfGeneAssociations(*this);
}


/* return nth item in list */
GeneAssociation *
ListOfGeneAssociations::get(unsigned int n)
{
  return static_cast<GeneAssociation*>(ListOf::get(n));
}


/* return nth item in list */
const GeneAssociation *
ListOfGeneAssociations::get(unsigned int n) const
{
  return static_cast<const GeneAssociation*>(ListOf::get(n));
}


/* return item by symbol */
GeneAssociation*
ListOfGeneAssociations::get (const std::string& symbol)
{
  return const_cast<GeneAssociation*>( 
    static_cast<const ListOfGeneAssociations&>(*this).get(symbol) );
}


/* return item by symbol */
const GeneAssociation*
ListOfGeneAssociations::get (const std::string& symbol) const
{
  vector<SBase*>::const_iterator result;

  result = find_if( mItems.begin(), mItems.end(), IdEq<GeneAssociation>(symbol) );
  return (result == mItems.end()) ? 0 : static_cast <GeneAssociation*> (*result);
}


/* Removes the nth item from this list */
GeneAssociation*
ListOfGeneAssociations::remove (unsigned int n)
{
   return static_cast<GeneAssociation*>(ListOf::remove(n));
}


/* Removes item in this list by symbol */
GeneAssociation*
ListOfGeneAssociations::remove (const std::string& symbol)
{
  SBase* item = 0;
  vector<SBase*>::iterator result;

  result = find_if( mItems.begin(), mItems.end(), IdEq<GeneAssociation>(symbol) );

  if (result != mItems.end())
  {
    item = *result;
    mItems.erase(result);
  }

  return static_cast <GeneAssociation*> (item);
}


/*
 * @return the typecode (int) of SBML objects contained in this ListOf or
 * SBML_UNKNOWN (default).
 */
int
ListOfGeneAssociations::getItemTypeCode () const
{
  return SBML_FBC_GENEASSOCIATION;
}


/*
 * Returns the XML element name of
 * this SBML object.
 */
const std::string&
ListOfGeneAssociations::getElementName () const
{
  static const std::string name = "listOfGeneAssociations";
  return name;
}


/** @cond doxygenLibsbmlInternal */
SBase*
ListOfGeneAssociations::createObject (XMLInputStream& stream)
{
  const std::string& name   = stream.peek().getName();
  SBase*        object = 0;


  if (name == "geneAssociation")
  {
    try
    {
      FBC_CREATE_NS(fbcns, getSBMLNamespaces());
      object = new GeneAssociation(fbcns);
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


#endif /* __cplusplus */
LIBSBML_CPP_NAMESPACE_END


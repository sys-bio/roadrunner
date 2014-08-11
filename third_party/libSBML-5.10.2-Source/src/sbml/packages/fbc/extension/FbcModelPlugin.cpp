/**
 * @file    FbcModelPlugin.cpp
 * @brief   Implementation of FbcModelPlugin, the plugin class of
 *          the fbc package for the Model element.
 * @author  Frank T. Bergmann
 *
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

#include <sbml/packages/fbc/extension/FbcModelPlugin.h>
#include <sbml/packages/fbc/extension/FbcExtension.h>
#include <sbml/packages/fbc/validator/FbcSBMLError.h>


#include <sbml/util/ElementFilter.h>

#include <iostream>
using namespace std;


LIBSBML_CPP_NAMESPACE_BEGIN
#ifdef __cplusplus

  /*
  * Constructor
  */
  FbcModelPlugin::FbcModelPlugin (const std::string &uri, 
  const std::string &prefix,
  FbcPkgNamespaces *fbcns)
  : SBasePlugin(uri,prefix, fbcns)
  , mBounds(fbcns)
  , mObjectives(fbcns)
  , mAssociations(fbcns)
{
  // connect child elements to this element.
  connectToChild();
}


/*
* Copy constructor. Creates a copy of this FbcModelPlugin object.
*/
FbcModelPlugin::FbcModelPlugin(const FbcModelPlugin& orig)
  : SBasePlugin(orig)
  , mBounds(orig.mBounds)
  , mObjectives(orig.mObjectives)
  , mAssociations(orig.mAssociations)
{
  // connect child elements to this element.
  connectToChild();
}


/*
* Destroy this object.
*/
FbcModelPlugin::~FbcModelPlugin () {}

/*
* Assignment operator for FbcModelPlugin.
*/
FbcModelPlugin& 
  FbcModelPlugin::operator=(const FbcModelPlugin& orig)
{
  if(&orig!=this)
  {
    this->SBasePlugin::operator =(orig);
    mBounds       = orig.mBounds;
    mObjectives   = orig.mObjectives;
    mAssociations = orig.mAssociations;

    // connect child elements to this element.
    connectToChild();
  }    
  return *this;
}


/*
* Creates and returns a deep copy of this FbcModelPlugin object.
* 
* @return a (deep) copy of this FbcModelPlugin object
*/
FbcModelPlugin* 
  FbcModelPlugin::clone () const
{
  return new FbcModelPlugin(*this);  
}


/** @cond doxygenLibsbmlInternal */
SBase*
  FbcModelPlugin::createObject(XMLInputStream& stream)
{
  SBase*        object = 0;

  const std::string&   name   = stream.peek().getName();
  const XMLNamespaces& xmlns  = stream.peek().getNamespaces();
  const std::string&   prefix = stream.peek().getPrefix();

  const std::string& targetPrefix = (xmlns.hasURI(mURI)) ? xmlns.getPrefix(mURI) : mPrefix;

  if (prefix == targetPrefix)
  {
    if ( name == "listOfFluxBounds" ) 
    {
      if (mBounds.size() != 0)
      {
        getErrorLog()->logPackageError("fbc", FbcOnlyOneEachListOf, 
          getPackageVersion(), getLevel(), getVersion());
      }
      
      object = &mBounds;

      if (targetPrefix.empty())
      {
        mBounds.getSBMLDocument()->enableDefaultNS(mURI,true);
      }
    }          
    else if ( name == "listOfObjectives" ) 
    {
      if (mObjectives.size() != 0)
      {
        getErrorLog()->logPackageError("fbc", FbcOnlyOneEachListOf, 
          getPackageVersion(), getLevel(), getVersion());
      }
      
      object = &mObjectives;

      if (targetPrefix.empty())
      {
        mObjectives.getSBMLDocument()->enableDefaultNS(mURI,true);
      }
    }          
    else if ( name == "listOfGeneAssociations" ) 
    {
      object = &mAssociations;

      if (targetPrefix.empty())
      {
        mAssociations.getSBMLDocument()->enableDefaultNS(mURI,true);
      }
    }          
  }    

  return object;
}
/** @endcond */

/** @cond doxygenLibsbmlInternal */
void
  FbcModelPlugin::writeElements (XMLOutputStream& stream) const
{
  if (getNumFluxBounds() > 0)
  {
    mBounds.write(stream);
  }    
  if (getNumObjectives() > 0)
  {
    mObjectives.write(stream);
  }    
  
  // gene associations did not make the cut for the first version,
  // so we can't save them here, instead do it in an annotation.
  /*if (getNumGeneAssociations() > 0)
  {
    mAssociations.write(stream);
  }*/   
}
/** @endcond */



LIBSBML_EXTERN
void
parseFbcAnnotation(XMLNode * annotation, ListOfGeneAssociations& associations,
                   FbcPkgNamespaces* fbcns)
{
  
  if (!annotation) return;
  
  const string&  name = annotation->getName();
  const XMLNode*  plOGATop = NULL;
  GeneAssociation* ga;
  unsigned int n = 0;
  
  // need to find the layout desciption opening annotation
  if (name == "annotation" && annotation->getNumChildren() > 0)
  {
    while (n < annotation->getNumChildren())
    {
      const string &name1 = annotation->getChild(n).getName();
      if (name1 == "listOfGeneAssociations") // also check the namespace
      {
        const XMLNamespaces& namespaces=annotation->getChild(n).getNamespaces();
        if(namespaces.getIndex(FbcExtension::getXmlnsL3V1V1())!=-1)
        {
          plOGATop = &(annotation->getChild(n));
          break;
        }
      }
      n++;
    }
  }
  
  // find qualifier nodes and create
  
  
  n = 0;
  if (plOGATop)
  {
    while (n < plOGATop->getNumChildren())
    {
      const string &name2 = plOGATop->getChild(n).getName();
      
      if (name2 == "annotation")
      {
        const XMLNode &annot = plOGATop->getChild(n);
        associations.setAnnotation(&annot);
      }
      
      if (name2 == "geneAssociation")
      {
        ga = new GeneAssociation(plOGATop->getChild(n), fbcns);
        associations.appendAndOwn(ga);
      }
      
      n++;
    }
  }
}


LIBSBML_EXTERN
XMLNode* deleteFbcAnnotation(XMLNode* pAnnotation)
{
  if (pAnnotation == NULL) 
    return NULL;
  
  const string&  name = pAnnotation->getName();
  unsigned int n = 0;
  
  if (!(name == "annotation" && pAnnotation->getNumChildren() > 0))
    return pAnnotation;

  
  // need to find each annotation and remove it if it is a list of gene associations
  while (n < pAnnotation->getNumChildren())
  {
    const string &name1 = pAnnotation->getChild(n).getName();
    if (name1 == "listOfGeneAssociations" ||
        pAnnotation->getChild(n).getNamespaces().getIndex(FbcExtension::getXmlnsL3V1V1()) !=-1)
    {
      // delete the anotation
      XMLNode* temp = pAnnotation->removeChild(n);
      if (temp != NULL)
        delete temp;

      continue;
    }
    n++;
  }
  
  return pAnnotation;
}


/** @cond doxygenLibsbmlInternal */
void
FbcModelPlugin::writeAttributes (XMLOutputStream& stream) const
{
  Model *parent = static_cast<Model*>(const_cast<SBase*>(getParentSBMLObject()));
  if (parent == NULL) return;
  
  
  XMLNode *parentAnnotation = parent->getAnnotation();
  if (parentAnnotation != NULL && parentAnnotation->getNumChildren() > 0)
  {
    deleteFbcAnnotation(parentAnnotation);
  }
  
  XMLToken ann_token = XMLToken(XMLTriple("annotation", "", ""), XMLAttributes());
  XMLNode* annt = new XMLNode(ann_token);

  
  
  if( mAssociations.size() > 0)
  {
    XMLAttributes loga_attr = XMLAttributes();
    loga_attr.add("xmlns", FbcExtension::getXmlnsL3V1V1());
    XMLToken loga_token = XMLToken(XMLTriple("listOfGeneAssociations", FbcExtension::getXmlnsL3V1V1(), ""), loga_attr);
    XMLNode loga = XMLNode(loga_token);

    for (unsigned int i = 0; i < mAssociations.size(); ++i)
      loga.addChild(mAssociations.get(i)->toXML());
    
    // then add the ones toXML()
    annt->addChild(loga);
  }

  
  if (annt && annt->getNumChildren() > 0)
  {
    parent->appendAnnotation(annt);
    delete annt;
  }
}
/** @endcond */


/** @cond doxygenLibsbmlInternal */
void 
FbcModelPlugin::parseAnnotation(SBase *parentObject, XMLNode *pAnnotation)
{
  mAssociations.setSBMLDocument(mSBML); 
  // don't read if we have an invalid node or already a gene associations object
  if (pAnnotation == NULL || mAssociations.size() > 0)
    return;

  // annotation element has been parsed by the parent element
  // (Model) of this plugin object, thus the annotation element 
  // set to the above pAnnotation variable is parsed in this block.
  
  XMLNode& listOfGeneAssociations = pAnnotation->getChild("listOfGeneAssociations");
  if (listOfGeneAssociations.getNumChildren() == 0)
    return;
 
  // read the xml node, overriding that all errors are flagged as 
  // warnings
  mAssociations.read(listOfGeneAssociations, LIBSBML_OVERRIDE_WARNING);
  // remove listOfLayouts annotation  
  parentObject->removeTopLevelAnnotationElement("listOfGeneAssociations", "", false);
}
/** @endcond */


/** @cond doxygenLibsbmlInternal */
bool
FbcModelPlugin::readOtherXML (SBase* parentObject, XMLInputStream& stream)
{
#ifndef ANNOATION
  return false;
#else
  bool readAnnotationFromStream = false;
  const string& name = stream.peek().getName();
  
  if (!(name.empty()) && name != "annotation")
  {
    return readAnnotationFromStream;
  }
  
  try
  {   
    XMLNode *pAnnotation = parentObject->getAnnotation();
    FBC_CREATE_NS(fbcns, getSBMLNamespaces());
    
    if (!pAnnotation)
    {
      //
      // (NOTES)
      //
      // annotation element has not been parsed by the parent element
      // (Model) of this plugin object, thus annotation element is
      // parsed via the given XMLInputStream object in this block.
      //
      
      const string& name = stream.peek().getName();
      
    
      if (name == "annotation")
      {
        pAnnotation = new XMLNode(stream);
        
        parseFbcAnnotation(pAnnotation, mAssociations, fbcns);
        
        if (mAssociations.size() > 0)
        {
          //
          // Removes the annotation for layout extension from the annotation
          // of parent element (pAnnotation) and then set the new annotation
          // (newAnnotation) to the parent element.
          //
          XMLNode *newAnnotation = deleteFbcAnnotation(pAnnotation);
          parentObject->setAnnotation(newAnnotation);
          delete newAnnotation;
        }
        else
        {
          //
          // No layout annotation is included in the read annotation
          // (pAnnotation) and thus just set the annotation to the parent
          // element.
          //
          parentObject->setAnnotation(pAnnotation);
        }
        
        delete pAnnotation;
        
        readAnnotationFromStream = true;
      }
      
    }
    else if (mAssociations.size() == 0)
    {
      //
      // (NOTES)
      //
      // annotation element has been parsed by the parent element
      // (Model) of this plugin object, thus the annotation element
      // set to the above pAnnotation variable is parsed in this block.
      //
      parseFbcAnnotation(pAnnotation, mAssociations, fbcns);
      
      if (mAssociations.size() > 0)
      {
        //
        // Removes the annotation for layout extension from the annotation
        // of parent element (pAnnotation) and then set the new annotation
        // (newAnnotation) to the parent element.
        //
        XMLNode *newAnnotation = deleteFbcAnnotation(pAnnotation);
        parentObject->setAnnotation(newAnnotation);
      }
      
      readAnnotationFromStream = true;
    }
    
    delete fbcns;
  }
  catch(...)
  {
    // an exception occured, most likely becase a namespace constructor
    // threw an exception, catching this here, and return false, to indicate
    // that the annotation wasn't read. 
    readAnnotationFromStream = false;
  }
  
  return readAnnotationFromStream;
#endif
}
/** @endcond */

/** @cond doxygenLibsbmlInternal */
/* default for components that have no required elements */
bool
  FbcModelPlugin::hasRequiredElements() const
{
  bool allPresent = true;

  if (mBounds.size() < 1)
  {
    allPresent = false;    
  }
  if (mObjectives.size() < 1)
  {
    allPresent = false;    
  }
  if (mAssociations.size() < 1)
  {
    allPresent = false;    
  }
  return allPresent;
}
/** @endcond */

SBase* 
FbcModelPlugin::getElementBySId(const std::string& id)
{
  if (id.empty()) return NULL;
  SBase* obj = mBounds.getElementBySId(id);
  if (obj != NULL) return obj;
  obj = mObjectives.getElementBySId(id);
  if (obj != NULL) return obj;
  obj = mAssociations.getElementBySId(id);
  return obj;
}


SBase*
FbcModelPlugin::getElementByMetaId(const std::string& metaid)
{
  if (metaid.empty()) return NULL;
  if (mBounds.getMetaId() == metaid) return &mBounds;
  if (mObjectives.getMetaId() == metaid) return &mObjectives;
  if (mAssociations.getMetaId() == metaid) return &mAssociations;

  SBase* obj = mBounds.getElementByMetaId(metaid);
  if (obj != NULL) return obj;
  obj = mObjectives.getElementByMetaId(metaid);
  if (obj != NULL) return obj;
  obj = mAssociations.getElementByMetaId(metaid);
  return obj;
}


List*
FbcModelPlugin::getAllElements(ElementFilter *filter)
{
  List* ret = new List();
  List* sublist = NULL;

  ADD_FILTERED_LIST(ret, sublist, mBounds, filter);
  ADD_FILTERED_LIST(ret, sublist, mObjectives, filter);
  ADD_FILTERED_LIST(ret, sublist, mAssociations, filter);
 
  return ret;
}


/** @cond doxygenLibsbmlInternal */
int 
FbcModelPlugin::appendFrom(const Model* model)
{
  int ret = LIBSBML_OPERATION_SUCCESS;

  if (model==NULL)
  {
    return LIBSBML_INVALID_OBJECT;
  }

  const FbcModelPlugin* modplug = 
    static_cast<const FbcModelPlugin*>(model->getPlugin(getPrefix()));
  
  if (modplug==NULL)
  {
    return LIBSBML_INVALID_OBJECT;
  }

  Model* parent = static_cast<Model*>(getParentSBMLObject());

  if (parent==NULL) 
  {
    return LIBSBML_INVALID_OBJECT;
  }
  
  ret = mBounds.appendFrom(modplug->getListOfFluxBounds());

  if (ret != LIBSBML_OPERATION_SUCCESS)
  {
    return ret;
  }

  ret = mObjectives.appendFrom(modplug->getListOfObjectives());
  
  return ret;
}
/** @endcond */





/*
*
*  (EXTENSION) Additional public functions
*
*/  



/*
* Returns the ListOfFluxBounds in this plugin object.
*
* @return ListOfFluxBounds object in this plugin object.
*/
const ListOfFluxBounds* 
  FbcModelPlugin::getListOfFluxBounds () const
{
  return &mBounds;
}


/*
* Returns the ListOfFluxBounds in this plugin object.
*
* @return ListOfFluxBounds object in this plugin object.
*/
ListOfFluxBounds* 
  FbcModelPlugin::getListOfFluxBounds ()
{
  return &mBounds;
}


/*
* Returns the FluxBound object that belongs to the given index. If the
* index is invalid, @c NULL is returned.
*
* @param n the index number of the FluxBound to get.
*
* @return the nth FluxBound in the ListOfFluxBounds.
*/
const FluxBound* 
  FbcModelPlugin::getFluxBound (unsigned int n) const
{
  return static_cast<const FluxBound*>(mBounds.get(n));
}


/*
* Returns the FluxBound object that belongs to the given index. If the
* index is invalid, @c NULL is returned.
*
* @param n the index number of the FluxBound to get.
*
* @return the nth FluxBound in the ListOfFluxBounds.
*/
FluxBound* 
  FbcModelPlugin::getFluxBound (unsigned int n)
{
  return static_cast<FluxBound*>(mBounds.get(n));
}


/*
* Returns the FluxBound object based on its identifier.
*
* @param sid a string representing the identifier 
* of the FluxBound to get.
* 
* @return FluxBound in the ListOfFluxBounds with the given @p id
* or NULL if no such FluxBound exists.
*
* @see get(unsigned int n)
* @see size()
*/
FluxBound* 
  FbcModelPlugin::getFluxBound (const std::string& sid)
{
  return static_cast<FluxBound*>(mBounds.get(sid));
}


/*
* Returns the FluxBound object based on its identifier.
*
* @param sid a string representing the identifier 
* of the FluxBound to get.
* 
* @return FluxBound in the ListOfFluxBounds with the given @p id 
* or NULL if no such FluxBound exists.
*
* @see get(unsigned int n)
* @see size()
*/
const FluxBound* 
  FbcModelPlugin::getFluxBound (const std::string& sid) const
{

  return static_cast<const FluxBound*>(mBounds.get(sid));

}

/*
* Adds a copy of the given FluxBound object to the list of FluxBounds.
*
* @param bound the FluxBound object to be added to the list of FluxBounds.
*
* @return integer value indicating success/failure of the
* function.  @if clike The value is drawn from the
* enumeration #OperationReturnValues_t. @endif The possible values
* returned by this function are:
* @li LIBSBML_OPERATION_SUCCESS
*/ 
int 
  FbcModelPlugin::addFluxBound (const FluxBound* bound)
{
  if (!bound)
  {
    return LIBSBML_OPERATION_FAILED;
  }    
  else if (!bound->hasRequiredElements())
  {
    return LIBSBML_INVALID_OBJECT;
  }
  else if (getLevel() != bound->getLevel())
  {
    return LIBSBML_LEVEL_MISMATCH;
  }
  else if (getVersion() != bound->getVersion())
  {
    return LIBSBML_VERSION_MISMATCH;
  }
  else if (getPackageVersion() != bound->getPackageVersion())
  {
    return LIBSBML_PKG_VERSION_MISMATCH;
  }
  else
  {
    mBounds.append(bound);
  }

  return LIBSBML_OPERATION_SUCCESS;
}


/*
* Creates a new FluxBound object and adds it to the list of FluxBound objects
* and returns it.
*
* @return a newly created FluxBound object
*/
FluxBound* 
  FbcModelPlugin::createFluxBound()
{
  FluxBound* bound = NULL;

  try
  {
    FBC_CREATE_NS(fbcns, getSBMLNamespaces());
    bound = new FluxBound(fbcns);
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

  if (bound) mBounds.appendAndOwn(bound);

  return bound;
}


/*
* Removes the nth FluxBound object from this plugin object and
* returns a pointer to it.
*
* The caller owns the returned object and is responsible for
*  deleting it.
*
* @param n the index of the FluxBound object to remove
*
* @return the FluxBound object removed.  As mentioned above, the 
* caller owns the returned object. @c NULL is returned if the 
* given index is out of range.
*/
FluxBound* 
  FbcModelPlugin::removeFluxBound (unsigned int n)
{
  return static_cast<FluxBound*>(mBounds.remove(n));
}


/*
* Removes the FluxBound object with the given @p id attribute from 
* this plugin object and returns a pointer to it.
*
* The caller owns the returned object and is responsible for
* deleting it.
*
* @param sid the id attribute of the FluxBound object to remove
*
* @return the FluxBound object removed.  As mentioned above, the 
* caller owns the returned object. @c NULL is returned if the 
* given index is out of range.
*/
FluxBound* 
  FbcModelPlugin::removeFluxBound (const std::string& sid)
{
  return static_cast<FluxBound*>(mBounds.remove(sid));
}


/*
* Returns the number of FluxBound object in this plugin object.
*
* @return the number of FluxBound object in this plugin object.
*/
unsigned int 
  FbcModelPlugin::getNumFluxBounds() const
{
  return mBounds.size();
}



/*
* Returns the ListOfObjectives in this plugin object.
*
* @return ListOfObjectives object in this plugin object.
*/
const ListOfObjectives* 
  FbcModelPlugin::getListOfObjectives () const
{
  return &mObjectives;
}

/*
* Returns the ListOfObjectives in this plugin object.
*
* @return ListOfObjectives object in this plugin object.
*/
ListOfObjectives* 
  FbcModelPlugin::getListOfObjectives ()
{
  return &mObjectives;
}



/*
* Returns the Objective object that belongs to the given index. If the
* index is invalid, @c NULL is returned.
*
* @param n the index number of the Objective to get.
*
* @return the nth Objective in the ListOfObjectives.
*/
const Objective* 
  FbcModelPlugin::getObjective (unsigned int n) const
{
  return static_cast<const Objective*>(mObjectives.get(n));
}


/*
* Returns the Objective object that belongs to the given index. If the
* index is invalid, @c NULL is returned.
*
* @param n the index number of the Objective to get.
*
* @return the nth Objective in the ListOfObjectives.
*/
Objective* 
  FbcModelPlugin::getObjective (unsigned int n)
{
  return static_cast<Objective*>(mObjectives.get(n));
}


/*
* Returns the Objective object based on its identifier.
*
* @param sid a string representing the identifier 
* of the Objective to get.
* 
* @return Objective in the ListOfObjectives with the given @p id
* or NULL if no such Objective exists.
*
* @see get(unsigned int n)
* @see size()
*/
Objective* 
  FbcModelPlugin::getObjective (const std::string& sid)
{
  return static_cast<Objective*>(mObjectives.get(sid));
}


/*
* Returns the Objective object based on its identifier.
*
* @param sid a string representing the identifier 
* of the Objective to get.
* 
* @return Objective in the ListOfObjectives with the given @p id 
* or NULL if no such Objective exists.
*
* @see get(unsigned int n)
* @see size()
*/
const Objective* 
  FbcModelPlugin::getObjective (const std::string& sid) const
{
  return static_cast<const Objective*>(mObjectives.get(sid));
}

/*
* Adds a copy of the given Objective object to the list of Objectives.
*
* @param objective the Objective object to be added to the list of Objectives.
*
* @return integer value indicating success/failure of the
* function.  @if clike The value is drawn from the
* enumeration #OperationReturnValues_t. @endif The possible values
* returned by this function are:
* @li LIBSBML_OPERATION_SUCCESS
*/ 
int 
  FbcModelPlugin::addObjective (const Objective* objective)
{
  if (!objective)
  {
    return LIBSBML_OPERATION_FAILED;
  }    
  else if (!objective->hasRequiredElements())
  {
    return LIBSBML_INVALID_OBJECT;
  }
  else if (getLevel() != objective->getLevel())
  {
    return LIBSBML_LEVEL_MISMATCH;
  }
  else if (getVersion() != objective->getVersion())
  {
    return LIBSBML_VERSION_MISMATCH;
  }
  else if (getPackageVersion() != objective->getPackageVersion())
  {
    return LIBSBML_PKG_VERSION_MISMATCH;
  }
  else
  {
    mObjectives.append(objective);
  }

  return LIBSBML_OPERATION_SUCCESS;
}


/*
* Creates a new Objective object and adds it to the list of Objective objects
* and returns it.
*
* @return a newly created Objective object
*/
Objective* 
  FbcModelPlugin::createObjective()
{
  Objective* objective = NULL;

  try
  {      
    FBC_CREATE_NS(fbcns, getSBMLNamespaces());
    objective = new Objective(fbcns);
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

  if (objective) mObjectives.appendAndOwn(objective);

  return objective;
}


/*
* Removes the nth Objective object from this plugin object and
* returns a pointer to it.
*
* The caller owns the returned object and is responsible for
*  deleting it.
*
* @param n the index of the Objective object to remove
*
* @return the Objective object removed.  As mentioned above, the 
* caller owns the returned object. @c NULL is returned if the 
* given index is out of range.
*/
Objective* 
  FbcModelPlugin::removeObjective (unsigned int n)
{
  return static_cast<Objective*>(mObjectives.remove(n));
}


/*
* Removes the Objective object with the given @p id attribute from 
* this plugin object and returns a pointer to it.
*
* The caller owns the returned object and is responsible for
* deleting it.
*
* @param sid the id attribute of the Objective object to remove
*
* @return the Objective object removed.  As mentioned above, the 
* caller owns the returned object. @c NULL is returned if the 
* given index is out of range.
*/
Objective* 
  FbcModelPlugin::removeObjective (const std::string& sid)
{
  return static_cast<Objective*>(mObjectives.remove(sid));
}


/*
* Returns the number of Objective object in this plugin object.
*
* @return the number of Objective object in this plugin object.
*/
unsigned int 
  FbcModelPlugin::getNumObjectives() const
{
  return mObjectives.size();
}


/* 
 * Returns the current active objective. 
 */
Objective* 
FbcModelPlugin::getActiveObjective()
{
  return getObjective(getActiveObjectiveId());
}

/* 
 * Returns the current active objective. 
 */
const Objective *
FbcModelPlugin::getActiveObjective() const
{
  return getObjective(getActiveObjectiveId());
}

/* 
 * Sets the id of the active objective.
 */
int 
FbcModelPlugin::setActiveObjectiveId(const std::string& objectiveId)
{
  return mObjectives.setActiveObjective(objectiveId);
}

/* 
 * returns the id of the current active objective.
 */  
std::string 
FbcModelPlugin::getActiveObjectiveId() const
{
  return mObjectives.getActiveObjective();
}

/* 
 * Unsets the active objective.
 */  
void 
FbcModelPlugin::unsetActiveObjectiveId()
{
  mObjectives.unsetActiveObjective();
}

/*
* Returns the ListOfObjectives in this plugin object.
*
* @return ListOfObjectives object in this plugin object.
*/
const ListOfGeneAssociations* 
  FbcModelPlugin::getListOfGeneAssociations () const
{
  return &mAssociations;
}

/*
* Returns the ListOfGeneAssociations in this plugin object.
*
* @return ListOfGeneAssociations object in this plugin object.
*/
ListOfGeneAssociations* 
  FbcModelPlugin::getListOfGeneAssociations ()
{
  return &mAssociations;
}


/*
* Returns the GeneAssociation object that belongs to the given index. If the
* index is invalid, @c NULL is returned.
*
* @param n the index number of the GeneAssociation to get.
*
* @return the nth GeneAssociation in the ListOfGeneAssociations.
*/
const GeneAssociation* 
  FbcModelPlugin::getGeneAssociation (unsigned int n) const
{
  return static_cast<const GeneAssociation*>(mAssociations.get(n));
}


/*
* Returns the GeneAssociation object that belongs to the given index. If the
* index is invalid, @c NULL is returned.
*
* @param n the index number of the GeneAssociation to get.
*
* @return the nth GeneAssociation in the ListOfGeneAssociations.
*/
GeneAssociation* 
  FbcModelPlugin::getGeneAssociation (unsigned int n)
{
  return static_cast<GeneAssociation*>(mAssociations.get(n));
}


/*
* Returns the GeneAssociation object based on its identifier.
*
* @param sid a string representing the identifier 
* of the GeneAssociation to get.
* 
* @return GeneAssociation in the ListOfGeneAssociations with the given @p id
* or NULL if no such GeneAssociation exists.
*
* @see get(unsigned int n)
* @see size()
*/
GeneAssociation* 
  FbcModelPlugin::getGeneAssociation (const std::string& sid)
{
  return static_cast<GeneAssociation*>(mAssociations.get(sid));
}


/*
* Returns the GeneAssociation object based on its identifier.
*
* @param sid a string representing the identifier 
* of the GeneAssociation to get.
* 
* @return GeneAssociation in the ListOfGeneAssociations with the given @p id 
* or NULL if no such GeneAssociation exists.
*
* @see get(unsigned int n)
* @see size()
*/
const GeneAssociation* 
  FbcModelPlugin::getGeneAssociation (const std::string& sid) const
{
  return static_cast<const GeneAssociation*>(mAssociations.get(sid));
}

/*
* Adds a copy of the given GeneAssociation object to the list of GeneAssociations.
*
* @param association the GeneAssociation object to be added to the list of GeneAssociations.
*
* @return integer value indicating success/failure of the
* function.  @if clike The value is drawn from the
* enumeration #OperationReturnValues_t. @endif The possible values
* returned by this function are:
* @li LIBSBML_OPERATION_SUCCESS
*/ 
int 
  FbcModelPlugin::addGeneAssociation (const GeneAssociation* association)
{
  if (!association)
  {
    return LIBSBML_OPERATION_FAILED;
  }    
  else if (!association->hasRequiredElements())
  {
    return LIBSBML_INVALID_OBJECT;
  }
  else if (getLevel() != association->getLevel())
  {
    return LIBSBML_LEVEL_MISMATCH;
  }
  else if (getVersion() != association->getVersion())
  {
    return LIBSBML_VERSION_MISMATCH;
  }
  else if (getPackageVersion() != association->getPackageVersion())
  {
    return LIBSBML_PKG_VERSION_MISMATCH;
  }
  else
  {
    mAssociations.append(association);
  }

  return LIBSBML_OPERATION_SUCCESS;
}


/*
* Creates a new GeneAssociation object and adds it to the list of GeneAssociation objects
* and returns it.
*
* @return a newly created GeneAssociation object
*/
GeneAssociation* 
  FbcModelPlugin::createGeneAssociation()
{
  GeneAssociation* association = NULL;

  try
  {
    FBC_CREATE_NS(fbcns, getSBMLNamespaces());
    association = new GeneAssociation(fbcns);
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

  if (association) mAssociations.appendAndOwn(association);

  return association;
}


/*
* Removes the nth GeneAssociation object from this plugin object and
* returns a pointer to it.
*
* The caller owns the returned object and is responsible for
*  deleting it.
*
* @param n the index of the GeneAssociation object to remove
*
* @return the GeneAssociation object removed.  As mentioned above, the 
* caller owns the returned object. @c NULL is returned if the 
* given index is out of range.
*/
GeneAssociation* 
  FbcModelPlugin::removeGeneAssociation (unsigned int n)
{
  return static_cast<GeneAssociation*>(mAssociations.remove(n));
}


/*
* Removes the GeneAssociation object with the given @p id attribute from 
* this plugin object and returns a pointer to it.
*
* The caller owns the returned object and is responsible for
* deleting it.
*
* @param sid the id attribute of the GeneAssociation object to remove
*
* @return the GeneAssociation object removed.  As mentioned above, the 
* caller owns the returned object. @c NULL is returned if the 
* given index is out of range.
*/
GeneAssociation* 
  FbcModelPlugin::removeGeneAssociation (const std::string& sid)
{
  return static_cast<GeneAssociation*>(mAssociations.remove(sid));
}


/*
* Returns the number of GeneAssociation object in this plugin object.
*
* @return the number of GeneAssociation object in this plugin object.
*/
int 
  FbcModelPlugin::getNumGeneAssociations() const
{
  return mAssociations.size();
}


/** @cond doxygenLibsbmlInternal */
/*
 * Sets the parent SBMLDocument of this SBML object.
 *
 * @param d the SBMLDocument object to use
 */
void 
  FbcModelPlugin::setSBMLDocument (SBMLDocument* d)
{
  SBasePlugin::setSBMLDocument(d);

  mBounds.setSBMLDocument(d);  
  mAssociations.setSBMLDocument(d);  
  mObjectives.setSBMLDocument(d);  
}
/** @endcond */


/** @cond doxygenLibsbmlInternal */
void
FbcModelPlugin::connectToChild()
{
  connectToParent(getParentSBMLObject());
}
/** @endcond */


/** @cond doxygenLibsbmlInternal */
/*
* Sets the parent SBML object of this plugin object to
* this object and child elements (if any).
* (Creates a child-parent relationship by this plugin object)
*/
void
  FbcModelPlugin::connectToParent (SBase* sbase)
{
  SBasePlugin::connectToParent(sbase);

  mAssociations.connectToParent(sbase);
  mBounds.connectToParent(sbase);
  mObjectives.connectToParent(sbase);
}
/** @endcond */


/** @cond doxygenLibsbmlInternal */
/*
* Enables/Disables the given package with child elements in this plugin
* object (if any).
*/
void
  FbcModelPlugin::enablePackageInternal(const std::string& pkgURI,
  const std::string& pkgPrefix, bool flag)
{
  mAssociations.enablePackageInternal(pkgURI,pkgPrefix,flag);
  mBounds.enablePackageInternal(pkgURI,pkgPrefix,flag);
  mObjectives.enablePackageInternal(pkgURI,pkgPrefix,flag);
}
/** @endcond */


/** @cond doxygenLibsbmlInternal */


bool 
FbcModelPlugin::accept(SBMLVisitor& v) const
{
  const Model * model = static_cast<const Model * >(this->getParentSBMLObject()); 
  
  v.visit(*model);
  v.leave(*model);

  for (unsigned int i = 0; i < getNumFluxBounds(); i++)
  {
    getFluxBound(i)->accept(v);
  }
  for (unsigned int i = 0; i < getNumObjectives(); i++)
  {
    getListOfObjectives()->accept(v);
    getObjective(i)->accept(v);
  }
  return true;
}


/** @endcond */

ListOfFluxBounds * 
FbcModelPlugin::getFluxBoundsForReaction(const std::string& reaction) const
{
  ListOfFluxBounds * loFB = new ListOfFluxBounds(getLevel(), getVersion(),
    getPackageVersion());
                                                

  for (unsigned int i = 0; i < getNumFluxBounds(); i++)
  {
    if (getFluxBound(i)->getReaction() == reaction)
    {
      loFB->append(getFluxBound(i)->clone());
    }
  }

  if (loFB->size() == 0)
  {
    delete loFB;
    loFB = NULL;
  }

  return loFB;
}

#endif /* __cplusplus */
/** @cond doxygenIgnored */


LIBSBML_EXTERN
int
FbcModelPlugin_addFluxBound(SBasePlugin_t * fbc, FluxBound_t * fb)
{
  return (fbc != NULL) ? static_cast<FbcModelPlugin*>(fbc)->addFluxBound(fb)
    : LIBSBML_INVALID_OBJECT;
}


LIBSBML_EXTERN
FluxBound_t *
FbcModelPlugin_getFluxBound(SBasePlugin_t * fbc, unsigned int n)
{
  return (fbc != NULL) ? static_cast<FbcModelPlugin*>(fbc)->getFluxBound(n)
    : NULL;
}


LIBSBML_EXTERN
unsigned int
FbcModelPlugin_getNumFluxBounds(SBasePlugin_t * fbc)
{
  return (fbc != NULL) ? static_cast<FbcModelPlugin*>(fbc)->getNumFluxBounds()
    : SBML_INT_MAX;
}


LIBSBML_EXTERN
int
FbcModelPlugin_addObjective(SBasePlugin_t * fbc, Objective_t * obj)
{
  return (fbc != NULL) ? static_cast<FbcModelPlugin*>(fbc)->addObjective(obj)
    : LIBSBML_INVALID_OBJECT;
}


LIBSBML_EXTERN
Objective_t *
FbcModelPlugin_getObjective(SBasePlugin_t * fbc, unsigned int n)
{
  return (fbc != NULL) ? static_cast<FbcModelPlugin*>(fbc)->getObjective(n)
    : NULL;
}


LIBSBML_EXTERN
unsigned int
FbcModelPlugin_getNumObjectives(SBasePlugin_t * fbc)
{
  return (fbc != NULL) ? static_cast<FbcModelPlugin*>(fbc)->getNumObjectives()
    : SBML_INT_MAX;
}


LIBSBML_EXTERN
char *
FbcModelPlugin_getActiveObjectiveId(SBasePlugin_t * fbc)
{
  if (fbc == NULL)
    return NULL;

  return static_cast<FbcModelPlugin *>(fbc)->getActiveObjectiveId().empty() 
    ? safe_strdup("") 
    : safe_strdup(static_cast<FbcModelPlugin *>(fbc)->getActiveObjectiveId().c_str());
}


LIBSBML_EXTERN
int
FbcModelPlugin_setActiveObjectiveId(SBasePlugin_t * fbc, const char * activeId)
{
  return (fbc != NULL) 
    ? static_cast<FbcModelPlugin *>(fbc)->setActiveObjectiveId(activeId) 
    : LIBSBML_INVALID_OBJECT;
}


/** @endcond */
LIBSBML_CPP_NAMESPACE_END

/**
 * @file    Submodel.h
 * @brief   Definition of Submodel, the SBase derived class of the comp package.
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
 * ------------------------------------------------------------------------ -->
 *
 * @class Submodel
 * @sbmlbrief{comp} A model instance inside another model.
 *
 * The Submodel class was introduced by the SBML Level&nbsp;3 @ref comp
 * @if java "Hierarchical %Model Composition"@endif@~ package
 * (&ldquo;comp&rdquo;) as the principle way by which models are structured
 * hierarchically.  Submodels are instantiations of models contained within
 * other models.  They reference another Model that is to be instantiated
 * within its parent Model, and additionally define how that Model is to be
 * modified before instantiation.
 *
 * The Submodel object class has a required attribute "modelRef", which must
 * reference another Model or ExternalModelDefinition object present in the
 * SBML Document.  This referenced Model is the model to be instantiated.
 * 
 * It also has a required attribute, "id", to give the submodel a unique
 * identifier by which other parts of an SBML model definition can refer to
 * it, and an optional "name" attribute of type @c string.  Identifiers and
 * names must be used according to the guidelines described in the SBML
 * specification.
 *
 * The Submodel class also provides constructs that define how the referenced
 * Model object is to be modified before it is instantiated in the enclosing
 * model.  If numerical values in the referenced model must be changed in order 
 * to fit them into their new context as part of the submodel, the changes can 
 * be handled through conversion factors.  If one or more structural features 
 * in the referenced model are undesirable and should be removed, the changes 
 * can be handled through deletions.  (For example, an initial assignment or 
 * reaction may not be relevant in its new context and should be removed.)
 *
 * In some cases, the referenced Model may have been written with different
 * units than the containing model.  For most model elements, this is not a
 * problem: it is already possible to have Species and Parameter objects with
 * different units in a single model, for example, so in this case the
 * resulting hierarchical model would be treated in exactly the same way as
 * any other model with Species and Parameters with different units.
 *
 * However, two units in SBML models are fixed and must not vary between SBML
 * elements: time and extent.  The units of time are set once per model, and
 * affect the core elements of RateRule, KineticLaw, Delay, and the
 * csymbols 'time' and 'delay'.  Even if the model does not explicitly state
 * what the units of time actually are, they are defined to be consistent
 * across the model, and therefore might differ from the units of time across
 * a parent model.  To correct this imbalance, the optional attribute
 * "timeConversionFactor" may be used, which, if defined, must reference a
 * constant parameter in the parent model. The value of the time conversion
 * factor should be defined such that a single unit of time in the Submodel
 * multiplied by the time conversion factor should equal a single unit of
 * time in the parent model.
 *
 * Extent is the unit in SBML that defines how the KineticLaw of a Reaction
 * affects species quantities: kinetic laws are defined to be in units of
 * extent/time.  No other SBML core construct is defined in terms of extent.
 * If the effective units of extent in a submodel differ from the effective
 * units of extent in the parent model (regardless of whether either defined
 * what those units actually are), the optional attribute
 * "extentConversionFactor" may be used, which, if defined, must reference a
 * constant parameter in the parent model. The value of the extent conversion
 * factor should be defined such that a single unit of extent in the Submodel
 * multiplied by the extent conversion factor should equal a single unit of
 * extent in the parent model.
 *
 * If features of the referenced model must be removed, a Deletion should be added 
 * to the Submodel object.  A Submodel may contain a child ListOfDeletions, which
 * in turn may contain one or more Deletion items.  Each Deletion references a single
 * element of the referenced Model that must be removed before instantiating that
 * Model as a submodel of the parent Model.
 */

#ifndef Submodel_H__
#define Submodel_H__


#include <sbml/common/extern.h>
#include <sbml/common/sbmlfwd.h>
#include <sbml/packages/comp/common/compfwd.h>

#ifdef __cplusplus

#include <string>

#include <sbml/SBase.h>
#include <sbml/packages/comp/extension/CompExtension.h>
#include <sbml/packages/comp/sbml/CompBase.h>
#include <sbml/packages/comp/sbml/ListOfDeletions.h>

LIBSBML_CPP_NAMESPACE_BEGIN

class ReplacedElement;

class LIBSBML_EXTERN Submodel : public CompBase
{
protected:

  /** @cond doxygenLibsbmlInternal */
  std::string   mId;
  std::string   mName;
  std::string   mModelRef;
  std::string   mTimeConversionFactor;
  std::string   mExtentConversionFactor;
  ListOfDeletions  mListOfDeletions;
  Model*        mInstantiatedModel;
  std::string   mInstantiationOriginalURI;
  /** @endcond */

public:

  /**
   * Creates a new Submodel with the given level, version, and package
   * version.
   *
   * @param level the SBML Level
   * @param version the Version within the SBML Level
   * @param pkgVersion the version of the package
   */
  Submodel(unsigned int level      = CompExtension::getDefaultLevel(),
           unsigned int version    = CompExtension::getDefaultVersion(),
           unsigned int pkgVersion = CompExtension::getDefaultPackageVersion());


  /**
   * Creates a new Submodel with the given CompPkgNamespaces object.
   *
   * @param compns the namespace to use
   */
  Submodel(CompPkgNamespaces* compns);


  /**
   * Copy constructor.
   */
  Submodel(const Submodel& source);


  /**
   * Assignment operator.
   */
  Submodel& operator=(const Submodel& source);


  /**
   * Creates and returns a deep copy of this Submodel object.
   * 
   * @return a (deep) copy of this Submodel object
   */
  virtual Submodel* clone () const;


  /**
   * Destructor.
   */ 
  virtual ~Submodel ();


  /**
   * Returns the first child element found that has the given @p id in the
   * model-wide SId namespace, or @c NULL if no such object is found.
   *
   * @param id string representing the id of objects to find
   *
   * @return a pointer to the SBase element with the given @p id.
   */
  virtual SBase* getElementBySId(const std::string& id);
  
  
  /**
   * Returns the first child element it can find with the given @p metaid, or
   * itself if it has the given @p metaid, or @c NULL if no such object is found.
   *
   * @param metaid string representing the metaid of objects to find
   *
   * @return a pointer to the SBase element with the given @p metaid.
   */
  virtual SBase* getElementByMetaId(const std::string& metaid);
  
  
  /**
   * Returns a List of all child SBase objects, including those nested to an
   * arbitrary depth.
   *
   * @return a List of pointers to all children objects.
   */
  virtual List* getAllElements(ElementFilter* filter=NULL);
  
  
  /**
   * Returns the value of the "id" attribute of this Submodel.
   *
   * @return the value of the "id" attribute of this Submodel.
   */
  virtual const std::string& getId() const;


  /**
   * Predicate returning @c true or @c false depending on whether this
   * Submodel's "id" attribute has been set.
   *
   * @return @c true if this Submodel's "id" attribute has been set, 
   * otherwise @c false is returned.
   */
  virtual bool isSetId() const;

  
  /**
   * Sets the value of the "id" attribute of this Submodel.  Fails if the id
   * is not a valid syntax for an SId.
   *
   * @return integer value indicating success/failure of the
   * operation. The possible return values are:
   * @li @sbmlconstant{LIBSBML_OPERATION_SUCCESS, OperationReturnValues_t}
   * @li @sbmlconstant{LIBSBML_INVALID_ATTRIBUTE_VALUE, OperationReturnValues_t}
   */
  virtual int setId(const std::string& id);


  /**
   * Unsets the value of the "id" attribute of this Submodel.
   *
   * @return integer value indicating success/failure of the
   * operation. The possible return values are:
   * @li @sbmlconstant{LIBSBML_OPERATION_SUCCESS, OperationReturnValues_t}
   * @li @sbmlconstant{LIBSBML_OPERATION_FAILED, OperationReturnValues_t}
   */
  virtual int unsetId();


  /**
   * Returns the value of the "name" attribute of this Submodel.
   *
   * @return the value of the "name" attribute of this Submodel.
   */
  virtual const std::string& getName() const;


  /**
   * Predicate returning @c true or @c false depending on whether this
   * Submodel's "name" attribute has been set.
   *
   * @return @c true if this Submodel's "name" attribute has been set, 
   * otherwise @c false is returned.
   */
  virtual bool isSetName() const;

  
  /**
   * Sets the value of the "name" attribute of this Submodel.  Fails if the
   * name is empty.
   *
   * @return integer value indicating success/failure of the
   * operation. The possible return values are:
   * @li @sbmlconstant{LIBSBML_OPERATION_SUCCESS, OperationReturnValues_t}
   * @li @sbmlconstant{LIBSBML_INVALID_ATTRIBUTE_VALUE, OperationReturnValues_t}
   */
  virtual int setName(const std::string& name);


  /**
   * Unsets the value of the "name" attribute of this Submodel.
   *
   * @return integer value indicating success/failure of the
   * operation. The possible return values are:
   * @li @sbmlconstant{LIBSBML_OPERATION_SUCCESS, OperationReturnValues_t}
   * @li @sbmlconstant{LIBSBML_OPERATION_FAILED, OperationReturnValues_t}
   */
  virtual int unsetName();


  /**
   * Returns the value of the "modelRef" attribute of this Submodel.
   *
   * @return the value of the "modelRef" attribute of this Submodel.
   */
  virtual const std::string& getModelRef() const;


  /**
   * Predicate returning @c true or @c false depending on whether this
   * Submodel's "modelRef" attribute has been set.
   *
   * @return @c true if this Submodel's "modelRef" attribute has been set, 
   * otherwise @c false is returned.
   */
  virtual bool isSetModelRef() const;

  
  /**
   * Sets the value of the "modelRef" attribute of this Submodel.  Fails if
   * the modelRef is not a valid syntax for an SIdRef.
   *
   * @return integer value indicating success/failure of the
   * operation. The possible return values are:
   * @li @sbmlconstant{LIBSBML_OPERATION_SUCCESS, OperationReturnValues_t}
   * @li @sbmlconstant{LIBSBML_INVALID_ATTRIBUTE_VALUE, OperationReturnValues_t}
   */
  virtual int setModelRef(const std::string& modelRef);


  /**
   * Unsets the value of the "modelRef" attribute of this Submodel.
   *
   * @return integer value indicating success/failure of the
   * operation. The possible return values are:
   * @li @sbmlconstant{LIBSBML_OPERATION_SUCCESS, OperationReturnValues_t}
   * @li @sbmlconstant{LIBSBML_OPERATION_FAILED, OperationReturnValues_t}
   */
  virtual int unsetModelRef();


  /**
   * Returns an empty string, since "substanceConversionFactor" is not a part of the comp spec.
   *
   * @return an empty string
   */
  virtual const std::string& getSubstanceConversionFactor () const;


  /**
   * Returns @c false, since "substanceConversionFactor" is not a part of the comp spec.
   *
   * @return @c false.
   */
  virtual bool isSetSubstanceConversionFactor () const;

  
  /**
   * Automatically fails, since "substanceConversionFactor" is not a part of the comp spec.
   *
   * @return integer value indicating success/failure of the
   * operation. The possible return value is:
   * @li @sbmlconstant{LIBSBML_INVALID_ATTRIBUTE_VALUE, OperationReturnValues_t}
   */
  virtual int setSubstanceConversionFactor (const std::string& id);


  /**
   * Automatically fails, since "substanceConversionFactor" is not a part of the comp spec.
   *
   * @return integer value indicating success/failure of the
   * operation. The possible return value is:
   * @li @sbmlconstant{LIBSBML_OPERATION_FAILED, OperationReturnValues_t}
   */
  virtual int unsetSubstanceConversionFactor ();


  /**
   * Returns the value of the "timeConversionFactor" attribute of this Submodel.
   *
   * @return the value of the "timeConversionFactor" attribute of this Submodel.
   */
  virtual const std::string& getTimeConversionFactor () const;


  /**
   * Predicate returning @c true or @c false depending on whether this
   * Submodel's "timeConversionFactor" attribute has been set.
   *
   * @return @c true if this Submodel's "timeConversionFactor" attribute has been set, 
   * otherwise @c false is returned.
   */
  virtual bool isSetTimeConversionFactor () const;

  
  /**
   * Sets the value of the "timeConversionFactor" attribute of this Submodel.
   * Fails if the id is not a valid syntax for an SIdRef.
   *
   * @return integer value indicating success/failure of the
   * operation. The possible return values are:
   * @li @sbmlconstant{LIBSBML_OPERATION_SUCCESS, OperationReturnValues_t}
   * @li @sbmlconstant{LIBSBML_INVALID_ATTRIBUTE_VALUE, OperationReturnValues_t}
   */
  virtual int setTimeConversionFactor (const std::string& id);


  /**
   * Unsets the value of the "timeConversionFactor" attribute of this Submodel.
   *
   * @return integer value indicating success/failure of the
   * operation. The possible return values are:
   * @li @sbmlconstant{LIBSBML_OPERATION_SUCCESS, OperationReturnValues_t}
   * @li @sbmlconstant{LIBSBML_OPERATION_FAILED, OperationReturnValues_t}
   */
  virtual int unsetTimeConversionFactor ();


  /**
   * Returns the value of the "extentConversionFactor" attribute of this Submodel.
   *
   * @return the value of the "extentConversionFactor" attribute of this Submodel.
   */
  virtual const std::string& getExtentConversionFactor () const;


  /**
   * Predicate returning @c true or @c false depending on whether this
   * Submodel's "extentConversionFactor" attribute has been set.
   *
   * @return @c true if this Submodel's "extentConversionFactor" attribute has been set, 
   * otherwise @c false is returned.
   */
  virtual bool isSetExtentConversionFactor () const;

  
  /**
   * Sets the value of the "extentConversionFactor" attribute of this
   * Submodel.  Fails if the id is not a valid syntax for an SIdRef.
   *
   * @return integer value indicating success/failure of the
   * operation. The possible return values are:
   * @li @sbmlconstant{LIBSBML_OPERATION_SUCCESS, OperationReturnValues_t}
   * @li @sbmlconstant{LIBSBML_INVALID_ATTRIBUTE_VALUE, OperationReturnValues_t}
   */
  virtual int setExtentConversionFactor (const std::string& id);


  /**
   * Unsets the value of the "extentConversionFactor" attribute of this
   * Submodel.
   *
   * @return integer value indicating success/failure of the
   * operation. The possible return values are:
   * @li @sbmlconstant{LIBSBML_OPERATION_SUCCESS, OperationReturnValues_t}
   * @li @sbmlconstant{LIBSBML_OPERATION_FAILED, OperationReturnValues_t}
   */
  virtual int unsetExtentConversionFactor ();


  /**
   * Returns the ListOf object that holds all deletions.
   *
   * @return the ListOf object that holds all deletions.
   */ 
  const ListOfDeletions* getListOfDeletions () const;


  /**
   * Returns the ListOf object that holds all deletions.
   *
   * @return the ListOf object that holds all deletions.
   */ 
  ListOfDeletions* getListOfDeletions ();


  /**
   * Returns the deletion with the given index.
   * If the index is invalid, @c NULL is returned.
   *
   * @param n the index number of the Deletion to get.
   *
   * @return the nth Deletion in the ListOfDeletions.
   */ 
  Deletion* getDeletion (unsigned int n);


  /**
   * Returns the deletion with the given index.
   * If the index is invalid, @c NULL is returned.
   *
   * @param n the index number of the Deletion to get.
   *
   * @return the nth Deletion in the ListOfDeletions.
   */ 
  const Deletion* getDeletion (unsigned int n) const;


  /**
   * Returns the deletion with the given @p id.
   * If the id is invalid, @c NULL is returned.
   *
   * @param id the id of the Deletion to get.
   *
   * @return the Deletion in the ListOfDeletions with the given @p id.
   */ 
  Deletion* getDeletion (std::string id);


  /**
   * Returns the deletion with the given @p id.
   * If the id is invalid, @c NULL is returned.
   *
   * @param id the id of the Deletion to get.
   *
   * @return the Deletion in the ListOfDeletions with the given @p id.
   */ 
  const Deletion* getDeletion (std::string id) const;


  /**
   * Adds a copy of the given Deletion object to the list of deletions.
   *
   * @param deletion the Deletion object to be added to the list of
   * deletions.  Fails if the added deletion is NULL, does not match the
   * level/version/package of the parent object, or cannot be added to the
   * list of deletions.
   *
   * @return integer value indicating success/failure of the
   * operation. The possible return values are:
   * @li @sbmlconstant{LIBSBML_OPERATION_SUCCESS, OperationReturnValues_t}
   * @li @sbmlconstant{LIBSBML_INVALID_OBJECT, OperationReturnValues_t}
   * @li @sbmlconstant{LIBSBML_LEVEL_MISMATCH, OperationReturnValues_t}
   * @li @sbmlconstant{LIBSBML_VERSION_MISMATCH, OperationReturnValues_t}
   * @li @sbmlconstant{LIBSBML_PKG_VERSION_MISMATCH, OperationReturnValues_t}
   */
  int addDeletion (const Deletion* deletion);


  /**
   * Returns the number of deletions for this Submodel.
   *
   * @return the number of deletions for this Submodel.
   */
  unsigned int getNumDeletions () const;


  /**
   * Creates a Deletion object, adds it to the end of the
   * deletion objects list and returns a pointer to the newly
   * created object.
   *
   * @return a newly created Deletion object
   */
  Deletion* createDeletion ();


  /**
   * Removes the deletion with the given index from the Submodel.
   * A pointer to the deletion that was removed is returned.
   * If no deletion has been removed, @c NULL is returned.
   *
   * @param index the index of the Deletion object to remove
   *
   * @return the Deletion object removed.  As mentioned above, 
   * the caller owns the returned object. @c NULL is returned if 
   * the given index is out of range.
   */
  Deletion* removeDeletion(unsigned int index);


  /**
   * Removes the deletion with the given identifier from the Submodel.
   * A pointer to the deletion that was removed is returned.
   * If no deletion has been removed, @c NULL is returned.
   *
   * @param sid string representing the identifier
   * of the Deletion object to remove
   *
   * @return the Deletion object removed.  As mentioned above, 
   * the caller owns the returned object. @c NULL is returned if 
   * the given @p sid is not found.
   */
  Deletion* removeDeletion(const std::string& sid);


  /**
   * Returns true if the 'submodel' attribute is set, and if getNumReferents() is exactly 1.
   *
   * @return boolean: 'true' if the attributes are correctly set; 'false' if not.
   */
  virtual bool hasRequiredAttributes() const;


  /**
   * Returns the XML element name of
   * this SBML object.
   *
   * @return the name of this element, as a text string.
   */
  virtual const std::string& getElementName () const ;


  /**
   * Renames the conversion factor attributes on this element if @p oldid matches.
   */
  virtual void renameSIdRefs(const std::string& oldid, const std::string& newid);


  /**
   * Returns the libSBML type code of this object instance.
   *
   * @copydetails doc_what_are_typecodes
   *
   * @return the SBML type code for this object:
   * @sbmlconstant{SBML_COMP_SUBMODEL, SBMLCompTypeCode_t}
   *
   * @copydetails doc_warning_typecodes_not_unique
   *
   * @see getElementName()
   * @see getPackageName()
   */
  virtual int getTypeCode () const;


  /** @cond doxygenLibsbmlInternal */
  /**
   * Subclasses should override this method to write out their contained
   * SBML objects as XML elements.  Be sure to call your parents
   * implementation of this method as well.  For example:
   *
   *   SBase::writeElements(stream);
   *   mReactants.write(stream);
   *   mProducts.write(stream);
   *   ...
   */
  virtual void writeElements (XMLOutputStream& stream) const;
  /** @endcond */


  /** @cond doxygenLibsbmlInternal */


  /**
   * Accepts the given SBMLVisitor.
   *
   * @return the result of calling <code>v.visit()</code>, which indicates
   * whether or not the Visitor would like to visit the SBML object's next
   * sibling object (if available).
   */
  virtual bool accept (SBMLVisitor& v) const;
  
  /** @endcond */


  /** @cond doxygenLibsbmlInternal */
  /**
   * Sets the parent SBMLDocument of this SBML object.
   *
   * @param d the SBMLDocument object to use
   */
  virtual void setSBMLDocument (SBMLDocument* d);
  /** @endcond */


  /** @cond doxygenLibsbmlInternal */
  /**
   * Sets this SBML object to child SBML objects (if any).
   * (Creates a child-parent relationship by the parent)
   *
   * Subclasses must override this function if they define
   * one ore more child elements.
   * Basically, this function needs to be called in
   * constructor, copy constructor, assignment operator.
   *
   * @see setSBMLDocument
   * @see enablePackageInternal
   */
  virtual void connectToChild ();
  /** @endcond */


  /** @cond doxygenLibsbmlInternal */
  /**
   * Create and return an SBML object of this class, if present.
   *
   * @return the SBML object corresponding to next XMLToken in the
   * XMLInputStream or NULL if the token was not recognized.
   */
  virtual SBase* createObject (XMLInputStream& stream);
  /** @endcond */


  /**
   * Find and create a local copy of the Model object referenced by this
   * Submodel.  Is recursive, in that if the instantiated Model contains any
   * Submodel objects, those Submodels will themselves be instantiated.  If
   * an instantiated model previously existed, it is deleted and a new one is
   * created.  For this reason, call this function only once, or 
   * call Submodel::getInstantiation().
   *
   * @return an integer value indicating success/failure of the operation.
   * Possible return values from this function are:
   * @li @sbmlconstant{LIBSBML_OPERATION_SUCCESS, OperationReturnValues_t}
   * @li @sbmlconstant{LIBSBML_INVALID_OBJECT, OperationReturnValues_t}
   * @li @sbmlconstant{LIBSBML_OPERATION_FAILED, OperationReturnValues_t}
   * In this case, 'invalid object' means that this Submodel itself is invalid, and no Model can be instantiated from it.
   */
  virtual int instantiate();


  /**
   * Delete elements in the instantiated submodel, based on any Deletions
   * from this Submodel's listOfDeletions.
   *
   * @return an integer value indicating success/failure of the operation.
   * Possible return values from this function are:
   * @li @sbmlconstant{LIBSBML_OPERATION_SUCCESS, OperationReturnValues_t}
   * @li @sbmlconstant{LIBSBML_INVALID_OBJECT, OperationReturnValues_t}
   * @li @sbmlconstant{LIBSBML_OPERATION_FAILED, OperationReturnValues_t}
   * In this case, 'invalid object' means that this Submodel itself is invalid, and no Model can be instantiated from it.
   */
  virtual int performDeletions();


  /**
   * Delete the element in question from the stored instantiated Model, and
   * replace all references to it with references to the replacement object.
   * @sbmlconstant{LIBSBML_INVALID_OBJECT, OperationReturnValues_t}
   * means that this Submodel itself or one of the passed-in objects are invalid.
   * @sbmlconstant{LIBSBML_OPERATION_FAILED, OperationReturnValues_t}
   * means that the routine failed for some othe reason.
   *
   * @return an integer value indicating success/failure of the operation.
   * Possible return values from this function are:
   * @li @sbmlconstant{LIBSBML_OPERATION_SUCCESS, OperationReturnValues_t}
   * @li @sbmlconstant{LIBSBML_INVALID_OBJECT, OperationReturnValues_t}
   * @li @sbmlconstant{LIBSBML_OPERATION_FAILED, OperationReturnValues_t}
   */
  virtual int replaceElement(SBase* toReplace, SBase* replacement);


  /**
   * Get the instantiated Model this Submodel contains rules to create.
   * Calls instantiate() automatically if this operation has not yet been
   * performed, and/or if the operation failed the last time it was called.
   * Any modifictions that have been performed with performDeletions(), 
   * replaceElement(), or convertTimeAndExtent() function calls will be included.
   *
   * @return the instantiated Model object: a clone of the original, modified
   * according to the performDeletions() and replaceElement() functions that
   * have been called.  Returns NULL if any error is encountered.
   */
  virtual Model* getInstantiation();


  /**
   * Get the instantiated Model this Submodel contains rules to create.
   * Calls instantiate() automatically if this operation has not yet been
   * performed, and/or if the operation failed the last time it was called.
   * Any modifictions that have been performed with performDeletions(), 
   * replaceElement(), or convertTimeAndExtent() function calls will be included.
   *
   * @return the instantiated Model object: a clone of the original, modified
   * according to the performDeletions() and replaceElement() functions that
   * have been called.  Returns NULL if any error is encountered.
   */
  virtual const Model* getInstantiation() const;


  /**
   * Delete the instantiated Model, if it exists.
   */
  virtual void clearInstantiation();

  
  /**
   * Get all instantiated sub-elements, including any elements from
   * instantiated submodels, etc.
   */
  virtual List* getAllInstantiatedElements();


  /**
   * Convert all references to time and extent in the instantiated
   * Model, according to the
   * timeConversionFactor and extentConversionFactor attributes.
   */
  virtual int convertTimeAndExtent();

protected:

  /** @cond doxygenLibsbmlInternal */
  /**
   * Subclasses should override this method to get the list of
   * expected attributes.
   * This function is invoked from corresponding readAttributes()
   * function.
   */
  virtual void addExpectedAttributes(ExpectedAttributes& attributes);
  /** @endcond */


  /** @cond doxygenLibsbmlInternal */
  /**
   * Subclasses should override this method to read values from the given
   * XMLAttributes set into their specific fields.  Be sure to call your
   * parents implementation of this method as well.
   */
  virtual void readAttributes (const XMLAttributes& attributes, 
                               const ExpectedAttributes& expectedAttributes);
  /** @endcond */


  /** @cond doxygenLibsbmlInternal */
  /**
   * Subclasses should override this method to write their XML attributes
   * to the XMLOutputStream.  Be sure to call your parents implementation
   * of this method as well.  For example:
   *
   *   SBase::writeAttributes(stream);
   *   stream.writeAttribute( "submodel", mId );
   *   ...
   */
  virtual void writeAttributes (XMLOutputStream& stream) const;
  /** @endcond */

private:
  /**
   * Internal function to convert time and extent with the given ASTNodes.
   */
  virtual int convertTimeAndExtentWith(const ASTNode* time, const ASTNode* xcf, const ASTNode* klmod);

  /**
   * Internal function that changes 'math' according to the passed-in time conversion factors (pre-set-up for convenience)
   */
  virtual void convertCSymbols(ASTNode*& math, const ASTNode* tcfdiv, const ASTNode* tcftimes);

  /**
   * Internal function that creates a new conversion factor in the given Model* based on newcf and oldcf, and sets 'cf' to be the name of that new conversion factor.
   */
  virtual void createNewConversionFactor(std::string& cf, const ASTNode* newcf, std::string oldcf, Model* model);
};


LIBSBML_CPP_NAMESPACE_END

#endif /* __cplusplus */


#ifndef SWIG

LIBSBML_CPP_NAMESPACE_BEGIN
BEGIN_C_DECLS

/**
 * Creates a new Submodel_t structure using the given SBML @p level
 * and @p version values.
 *
 * @param level an unsigned int, the SBML Level to assign to this
 * Submodel_t
 * @param version an unsigned int, the SBML Version to assign to this
 * Submodel_t
 * @param pkgVersion an unsigned int, the SBML 'Qual' package Version to assign to this
 * Submodel_t
 *
 * @return a pointer to the newly created Submodel_t structure.
 *
 * @memberof Submodel_t
 */
LIBSBML_EXTERN
Submodel_t *
Submodel_create(unsigned int level, unsigned int version,
                unsigned int pkgVersion);


/**
 * Frees the given Submodel_t structure.
 *
 * @param s the Submodel_t structure to free.
 *
 * @memberof Submodel_t
 */
LIBSBML_EXTERN
void
Submodel_free(Submodel_t * s);


/**
 * Returns a copy of the given Submodel_t structure.
 *
 * @param s the Submodel_t structure to copy.
 * 
 * @return a (deep) copy of the Submodel_t.
 *
 * @memberof Submodel_t
 */
LIBSBML_EXTERN
Submodel_t *
Submodel_clone(Submodel_t * s);


/**
 * Takes an Submodel_t structure and returns its identifier.
 *
 * @param s the Submodel_t structure whose identifier is sought
 * 
 * @return the identifier of the given Submodel_t, as a pointer to a string.
 *
 * @memberof Submodel_t
 */
LIBSBML_EXTERN
char *
Submodel_getId(Submodel_t * s);


/**
 * Takes a Submodel_t structure and returns its name.
 *
 * @param s the Submodel_t whose name is sought.
 *
 * @return the name of the given Submodel_t, as a pointer to a string.
 *
 * @memberof Submodel_t
 */
LIBSBML_EXTERN
char *
Submodel_getName(Submodel_t * s);


/**
 * Takes a Submodel_t structure and returns its modelRef.
 *
 * @param s the Submodel_t whose modelRef is sought.
 *
 * @return the modelRef of the given Submodel_t, as a pointer to a string.
 *
 * @memberof Submodel_t
 */
LIBSBML_EXTERN
char *
Submodel_getModelRef(Submodel_t * s);


/**
 * Returns NULL, since "substanceConversionFactor" is not a part of the comp spec.
 *
 * @param s the Submodel_t whose substanceConversionFactor is sought.
 *
 * @return NULL
 *
 * @memberof Submodel_t
 */
LIBSBML_EXTERN
char *
Submodel_getSubstanceConversionFactor(Submodel_t * s);


/**
 * Takes a Submodel_t structure and returns its timeConversionFactor.
 *
 * @param s the Submodel_t whose timeConversionFactor is sought.
 *
 * @return the timeConversionFactor of the given Submodel_t, as a pointer to a string.
 *
 * @memberof Submodel_t
 */
LIBSBML_EXTERN
char *
Submodel_getTimeConversionFactor(Submodel_t * s);


/**
 * Takes a Submodel_t structure and returns its extentConversionFactor.
 *
 * @param s the Submodel_t whose extentConversionFactor is sought.
 *
 * @return the extentConversionFactor of the given Submodel_t, as a pointer to a string.
 *
 * @memberof Submodel_t
 */
LIBSBML_EXTERN
char *
Submodel_getExtentConversionFactor(Submodel_t * s);


/**
 * Predicate returning @c true or @c false depending on whether the given
 * Submodel_t structure's identifier is set.
 *
 * @param s the Submodel_t structure to query
 * 
 * @return @c non-zero (true) if the "id" attribute of the given
 * Submodel_t structure is set, zero (false) otherwise.
 *
 * @memberof Submodel_t
 */
LIBSBML_EXTERN
int
Submodel_isSetId(Submodel_t * s);


/**
 * Predicate returning @c true or @c false depending on whether the given
 * Submodel_t structure's name is set.
 *
 * @param s the Submodel_t structure to query
 * 
 * @return @c non-zero (true) if the "name" attribute of the given
 * Submodel_t structure is set, zero (false) otherwise.
 *
 * @memberof Submodel_t
 */
LIBSBML_EXTERN
int
Submodel_isSetName(Submodel_t * s);


/**
 * Predicate returning @c true or @c false depending on whether the given
 * Submodel_t structure's modelRef is set.
 *
 * @param s the Submodel_t structure to query
 * 
 * @return @c non-zero (true) if the "modelRef" attribute of the given
 * Submodel_t structure is set, zero (false) otherwise.
 *
 * @memberof Submodel_t
 */
LIBSBML_EXTERN
int
Submodel_isSetModelRef(Submodel_t * s);


/**
 * Returns @c false, since "substanceConversionFactor" is not a part of the comp spec.
 *
 * @param s the Submodel_t structure to query
 * 
 * @return @c 0 (false)
 *
 * @memberof Submodel_t
 */
LIBSBML_EXTERN
int
Submodel_isSetSubstanceConversionFactor(Submodel_t * s);


/**
 * Predicate returning @c true or @c false depending on whether the given
 * Submodel_t structure's timeConversionFactor is set.
 *
 * @param s the Submodel_t structure to query
 * 
 * @return @c non-zero (true) if the "timeConversionFactor" attribute of the given
 * Submodel_t structure is set, zero (false) otherwise.
 *
 * @memberof Submodel_t
 */
LIBSBML_EXTERN
int
Submodel_isSetTimeConversionFactor(Submodel_t * s);


/**
 * Predicate returning @c true or @c false depending on whether the given
 * Submodel_t structure's extentConversionFactor is set.
 *
 * @param s the Submodel_t structure to query
 * 
 * @return @c non-zero (true) if the "extentConversionFactor" attribute of the given
 * Submodel_t structure is set, zero (false) otherwise.
 *
 * @memberof Submodel_t
 */
LIBSBML_EXTERN
int
Submodel_isSetExtentConversionFactor(Submodel_t * s);


/**
 * Assigns the identifier of an Submodel_t structure.
 *
 * This makes a copy of the string passed in the param @p sid.
 *
 * @param s the Submodel_t structure to set.
 * @param sid the string to use as the identifier.
 *
 * @return integer value indicating success/failure of the
 * function.  @if clike The value is drawn from the
 * enumeration #OperationReturnValues_t. @endif@~ The possible values
 * returned by this function are:
 * @li @sbmlconstant{LIBSBML_OPERATION_SUCCESS, OperationReturnValues_t}
 * @li @sbmlconstant{LIBSBML_INVALID_ATTRIBUTE_VALUE, OperationReturnValues_t}
 *
 * @note Using this function with an id of NULL is equivalent to
 * unsetting the "id" attribute.
 *
 * @memberof Submodel_t
 */
LIBSBML_EXTERN
int
Submodel_setId(Submodel_t * s, const char * sid);


/**
 * Sets the name of the given Submodel_t to a copy of @p name.
 *
 * @param s the Submodel_t structure to set
 * @param name the name to assign to the given Submodel_t's "name" attribute.
 *
 * @return integer value indicating success/failure of the
 * function.  @if clike The value is drawn from the
 * enumeration #OperationReturnValues_t. @endif@~ The possible values
 * returned by this function are:
 * @li @sbmlconstant{LIBSBML_OPERATION_SUCCESS, OperationReturnValues_t}
 * @li @sbmlconstant{LIBSBML_INVALID_ATTRIBUTE_VALUE, OperationReturnValues_t}
 *
 * @note Using this function with the name set to NULL is equivalent to
 * unsetting the "name" attribute.
 *
 * @memberof Submodel_t
 */
LIBSBML_EXTERN
int
Submodel_setName(Submodel_t * s, const char * name);


/**
 * Sets the modelRef of the given Submodel_t to a copy of @p modelRef.
 *
 * @param s the Submodel_t structure to set
 * @param modelRef the modelRef to assign to the given Submodel_t's "modelRef" attribute.
 *
 * @return integer value indicating success/failure of the
 * function.  @if clike The value is drawn from the
 * enumeration #OperationReturnValues_t. @endif@~ The possible values
 * returned by this function are:
 * @li @sbmlconstant{LIBSBML_OPERATION_SUCCESS, OperationReturnValues_t}
 * @li @sbmlconstant{LIBSBML_INVALID_ATTRIBUTE_VALUE, OperationReturnValues_t}
 *
 * @note Using this function with the name set to NULL is equivalent to
 * unsetting the "modelRef" attribute.
 *
 * @memberof Submodel_t
 */
LIBSBML_EXTERN
int
Submodel_setModelRef(Submodel_t * s, const char * modelRef);


/**
 * Automatically fails, since "substanceConversionFactor" is not a part of the comp spec.
 *
 * @param s the Submodel_t structure to set
 * @param substanceConversionFactor the substanceConversionFactor to ignore
 *
 * @return integer value indicating success/failure of the
 * function.  @if clike The value is drawn from the
 * enumeration #OperationReturnValues_t. @endif@~ The possible value
 * returned by this function is:
 * @li @sbmlconstant{LIBSBML_INVALID_ATTRIBUTE_VALUE, OperationReturnValues_t}
 *
 * @memberof Submodel_t
 */
LIBSBML_EXTERN
int
Submodel_setSubstanceConversionFactor(Submodel_t * s, const char * substanceConversionFactor);


/**
 * Sets the timeConversionFactor of the given Submodel_t to a copy of @p timeConversionFactor.
 *
 * @param s the Submodel_t structure to set
 * @param timeConversionFactor the timeConversionFactor to assign to the given Submodel_t's "timeConversionFactor" attribute.
 *
 * @return integer value indicating success/failure of the
 * function.  @if clike The value is drawn from the
 * enumeration #OperationReturnValues_t. @endif@~ The possible values
 * returned by this function are:
 * @li @sbmlconstant{LIBSBML_OPERATION_SUCCESS, OperationReturnValues_t}
 * @li @sbmlconstant{LIBSBML_INVALID_ATTRIBUTE_VALUE, OperationReturnValues_t}
 *
 * @note Using this function with the name set to NULL is equivalent to
 * unsetting the "timeConversionFactor" attribute.
 *
 * @memberof Submodel_t
 */
LIBSBML_EXTERN
int
Submodel_setTimeConversionFactor(Submodel_t * s, const char * timeConversionFactor);


/**
 * Sets the extentConversionFactor of the given Submodel_t to a copy of @p extentConversionFactor.
 *
 * @param s the Submodel_t structure to set
 * @param extentConversionFactor the extentConversionFactor to assign to the given Submodel_t's "extentConversionFactor" attribute.
 *
 * @return integer value indicating success/failure of the
 * function.  @if clike The value is drawn from the
 * enumeration #OperationReturnValues_t. @endif@~ The possible values
 * returned by this function are:
 * @li @sbmlconstant{LIBSBML_OPERATION_SUCCESS, OperationReturnValues_t}
 * @li @sbmlconstant{LIBSBML_INVALID_ATTRIBUTE_VALUE, OperationReturnValues_t}
 *
 * @note Using this function with the name set to NULL is equivalent to
 * unsetting the "extentConversionFactor" attribute.
 *
 * @memberof Submodel_t
 */
LIBSBML_EXTERN
int
Submodel_setExtentConversionFactor(Submodel_t * s, const char * extentConversionFactor);


/**
 * Unsets the "id" attribute of the given Submodel_t structure.
 *
 * @param s the Submodel_t structure to unset
 *
 * @return integer value indicating success/failure of the
 * function.  @if clike The value is drawn from the
 * enumeration #OperationReturnValues_t. @endif@~ The possible values
 * returned by this function are:
 * @li @sbmlconstant{LIBSBML_OPERATION_SUCCESS, OperationReturnValues_t}
 * @li @sbmlconstant{LIBSBML_OPERATION_FAILED, OperationReturnValues_t}
 *
 * @memberof Submodel_t
 */
LIBSBML_EXTERN
int
Submodel_unsetId(Submodel_t * s);


/**
 * Unsets the "name" attribute of the given Submodel_t structure.
 *
 * @param s the Submodel_t structure to unset
 *
 * @return integer value indicating success/failure of the
 * function.  @if clike The value is drawn from the
 * enumeration #OperationReturnValues_t. @endif@~ The possible values
 * returned by this function are:
 * @li @sbmlconstant{LIBSBML_OPERATION_SUCCESS, OperationReturnValues_t}
 * @li @sbmlconstant{LIBSBML_OPERATION_FAILED, OperationReturnValues_t}
 *
 * @memberof Submodel_t
 */
LIBSBML_EXTERN
int
Submodel_unsetName(Submodel_t * s);


/**
 * Unsets the "modelRef" attribute of the given Submodel_t structure.
 *
 * @param s the Submodel_t structure to unset
 *
 * @return integer value indicating success/failure of the
 * function.  @if clike The value is drawn from the
 * enumeration #OperationReturnValues_t. @endif@~ The possible values
 * returned by this function are:
 * @li @sbmlconstant{LIBSBML_OPERATION_SUCCESS, OperationReturnValues_t}
 * @li @sbmlconstant{LIBSBML_OPERATION_FAILED, OperationReturnValues_t}
 *
 * @memberof Submodel_t
 */
LIBSBML_EXTERN
int
Submodel_unsetModelRef(Submodel_t * s);


/**
 * Automatically fails, since "substanceConversionFactor" is not a part of the comp spec.
 *
 * @param s the Submodel_t structure to unset
 *
 * @return integer value indicating success/failure of the
 * function.  @if clike The value is drawn from the
 * enumeration #OperationReturnValues_t. @endif@~ The possible value
 * returned by this function is:
 * @li @sbmlconstant{LIBSBML_OPERATION_FAILED, OperationReturnValues_t}
 *
 * @memberof Submodel_t
 */
LIBSBML_EXTERN
int
Submodel_unsetSubstanceConversionFactor(Submodel_t * s);


/**
 * Unsets the "timeConversionFactor" attribute of the given Submodel_t structure.
 *
 * @param s the Submodel_t structure to unset
 *
 * @return integer value indicating success/failure of the
 * function.  @if clike The value is drawn from the
 * enumeration #OperationReturnValues_t. @endif@~ The possible values
 * returned by this function are:
 * @li @sbmlconstant{LIBSBML_OPERATION_SUCCESS, OperationReturnValues_t}
 * @li @sbmlconstant{LIBSBML_OPERATION_FAILED, OperationReturnValues_t}
 *
 * @memberof Submodel_t
 */
LIBSBML_EXTERN
int
Submodel_unsetTimeConversionFactor(Submodel_t * s);


/**
 * Unsets the "extentConversionFactor" attribute of the given Submodel_t structure.
 *
 * @param s the Submodel_t structure to unset
 *
 * @return integer value indicating success/failure of the
 * function.  @if clike The value is drawn from the
 * enumeration #OperationReturnValues_t. @endif@~ The possible values
 * returned by this function are:
 * @li @sbmlconstant{LIBSBML_OPERATION_SUCCESS, OperationReturnValues_t}
 * @li @sbmlconstant{LIBSBML_OPERATION_FAILED, OperationReturnValues_t}
 *
 * @memberof Submodel_t
 */
LIBSBML_EXTERN
int
Submodel_unsetExtentConversionFactor(Submodel_t * s);


/**
 * Appends a copy of the given Deletion_t structure to the given Submodel_t
 * structure.
 *
 * @param s the Submodel_t structure to which the Deletion_t should be
 * added
 *
 * @param d a Deletion_t structure to add
 *
 * @return integer value indicating success/failure of the
 * function.  @if clike The value is drawn from the
 * enumeration #OperationReturnValues_t. @endif@~ The possible values
 * returned by this function are:
 * @li @sbmlconstant{LIBSBML_OPERATION_SUCCESS, OperationReturnValues_t}
 * @li @sbmlconstant{LIBSBML_LEVEL_MISMATCH, OperationReturnValues_t}
 * @li @sbmlconstant{LIBSBML_VERSION_MISMATCH, OperationReturnValues_t}
 * @li @sbmlconstant{LIBSBML_PKG_VERSION_MISMATCH, OperationReturnValues_t}
 * @li @sbmlconstant{LIBSBML_DUPLICATE_OBJECT_ID, OperationReturnValues_t}
 * @li @sbmlconstant{LIBSBML_OPERATION_FAILED, OperationReturnValues_t}
 *
 * @memberof Submodel_t
 */
LIBSBML_EXTERN
int
Submodel_addDeletion(Submodel_t * s, Deletion_t * d);


/**
 * Creates a new, empty Deletion_t structure, adds it to the given
 * Submodel_t, and returns the Deletion_t.
 *
 * @param s the Submodel_t structure to which the Deletion_t should be
 * added
 *
 * @return the newly-created empty Deletion_t.
 *
 * @memberof Submodel_t
 */
LIBSBML_EXTERN
Deletion_t *
Submodel_createDeletion(Submodel_t * s);


/**
 * Get the list of Deletion_t structures from the given Submodel_t
 * structure.
 *
 * @param s the Submodel_t structure to use.
 *
 * @return the list of CHILDTYPEs for the given Submodel_t.
 *
 * @memberof Submodel_t
 */
LIBSBML_EXTERN
ListOf_t *
Submodel_getListOfDeletions(Submodel_t * s) ;


/**
 * Return a specific Deletion_t structure of the given Submodel_t.
 *
 * @param s the Submodel_t structure to use
 *
 * @param n an integer, the index of the Deletion_t structure to return
 * 
 * @return the nth Deletion_t of the given Submodel_t, or @c NULL if no such Deletion_t exists.
 *
 * @memberof Submodel_t
 */
LIBSBML_EXTERN
Deletion_t *
Submodel_getDeletion(Submodel_t * s, unsigned int n);


/**
 * Return the Deletion_t indicated by the given @p sid.
 *
 * @param s the Submodel_t structure to use
 *
 * @param sid a string, the identifier of the
 * Deletion_t is being sought.
 *
 * @return the Deletion_t for the given variable, or @c NULL if no such
 * Deletion_t exits.
 *
 * @memberof Submodel_t
 */
LIBSBML_EXTERN
Deletion_t *
Submodel_getDeletionById(Submodel_t * s, const char * sid);


/**
 * Returns the number of EventAssignment_t structures attached to the given
 * Submodel_t.
 *
 * @param s the Submodel_t structure to use
 * 
 * @return the number of EventAssignment_t structures in the given Submodel_t.
 *
 * @memberof Submodel_t
 */
LIBSBML_EXTERN
unsigned int
Submodel_getNumDeletions(Submodel_t * s);


/**
 * Removes the nth Deletion_t structure from the given Submodel_t structure and
 * returns a pointer to it.
 *
 * * The caller owns the returned structure and is responsible for deleting it.
 *
 * @param s the Submodel_t structure
 * @param n the integer index of the Deletion_t sought
 *
 * @return the Deletion_t structure removed.  As mentioned above, 
 * the caller owns the returned item. @c NULL is returned if the given index 
 * is out of range.
 *
 * @memberof Submodel_t
 */
LIBSBML_EXTERN
Deletion_t *
Submodel_removeDeletion(Submodel_t * s, unsigned int n);


/**
 * Removes the Deletion_t structure with the given @p sid
 * from the given Submodel_t structure and returns a pointer to it.
 *
 * * The caller owns the returned structure and is responsible for deleting it.
 *
 * @param s the Submodel_t structure
 * @param sid the string of the "id" attribute of the Deletion_t sought
 *
 * @return the Deletion_t structure removed.  As mentioned above, the 
 * caller owns the returned structure. @c NULL is returned if no Deletion_t
 * structure with the "id" attribute exists in the given Submodel_t structure.
 *
 * @memberof Submodel_t
 */
LIBSBML_EXTERN
Deletion_t *
Submodel_removeDeletionById(Submodel_t * s, const char * sid);


/**
  * Predicate returning @c true or @c false depending on whether
  * all the required attributes for the given Submodel_t structure
  * have been set.
  *
  * @note The required attributes for a Submodel_t structure are:
  * @li useValuesfromTriggerTime ( L3 onwards )
  *
 * @memberof Submodel_t
 */
LIBSBML_EXTERN
int
Submodel_hasRequiredAttributes(Submodel_t * s);


/**
  * Predicate returning @c true or @c false depending on whether
  * all the required elements for the given Submodel_t structure
  * have been set.
  *
  * @note The required elements for an Submodel_t structure are:
  * @li trigger
  * @li listOfEventAssignments (requirement removed in L3)
  *
 * @memberof Submodel_t
 */
LIBSBML_EXTERN
int
Submodel_hasRequiredElements(Submodel_t * s);


/**
 * Return the Submodel_t indicated by the given @p sid.
 *
 * @param lo the ListOf_t structure to use
 *
 * @param sid a string, the identifier of the
 * Submodel_t is being sought.
 *
 * @return the Submodel_t for the given variable, or @c NULL if no such
 * Submodel_t exits.
 *
 * @memberof Submodel_t
 */
LIBSBML_EXTERN
Submodel_t *
ListOfSubmodels_getById(ListOf_t * lo, const char * sid);


/**
 * Removes the Submodel_t structure with the given @p sid
 * from the given ListOf_t structure and returns a pointer to it.
 *
 * * The caller owns the returned structure and is responsible for deleting it.
 *
 * @param lo the ListOf_t structure
 * @param sid the string of the "id" attribute of the Submodel_t sought
 *
 * @return the Submodel_t structure removed.  As mentioned above, the 
 * caller owns the returned structure. @c NULL is returned if no Submodel_t
 * structure with the "id" attribute exists in the given ListOf_t structure.
 *
 * @memberof Submodel_t
 */
LIBSBML_EXTERN
Submodel_t *
ListOfSubmodels_removeById(ListOf_t * lo, const char * sid);



END_C_DECLS
LIBSBML_CPP_NAMESPACE_END


#endif  /* !SWIG */
#endif  /* Submodel_H__ */

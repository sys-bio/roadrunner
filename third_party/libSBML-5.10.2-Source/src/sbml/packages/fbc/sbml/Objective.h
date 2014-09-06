/**
 * @file    Objective.h
 * @brief   Definition of Objective, the SBase derived class of the fbc package.
 * @author  Frank T. Bergmann
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
 *
 * @class Objective
 * @sbmlbrief{fbc} An objective function.
 *
 * An integral component in a complete description of a steady-state model is
 * the so-called <em>objective function</em>, which generally consists of a
 * linear combination of model variables (fluxes) and a sense (direction). In
 * the SBML Level&nbsp;3 FBC package, this concept is succinctly captured in
 * the Objective class. The FBC Objective class is derived from the normal
 * SBML SBase class and inherits the 'metaid' and 'sboTerm' attributes, as
 * well as the subcomponents for Annotation and Notes.
 */

#ifndef Objective_H__
#define Objective_H__

#include <sbml/common/extern.h>
#include <sbml/common/sbmlfwd.h>
#include <sbml/packages/fbc/common/fbcfwd.h>

LIBSBML_CPP_NAMESPACE_BEGIN

/** 
 * @enum ObjectiveType_t
 * @brief Enumeration of possible values for the 'type' attribute of an Objective.
 */
typedef enum
{
    OBJECTIVE_TYPE_MAXIMIZE /*!< 'maximize' */
  , OBJECTIVE_TYPE_MINIMIZE /*!< 'minimize' */
  , OBJECTIVE_TYPE_UNKNOWN /*!< Unset or illegal value: anything other than 'maximize' or 'minimize'. */
} ObjectiveType_t;

LIBSBML_CPP_NAMESPACE_END


#ifdef __cplusplus

#include <string>

#include <sbml/SBase.h>
#include <sbml/ListOf.h>
#include <sbml/packages/fbc/extension/FbcExtension.h>
#include <sbml/packages/fbc/sbml/FluxObjective.h>

LIBSBML_CPP_NAMESPACE_BEGIN


class LIBSBML_EXTERN Objective : public SBase
{
protected:
  /** @cond doxygenLibsbmlInternal */
  std::string mId;
  std::string mName;
  ObjectiveType_t mType;
  std::string mTypeString;
  ListOfFluxObjectives mFluxes;
  bool mIsSetListOfFluxObjectives;
  /** @endcond */


public:

  /**
   * Creates a new Objective with the given level, version, and package version.
   */
   Objective(unsigned int level      = FbcExtension::getDefaultLevel(),
          unsigned int version    = FbcExtension::getDefaultVersion(),
          unsigned int pkgVersion = FbcExtension::getDefaultPackageVersion());


  /**
   * Creates a new Objective with the given FbcPkgNamespaces object.
   */
   Objective(FbcPkgNamespaces* fbcns);


  /**
   * Copy constructor.
   */
   Objective(const Objective& source);


  /**
   * Assignment operator.
   */
   Objective& operator=(const Objective& source);


  /**
   * Destructor.
   */ 
  virtual ~Objective ();

  
  
  /**
   * Returns the first child element found that has the given @p id in the model-wide SId namespace, or @c NULL if no such object is found.
   *
   * @param id string representing the id of objects to find
   *
   * @return a pointer to the SBase element with the given @p id.
   */
  virtual SBase* getElementBySId(const std::string& id);
  
  
  /**
   * Returns the first child element it can find with the given @p metaid, or itself if it has the given @p metaid, or @c NULL if no such object is found.
   *
   * @param metaid string representing the metaid of objects to find
   *
   * @return a pointer to the SBase element with the given @p metaid.
   */
  virtual SBase* getElementByMetaId(const std::string& metaid);
  
  /**
   * Returns a List of all child SBase* objects, including those nested to an arbitrary depth
   *
   * @return a List* of pointers to all children objects.
   */
  virtual List* getAllElements(ElementFilter* filter=NULL);
  
  
  /**
   * Returns the value of the "name" attribute of this Objective.
   *
   * @return the value of the "name" attribute of this Objective.
   */
  virtual const std::string& getName () const;
  
  
  /**
   * Predicate returning @c true or @c false depending on whether this
   * Objective's "name" attribute has been set.
   *
   * @return @c true if this Objective's "id" attribute has been set,
   * otherwise @c false is returned.
   */
  virtual bool isSetName () const;
  
  
  /**
   * Sets the value of the "name" attribute of this Objective.
   *
   * @return integer value indicating success/failure of the
   * operation. The possible return values are:
   * @li @sbmlconstant{LIBSBML_OPERATION_SUCCESS, OperationReturnValues_t}
   * @li @sbmlconstant{LIBSBML_INVALID_ATTRIBUTE_VALUE, OperationReturnValues_t}
   */
  virtual int setName (const std::string& name);
  
  
  /**
   * Unsets the value of the "name" attribute of this Objective.
   *
   * @return integer value indicating success/failure of the
   * operation. The possible return values are:
   * @li @sbmlconstant{LIBSBML_OPERATION_SUCCESS, OperationReturnValues_t}
   * @li @sbmlconstant{LIBSBML_OPERATION_FAILED, OperationReturnValues_t}
   */
  virtual int unsetName ();


  /**
   * Returns the string of the "type" attribute of this Objective.
   *
   * @return the string of the "type" attribute of this Objective.
   */
  const std::string& getType ();


  /**
   * Returns the ObjectiveType_t of the "type" attribute of this Objective.
   *
   * @return the ObjectiveType_t of the "type" attribute of this Objective.
   */
  ObjectiveType_t getObjectiveType () const;


  /**
   * Predicate returning @c true or @c false depending on whether this
   * Objective's "type" attribute has been set.
   *
   * @return @c true if this Objective's "type" attribute has been set, 
   * otherwise @c false is returned.
   */
  virtual bool isSetType () const;

  
  /**
   * Sets the SIdRef string of the "type" attribute of this Objective.
   *
   * @param type a SIdRef string to be set.
   *
   * @return integer value indicating success/failure of the
   * operation. The possible return values are:
   * @li @sbmlconstant{LIBSBML_OPERATION_SUCCESS, OperationReturnValues_t}
   * @li @sbmlconstant{LIBSBML_INVALID_ATTRIBUTE_VALUE, OperationReturnValues_t}
   */
  virtual int setType (const std::string& type);


  /**
   * Sets the SIdRef string of the "type" attribute of this Objective.
   *
   * @param type a SIdRef string to be set.
   *
   * @return integer value indicating success/failure of the
   * operation. The possible return values are:
   * @li @sbmlconstant{LIBSBML_OPERATION_SUCCESS, OperationReturnValues_t}
   * @li @sbmlconstant{LIBSBML_INVALID_ATTRIBUTE_VALUE, OperationReturnValues_t}
   */
  int setType (ObjectiveType_t type);


  /**
   * Unsets the value of the "id" attribute of this Objective.
   *
   * @return integer value indicating success/failure of the
   * operation. The possible return values are:
   * @li @sbmlconstant{LIBSBML_OPERATION_SUCCESS, OperationReturnValues_t}
   * @li @sbmlconstant{LIBSBML_OPERATION_FAILED, OperationReturnValues_t}
   */
  virtual int unsetType ();

  
  /**
   * Returns the string of the "id" attribute of this Objective.
   *
   * @return the string of the "id" attribute of this Objective.
   */
  virtual const std::string& getId () const;


  /**
   * Predicate returning @c true or @c false depending on whether this
   * Objective's "id" attribute has been set.
   *
   * @return @c true if this Objective's "id" attribute has been set, 
   * otherwise @c false is returned.
   */
  virtual bool isSetId () const;

  
  /**
   * Sets the SIdRef string of the "id" attribute of this Objective.
   *
   * @param id a SIdRef string to be set.
   *
   * @return integer value indicating success/failure of the
   * operation. The possible return values are:
   * @li @sbmlconstant{LIBSBML_OPERATION_SUCCESS, OperationReturnValues_t}
   * @li @sbmlconstant{LIBSBML_INVALID_ATTRIBUTE_VALUE, OperationReturnValues_t}
   */
  virtual int setId (const std::string& id);


  /**
   * Unsets the value of the "id" attribute of this Objective.
   *
   * @return integer value indicating success/failure of the
   * operation. The possible return values are:
   * @li @sbmlconstant{LIBSBML_OPERATION_SUCCESS, OperationReturnValues_t}
   * @li @sbmlconstant{LIBSBML_OPERATION_FAILED, OperationReturnValues_t}
   */
  virtual int unsetId ();

  
  /**
   * Returns the ListOf object that holds all members.
   *
   * @return the ListOf object that holds all members.
   */ 
  const ListOfFluxObjectives* getListOfFluxObjectives () const;

  /**
   * Returns the member with the given index.
   * If the index is invalid, @c NULL is returned.
   *
   * @param n the index number of the FluxObjective to get.
   *
   * @return the nth FluxObjective in the ListOfFluxObjectives.
   */ 
  FluxObjective* getFluxObjective (unsigned int n);

  /**
   * Returns the member with the given index.
   * If the index is invalid, @c NULL is returned.
   *
   * @param n the index number of the FluxObjective to get.
   *
   * @return the nth FluxObjective in the ListOfFluxObjectives.
   */ 
  const FluxObjective* getFluxObjective (unsigned int n) const;

  /**
   * Returns the member with the given symbol.
   * If the index is invalid, @c NULL is returned.
   *
   * @param symbol a string representing the symbol attribute
   * of the FluxObjective to get.
   * 
   * @return FluxObjective in the ListOfFluxObjectives with the given symbol
   * or NULL if no such FluxObjective exists.
   */
  FluxObjective* getFluxObjective (const std::string& symbol);


  /**
   * Returns the member with the given symbol.
   * If the index is invalid, @c NULL is returned.
   *
   * @param symbol a string representing the symbol attribute
   * of the FluxObjective to get.
   * 
   * @return FluxObjective in the ListOfFluxObjectives with the given symbol
   * or NULL if no such FluxObjective exists.
   */
  const FluxObjective* getFluxObjective (const std::string& symbol) const;


  /**
   * Adds a copy of the given FluxObjective object to the list of members.
   *
   * @param member the FluxObjective object to be added to the list of 
   * members.
   *
   * @return integer value indicating success/failure of the
   * operation. The possible return values are:
   * @li @sbmlconstant{LIBSBML_OPERATION_SUCCESS, OperationReturnValues_t}
   * @li @sbmlconstant{LIBSBML_LEVEL_MISMATCH, OperationReturnValues_t}
   * @li @sbmlconstant{LIBSBML_VERSION_MISMATCH, OperationReturnValues_t}
   * @li @sbmlconstant{LIBSBML_PKG_VERSION_MISMATCH, OperationReturnValues_t}
   * @li @sbmlconstant{LIBSBML_DUPLICATE_OBJECT_ID, OperationReturnValues_t}
   * @li @sbmlconstant{LIBSBML_OPERATION_FAILED, OperationReturnValues_t}
   */
  int addFluxObjective (const FluxObjective* member);


  /**
   * Returns the number of members for this objective.
   *
   * @return the number of members for this objective.
   */
  unsigned int getNumFluxObjectives () const;


  /**
   * Creates a FluxObjective object, adds it to the end of the
   * member objects list and returns a pointer to the newly
   * created object.
   *
   * @return a newly created FluxObjective object
   */
  FluxObjective* createFluxObjective ();


  /**
   * Removes the member with the given index from the objective.
   * A pointer to the member that was removed is returned.
   * If no member has been removed, @c NULL is returned.
   *
   * @param index the index of the FluxObjective object to remove
   *
   * @return the FluxObjective object removed.  As mentioned above, 
   * the caller owns the returned object. @c NULL is returned if 
   * the given index is out of range.
   */
  FluxObjective* removeFluxObjective(unsigned int index);


  /**
   * Removes the member with the given symbol from the objective.
   * A pointer to the member that was removed is returned.
   * If no member has been removed, @c NULL is returned.
   *
   * @param symbol the symbol attribute of the FluxObjective object to remove
   *
   * @return the FluxObjective object removed.  As mentioned above, 
   * the caller owns the returned object. @c NULL is returned if 
   * the given index is out of range.
   */
  FluxObjective* removeFluxObjective(const std::string& symbol);

  /**
   * Returns the XML element name of
   * this SBML object.
   *
   * @return the name of this element, as a text string.
   */
  virtual const std::string& getElementName () const ;


  /**
   * Creates and returns a deep copy of this Objective object.
   *
   * @return a (deep) copy of this Objective.
   */
  virtual Objective* clone () const;


  /**
   * Returns the libSBML type code of this object instance.
   *
   * @copydetails doc_what_are_typecodes
   *
   * @return the SBML type code for this object:
   * @sbmlconstant{SBML_FBC_OBJECTIVE, SBMLFbcTypeCode_t}
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


  /**
   * Accepts the given SBMLVisitor.
   *
   * @return the result of calling <code>v.visit()</code>, which indicates
   * whether or not the Visitor would like to visit the SBML object's next
   * sibling object (if available).
   */
  virtual bool accept (SBMLVisitor& v) const;

  
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
   * Enables/Disables the given package with this element and child
   * elements (if any).
   * (This is an internal implementation for enablePakcage function)
   *
   * @note Subclasses in which one or more child elements are defined
   * must override this function.
   */
  virtual void enablePackageInternal(const std::string& pkgURI,
                                     const std::string& pkgPrefix, bool flag);
  /** @endcond */


  /** @cond doxygenLibsbmlInternal */
  /* function returns true if component has all the required
   * elements
   * needs to be overloaded for each component
   */
  virtual bool hasRequiredElements() const ;
  /** @endcond */


  /** @cond doxygenLibsbmlInternal */
  bool getIsSetListOfFluxObjectives() const;
  /** @endcond */
    
protected:
  /** @cond doxygenLibsbmlInternal */
  /**
   * Create and return an SBML object of this class, if present.
   *
   * @return the SBML object corresponding to next XMLToken in the
   * XMLInputStream or NULL if the token was not recognized.
   */
  virtual SBase*
  createObject (XMLInputStream& stream);
  /** @endcond */


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
   *   stream.writeAttribute( "id"  , mId   );
   *   stream.writeAttribute( "name", mName );
   *   ...
   */
  virtual void writeAttributes (XMLOutputStream& stream) const;
  /** @endcond */
};


/**
 * @class ListOfObjectives
 * @sbmlbrief{fbc} A list of Objective objects.
 * 
 * The ListOfObjectives is a container for the Objective elements of Model.
 * 
 * @copydetails doc_what_is_listof
 *
 * @see Objective
 */
class LIBSBML_EXTERN ListOfObjectives : public ListOf
{
public:

  /**
   * Creates and returns a deep copy of this ListOfObjectives object.
   *
   * @return a (deep) copy of this ListOfObjectives.
   */
  virtual ListOfObjectives* clone () const;


  /**
   * Creates a new ListOfObjectives with the given level, version, and package version.
   */
   ListOfObjectives(unsigned int level      = FbcExtension::getDefaultLevel(), 
                 unsigned int version    = FbcExtension::getDefaultVersion(), 
                 unsigned int pkgVersion = FbcExtension::getDefaultPackageVersion());


  /**
   * Creates a new ListOfObjectives with the given FbcPkgNamespaces object.
   */
   ListOfObjectives(FbcPkgNamespaces* fbcns);


  /**
   * Get a Objective from the ListOfObjectives.
   *
   * @param n the index number of the Objective to get.
   * 
   * @return the nth Objective in this ListOfObjectives.
   *
   * @see size()
   */
  virtual Objective * get(unsigned int n); 


  /**
   * Get a Objective from the ListOfObjectives.
   *
   * @param n the index number of the Objective to get.
   * 
   * @return the nth Objective in this ListOfObjectives.
   *
   * @see size()
   */
  virtual const Objective * get(unsigned int n) const; 

  /**
   * Get a Objective from the ListOfObjectives
   * based on its identifier.
   *
   * @param sid a string representing the identifier 
   * of the Objective to get.
   * 
   * @return Objective in this ListOfObjectives
   * with the given @p sid or @c NULL if no such
   * Objective exists.
   *
   * @see get(unsigned int n)
   * @see size()
   */
  virtual Objective* get (const std::string& sid);


  /**
   * Get a Objective from the ListOfObjectives
   * based on its identifier.
   *
   * @param sid a string representing the identifier 
   * of the Objective to get.
   * 
   * @return Objective in this ListOfObjectives
   * with the given @p sid or @c NULL if no such
   * Objective exists.
   *
   * @see get(unsigned int n)
   * @see size()
   */
  virtual const Objective* get (const std::string& sid) const;


  /**
   * Removes the nth item from this ListOfObjectives items and returns a pointer to
   * it.
   *
   * The caller owns the returned item and is responsible for deleting it.
   *
   * @param n the index of the item to remove
   * @return the item removed.  As mentioned above, the caller owns the
   * returned item.
   *
   * @see size()
   */
  virtual Objective* remove (unsigned int n);


  /**
   * Removes item in this ListOfObjectives items with the given identifier.
   *
   * The caller owns the returned item and is responsible for deleting it.
   * If none of the items in this list have the identifier @p sid, then @c
   * NULL is returned.
   *
   * @param sid the identifier of the item to remove
   *
   * @return the item removed.  As mentioned above, the caller owns the
   * returned item.
   */
  virtual Objective* remove (const std::string& sid);


  /**
   * Returns the libSBML type code for the objects contained in this ListOf
   * (i.e., @sbmlconstant{SBML_FBC_OBJECTIVE, SBMLFbcTypeCode_t}).
   * 
   * @copydetails doc_what_are_typecodes
   *
   * @return the SBML type code for objects contained in this list:
   * @sbmlconstant{SBML_FBC_OBJECTIVE, SBMLFbcTypeCode_t}
   *
   * @see getElementName()
   * @see getPackageName()
   */
  virtual int getItemTypeCode () const;

  /**
   * Returns the XML element name of
   * this SBML object.
   *
   * @return the name of this element, as a text string.
   */
  virtual const std::string& getElementName () const;


  /**
   * Predicate returning @c true or @c false depending on whether this
   * ListOfObjective's "activeObjective" attribute has been set.
   *
   * @return @c true if this ListOfObjective's "activeObjective" attribute has been set, 
   * otherwise @c false is returned.
   */
  virtual bool isSetActiveObjective() const;


  /**
   * Sets the value of the "activeObjective" attribute of this ListOfObjectives.
   *
   * @return integer value indicating success/failure of the
   * operation. The possible return values are:
   * @li @sbmlconstant{LIBSBML_OPERATION_SUCCESS, OperationReturnValues_t}
   * @li @sbmlconstant{LIBSBML_INVALID_ATTRIBUTE_VALUE, OperationReturnValues_t}
   */
  virtual int setActiveObjective(const std::string &activeObjective);


  /**
   * Returns the value of the "activeObjective" attribute of this ListOfObjectives.
   *
   * @return the value of the "activeObjective" attribute of this ListOfObjectives.
   */
  virtual const std::string& getActiveObjective() const;


  /**
   * Unsets the value of the "activeObjective" attribute of this ListOfObjectives.
   *
   * @return integer value indicating success/failure of the
   * operation. The possible return values are:
   * @li @sbmlconstant{LIBSBML_OPERATION_SUCCESS, OperationReturnValues_t}
   * @li @sbmlconstant{LIBSBML_OPERATION_FAILED, OperationReturnValues_t}
   */
  virtual int unsetActiveObjective();

  /** @cond doxygenLibsbmlInternal */

  virtual bool accept (SBMLVisitor& v) const;

  /** @endcond */
    
  /**
   * Adds a clone of all items in the provided ListOf to this object.  This means that when this ListOf is destroyed, the original items will not be destroyed.  In addition, copy over the input ListOfObjectives' 'activeObjective' attribute, if none is set for this element.
   *
   * @param list A list of items to be added.
   *
   * @see append(const SBase* item)
   */
  virtual int appendFrom(const ListOf* list);
  

  /**
   * @copydoc doc_renamesidref_common
   */
  virtual void renameSIdRefs(const std::string& oldid, const std::string& newid);



protected:
  /** @cond doxygenLibsbmlInternal */
  /**
   * Create and return an SBML object of this class, if present.
   *
   * @return the SBML object corresponding to next XMLToken in the
   * XMLInputStream or NULL if the token was not recognized.
   */
  virtual SBase* createObject (XMLInputStream& stream);
  /** @endcond */


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
   * Reads the attributes of corresponding package in SBMLDocument element.
   */
  virtual void readAttributes (const XMLAttributes& attributes,
                               const ExpectedAttributes& expectedAttributes);
  /** @endcond */


  /** @cond doxygenLibsbmlInternal */
  /**
   * Writes the attributes of corresponding package in SBMLDocument element.
   */
  virtual void writeAttributes (XMLOutputStream& stream) const;  
  
  std::string mActiveObjective;
  /** @endcond */

};

/** @cond doxygenLibsbmlInternal */
/**
 * Used by ListOfObjectives::get() to lookup an SBase based by its 
 * symbol
 */
#ifndef SWIG
template<>
struct IdEq<Objective> : public std::unary_function<SBase*, bool>
{
  const std::string& id;

  IdEq (const std::string& id) : id(id) { }
  bool operator() (SBase* sb) 
       { return static_cast <Objective*> (sb)->getId() == id; }
};
#endif
/** @endcond */

LIBSBML_CPP_NAMESPACE_END

#endif /* __cplusplus */


#ifndef SWIG

LIBSBML_CPP_NAMESPACE_BEGIN
BEGIN_C_DECLS

/**
 * Creates a new Objective_t structure using the given SBML @p level
 * and @p version values.
 *
 * @param level an unsigned int, the SBML Level to assign to this
 * Objective_t
 * @param version an unsigned int, the SBML Version to assign to this
 * Objective_t
 * @param pkgVersion an unsigned int, the SBML 'Qual' package Version to assign to this
 * Objective_t
 *
 * @return a pointer to the newly created Objective_t structure.
 *
 * @memberof Objective_t
 */
LIBSBML_EXTERN
Objective_t *
Objective_create(unsigned int level, unsigned int version, unsigned int pkgVersion);


/**
 * Takes an Objective_t structure and returns its identifier.
 *
 * @param obj the Objective_t structure whose identifier is sought
 * 
 * @return the identifier of the given Objective_t, as a pointer to a string.
 *
 * @memberof Objective_t
 */
LIBSBML_EXTERN
const char *
Objective_getId(Objective_t * obj);


/**
 * Predicate returning @c true or @c false depending on whether the given
 * Objective_t structure's identifier is set.
 *
 * @param obj the Objective_t structure to query
 * 
 * @return @c non-zero (true) if the "id" attribute of the given
 * Objective_t structure is set, zero (false) otherwise.
 *
 * @memberof Objective_t
 */
LIBSBML_EXTERN
int
Objective_isSetId(Objective_t * obj);


/**
 * Assigns the identifier of an Objective_t structure.
 *
 * This makes a copy of the string passed in the param @p sid.
 *
 * @param obj the Objective_t structure to set.
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
 * @memberof Objective_t
 */
LIBSBML_EXTERN
int
Objective_setId(Objective_t * obj, const char * sid);


/**
 * Unsets the "id" attribute of the given Objective_t structure.
 *
 * @param obj the Objective_t structure to unset
 *
 * @return integer value indicating success/failure of the
 * function.  @if clike The value is drawn from the
 * enumeration #OperationReturnValues_t. @endif@~ The possible values
 * returned by this function are:
 * @li @sbmlconstant{LIBSBML_OPERATION_SUCCESS, OperationReturnValues_t}
 * @li @sbmlconstant{LIBSBML_OPERATION_FAILED, OperationReturnValues_t}
 *
 * @memberof Objective_t
 */
LIBSBML_EXTERN
int
Objective_unsetId(Objective_t * obj);

/**
 * Takes a Objective_t structure and returns its name.
 *
 * @param obj the Objective_t whose name is sought.
 *
 * @return the name of the given Objective_t, as a pointer to a string.
 *
 * @memberof Objective_t
 */
LIBSBML_EXTERN
const char *
Objective_getName(Objective_t * obj);


/**
 * Predicate returning @c true or @c false depending on whether the given
 * Objective_t structure's name is set.
 *
 * @param obj the Objective_t structure to query
 * 
 * @return @c non-zero (true) if the "name" attribute of the given
 * Objective_t structure is set, zero (false) otherwise.
 *
 * @memberof Objective_t
 */
LIBSBML_EXTERN
int
Objective_isSetName(Objective_t * obj);


/**
 * Sets the name of the given Objective_t to a copy of @p name.
 *
 * @param obj the Objective_t structure to set
 * @param name the name to assign to the given Objective_t's "name" attribute.
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
 * @memberof Objective_t
 */
LIBSBML_EXTERN
int
Objective_setName(Objective_t * obj, const char * name);


/**
 * Unsets the "name" attribute of the given Objective_t structure.
 *
 * @param obj the Objective_t structure to unset
 *
 * @return integer value indicating success/failure of the
 * function.  @if clike The value is drawn from the
 * enumeration #OperationReturnValues_t. @endif@~ The possible values
 * returned by this function are:
 * @li @sbmlconstant{LIBSBML_OPERATION_SUCCESS, OperationReturnValues_t}
 * @li @sbmlconstant{LIBSBML_OPERATION_FAILED, OperationReturnValues_t}
 *
 * @memberof Objective_t
 */
LIBSBML_EXTERN
int
Objective_unsetName(Objective_t * obj);


/**
 * Takes a Objective_t structure and returns its type.
 *
 * @param obj the Objective_t whose type is sought.
 *
 * @return the type of the given Objective_t, as a pointer to a string.
 *
 * @memberof Objective_t
 */
LIBSBML_EXTERN
const char *
Objective_getType(Objective_t * obj);


/**
 * Predicate returning @c true or @c false depending on whether the given
 * Objective_t structure's type is set.
 *
 * @param obj the Objective_t structure to query
 * 
 * @return @c non-zero (true) if the "type" attribute of the given
 * Objective_t structure is set, zero (false) otherwise.
 *
 * @memberof Objective_t
 */
LIBSBML_EXTERN
int
Objective_isSetType(Objective_t * obj);


/**
 * Sets the type of the given Objective_t to a copy of @p type.
 *
 * @param obj the Objective_t structure to set
 * @param type the type to assign to the given Objective_t's "type" attribute.
 *
 * @return integer value indicating success/failure of the
 * function.  @if clike The value is drawn from the
 * enumeration #OperationReturnValues_t. @endif@~ The possible values
 * returned by this function are:
 * @li @sbmlconstant{LIBSBML_OPERATION_SUCCESS, OperationReturnValues_t}
 * @li @sbmlconstant{LIBSBML_INVALID_ATTRIBUTE_VALUE, OperationReturnValues_t}
 *
 * @note Using this function with the name set to NULL is equivalent to
 * unsetting the "type" attribute.
 *
 * @memberof Objective_t
 */
LIBSBML_EXTERN
int
Objective_setType(Objective_t * obj, const char * type);


/**
 * Unsets the "type" attribute of the given Objective_t structure.
 *
 * @param obj the Objective_t structure to unset
 *
 * @return integer value indicating success/failure of the
 * function.  @if clike The value is drawn from the
 * enumeration #OperationReturnValues_t. @endif@~ The possible values
 * returned by this function are:
 * @li @sbmlconstant{LIBSBML_OPERATION_SUCCESS, OperationReturnValues_t}
 * @li @sbmlconstant{LIBSBML_OPERATION_FAILED, OperationReturnValues_t}
 *
 * @memberof Objective_t
 */
LIBSBML_EXTERN
int
Objective_unsetType(Objective_t * obj);



/**
 * Appends a copy of the given FluxObjective_t structure to the given Objective_t
 * structure.
 *
 * @param obj the Objective_t structure to which the FluxObjective_t should be
 * added
 *
 * @param fo a FluxObjective_t structure to add
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
 * @memberof Objective_t
 */
LIBSBML_EXTERN
int
Objective_addFluxObjective(Objective_t * obj, FluxObjective_t * fo);


/**
 * Return a specific FluxObjective_t structure of the given Objective_t.
 *
 * @param obj the Objective_t structure to use
 *
 * @param n an integer, the index of the FluxObjective_t structure to return
 * 
 * @return the nth FluxObjective_t of the given Objective_t, or @c NULL if no such FluxObjective_t exists.
 *
 * @memberof Objective_t
 */
LIBSBML_EXTERN
FluxObjective_t *
Objective_getFluxObjective(Objective_t * obj, unsigned int n);


/**
 * Returns the number of FluxObjective_t structures attached to the given
 * Objective_t.
 *
 * @param obj the Objective_t structure to use
 * 
 * @return the number of FluxObjective_t structures in the given Objective_t.
 *
 * @memberof Objective_t
 */
LIBSBML_EXTERN
unsigned int
Objective_getNumFluxObjectives(Objective_t * obj);


/**
 * Takes a Objective_t structure and returns its activeObjective.
 *
 * @param lo the ListOf_t whose activeObjective is sought.
 *
 * @return the activeObjective of the given Objective_t, as a pointer to a string.
 *
 * @memberof Objective_t
 */
LIBSBML_EXTERN
const char *
ListOfObjectives_getActiveObjective(ListOf_t * lo);


/**
 * Predicate returning @c true or @c false depending on whether the given
 * Objective_t structure's activeObjective is set.
 *
 * @param lo the ListOf_t structure to query
 * 
 * @return @c non-zero (true) if the "activeObjective" attribute of the given
 * Objective_t structure is set, zero (false) otherwise.
 *
 * @memberof Objective_t
 */
LIBSBML_EXTERN
int
ListOfObjectives_isSetActiveObjective(ListOf_t * lo);


/**
 * Sets the activeObjective of the given Objective_t to a copy of @p activeObjective.
 *
 * @param lo the ListOf_t structure to set
 * @param activeObjective the activeObjective to assign to the given Objective_t's "activeObjective" attribute.
 *
 * @return integer value indicating success/failure of the
 * function.  @if clike The value is drawn from the
 * enumeration #OperationReturnValues_t. @endif@~ The possible values
 * returned by this function are:
 * @li @sbmlconstant{LIBSBML_OPERATION_SUCCESS, OperationReturnValues_t}
 * @li @sbmlconstant{LIBSBML_INVALID_ATTRIBUTE_VALUE, OperationReturnValues_t}
 *
 * @note Using this function with the name set to NULL is equivalent to
 * unsetting the "activeObjective" attribute.
 *
 * @memberof Objective_t
 */
LIBSBML_EXTERN
int
ListOfObjectives_setActiveObjective(ListOf_t * lo, const char * activeObjective);


/**
 * Unsets the "activeObjective" attribute of the given Objective_t structure.
 *
 * @param lo the ListOf_t structure to unset
 *
 * @return integer value indicating success/failure of the
 * function.  @if clike The value is drawn from the
 * enumeration #OperationReturnValues_t. @endif@~ The possible values
 * returned by this function are:
 * @li @sbmlconstant{LIBSBML_OPERATION_SUCCESS, OperationReturnValues_t}
 * @li @sbmlconstant{LIBSBML_OPERATION_FAILED, OperationReturnValues_t}
 *
 * @memberof Objective_t
 */
LIBSBML_EXTERN
int
ListOfObjectives_unsetActiveObjective(ListOf_t * lo);

/**
 * Returns the string version of the provided ObjectiveType_t enumeration.
 *
 * @param type The ObjectiveType_t enumeration to convert
 *
 * @return A string corresponding to the given effect:  "maximize", 
 * "minimize", or NULL if the value is OBJECTIVE_TYPE_UNKNOWN 
 * or another invalid enumeration value.
 *
 * @memberof Objective_t
 */
LIBSBML_EXTERN
const char* 
ObjectiveType_toString(ObjectiveType_t type);


/**
 * Returns the ObjectiveType_t enumeration corresponding to 
 * the given string, or OBJECTIVE_TYPE_UNKNOWN if there is 
 * no such match.  The matching is case-sensitive:  "maximize" will 
 * return OBJECTIVE_TYPE_MAXIMIZE, but "Maximize" will return 
 * OBJECTIVE_TYPE_UNKNOWN.
 *
 * @param s The string to convert to an ObjectiveType_t
 *
 * @return The corresponding ObjectiveType_t, or 
 * OBJECTIVE_TYPE_UNKNOWN if no match found.
 *
 * @memberof Objective_t
 */
LIBSBML_EXTERN
ObjectiveType_t 
ObjectiveType_fromString(const char* s);


/**
 * Predicate returning @c true (non-zero) or @c false (zero) depending on whether the given
 * ObjectiveType_t is valid.
 *
 * @param type the ObjectiveType_t enumeration to query
 * 
 * @return @c non-zero (true) if the ObjectiveType_t is
 * OBJECTIVE_TYPE_MAXIMIZE or OBJECTIVE_TYPE_MINIMIZE;
 * zero (false) otherwise (including OBJECTIVE_TYPE_UNKNOWN).
 *
 * @memberof Objective_t
 */
LIBSBML_EXTERN
int 
ObjectiveType_isValidObjectiveType(ObjectiveType_t type);


/**
 * Predicate returning @c true (non-zero) or @c false (zero) depending 
 * on whether the given string is a valid ObjectiveType_t.  
 * The matching is case-sensitive:  "maximize" will return @c true, but 
 * "Maximize" will return @c false.
 *
 * @param s The string to query
 * 
 * @return @c non-zero (true) if the string is
 * "maximize" or "minimize"; zero (false) otherwise.
 *
 * @memberof Objective_t
 */
LIBSBML_EXTERN
int 
ObjectiveType_isValidObjectiveTypeString(const char* s);


END_C_DECLS
LIBSBML_CPP_NAMESPACE_END


#endif  /* !SWIG */
#endif  /* Objective_H__ */

/**
 * @file    FluxBound.h
 * @brief   Definition of FluxBound, the SBase derived class of the fbc package.
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
 * @class FluxBound
 * @sbmlbrief{fbc} Max or min value for a reaction flux.
 *
 * The FluxBound object holds a single (in)equality that provides the maximum
 * or minimum value that a reaction flux can obtain at steady state.
 */ 

#ifndef FluxBound_H__
#define FluxBound_H__


#include <sbml/common/extern.h>
#include <sbml/common/sbmlfwd.h>
#include <sbml/packages/fbc/common/fbcfwd.h>

LIBSBML_CPP_NAMESPACE_BEGIN

  /**
   * The FluxBoundOperation_t enumeration contains the possible values for
   * the FluxBound 'operation' attribute.  The possible legal values are less
   * than or equal to, greater than or equal to, or equal to.  The two
   * options 'less than' and 'greater than' are not legal values for the
   * FluxBound 'operation' attribute, but are provided to allow backwards
   * compatibility with an earlier version of the draft specification.
   */
typedef enum
{
    FLUXBOUND_OPERATION_LESS_EQUAL /** Less than or equal to. */ 
  , FLUXBOUND_OPERATION_GREATER_EQUAL /** Greater than or equal to.*/ 
  , FLUXBOUND_OPERATION_LESS /** Less than. NOTE:  ILLEGAL VALUE */ 
  , FLUXBOUND_OPERATION_GREATER /** Greater than. NOTE:  ILLEGAL VALUE*/ 
  , FLUXBOUND_OPERATION_EQUAL /** Equal to. */ 
  , FLUXBOUND_OPERATION_UNKNOWN  /** Unknown operation. */
} FluxBoundOperation_t;

LIBSBML_CPP_NAMESPACE_END


#ifdef __cplusplus

#include <string>

#include <sbml/SBase.h>
#include <sbml/ListOf.h>
#include <sbml/packages/fbc/extension/FbcExtension.h>

LIBSBML_CPP_NAMESPACE_BEGIN


class LIBSBML_EXTERN FluxBound : public SBase
{
protected:
  /** @cond doxygenLibsbmlInternal */
  std::string   mId;
  std::string   mName;
  std::string   mReaction;
  FluxBoundOperation_t   mOperation;
  std::string   mOperationString;
  double        mValue;
  /** @endcond */

public:

  /**
   * Creates a new FluxBound with the given level, version, and package version.
   */
   FluxBound(unsigned int level      = FbcExtension::getDefaultLevel(),
         unsigned int version    = FbcExtension::getDefaultVersion(),
         unsigned int pkgVersion = FbcExtension::getDefaultPackageVersion());


  /**
   * Creates a new FluxBound with the given FbcPkgNamespaces object.
   */
   FluxBound(FbcPkgNamespaces* fbcns);


  /**
   * Copy constructor.
   */
   FluxBound(const FluxBound& source);

  /**
   * Assignment operator.
   */
   FluxBound& operator=(const FluxBound& source);


  /**
   * Destructor.
   */ 
  virtual ~FluxBound ();


  /**
   * Returns the value of the "id" attribute of this FluxBound.
   *
   * @return the value of the "id" attribute of this FluxBound.
   */
  virtual const std::string& getId () const;


  /**
   * Predicate returning @c true or @c false depending on whether this
   * FluxBound's "id" attribute has been set.
   *
   * @return @c true if this FluxBound's "id" attribute has been set, 
   * otherwise @c false is returned.
   */
  virtual bool isSetId () const;

  
  /**
   * Sets the value of the "id" attribute of this FluxBound.
   *
   * @return integer value indicating success/failure of the
   * operation. The possible return values are:
   * @li @sbmlconstant{LIBSBML_OPERATION_SUCCESS, OperationReturnValues_t}
   * @li @sbmlconstant{LIBSBML_INVALID_ATTRIBUTE_VALUE, OperationReturnValues_t}
   */
  virtual int setId (const std::string& id);


  /**
   * Unsets the value of the "id" attribute of this FluxBound.
   *
   * @return integer value indicating success/failure of the
   * operation. The possible return values are:
   * @li @sbmlconstant{LIBSBML_OPERATION_SUCCESS, OperationReturnValues_t}
   * @li @sbmlconstant{LIBSBML_OPERATION_FAILED, OperationReturnValues_t}
   */
  virtual int unsetId ();
  
  
  /**
   * Returns the value of the "name" attribute of this FluxBound.
   *
   * @return the value of the "name" attribute of this FluxBound.
   */
  virtual const std::string& getName () const;
  
  
  /**
   * Predicate returning @c true or @c false depending on whether this
   * FluxBound's "name" attribute has been set.
   *
   * @return @c true if this FluxBound's "id" attribute has been set,
   * otherwise @c false is returned.
   */
  virtual bool isSetName () const;
  
  
  /**
   * Sets the value of the "name" attribute of this FluxBound.
   *
   * @return integer value indicating success/failure of the
   * operation. The possible return values are:
   * @li @sbmlconstant{LIBSBML_OPERATION_SUCCESS, OperationReturnValues_t}
   * @li @sbmlconstant{LIBSBML_INVALID_ATTRIBUTE_VALUE, OperationReturnValues_t}
   */
  virtual int setName (const std::string& name);
  
  
  /**
   * Unsets the value of the "name" attribute of this FluxBound.
   *
   * @return integer value indicating success/failure of the
   * operation. The possible return values are:
   * @li @sbmlconstant{LIBSBML_OPERATION_SUCCESS, OperationReturnValues_t}
   * @li @sbmlconstant{LIBSBML_OPERATION_FAILED, OperationReturnValues_t}
   */
  virtual int unsetName ();

  /**
   * Returns the value of the "reaction" attribute of this FluxBound.
   *
   * @return the value of the "reaction" attribute of this FluxBound.
   */
  virtual const std::string& getReaction () const;


  /**
   * Predicate returning @c true or @c false depending on whether this
   * FluxBound's "reaction" attribute has been set.
   *
   * @return @c true if this FluxBound's "reaction" attribute has been set, 
   * otherwise @c false is returned.
   */
  virtual bool isSetReaction () const;

  
  /**
   * Sets the value of the "reaction" attribute of this FluxBound.
   *
   * @return integer value indicating success/failure of the
   * operation. The possible return values are:
   * @li @sbmlconstant{LIBSBML_OPERATION_SUCCESS, OperationReturnValues_t}
   * @li @sbmlconstant{LIBSBML_INVALID_ATTRIBUTE_VALUE, OperationReturnValues_t}
   */
  virtual int setReaction (const std::string& reaction);


  /**
   * Unsets the value of the "reaction" attribute of this FluxBound.
   *
   * @return integer value indicating success/failure of the
   * operation. The possible return values are:
   * @li @sbmlconstant{LIBSBML_OPERATION_SUCCESS, OperationReturnValues_t}
   * @li @sbmlconstant{LIBSBML_OPERATION_FAILED, OperationReturnValues_t}
   */
  virtual int unsetReaction ();


  /**
   * Returns the value of the "operation" attribute of this FluxBound.
   *
   * @return the value of the "operation" attribute of this FluxBound.
   */
  const std::string& getOperation ();


  /**
   * Returns the value of the "operation" attribute of this FluxBound.
   *
   * @return the value of the "operation" attribute of this FluxBound.
   */
  FluxBoundOperation_t getFluxBoundOperation () const;


  /**
   * Predicate returning @c true or @c false depending on whether this
   * FluxBound's "operation" attribute has been set.
   *
   * @return @c true if this FluxBound's "operation" attribute has been set, 
   * otherwise @c false is returned.
   */
  virtual bool isSetOperation () const;

  
  /**
   * Sets the value of the "operation" attribute of this FluxBound.
   *
   * @return integer value indicating success/failure of the
   * operation. The possible return values are:
   * @li @sbmlconstant{LIBSBML_OPERATION_SUCCESS, OperationReturnValues_t}
   * @li @sbmlconstant{LIBSBML_INVALID_ATTRIBUTE_VALUE, OperationReturnValues_t}
   */
  virtual int setOperation (const std::string& operation);


  /**
   * Sets the value of the "operation" attribute of this FluxBound.
   *
   * @return integer value indicating success/failure of the
   * operation. The possible return values are:
   * @li @sbmlconstant{LIBSBML_OPERATION_SUCCESS, OperationReturnValues_t}
   * @li @sbmlconstant{LIBSBML_INVALID_ATTRIBUTE_VALUE, OperationReturnValues_t}
   */
  virtual int setOperation (FluxBoundOperation_t operation);


  /**
   * Unsets the value of the "operation" attribute of this FluxBound.
   *
   * @return integer value indicating success/failure of the
   * operation. The possible return values are:
   * @li @sbmlconstant{LIBSBML_OPERATION_SUCCESS, OperationReturnValues_t}
   * @li @sbmlconstant{LIBSBML_OPERATION_FAILED, OperationReturnValues_t}
   */
  virtual int unsetOperation ();

  /**
   * Returns the value of the "value" attribute of this FluxBound.
   *
   * @return the value of the "value" attribute of this FluxBound.
   */
  virtual const double getValue () const;


  /**
   * Predicate returning @c true or @c false depending on whether this
   * FluxBound's "value" attribute has been set.
   *
   * @return @c true if this FluxBound's "value" attribute has been set, 
   * otherwise @c false is returned.
   */
  virtual bool isSetValue () const;

  
  /**
   * Sets the value of the "value" attribute of this FluxBound.
   *
   * @return integer value indicating success/failure of the
   * operation. The possible return values are:
   * @li @sbmlconstant{LIBSBML_OPERATION_SUCCESS, OperationReturnValues_t}
   * @li @sbmlconstant{LIBSBML_INVALID_ATTRIBUTE_VALUE, OperationReturnValues_t}
   */
  virtual int setValue (const double value);


  /**
   * Unsets the value of the "value" attribute of this FluxBound.
   *
   * @return integer value indicating success/failure of the
   * operation. The possible return values are:
   * @li @sbmlconstant{LIBSBML_OPERATION_SUCCESS, OperationReturnValues_t}
   * @li @sbmlconstant{LIBSBML_OPERATION_FAILED, OperationReturnValues_t}
   */
  virtual int unsetValue ();


  /**
   * @copydoc doc_renamesidref_common
   */
   virtual void renameSIdRefs(const std::string& oldid, const std::string& newid);


  /**
   * Returns the XML element name of
   * this SBML object.
   *
   * @return the name of this element, as a text string.
   */
  virtual const std::string& getElementName () const ;


  /**
   * Creates and returns a deep copy of this FluxBound.
   * 
   * @return a (deep) copy of this FluxBound.
   */
  virtual FluxBound* clone () const;


  /**
   * Returns the libSBML type code of this object instance.
   *
   * @copydetails doc_what_are_typecodes
   *
   * @return the SBML type code for this object:
   * @sbmlconstant{SBML_FBC_FLUXBOUND, SBMLFbcTypeCode_t}
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
 * @class ListOfFluxBounds
 * @sbmlbrief{fbc} A list of FluxBound objects.
 * 
 * The ListOfFluxBounds is a container for the FluxBound elements of a Model.
 * 
 * @copydetails doc_what_is_listof
 */
class LIBSBML_EXTERN ListOfFluxBounds : public ListOf
{
public:

  /**
   * Creates and returns a deep copy of this ListOfFluxBounds.
   * 
   * @return a (deep) copy of this ListOfFluxBounds.
   */
  virtual ListOfFluxBounds* clone () const;


  /**
   * Creates a new ListOfFluxBounds with the given level, version, and package version.
   */
   ListOfFluxBounds(unsigned int level      = FbcExtension::getDefaultLevel(), 
                unsigned int version    = FbcExtension::getDefaultVersion(), 
                unsigned int pkgVersion = FbcExtension::getDefaultPackageVersion());


  /**
   * Creates a new ListOfFluxBounds with the given FbcPkgNamespaces object.
   */
   ListOfFluxBounds(FbcPkgNamespaces* fbcns);


  /**
   * Get a FluxBound from the ListOfFluxBounds.
   *
   * @param n the index number of the FluxBound to get.
   * 
   * @return the nth FluxBound in this ListOfFluxBounds.
   *
   * @see size()
   */
  virtual FluxBound* get(unsigned int n); 


  /**
   * Get a FluxBound from the ListOfFluxBounds.
   *
   * @param n the index number of the FluxBound to get.
   * 
   * @return the nth FluxBound in this ListOfFluxBounds.
   *
   * @see size()
   */
  virtual const FluxBound * get(unsigned int n) const; 


  /**
   * Get a FluxBound from the ListOfFluxBounds
   * based on its identifier.
   *
   * @param sid a string representing the identifier 
   * of the FluxBound to get.
   * 
   * @return FluxBound in this ListOfFluxBounds
   * with the given @p sid or @c NULL if no such
   * FluxBound exists.
   *
   * @see get(unsigned int n)
   * @see size()
   */
  virtual FluxBound* get (const std::string& sid);


  /**
   * Get a FluxBound from the ListOfFluxBounds
   * based on its identifier.
   *
   * @param sid a string representing the identifier 
   * of the FluxBound to get.
   * 
   * @return FluxBound in this ListOfFluxBounds
   * with the given @p sid or @c NULL if no such
   * FluxBound exists.
   *
   * @see get(unsigned int n)
   * @see size()
   */
  virtual const FluxBound* get (const std::string& sid) const;


  /**
   * Removes the nth item from this ListOfFluxBounds items and returns a pointer to
   * it.
   *
   * The caller owns the returned item and is responsible for deleting it.
   *
   * @param n the index of the item to remove
   *
   * @see size()
   */
  virtual FluxBound* remove (unsigned int n);


  /**
   * Removes item in this ListOfFluxBounds items with the given identifier.
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
  virtual FluxBound* remove (const std::string& sid);


  /**
   * Returns the libSBML type code for the SBML objects
   * contained in this ListOf object.
   * 
   * @copydetails doc_what_are_typecodes
   *
   * @return the SBML type code for objects contained in this list:
   * @sbmlconstant{SBML_FBC_FLUXBOUND, SBMLTypeCode_t} (default).
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


protected:

  /** @cond doxygenLibsbmlInternal */
  /**
   * Create and return an SBML object of this class, if present.
   *
   * @return the SBML object corresponding to next XMLToken in the
   * XMLInputStream or NULL if the token was not recognized.
   */
  virtual SBase* createObject (XMLInputStream& stream);

  virtual void writeXMLNS (XMLOutputStream& stream) const;
  /** @endcond */
};


LIBSBML_CPP_NAMESPACE_END

#endif /* __cplusplus */


#ifndef SWIG

LIBSBML_CPP_NAMESPACE_BEGIN
BEGIN_C_DECLS

/**
 * Creates a new FluxBound_t structure using the given SBML @p level
 * and @p version values.
 *
 * @param level an unsigned int, the SBML Level to assign to this
 * FluxBound_t
 * @param version an unsigned int, the SBML Version to assign to this
 * FluxBound_t
 * @param pkgVersion an unsigned int, the SBML 'Qual' package Version to assign to this
 * FluxBound_t
 *
 * @return a pointer to the newly created FluxBound_t structure.
 *
 * @memberof FluxBound_t
 */
LIBSBML_EXTERN
FluxBound_t *
FluxBound_create(unsigned int level, unsigned int version, unsigned int pkgVersion);


/**
 * Takes an FluxBound_t structure and returns its identifier.
 *
 * @param fb the FluxBound_t structure whose identifier is sought
 * 
 * @return the identifier of the given FluxBound_t, as a pointer to a string.
 *
 * @memberof FluxBound_t
 */
LIBSBML_EXTERN
const char *
FluxBound_getId(FluxBound_t * fb);


/**
 * Predicate returning @c true or @c false depending on whether the given
 * FluxBound_t structure's identifier is set.
 *
 * @param fb the FluxBound_t structure to query
 * 
 * @return @c non-zero (true) if the "id" attribute of the given
 * FluxBound_t structure is set, zero (false) otherwise.
 *
 * @memberof FluxBound_t
 */
LIBSBML_EXTERN
int
FluxBound_isSetId(FluxBound_t * fb);


/**
 * Assigns the identifier of an FluxBound_t structure.
 *
 * This makes a copy of the string passed in the param @p sid.
 *
 * @param fb the FluxBound_t structure to set.
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
 * @memberof FluxBound_t
 */
LIBSBML_EXTERN
int
FluxBound_setId(FluxBound_t * fb, const char * sid);


/**
 * Unsets the "id" attribute of the given FluxBound_t structure.
 *
 * @param fb the FluxBound_t structure to unset
 *
 * @return integer value indicating success/failure of the
 * function.  @if clike The value is drawn from the
 * enumeration #OperationReturnValues_t. @endif@~ The possible values
 * returned by this function are:
 * @li @sbmlconstant{LIBSBML_OPERATION_SUCCESS, OperationReturnValues_t}
 * @li @sbmlconstant{LIBSBML_OPERATION_FAILED, OperationReturnValues_t}
 *
 * @memberof FluxBound_t
 */
LIBSBML_EXTERN
int
FluxBound_unsetId(FluxBound_t * fb);

/**
 * Takes a FluxBound_t structure and returns its name.
 *
 * @param fb the FluxBound_t whose name is sought.
 *
 * @return the name of the given FluxBound_t, as a pointer to a string.
 *
 * @memberof FluxBound_t
 */
LIBSBML_EXTERN
const char *
FluxBound_getName(FluxBound_t * fb);


/**
 * Predicate returning @c true or @c false depending on whether the given
 * FluxBound_t structure's name is set.
 *
 * @param fb the FluxBound_t structure to query
 * 
 * @return @c non-zero (true) if the "name" attribute of the given
 * FluxBound_t structure is set, zero (false) otherwise.
 *
 * @memberof FluxBound_t
 */
LIBSBML_EXTERN
int
FluxBound_isSetName(FluxBound_t * fb);


/**
 * Sets the name of the given FluxBound_t to a copy of @p name.
 *
 * @param fb the FluxBound_t structure to set
 * @param name the name to assign to the given FluxBound_t's "name" attribute.
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
 * @memberof FluxBound_t
 */
LIBSBML_EXTERN
int
FluxBound_setName(FluxBound_t * fb, const char * name);


/**
 * Unsets the "name" attribute of the given FluxBound_t structure.
 *
 * @param fb the FluxBound_t structure to unset
 *
 * @return integer value indicating success/failure of the
 * function.  @if clike The value is drawn from the
 * enumeration #OperationReturnValues_t. @endif@~ The possible values
 * returned by this function are:
 * @li @sbmlconstant{LIBSBML_OPERATION_SUCCESS, OperationReturnValues_t}
 * @li @sbmlconstant{LIBSBML_OPERATION_FAILED, OperationReturnValues_t}
 *
 * @memberof FluxBound_t
 */
LIBSBML_EXTERN
int
FluxBound_unsetName(FluxBound_t * fb);


/**
 * Takes a FluxBound_t structure and returns its reaction.
 *
 * @param fb the FluxBound_t whose reaction is sought.
 *
 * @return the reaction of the given FluxBound_t, as a pointer to a string.
 *
 * @memberof FluxBound_t
 */
LIBSBML_EXTERN
const char *
FluxBound_getReaction(FluxBound_t * fb);


/**
 * Predicate returning @c true or @c false depending on whether the given
 * FluxBound_t structure's reaction is set.
 *
 * @param fb the FluxBound_t structure to query
 * 
 * @return @c non-zero (true) if the "reaction" attribute of the given
 * FluxBound_t structure is set, zero (false) otherwise.
 *
 * @memberof FluxBound_t
 */
LIBSBML_EXTERN
int
FluxBound_isSetReaction(FluxBound_t * fb);


/**
 * Sets the reaction of the given FluxBound_t to a copy of @p reaction.
 *
 * @param fb the FluxBound_t structure to set
 * @param reaction the reaction to assign to the given FluxBound_t's "reaction" attribute.
 *
 * @return integer value indicating success/failure of the
 * function.  @if clike The value is drawn from the
 * enumeration #OperationReturnValues_t. @endif@~ The possible values
 * returned by this function are:
 * @li @sbmlconstant{LIBSBML_OPERATION_SUCCESS, OperationReturnValues_t}
 * @li @sbmlconstant{LIBSBML_INVALID_ATTRIBUTE_VALUE, OperationReturnValues_t}
 *
 * @note Using this function with the name set to NULL is equivalent to
 * unsetting the "reaction" attribute.
 *
 * @memberof FluxBound_t
 */
LIBSBML_EXTERN
int
FluxBound_setReaction(FluxBound_t * fb, const char * reaction);


/**
 * Unsets the "reaction" attribute of the given FluxBound_t structure.
 *
 * @param fb the FluxBound_t structure to unset
 *
 * @return integer value indicating success/failure of the
 * function.  @if clike The value is drawn from the
 * enumeration #OperationReturnValues_t. @endif@~ The possible values
 * returned by this function are:
 * @li @sbmlconstant{LIBSBML_OPERATION_SUCCESS, OperationReturnValues_t}
 * @li @sbmlconstant{LIBSBML_OPERATION_FAILED, OperationReturnValues_t}
 *
 * @memberof FluxBound_t
 */
LIBSBML_EXTERN
int
FluxBound_unsetReaction(FluxBound_t * fb);


/**
 * Takes a FluxBound_t structure and returns its operation.
 *
 * @param fb the FluxBound_t whose operation is sought.
 *
 * @return the operation of the given FluxBound_t, as a pointer to a string.
 *
 * @memberof FluxBound_t
 */
LIBSBML_EXTERN
const char *
FluxBound_getOperation(FluxBound_t * fb);


/**
 * Predicate returning @c true or @c false depending on whether the given
 * FluxBound_t structure's operation is set.
 *
 * @param fb the FluxBound_t structure to query
 * 
 * @return @c non-zero (true) if the "operation" attribute of the given
 * FluxBound_t structure is set, zero (false) otherwise.
 *
 * @memberof FluxBound_t
 */
LIBSBML_EXTERN
int
FluxBound_isSetOperation(FluxBound_t * fb);


/**
 * Sets the operation of the given FluxBound_t to a copy of @p operation.
 *
 * @param fb the FluxBound_t structure to set
 * @param operation the operation to assign to the given FluxBound_t's "operation" attribute.
 *
 * @return integer value indicating success/failure of the
 * function.  @if clike The value is drawn from the
 * enumeration #OperationReturnValues_t. @endif@~ The possible values
 * returned by this function are:
 * @li @sbmlconstant{LIBSBML_OPERATION_SUCCESS, OperationReturnValues_t}
 * @li @sbmlconstant{LIBSBML_INVALID_ATTRIBUTE_VALUE, OperationReturnValues_t}
 *
 * @note Using this function with the name set to NULL is equivalent to
 * unsetting the "operation" attribute.
 *
 * @memberof FluxBound_t
 */
LIBSBML_EXTERN
int
FluxBound_setOperation(FluxBound_t * fb, const char * operation);


/**
 * Unsets the "operation" attribute of the given FluxBound_t structure.
 *
 * @param fb the FluxBound_t structure to unset
 *
 * @return integer value indicating success/failure of the
 * function.  @if clike The value is drawn from the
 * enumeration #OperationReturnValues_t. @endif@~ The possible values
 * returned by this function are:
 * @li @sbmlconstant{LIBSBML_OPERATION_SUCCESS, OperationReturnValues_t}
 * @li @sbmlconstant{LIBSBML_OPERATION_FAILED, OperationReturnValues_t}
 *
 * @memberof FluxBound_t
 */
LIBSBML_EXTERN
int
FluxBound_unsetOperation(FluxBound_t * fb);


/**
 * Takes a FluxBound_t structure and returns its value.
 *
 * @param fb the FluxBound_t whose value is sought.
 *
 * @return the value attribute of the given FluxBound_t, as a @c double.
 *
 * @memberof FluxBound_t
 */
LIBSBML_EXTERN
double
FluxBound_getValue(FluxBound_t * fb);


/**
 * Predicate returning @c true or @c false depending on whether the given
 * FluxBound_t structure's value is set.
 *
 * @param fb the FluxBound_t structure to query
 * 
 * @return @c non-zero (true) if the "value" attribute of the given
 * FluxBound_t structure is set, zero (false) otherwise.
 *
 * @memberof FluxBound_t
 */
LIBSBML_EXTERN
int
FluxBound_isSetValue(FluxBound_t * fb);


/**
 * Sets the "value" attribute of the given FluxBound_t
 * structure.
 *
 * @param fb the FluxBound_t structure
 * 
 * @param value the value of value to assign to the "value" attribute
 *
 * @return integer value indicating success/failure of the
 * function.  @if clike The value is drawn from the
 * enumeration #OperationReturnValues_t. @endif@~ The possible values
 * returned by this function are:
 * @li @sbmlconstant{LIBSBML_OPERATION_SUCCESS, OperationReturnValues_t}
 * @li @sbmlconstant{LIBSBML_UNEXPECTED_ATTRIBUTE, OperationReturnValues_t}
 *
 * @memberof FluxBound_t
 */
LIBSBML_EXTERN
int
FluxBound_setValue(FluxBound_t * fb, double value);


/**
 * Unsets the "value" attribute of the given FluxBound_t structure.
 *
 * @param fb the FluxBound_t structure to unset
 *
 * @return integer value indicating success/failure of the
 * function.  @if clike The value is drawn from the
 * enumeration #OperationReturnValues_t. @endif@~ The possible values
 * returned by this function are:
 * @li @sbmlconstant{LIBSBML_OPERATION_SUCCESS, OperationReturnValues_t}
 * @li @sbmlconstant{LIBSBML_OPERATION_FAILED, OperationReturnValues_t}
 *
 * @memberof FluxBound_t
 */
LIBSBML_EXTERN
int
FluxBound_unsetValue(FluxBound_t * fb);

/**
 * Returns the string version of the provided FluxBoundOperation_t enumeration.
 *
 * @param type The FluxBoundOperation_t enumeration to convert
 *
 * @return A string corresponding to the given effect:  "lessEqual", 
 * "greaterEqual", "equal", or NULL if the value is FLUXBOUND_OPERATION_UNKNOWN 
 * or another invalid enumeration value.
 *
 * @note In an earlier version of this specification, "less" and "greater" were
 * options that were dropped in the final version of the specification.
 * Accordingly, "less" is always converted to "lessEqual", and "greater" is 
 * always converted to "greaterEqual".
 *
 * @memberof FluxBound_t
 */
LIBSBML_EXTERN
const char* 
FluxBoundOperation_toString(FluxBoundOperation_t type);


/**
 * Returns the FluxBoundOperation_t enumeration corresponding to 
 * the given string, or FLUXBOUND_OPERATION_UNKNOWN if there is 
 * no such match.  The matching is case-sensitive:  "lessEqual" will 
 * return FLUXBOUND_OPERATION_LESS_EQUAL, but "lessequal" will return 
 * FLUXBOUND_OPERATION_UNKNOWN.
 *
 * @param s The string to convert to an FluxBoundOperation_t
 *
 * @return The corresponding FluxBoundOperation_t, or 
 * FLUXBOUND_OPERATION_UNKNOWN if no match found.
 *
 * @note In an earlier version of this specification, "less" and "greater" were
 * options that were dropped in the final version of the specification.
 * Accordingly, "less" is always converted to "lessEqual", and "greater" is 
 * always converted to "greaterEqual".
 *
 * @memberof FluxBound_t
 */
LIBSBML_EXTERN
FluxBoundOperation_t 
FluxBoundOperation_fromString(const char* s);


/**
 * Predicate returning @c true (non-zero) or @c false (zero) depending on whether the given
 * FluxBoundOperation_t is valid.
 *
 * @param type the FluxBoundOperation_t enumeration to query
 * 
 * @return @c non-zero (true) if the FluxBoundOperation_t is
 * FLUXBOUND_OPERATION_LESS_EQUAL, FLUXBOUND_OPERATION_GREATER_EQUAL, 
 * FLUXBOUND_OPERATION_LESS, FLUXBOUND_OPERATION_GREATER, or 
 * FLUXBOUND_OPERATION_EQUAL;
 * zero (false) otherwise (including FLUXBOUND_OPERATION_UNKNOWN).
 *
 * @note In an earlier version of this specification, "less" and "greater" were
 * options that were dropped in the final version of the specification.
 * Accordingly, "less" is always converted to "lessEqual", and "greater" is 
 * always converted to "greaterEqual".
 *
 * @memberof FluxBound_t
 */
LIBSBML_EXTERN
int 
FluxBoundOperation_isValidFluxBoundOperation(FluxBoundOperation_t type);


/**
 * Predicate returning @c true (non-zero) or @c false (zero) depending 
 * on whether the given string is a valid FluxBoundOperation_t.  
 * The matching is case-sensitive:  "lessEqual" will return @c true, but 
 * "lessequal" will return @c false.
 *
 * @param s The string to query
 * 
 * @return @c non-zero (true) if the string is
 * "lessEqual", "greaterEqual", "less", "greater", or "equal"; zero (false) otherwise.
 *
 * @note In an earlier version of this specification, "less" and "greater" were
 * options that were dropped in the final version of the specification.
 * Accordingly, "less" is always converted to "lessEqual", and "greater" is 
 * always converted to "greaterEqual".
 *
 * @memberof FluxBound_t
 */
LIBSBML_EXTERN
int 
FluxBoundOperation_isValidFluxBoundOperationString(const char* s);


END_C_DECLS
LIBSBML_CPP_NAMESPACE_END


#endif  /* !SWIG */
#endif  /* FluxBound_H__ */

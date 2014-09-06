/**
 * @file    GeneAssociation.h
 * @brief   Definition of GeneAssociation, the SBase derived class of the fbc package.
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
 * @class GeneAssociation
 * @sbmlbrief{fbc} Proposed annotation for gene associations.
 *
 * Gene associations are not part of the core FBC specification, but rather
 * are a proposed annotation.
 */ 

#ifndef GeneAssociation_H__
#define GeneAssociation_H__

#include <sbml/common/extern.h>
#include <sbml/common/sbmlfwd.h>
#include <sbml/packages/fbc/common/fbcfwd.h>

#ifdef __cplusplus

#include <string>

#include <sbml/SBase.h>
#include <sbml/xml/XMLNode.h>
#include <sbml/ListOf.h>
#include <sbml/packages/fbc/extension/FbcExtension.h>
#include <sbml/packages/fbc/sbml/Association.h>

LIBSBML_CPP_NAMESPACE_BEGIN


class LIBSBML_EXTERN GeneAssociation : public SBase
{
protected:
  /** @cond doxygenLibsbmlInternal */
  std::string mId;
  std::string mReaction;
  Association* mAssociation;
  /** @endcond */

public:

  /**
   * Creates a new GeneAssociation with the given @p level, @p version, and @p pkgVersion.
   */
  GeneAssociation(unsigned int level      = FbcExtension::getDefaultLevel(),
                  unsigned int version    = FbcExtension::getDefaultVersion(),
                  unsigned int pkgVersion = FbcExtension::getDefaultPackageVersion());

  /**
   * Creates a new GeneAssociation with the given @p node and FbcPkgNamespaces @p fbcns.
   */
  GeneAssociation(const XMLNode& node, FbcPkgNamespaces* fbcns);

  /**
   * Creates a new GeneAssociation with the given FbcPkgNamespaces object.
   */
  GeneAssociation(FbcPkgNamespaces* fbcns);


  /**
   * Copy constructor.
   */
  GeneAssociation(const GeneAssociation& source);


  /**
   * Assignment operator.
   */
  GeneAssociation& operator=(const GeneAssociation& source);


  /**
   * Destructor.
   */ 
  virtual ~GeneAssociation ();

  
  /**
   * Returns the string of the "id" attribute of this GeneAssociation.
   *
   * @return the string of the "id" attribute of this GeneAssociation.
   */
  virtual const std::string& getId () const;


  /**
   * Predicate returning @c true or @c false depending on whether this
   * GeneAssociation's "id" attribute has been set.
   *
   * @return @c true if this GeneAssociation's "id" attribute has been set, 
   * otherwise @c false is returned.
   */
  virtual bool isSetId () const;

  
  /**
   * Sets the SIdRef string of the "id" attribute of this GeneAssociation.
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
   * Unsets the value of the "id" attribute of this GeneAssociation.
   *
   * @return integer value indicating success/failure of the
   * operation. The possible return values are:
   * @li @sbmlconstant{LIBSBML_OPERATION_SUCCESS, OperationReturnValues_t}
   * @li @sbmlconstant{LIBSBML_OPERATION_FAILED, OperationReturnValues_t}
   */
  virtual int unsetId ();

  
  /**
   * Returns the string of the "reaction" attribute of this GeneAssociation.
   *
   * @return the string of the "reaction" attribute of this GeneAssociation.
   */
  virtual const std::string& getReaction () const;


  /**
   * Predicate returning @c true or @c false depending on whether this
   * GeneAssociation's "reaction" attribute has been set.
   *
   * @return @c true if this GeneAssociation's "reaction" attribute has been set, 
   * otherwise @c false is returned.
   */
  virtual bool isSetReaction () const;

  
  /**
   * Sets the SIdRef string of the "reaction" attribute of this GeneAssociation.
   *
   * @param reaction a SIdRef string to be set.
   *
   * @return integer value indicating success/failure of the
   * operation. The possible return values are:
   * @li @sbmlconstant{LIBSBML_OPERATION_SUCCESS, OperationReturnValues_t}
   * @li @sbmlconstant{LIBSBML_INVALID_ATTRIBUTE_VALUE, OperationReturnValues_t}
   */
  virtual int setReaction (const std::string& reaction);


  /**
   * Unsets the value of the "id" attribute of this GeneAssociation.
   *
   * @return integer value indicating success/failure of the
   * operation. The possible return values are:
   * @li @sbmlconstant{LIBSBML_OPERATION_SUCCESS, OperationReturnValues_t}
   * @li @sbmlconstant{LIBSBML_OPERATION_FAILED, OperationReturnValues_t}
   */
  virtual int unsetReaction ();

  
  /** 
   * Creates a new association, sets it to this element and returns it. 
   */
  Association* createAssociation();

  /**
   * Returns Association object of this GeneAssociation.
   *
   * @return Association object of this GeneAssociation.
   */
  virtual const Association* getAssociation () const;

  /**
   * Returns Association object of this GeneAssociation.
   *
   * @return Association object of this GeneAssociation.
   */
  virtual Association* getAssociation ();


  /**
   * Predicate returning @c true or @c false depending on whether this
   * GeneAssociation's "association" element has been set.
   *
   * @return @c true if this GeneAssociation's "association" element has been set, 
   * otherwise @c false is returned.
   */
  virtual bool isSetAssociation () const;

  
  /**
   * Sets the Association object of this GeneAssociation.
   *
   * @param association a Association object to be set.
   *
   * @return integer value indicating success/failure of the
   * operation. The possible return values are:
   * @li @sbmlconstant{LIBSBML_OPERATION_SUCCESS, OperationReturnValues_t}
   * @li @sbmlconstant{LIBSBML_INVALID_ATTRIBUTE_VALUE, OperationReturnValues_t}
   */
  virtual int setAssociation (const Association* association);


  /**
   * Unsets the Association object of this GeneAssociation.
   *
   * @return integer value indicating success/failure of the
   * operation. The possible return values are:
   * @li @sbmlconstant{LIBSBML_OPERATION_SUCCESS, OperationReturnValues_t}
   * @li @sbmlconstant{LIBSBML_OPERATION_FAILED, OperationReturnValues_t}
   */
  virtual int unsetAssociation ();


  /**
   * Returns the XML element name of
   * this SBML object.
   *
   * @return the name of this element, as a text string.
   */
  virtual const std::string& getElementName () const ;


  /**
   * Creates and returns a deep copy of this GeneAssociation.
   *
   * @return a (deep) copy of this GeneAssociation.
   */
  virtual GeneAssociation* clone () const;


  /**
   * Returns the libSBML type code of this object instance.
   *
   * @copydetails doc_what_are_typecodes
   *
   * @return the SBML type code for this object:
   * @sbmlconstant{SBML_FBC_GENEASSOCIATION, SBMLFbcTypeCode_t}
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
  
  
  /**
   * Creates an XMLNode object from this.
   */
  XMLNode toXML() const;

  
protected:
  /** @cond doxygenLibsbmlInternal */
  /**
   * Creates and returns an SBML object corresponding to the next
   * XMLToken in the @p stream, or NULL if the token was not recognized.
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
 * @class ListOfGeneAssociations
 * @sbmlbrief{fbc} A list of GeneAssociation objects.
 * 
 * The ListOfGeneAssociations is a container for the GeneAssociation elements of the proposed Model annotation, and is not part of the official FBC specification.
 * 
 * @copydetails doc_what_is_listof
 *
 * @see GeneAssociation
 */
class LIBSBML_EXTERN ListOfGeneAssociations : public ListOf
{
public:

  /**
   * Creates and returns a deep copy of this ListOfGeneAssociations.
   *
   * @return a (deep) copy of this ListOfGeneAssociations.
   */
  virtual ListOfGeneAssociations* clone () const;


  /**
   * Creates a new ListOfGeneAssociations with the given level, version, and package version.
   */
   ListOfGeneAssociations(unsigned int level      = FbcExtension::getDefaultLevel(), 
                 unsigned int version    = FbcExtension::getDefaultVersion(), 
                 unsigned int pkgVersion = FbcExtension::getDefaultPackageVersion());


  /**
   * Creates a new ListOfGeneAssociations with the given FbcPkgNamespaces object.
   */
   ListOfGeneAssociations(FbcPkgNamespaces* fbcns);


  /**
   * Get a GeneAssociation from the ListOfGeneAssociations.
   *
   * @param n the index number of the GeneAssociation to get.
   * 
   * @return the nth GeneAssociation in this ListOfGeneAssociations.
   *
   * @see size()
   */
  virtual GeneAssociation * get(unsigned int n); 


  /**
   * Get a GeneAssociation from the ListOfGeneAssociations.
   *
   * @param n the index number of the GeneAssociation to get.
   * 
   * @return the nth GeneAssociation in this ListOfGeneAssociations.
   *
   * @see size()
   */
  virtual const GeneAssociation * get(unsigned int n) const; 

  /**
   * Get a GeneAssociation from the ListOfGeneAssociations
   * based on its identifier.
   *
   * @param sid a string representing the identifier 
   * of the GeneAssociation to get.
   * 
   * @return GeneAssociation in this ListOfGeneAssociations
   * with the given @p sid or @c NULL if no such
   * GeneAssociation exists.
   *
   * @see get(unsigned int n)
   * @see size()
   */
  virtual GeneAssociation* get (const std::string& sid);


  /**
   * Get a GeneAssociation from the ListOfGeneAssociations
   * based on its identifier.
   *
   * @param sid a string representing the identifier 
   * of the GeneAssociation to get.
   * 
   * @return GeneAssociation in this ListOfGeneAssociations
   * with the given @p sid or @c NULL if no such
   * GeneAssociation exists.
   *
   * @see get(unsigned int n)
   * @see size()
   */
  virtual const GeneAssociation* get (const std::string& sid) const;


  /**
   * Removes the nth item from this ListOfGeneAssociations items and returns a pointer to
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
  virtual GeneAssociation* remove (unsigned int n);


  /**
   * Removes item in this ListOfGeneAssociations items with the given identifier.
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
  virtual GeneAssociation* remove (const std::string& sid);


  /**
   * Returns the libSBML type code for the SBML objects
   * contained in this ListOf object.
   * 
   * @copydetails doc_what_are_typecodes
   *
   * @return the SBML type code for objects contained in this list:
   * @sbmlconstant{SBML_FBC_GENEASSOCIATION, SBMLTypeCode_t} (default).
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
   * Create and return a geneAssociation object, if present.
   *
   * @return the SBML object corresponding to next XMLToken in the
   * XMLInputStream or NULL if the token was not recognized.
   */
  virtual SBase* createObject (XMLInputStream& stream);
  /** @endcond */
};

/** @cond doxygenLibsbmlInternal */
/**
 * Used by ListOfGeneAssociations::get() to lookup an SBase based by its 
 * symbol
 */
#ifndef SWIG
template<>
struct IdEq<GeneAssociation> : public std::unary_function<SBase*, bool>
{
  const std::string& id;

  IdEq (const std::string& id) : id(id) { }
  bool operator() (SBase* sb) 
       { return static_cast <GeneAssociation*> (sb)->getId() == id; }
};
#endif
/** @endcond */

LIBSBML_CPP_NAMESPACE_END

#endif /* __cplusplus */


#ifndef SWIG

LIBSBML_CPP_NAMESPACE_BEGIN
BEGIN_C_DECLS

/*
 * C API will be added here.
 */

END_C_DECLS
LIBSBML_CPP_NAMESPACE_END


#endif  /* !SWIG */
#endif  /* GeneAssociation_H__ */

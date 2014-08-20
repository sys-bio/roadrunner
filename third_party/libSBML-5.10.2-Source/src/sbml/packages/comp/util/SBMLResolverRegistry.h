/**
 * @file    SBMLResolverRegistry.h
 * @brief   Definition of SBMLResolverRegistry, a registry of available resolvers.
 * @author  Frank Bergmann
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
 *
 * @class SBMLResolverRegistry
 * @sbmlbrief{comp} Registry of all SBML resolvers.
 *
 * @htmlinclude libsbml-facility-only-warning.html
 *
 * LibSBML provides facilities for resolving SBML documents in various ways
 * from a given URI. Resolvers are implemented as objects derived from the
 * class SBMLResolver.
 *
 * The resolver registry maintains a list of known resolvers and provides
 * methods for discovering them.  It is implemented as a singleton object of
 * class SBMLResolverRegistry.  Callers can use the method
 * SBMLResolverRegistry::getNumResolvers() to find out how many resolvers are
 * registered, then use SBMLResolverRegistry::getResolverByIndex(@if java
 * int@endif) to iterate over each one;
 *
 * @see SBMLFileResolver
 */

#ifndef SBMLResolverRegistry_h
#define SBMLResolverRegistry_h


#include <sbml/common/sbmlfwd.h>
#include <sbml/packages/comp/util/SBMLResolver.h>


#ifdef __cplusplus

#include <map>
#include <vector>
#include <string>


LIBSBML_CPP_NAMESPACE_BEGIN


class LIBSBML_EXTERN SBMLResolverRegistry
{
public:

  /**
   * Returns the singleton instance for the resolver registry.
   *
   * Prior to using the registry, callers have to obtain a copy of the
   * registry.  This static method provides the means for doing that.
   *
   * @return the singleton for the resolver registry.
   */
  static SBMLResolverRegistry& getInstance();


  /**
   * Adds the given resolver to the registry of SBML resolvers.
   *
   * @param resolver the resolver to add to the registry.
   *
   * @return integer value indicating the success/failure of the operation.
   * @if clike The value is drawn from the enumeration
   * #OperationReturnValues_t. @endif@~ The possible values are:
   * @li @sbmlconstant{LIBSBML_OPERATION_SUCCESS, OperationReturnValues_t}
   * @li @sbmlconstant{LIBSBML_INVALID_OBJECT, OperationReturnValues_t}
   */
  int addResolver (const SBMLResolver* resolver);


  /**
   * Removes the resolver with the given index.
   * 
   * @param index the index of the resolver to be removed
   *
   * @return integer value indicating the success/failure of the operation.
   * @if clike The value is drawn from the enumeration
   * #OperationReturnValues_t. @endif@~ The possible values are:
   * @li @sbmlconstant{LIBSBML_OPERATION_SUCCESS, OperationReturnValues_t}
   * @li @sbmlconstant{LIBSBML_INVALID_OBJECT, OperationReturnValues_t}
   */
  int removeResolver(int index);


  /**
   * Returns the resolver with the given index number.
   *
   * Resolvers are given arbitrary index numbers by the registry.  Callers
   * can use the method SBMLResolverRegistry::getNumResolvers() to find
   * out how many resolvers are registered, then use this method to
   * iterate over the list and obtain each one in turn.
   *
   * @param index the zero-based index of the resolver to fetch.
   *
   * @return the resolver with the given index number, or @c NULL if the
   * number is less than @c 0 or there is no resolver at the given index
   * position.
   */
  SBMLResolver* getResolverByIndex(int index) const;


  /**
   * Returns the number of resolvers known by the registry.
   *
   * @return the number of registered resolvers.
   *
   * @see getResolverByIndex(@if java int@endif)
   */
  int getNumResolvers() const;


  /**
   * Destructor
   */
  virtual ~SBMLResolverRegistry();


  /**
   * Resolves the document for the given URI.
   *
   * @param uri the URI to the target document
   * @param baseUri base URI, in case the URI is a relative one
   *
   * @return  the document, if this resolver can resolve the document or NULL.
   */
  virtual SBMLDocument* resolve(const std::string &uri, const std::string& baseUri="") const;

  
  /**
   * Resolves the full URI for the given URI without actually reading the
   * document.
   *
   * @param uri the URI to the target document
   * @param baseUri base URI, in case the URI is a relative one
   *
   * @return  the full URI to the document, if this resolver can resolve the document or NULL.
   */
  virtual SBMLUri* resolveUri(const std::string &uri, const std::string& baseUri="") const;


protected:

  /** @cond doxygenLibsbmlInternal */
  /**
   * protected constructor, use the getInstance() method to access the registry.
   */
  SBMLResolverRegistry();
  /** @endcond */


protected:
  /** @cond doxygenLibsbmlInternal */
  std::vector<const SBMLResolver*>  mResolvers;
  /** @endcond */
};

LIBSBML_CPP_NAMESPACE_END

#endif /* __cplusplus */

#ifndef SWIG

LIBSBML_CPP_NAMESPACE_BEGIN
BEGIN_C_DECLS

END_C_DECLS
LIBSBML_CPP_NAMESPACE_END

#endif  /* !SWIG */
#endif /* !SBMLResolverRegistry_h */


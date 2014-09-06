/**
 * @file    FbcValidator.h
 * @brief   Base class for SBML CompValidators
 * @author  Sarah Keating
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
 * Copyright (C) 2009-2013 jointly by the following organizations: 
 *     1. California Institute of Technology, Pasadena, CA, USA
 *     2. EMBL European Bioinformatics Institute (EMBL-EBI), Hinxton, UK
 *
 * This library is free software; you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation.  A copy of the license agreement is provided
 * in the file named "LICENSE.txt" included with this software distribution and
 * also available online as http://sbml.org/software/libsbml/license.html
 * ---------------------------------------------------------------------- -->
 *
 * @class FbcValidator
 * @sbmlbrief{fbc} Entry point for package validation.
 * 
 * @htmlinclude not-sbml-warning.html
 *
 * LibSBML implements facilities for verifying that a given SBML document
 * is valid according to the SBML specifications; it also exposes the
 * validation interface so that user programs and SBML Level&nbsp;3 package
 * authors may use the facilities to implement new validators.  There are
 * two main interfaces to libSBML's validation facilities, based on the
 * classes CompValidator and SBMLCompValidator.
 *
 * The CompValidator class is the basis of the system for validating an SBML
 * document against the validation rules defined in the SBML
 * specifications.  The scheme used by CompValidator relies is compact and uses
 * the @em visitor programming pattern, but it relies on C/C++ features and
 * is not directly accessible from language bindings.  SBMLCompValidator offers
 * a framework for straightforward class-based extensibility, so that user
 * code can subclass SBMLCompValidator to implement new validation systems,
 * different validators can be introduced or turned off at run-time, and
 * interfaces can be provided in the libSBML language bindings.
 * SBMLCompValidator can call CompValidator functionality internally (as is the
 * case in the current implementation of SBMLInternalCompValidator) or use
 * entirely different implementation approaches, as necessary.
 */

#ifndef FbcValidator_h
#define FbcValidator_h


#ifdef __cplusplus


/** @cond doxygenLibsbmlInternal */

#include <list>
#include <string>

/** @endcond */


#include <sbml/SBMLError.h>
#include <sbml/validator/Validator.h>

LIBSBML_CPP_NAMESPACE_BEGIN

class VConstraint;
struct FbcValidatorConstraints;
class SBMLDocument;


class FbcValidator : public Validator
{
public:

  /**
   * Constructor; creates a new FbcValidator object for the given
   * category of validation.
   *
   * @param category code indicating the kind of validations that this
   * validator will perform.  The category code value must be
   * @if clike taken from the enumeration #SBMLErrorCategory_t @endif@~
   * @if java one of of the values from the set of constants whose names
   * begin with the characters <code>LIBSBML_CAT_</code> in the interface
   * class {@link libsbmlConstants}.@endif@~
   * @if python one of of the values from the set of constants whose names
   * begin with the characters <code>LIBSBML_CAT_</code> in the interface
   * class @link libsbml libsbml@endlink.@endif@~
   */
  FbcValidator ( SBMLErrorCategory_t category = LIBSBML_CAT_SBML );


  /**
   * Destroys this FbcValidator object.
   */
  virtual ~FbcValidator ();


  /**
   * Initializes this FbcValidator object.
   *
   * When creating a subclass of FbcValidator, override this method to add
   * your own validation code.
   */
  virtual void init () = 0;


  /**
   * Adds the given VContraint object to this validator.
   *
   * @param c the VConstraint ("validator constraint") object to add.
   */
  virtual void addConstraint (VConstraint* c);


  /**
   * Validates the given SBML document.
   *
   * @param d the SBMLDocument object to be validated.
   *
   * @return the number of validation failures that occurred.  The objects
   * describing the actual failures can be retrieved using getFailures().
   */
  virtual unsigned int validate (const SBMLDocument& d);


  /**
   * Validates the SBML document located at the given file name.
   *
   * @param filename the path to the file to be read and validated.
   *
   * @return the number of validation failures that occurred.  The objects
   * describing the actual failures can be retrieved using getFailures().
   */
  virtual unsigned int validate (const std::string& filename);


protected:
  /** @cond doxygenLibsbmlInternal */

  FbcValidatorConstraints* mFbcConstraints;

  friend class FbcValidatingVisitor;

  /** @endcond */
};

LIBSBML_CPP_NAMESPACE_END

#endif  /* __cplusplus */
#endif  /* FbcValidator_h */

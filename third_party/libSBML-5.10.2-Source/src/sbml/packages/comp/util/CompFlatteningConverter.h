/**
 * @file    CompFlatteningConverter.h
 * @brief   Definition of a first flattening converter.
 * @author  Frank T. Bergmann
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
 * ---------------------------------------------------------------------- -->
 *
 * @class CompFlatteningConverter
 * @sbmlbrief{comp} "Flattens" a model, removing composition.
 *
 * @htmlinclude libsbml-facility-only-warning.html
 *
 * This converter translates a hierarchical model defined with the 
 * Hierarchical %Model Composition package to a 'flattened' version
 * of the same model defined without use of the Hierarchical %Model
 * Composition package.  All of the mathematics of the model will
 * remain, but the hierarchical structure will be removed.
 *
 * Specifically, the following actions are carried out:
 * @li Each Submodel is instantiated, that is, copies of the Model it points to are created.  The IDs of their elements are changed such that the ID of the Submodel is prepended to all IDs, plus a digit if needed to ensure uniqueness, plus two underscores ("__").  Typically, this results in IDs of the form "SUBMODELID__ORIGINALID".  If that instantiated Model itself has Submodel children, they too are instantiated.
 * @li All deleted elements are removed from the Model and all instantiated Submodels.
 * @li All replaced elements are removed from the Model and all instantiated Submodels.
 * @li All references to replaced elements are changed to point to the replacement element.
 * @li All remaining elements are placed in a single Model object, which is made the new child of the SBMLDocument object.  The original Model, ModelDefinition, and ExternalModelDefinition objects are all deleted.
 *
 * Note that this means that if this converter is successful, all old pointers
 * to the document's Model and any of its children will be rendered 
 * obsolete, and will no longer work.  
 *
 * If package information is present, the same rules apply to that package,
 * assuming a flattening implementation exists for that package information.
 * If not, the behavior of the converter depends on the states of the
 * @c 'abortIfUnflattenable' and @c 'stripUnflattenablePackages' settings.  Specifically:
 *
 * @li If @c 'abortIfUnflattenable' is set to @c 'all', if any package information
 *     is found for which there is no flattening algorithm available, the
 *     converter will abort, return failure, and avoid changing the original
 *     SBMLDocument
 *
 * @li If @c 'abortIfUnflattenable' is set to @c 'requiredOnly' (the default), if
 *     any package information is found for a package set @c 'required=true'
 *     for which there is no flattening algorithm available, the converter
 *     will abort, return failure, and avoid changing the original
 *     SBMLDocument.  Package information from packages set @c 'required=false'
 *     for which there is no flattening algorithm available will be ignored,
 *     and that information will stay or be removed according to the 
 *     status of the @c 'stripUnflattenablePackages' setting.
 * 
 * @li If @c 'abortIfUnflattenable' is set to @c 'none', all unflattenable packages
 *     are ignored, and their information will stay or be removed according to the 
 *     status of the @c 'stripUnflattenablePackages' setting.
 *
 * For all cases above where the package is ignored, the @c 'stripUnflattenablePackages'
 * option is examined:
 *
 * @li If @c 'stripUnflattenablePackages' is set to @c 'true', any unflattenable 
 *     package information ignored because of the @c 'abortIfUnflattenable' option 
 *     will be removed from the SBMLDocument entirely, including the declaration 
 *     of the package's namespace.
 *
 * @li If @c 'stripUnflattenablePackages' is set to @c 'false' (the default), any
 *     ignored unflattenable package information will remain if it was present
 *     in the original Model object of the SBMLDocument.  Any package information
 *     from an instantiated Submodel that was not a child of a retained element
 *     will be lost.
 *
 * Other options are also available, though all have default values:
 *
 * @li @c 'basePath':
 *
 * @li @li If there are ExternalModelDefinitions that are to be instantiated in a
 *     flattened Submodel, the @c 'basePath' option may be set to a location where
 *     those external models may be found.  The default is the working directory
 *     ('.').
 *
 * @c 'leavePorts':
 *
 * @li If @c 'leavePorts' is set to @c 'false' (the default), the flattened model will
 *     have no Port elements in it.  If set to @c 'true', any Port objects not 
 *     referenced by any Replacement or Deletion will be left in the resulting 
 *     flattened Model.
 *
 * @c 'listModelDefinitions':
 * 
 * @li If @c 'listModelDefinitions' is set to @c 'false' (the default), no 
 *     ModelDefinition or ExternalModelDefinition objects will be present in the
 *     flattened SBMLDocument.  If set to @c 'true', they will remain, though they
 *     will no longer be referenced by any Submodel in the flattened Model
 *     child of the SBMLDocument.
 *
 * @note If both @c 'leavePorts' and @c 'listModelDefinitions' are set to @c 'false'
 * (which they are by default), the Hierarchical %Model Composition namespace will
 * be removed from the resulting SBMLDocument.
 *
 * @c 'performValidation':
 *
 * @li If @c 'performValidation' is set to @c 'true' (the default), the SBMLDocument
 *     will first be validated before flattening is attempted.  If there are any
 *     validation errors, those errors will be set on the SBMLDocument, which will
 *     remain otherwise unchanged, and the conversion attempt will return failure.
 *     Similarly, if the flattened Model is not valid, those validation errors will
 *     be added to the SBMLDocument, which will remain otherwise unchanged, and the
 *     conversion attempt will return failure.
 *
 * @li If @c 'performValidation' is set to @c 'false', the SBMLDocument will be flattened
 *     irrespective of any validation errors that may exist.  The conversion may yet
 *     fail if insurmountable errors are encountered in the course of trying to
 *     flatten the model (for instance, if an element is replaced by something that
 *     does not exist), but no separate validation steps are performed.
 *
 * @section CompFlatteningConverter-usage Configuration and use of SBMLIdConverter
 *
 * CompFlatteningConverter is enabled by creating a ConversionProperties
 * object with the option @c "flatten comp", and passing this properties
 * object to SBMLDocument::convert(@if java ConversionProperties@endif).
 * The converter also accepts the following options, all of
 * which are optional, and have default values:
 *
 * @li @c "abortIfUnflattenable": @c 'all', @c 'requiredOnly' (the default), or @c 'none'
 * @li @c "stripUnflattenablePackages": @c 'true' or @c 'false' (the default).
 * @li @c "basePath": A string representing the path where the converter should search for any ExternalModelDefinitions (default ".")
 * @li @c "leavePorts": @c 'true' or @c 'false' (the default).
 * @li @c "listModelDefinitions": @c 'true' or @c 'false' (the default).
 * @li @c "performValidation": @c 'true' (the default) or @c 'false'
 *
 * @copydetails doc_section_using_sbml_converters
 */

#ifndef CompFlatteningConverter_h
#define CompFlatteningConverter_h

#include <sbml/SBMLNamespaces.h>
#include <sbml/packages/comp/extension/CompModelPlugin.h>
#include <sbml/conversion/SBMLConverter.h>
#include <sbml/conversion/SBMLConverterRegister.h>


#ifdef __cplusplus


LIBSBML_CPP_NAMESPACE_BEGIN


class LIBSBML_EXTERN CompFlatteningConverter : public SBMLConverter
{
public:

  /** @cond doxygenLibsbmlInternal */
  
  /* register with the ConversionRegistry */
  static void init();  

  /** @endcond */


  /**
   * Constructor.
   */
  CompFlatteningConverter();


  /**
   * Copy constructor.
   */
  CompFlatteningConverter(const CompFlatteningConverter&);


  /**
   * Creates and returns a deep copy of this CompFlatteningConverter.
   * 
   * @return a (deep) copy of this CompFlatteningConverter.
   */
  virtual CompFlatteningConverter* clone() const;


  /**
   * Destroy this CompFlatteningConverter object.
   */
  virtual ~CompFlatteningConverter ();


  /**
   * This function determines whether a given converter matches the 
   * configuration properties given. 
   * 
   * @param props the properties to match
   * 
   * @return @c true if this converter is a match, @c false otherwise.
   */
  virtual bool matchesProperties(const ConversionProperties &props) const;


  /**
   * Performs the actual conversion.
   * 
   * @return status code represeting success/failure/conversion impossible
   */
  virtual int convert();


  /**
   * Returns the default properties of this converter.
   * 
   * A given converter exposes one or more properties that can be adjusted
   * in order to influence the behavior of the converter.  This method
   * returns the @em default property settings for this converter.  It is
   * meant to be called in order to discover all the settings for the
   * converter object.
   *
   * The properties for the CompFlatteningConverter are:
   * @li "flatten comp": the name of this converter
   * @li "basePath": the base directory to find external references in
   * @li "leavePorts": boolean indicating whether unused ports 
   *   should be listed in the flattened model; default = false
   * @li "listModelDefinitions": boolean indicating whether the model 
   *   definitions should be listed in the flattened model; default = false
   * @li "stripUnflattenablePackages": boolean indicating whether packages 
   *   that cannot be flattened should be removed; default = true
   * @li "performValidation": boolean indicating whether validation should be 
   *   performed. When @c true either an invalid source document or 
   *   an invalid flattened document will cause flattening to fail; default = true
   * @li "abortIfUnflattenable": string indicating the required status of
   *   any unflattenable packages that should cause flattening to fail.
   *   Possible values are "none", "requiredOnly" and "all"; default = requiredOnly
   *
   * @note previously there was an "ignorePackages" option; whose name
   * proved to be very misleading. This option has been deprecated and 
   * replaced by the "stripUnflattenablePackages" but will still work.
   *
   * @return the ConversionProperties object describing the default properties
   * for this converter.
   */
  virtual ConversionProperties getDefaultProperties() const;

private:

  int reconstructDocument(Model* flatmodel); 

  int reconstructDocument(Model* flatmodel, 
                          SBMLDocument &dummyDoc,  bool dummyRecon = false);

  void stripUnflattenablePackages();

  bool getLeavePorts() const;

  bool getLeaveDefinitions() const;

  bool getIgnorePackages() const;

  bool getStripUnflattenablePackages() const;

  bool getPerformValidation() const;

  bool getAbortForAll() const;

  bool getAbortForRequired() const;

  bool getAbortForNone() const;

  bool canBeFlattened();

  void restoreNamespaces();

  std::set<std::pair<std::string, std::string> > mDisabledPackages;


#ifndef SWIG
  typedef std::vector<bool>                     ValueSet;
  typedef std::map<const std::string, ValueSet> PackageValueMap;
  typedef PackageValueMap::iterator             PackageValueIter;
#endif

  PackageValueMap mPackageValues;

  void analyseDocument();

  bool getRequiredStatus(const std::string & package);

  bool getKnownStatus(const std::string& package);

  bool getFlattenableStatus(const std::string& package);

  bool haveUnknownRequiredPackages();

  bool haveUnknownUnrequiredPackages();

  bool haveUnflattenableRequiredPackages();

  bool haveUnflattenableUnrequiredPackages();

};

LIBSBML_CPP_NAMESPACE_END

#endif  /* __cplusplus */

  
#ifndef SWIG

LIBSBML_CPP_NAMESPACE_BEGIN
BEGIN_C_DECLS


END_C_DECLS
LIBSBML_CPP_NAMESPACE_END

#endif  /* !SWIG */
#endif  /* CompFlatteningConverter_h*/


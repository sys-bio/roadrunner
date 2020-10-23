/**
 * @file    ConservationExtension.h
 * @brief   Definition of ConservationExtension, the core module of groups package.
 * @author  Andy Somogyi
 *
 * The Conservation Extension performes moiety conservation analysis on
 * SBML documents.
 *
 * This file is based on the GroupsExtension, written by Akiya Jouraku
 *
 * This library is free software; you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation.  A copy of the license agreement is provided
 * in the file named "LICENSE.txt" included with this software distribution
 * and also available online as http://sbml.org/software/libsbml/license.html
 */

#ifndef ConservationExtension_h
#define ConservationExtension_h

#include "rrExporter.h"
#include <sbml/common/extern.h>
#include <sbml/SBMLTypeCodes.h>
#include <sbml/extension/SBMLExtension.h>
#include <sbml/extension/SBMLExtensionNamespaces.h>
#include <sbml/extension/SBMLExtensionRegister.h>

#include <string>
#include <vector>



namespace rr { namespace conservation {

#ifndef CONSERVATION_CREATE_NS
#define CONSERVATION_CREATE_NS(variable,sbmlns)\
  EXTENSION_CREATE_NS(ConservationPkgNamespaces,variable,sbmlns);
#endif

bool RR_DECLSPEC conservation_getInit();


class RR_DECLSPEC ConservationExtension : public libsbml::SBMLExtension
{
public:

  //---------------------------------------------------------------
  //
  // Required class methods
  //
  //---------------------------------------------------------------

  /**
   * Returns the package name of this extension.
   */
  static const std::string& getPackageName ();

  /**
   * Returns the default SBML Level this extension.
   */
  static unsigned int getDefaultLevel();

  /**
   * Returns the default SBML Version this extension.
   */
  static unsigned int getDefaultVersion();

  /**
   * Returns the default SBML version this extension.
   */
  static unsigned int getDefaultPackageVersion();

  /**
   * Returns URI of supported versions of this package.
   */
  static const std::string&  getXmlnsL3V1V1();


  /**
   * check if the Species is a conserved moiety,
   * a convenience function that just checks the ConservedMoietyPlugin
   */
  static bool getConservedMoiety(const libsbml::Species& s);

  /**
   * check if the Parameter is a conserved moiety,
   * a convenience function that just checks the ConservedMoietyPlugin
   */
  static bool getConservedMoiety(const libsbml::Parameter& p);


  /**
   * get the conserved moiety global parameter associated with this species
   */
  static std::string getConservedQuantity(const libsbml::Species& s);


  /**
   * get the conserved moiety global parameter associated with this species
   */
  static std::vector<std::string> getConservedQuantities(const libsbml::Species& s);


  /**
   * check if the document is already a conserved moeity document
   */
  static bool isConservedMoietyDocument(const libsbml::SBMLDocument*);



  /**
   * Constructor
   */
  ConservationExtension ();


  /**
   * Copy constructor.
   */
  ConservationExtension(const ConservationExtension&);


  /**
   * Destroy this object.
   */
  virtual ~ConservationExtension ();


  /**
   * Assignment operator for ConservationExtension.
   */
  ConservationExtension& operator=(const ConservationExtension&);


  /**
   * Creates and returns a deep copy of this ConservationExtension object.
   *
   * @return a (deep) copy of this SBase object
   */
  virtual ConservationExtension* clone () const;


  /**
   * Returns the name of this package ("groups")
   *
   * @pram the name of this package ("groups")
   */
  virtual const std::string& getName() const;


  /**
   * Returns the URI (namespace) of the package corresponding to the combination of
   * the given sbml level, sbml version, and package version.
   * Empty string will be returned if no corresponding URI exists.
   *
   * @param sbmlLevel the level of SBML
   * @param sbmlVersion the version of SBML
   * @param pkgVersion the version of package
   *
   * @return a string of the package URI
   */
  virtual const std::string& getURI(unsigned int sbmlLevel, unsigned int sbmlVersion,
                                    unsigned int pkgVersion) const;


  /**
   * Returns the SBML level with the given URI of this package.
   *
   * @param uri the string of URI that represents one of versions of layout package
   *
   * @return the SBML level with the given URI of this package. 0 will be returned
   * if the given URI is invalid.
   *
   */
  virtual unsigned int getLevel(const std::string &uri) const;


  /**
   * Returns the SBML version with the given URI of this package.
   *
   * @param uri the string of URI that represents one of versions of layout package
   *
   * @return the SBML version with the given URI of this package. 0 will be returned
   * if the given URI is invalid.
   */
  virtual unsigned int getVersion(const std::string &uri) const;


  /**
   * Returns the package version with the given URI of this package.
   *
   * @param uri the string of URI that represents one of versions of layout package
   *
   * @return the package version with the given URI of this package. 0 will be returned
   * if the given URI is invalid.
   */
  virtual unsigned int getPackageVersion(const std::string &uri) const;


  /**
   * Returns an SBMLExtensionNamespaces<ConservationExtension> object whose alias type is
   * LayoutPkgNamespace.
   * Null will be returned if the given uri is not defined in the layout package.
   *
   * @param uri the string of URI that represents one of versions of layout package
   *
   * @return an LayoutPkgNamespace object corresponding to the given uri. NULL will
   * be returned if the given URI is not defined in layout package.
   */
  virtual libsbml::SBMLNamespaces* getSBMLExtensionNamespaces(const std::string &uri) const;


  /**
   * This method takes a type code of groups package and returns a string representing
   * the code.
   */
  virtual const char* getStringFromTypeCode(int typeCode) const;


  /** @cond doxygen-libsbml-internal */
  /**
   * Initializes layout extension by creating an object of this class with
   * required SBasePlugin derived objects and registering the object
   * to the SBMLExtensionRegistry class.
   *
   * (NOTE) This function is automatically invoked when creating the following
   *        global object in ConservationExtension.cpp
   *
   *        static SBMLExtensionRegister<ConservationExtension> ConservationExtensionRegistry;
   *
   */

  static void init();

  /** @endcond doxygen-libsbml-internal */

};




// --------------------------------------------------------------------
//
// Required typedef definitions
//
// LayoutPkgNamespaces is derived from the SBMLNamespaces class and
// used when creating an object of SBase derived classes defined in
// layout package.
//
// --------------------------------------------------------------------

//
// (NOTE)
//
// SBMLExtensionNamespaces<ConservationExtension> must be instantiated
// in ConservationExtension.cpp for DLL.
//
typedef libsbml::SBMLExtensionNamespaces<ConservationExtension> ConservationPkgNamespaces;

typedef enum
{
   SBML_GROUPS_GROUP  = 200
 , SBML_GROUPS_MEMBER = 201
} SBMLGroupsTypeCode_t;


} } // namespace rr } namespace conservation }

#endif  /* ConservationExtension_h */

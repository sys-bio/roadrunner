/**
 * @file    ConservationExtension.cpp
 * @brief   Implementation of ConservationExtension, the core module of conservation package.
 * @author  Andy Somogyi
 *
 * This files the file format of the GroupsExtension by Akiya Jouraku
 *
 *
 * This library is free software; you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation.  A copy of the license agreement is provided
 * in the file named "LICENSE.txt" included with this software distribution
 * and also available online as http://sbml.org/software/libsbml/license.html
 */

#include "ConservationExtension.h"
#include "ConservationDocumentPlugin.h"
#include "ConservedMoietyPlugin.h"



#include <sbml/extension/SBMLExtensionRegister.h>
#include <sbml/extension/SBMLExtensionRegistry.h>
#include <sbml/extension/SBasePluginCreator.h>
#include <sbml/extension/SBMLDocumentPluginNotRequired.h>

#include <iostream>
#include <exception>
#include <sstream>

#include "rrLogger.h"
#include "rrExporter.h"
#include "rrOSSpecifics.h"


// --------------------------------------------------------
//
// Instantiate SBMLExtensionNamespaces<ConservationExtension>
// (GroupsPkgNamespaces) for DLL.
//
// --------------------------------------------------------


static bool initilized = false;


template class RR_DECLSPEC libsbml::SBMLExtensionNamespaces<rr::conservation::ConservationExtension> ;

namespace rr
{
namespace conservation
{

bool conservation_getInit()
{
    return initilized;
}



using namespace libsbml;

// -------------------------------------------------------------------------
//
// This block is global initialization code which should be automatically
// executed before invoking main() block.
//
// -------------------------------------------------------------------------

//------------- (START) -----------------------------------

// The name of this package

const std::string& ConservationExtension::getPackageName()
{
    static const std::string pkgName = "conservation";
    return pkgName;
}

//
// Default SBML level, version, and package version
//
unsigned int ConservationExtension::getDefaultLevel()
{
    return 3;
}

unsigned int ConservationExtension::getDefaultVersion()
{
    return 2;
}

unsigned int ConservationExtension::getDefaultPackageVersion()
{
    return 1;
}

//
// XML namespaces of (1) package versions of groups extension, and
// (2) another XML namespace(XMLSchema-instance) required in the groups
//  extension.
//

const std::string& ConservationExtension::getXmlnsL3V1V1()
{
    static const std::string xmlns =
            "http://www.sbml.org/sbml/level3/version2/conservation/version1";
    return xmlns;
}

//
// Adds this ConservationExtension object to the SBMLExtensionRegistry class.
// ConservationExtension::init() function is automatically invoked when this
// object is instantiated.
//
static libsbml::SBMLExtensionRegister<ConservationExtension> conservationExtensionRegistry;

static
const char* SBML_GROUPS_TYPECODE_STRINGS[] =
{ "Group", "Member" };

//------------- (END) -----------------------------------


ConservationExtension::ConservationExtension()
{
}

/*
 * Copy constructor.
 */
ConservationExtension::ConservationExtension(const ConservationExtension& orig) :
        SBMLExtension(orig)
{
}

/*
 * Destroy this object.
 */
ConservationExtension::~ConservationExtension()
{
}

/*
 * Assignment operator for ConservationExtension.
 */
ConservationExtension&
ConservationExtension::operator=(const ConservationExtension& orig)
{
    SBMLExtension::operator=(orig);
    return *this;
}

/*
 * Creates and returns a deep copy of this ConservationExtension object.
 *
 * @return a (deep) copy of this SBase object
 */
ConservationExtension*
ConservationExtension::clone() const
{
    return new ConservationExtension(*this);
}

const std::string&
ConservationExtension::getName() const
{
    return getPackageName();
}

/*
 * Returns the URI (namespace) of the package corresponding to the combination of the given sbml level,
 * sbml version, and package version.
 * Empty string will be returned if no corresponding URI exists.
 *
 * @return a string of the package URI
 */
const std::string&
ConservationExtension::getURI(unsigned int sbmlLevel, unsigned int sbmlVersion,
        unsigned int pkgVersion) const
{
    if (sbmlLevel == 3)
    {
        if (sbmlVersion == 2)
        {
            if (pkgVersion == 1)
            {
                return getXmlnsL3V1V1();
            }
        }
    }

    static std::string empty = "";

    return empty;
}

/*
 * Returns the SBML level with the given URI of this package.
 *
 *  (NOTICE) Package developers MUST OVERRIDE this pure virtual function in their derived class.
 *
 */
unsigned int ConservationExtension::getLevel(const std::string &uri) const
{
    if (uri == getXmlnsL3V1V1())
    {
        return 3;
    }

    return 0;
}

/*
 * Returns the SBML version with the given URI of this package.
 *
 *  (NOTICE) Package developers MUST OVERRIDE this pure virtual function in their derived class.
 *
 */
unsigned int ConservationExtension::getVersion(const std::string &uri) const
{
    if (uri == getXmlnsL3V1V1())
    {
        return 2;
    }

    return 0;
}

/*
 * Returns the package version with the given URI of this package.
 *
 *  (NOTICE) Package developers MUST OVERRIDE this pure virtual function in their derived class.
 *
 */
unsigned int ConservationExtension::getPackageVersion(
        const std::string &uri) const
{
    if (uri == getXmlnsL3V1V1())
    {
        return 1;
    }

    return 0;
}

/*
 * Returns an SBMLExtensionNamespaces<class SBMLExtensionType> object
 * (e.g. SBMLExtensionNamespaces<ConservationExtension> whose alias type is
 * GroupsPkgNamespaces) corresponding to the given uri.
 * NULL will be returned if the given uri is not defined in the corresponding package.
 *
 *  (NOTICE) Package developers MUST OVERRIDE this pure virtual function in their derived class.
 *
 */
SBMLNamespaces* ConservationExtension::getSBMLExtensionNamespaces(
        const std::string &uri) const
{
    ConservationPkgNamespaces* pkgns = 0;
    if (uri == getXmlnsL3V1V1())
    {
        pkgns = new ConservationPkgNamespaces(3, 1, 1);
    }
    return pkgns;
}

/*
 * This method takes a type code of groups package and returns a string representing
 * the code.
 */
const char* ConservationExtension::getStringFromTypeCode(int typeCode) const
{
    int min = SBML_GROUPS_GROUP;
    int max = SBML_GROUPS_MEMBER;

    if (typeCode < min || typeCode > max)
    {
        return "(Unknown SBML Groups Type)";
    }

    return SBML_GROUPS_TYPECODE_STRINGS[typeCode - min];
}

bool ConservationExtension::getConservedMoiety(const libsbml::Species& s)
{
    const ConservedMoietyPlugin* plugin =
            dynamic_cast<const ConservedMoietyPlugin*>(
                    s.getPlugin("conservation"));
    return plugin ? plugin->getConservedMoiety() : false;
}

bool ConservationExtension::getConservedMoiety(const libsbml::Parameter& s)
{
    const ConservedMoietyPlugin* plugin =
            dynamic_cast<const ConservedMoietyPlugin*>(
                    s.getPlugin("conservation"));
    return plugin ? plugin->getConservedMoiety() : false;
}

std::string ConservationExtension::getConservedQuantity(const libsbml::Species& s)
{
    const ConservedMoietyPlugin* plugin =
            dynamic_cast<const ConservedMoietyPlugin*>(
                    s.getPlugin("conservation"));
    return plugin ? plugin->getConservedQuantity() : "";
}

std::vector<std::string> ConservationExtension::getConservedQuantities(const libsbml::Species& s)
{
    const ConservedMoietyPlugin* plugin =
            dynamic_cast<const ConservedMoietyPlugin*>(
                    s.getPlugin("conservation"));
    return plugin ? plugin->getConservedQuantities() : std::vector<std::string>();
}


bool ConservationExtension::isConservedMoietyDocument(
        const libsbml::SBMLDocument* doc)
{
    const SBasePlugin* p = doc->getPlugin("conservation");

    return dynamic_cast<const ConservationDocumentPlugin*>(p) != 0;
}


/*
 *
 * Initialization function of groups extension module which is automatically invoked
 * by SBMLExtensionRegister class before main() function invoked.
 *
 */
void ConservationExtension::init()
{
    //std::cout << __FUNC__ << std::endl;

    initilized = true;
    //-------------------------------------------------------------------------
    //
    // 1. Checks if the groups pacakge has already been registered.
    //
    //-------------------------------------------------------------------------

    if (SBMLExtensionRegistry::getInstance().isRegistered(getPackageName()))
    {
        // do nothing;
        return;
    }

    //-------------------------------------------------------------------------
    //
    // 2. Creates an SBMLExtension derived object.
    //
    //-------------------------------------------------------------------------

    ConservationExtension conservationExtension;

    //-------------------------------------------------------------------------------------
    //
    // 3. Creates SBasePluginCreatorBase derived objects required for this
    //    extension. The derived classes can be instantiated by using the following
    //     template class.
    //
    //    temaplate<class SBasePluginType> class SBasePluginCreator
    //
    //    The constructor of the creator class has two arguments:
    //
    //        (1) SBaseExtensionPoint : extension point to which the plugin object connected
    //        (2) std::vector<std::string> : a std::vector object that contains a list of URI
    //                                       (package versions) supported by the plugin object.
    //
    //    For example, two plugin objects (plugged in SBMLDocument and Model elements) are
    //    required for the groups extension.
    //
    //    Since only 'required' attribute is used in SBMLDocument by the groups package, existing
    //    SBMLDocumentPluginNotRequired class can be used as-is for the plugin.
    //
    //    Since the lists of supported package versions (currently only L3V1-groups-V1 supported )
    //    are equal in the both plugin objects, the same vector object is given to each
    //    constructor.
    //
    //---------------------------------------------------------------------------------------

    std::vector<std::string> packageURIs;
    packageURIs.push_back(getXmlnsL3V1V1());

    // the plugins will be plugged in at these extension points
    SBaseExtensionPoint documentExtPoint("core", SBML_DOCUMENT);
    SBaseExtensionPoint parameterExtPoint("core", SBML_PARAMETER);
    SBaseExtensionPoint speciesExtPoint("core", SBML_SPECIES);

    SBasePluginCreator<ConservationDocumentPlugin, ConservationExtension>
        documentPluginCreator(documentExtPoint, packageURIs);

    SBasePluginCreator<ConservedMoietyPlugin, ConservationExtension>
        parameterPluginCreator(parameterExtPoint, packageURIs);

    SBasePluginCreator<ConservedMoietyPlugin, ConservationExtension>
        speciesPluginCreator(speciesExtPoint, packageURIs);

    //--------------------------------------------------------------------------------------
    //
    // 3. Adds the above SBasePluginCreatorBase derived objects to the SBMLExtension derived object.
    //
    //--------------------------------------------------------------------------------------

    conservationExtension.addSBasePluginCreator(&documentPluginCreator);
    conservationExtension.addSBasePluginCreator(&parameterPluginCreator);
    conservationExtension.addSBasePluginCreator(&speciesPluginCreator);

    //-------------------------------------------------------------------------
    //
    // 4. Registers the SBMLExtension derived object to SBMLExtensionRegistry
    //
    //-------------------------------------------------------------------------

    int result = SBMLExtensionRegistry::getInstance().addExtension(
            &conservationExtension);

    if (result != LIBSBML_OPERATION_SUCCESS)
    {
        std::stringstream ss;
        ss << "ConservationExtension::init() failed, result: " << result;
        throw std::runtime_error(ss.str().c_str());
    }
    else
    {
        //std::cout << "successfully registered conservation extension" << std::endl;
    }
}

}
} // namespace rr } namespace conservation }


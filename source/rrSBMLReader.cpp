// The poco files have issues on windows and have to be included before
// any potential 'windows.h' is included. There is a compile problem
// when windows.h is included BEFORE Poco HTTP Factory
#pragma hdrstop
#include <Poco/URI.h>
#include <Poco/URIStreamOpener.h>
#include <Poco/Net/HTTPStreamFactory.h>
#include <Poco/RegularExpression.h>


#include <Poco/Mutex.h>
#include <Poco/Path.h>
#include <Poco/File.h>

#include "rrOSSpecifics.h"
#include "rrLogger.h"
#include "rrSBMLReader.h"
#include "rrException.h"
#include <iostream>
#include <sstream>
#include <string>
#include <algorithm>
#include <cctype>

#include <sbml/SBMLReader.h>
#include <sbml/conversion/SBMLConverterRegistry.h>
#include <sbml/SBMLWriter.h>

using Poco::RegularExpression;


using Poco::Mutex;
using std::endl;
using std::stringstream;
using libsbml::SBMLDocument;
using libsbml::SBMLConverterRegistry;

namespace rr {

// asking for trouble statically registering the factory
static Mutex httpFactoryMutex;
static bool httpFactoryRegistered = false;

/**
 * performs some basic checks to see if the std::string is an sbml std::string.
 *
 * This just checks the basic format and checks on characters that are
 * not valid file path characters. So if this std::string matches this function,
 * there should basically be no way that this is a valid path or uri.
 */
bool SBMLReader::is_sbml(const std::string& str)
{
    //<?xml version="1.0" encoding="UTF-8"?> <sbml level="3" version="1" xmlns="http://www.sbml.org/sbml/level3/version1/core">
    // try first with xml declaration
    size_t pos = str.find("<");
    if (pos != std::string::npos)
    {
        pos = str.find("?", pos + 1);

        if (pos != std::string::npos)
        {
            pos = str.find("xml", pos + 1);

            if (pos != std::string::npos)
            {
                pos = str.find("?", pos + 3);

                if (pos != std::string::npos)
                {
                    pos = str.find(">", pos + 1);

                    if (pos != std::string::npos)
                    {
                        pos = str.find("<", pos + 1);

                        if (pos != std::string::npos)
                        {
                            pos = str.find("sbml", pos + 1);
                            return pos != std::string::npos;
                        }
                    }
                }
            }
        }
    }
    // try without xml declaration
    pos = str.find("<");
    if (pos != std::string::npos)
    {
        pos = str.find("sbml", pos + 1);
        return pos != std::string::npos;
    }
    return false;
}

/**
 * extract the <sbml...> tag from an sbml document.
 * This regex should stop at the first match.
 */
static const Poco::RegularExpression sbml_re("<\\s*sbml\\s*.*?>",
        RegularExpression::RE_UNGREEDY);

/**
 * Check if the given sbml std::string uses the composite extension.
 *
 * We only need to scan inside the <sbml ... > tag, which is
 * as the start of the doc, so this should be a very fast search.
 *
 * <sbml xmlns="http://www.sbml.org/sbml/level3/version1/core"
 *     xmlns:comp="http://www.sbml.org/sbml/level3/version1/comp/version1"
 *     level="3" version="1" comp:required="false">
 */
static bool has_comp(const std::string& sbml) {
    std::string sbmlss; // sbml substring, avoid looking in entire document.
    sbml_re.extract(sbml, sbmlss);
    static const std::string compns =
            "http://www.sbml.org/sbml/level3/version1/comp/version1";
    return sbmlss.find(compns) != std::string::npos;
}

/**
 * Check if the given sbml std::string uses the qual extension.
 *
 * We only need to scan inside the <sbml ... > tag, which is
 * as the start of the doc, so this should be a very fast search.
 *
 * <sbml xmlns="http://www.sbml.org/sbml/level3/version1/core"
 *     xmlns:qual="http://www.sbml.org/sbml/level3/version1/qual/version1"
 *     level="3" version="1" comp:required="false">
 */
static bool has_qual(const std::string& sbml) {
    std::string sbmlss; // sbml substring, avoid looking in entire document.
    sbml_re.extract(sbml, sbmlss);
    static const std::string qualns =
        "http://www.sbml.org/sbml/level3/version1/qual";
    return sbmlss.find(qualns) != std::string::npos;
}

/**
 * Check if the given sbml std::string uses the spatial extension.
 *
 * We only need to scan inside the <sbml ... > tag, which is
 * as the start of the doc, so this should be a very fast search.
 *
 * <sbml xmlns="http://www.sbml.org/sbml/level3/version1/core"
 *     xmlns:spatial="http://www.sbml.org/sbml/level3/version1/spatial/version1"
 *     level="3" version="1" comp:required="false">
 */
static bool has_spatial(const std::string& sbml) {
    std::string sbmlss; // sbml substring, avoid looking in entire document.
    sbml_re.extract(sbml, sbmlss);
    static const std::string spatialns =
        "http://www.sbml.org/sbml/level3/version1/spatial";
    return sbmlss.find(spatialns) != std::string::npos;
}

/**
 * Check if the given sbml std::string uses the multi extension.
 *
 * We only need to scan inside the <sbml ... > tag, which is
 * as the start of the doc, so this should be a very fast search.
 *
 * <sbml xmlns="http://www.sbml.org/sbml/level3/version1/core"
 *     xmlns:multi="http://www.sbml.org/sbml/level3/version1/multi/version1"
 *     level="3" version="1" comp:required="false">
 */
static bool has_multi(const std::string& sbml) {
    std::string sbmlss; // sbml substring, avoid looking in entire document.
    sbml_re.extract(sbml, sbmlss);
    static const std::string multins =
        "http://www.sbml.org/sbml/level3/version1/multi";
    return sbmlss.find(multins) != std::string::npos;
}

/**
 * flatten a comp model.
 */
static std::string flatten_comp(const std::string& sbml, const std::string fname)
{
    SBMLDocument *doc = libsbml::readSBMLFromString(sbml.c_str());
    libsbml::SBMLConverter* converter = NULL;

    // resolve external sub-models.
    doc->setLocationURI(fname);

    if (doc->getPlugin("comp") == NULL)
    {
        // this should not happen
        delete doc;
        throw std::runtime_error("Detected SBML document \"" + fname +
                "\" should have comp extension, but the libSBML::Document "
                "does not have the \"comp\" plugin");
    }

    libsbml::ConversionProperties props;
    props.addOption("flatten comp");
    props.addOption("performValidation", false);

    // a cloned converter, needs to be deleted.
    converter = SBMLConverterRegistry::getInstance().getConverterFor(props);

    if (converter == NULL)
    {
        delete doc;
        throw std::runtime_error("Could not get SBMLConverter for comp model "
                "flattening for model \"" + fname + "\"");
    }

    // ready to flatten model
    converter->setDocument(doc);
    if (converter->convert() != libsbml::LIBSBML_OPERATION_SUCCESS)
    {
        libsbml::SBMLErrorLog* log = doc->getErrorLog();
        if (log->getNumFailsWithSeverity(libsbml::LIBSBML_SEV_ERROR) != 0)
        {
            std::stringstream msg;
            msg << "Errors durring model flattening, model not converted:" << std::endl;

            for (int i = 0; i < log->getNumErrors(); ++i)
            {
                const libsbml::SBMLError* error = log->getError(i);
                if (error->getSeverity() >= libsbml::LIBSBML_SEV_ERROR)
                {
                    msg << "SBMLError(" << i << "): " << error->getMessage();
                }
            }
            throw std::runtime_error(msg.str());
        }
    }

    libsbml::SBMLWriter writer;

    std::stringstream stream;

    writer.writeSBML(doc, stream);

    delete converter;
    delete doc;

    return stream.str();
}

// C++ standard library isn't self-consistent
char mytolower(char x) {
    return (char)std::tolower(x);
}

std::string SBMLReader::read(const std::string& str)
{
    // bail here (empty str is valid sometimes), but messes
    // with POCO readers.
    if(str.empty()) {
        return str;
    }

    if (is_sbml(str))
    {
        return str;
    }

    // at this point, we have to assume that the std::string is to be interpreted as a
    // path or uri

    // just in case there are multiple threads...
    httpFactoryMutex.lock();

    // not working on windows yet.

    if (!httpFactoryRegistered)
    {
        Poco::Net::HTTPStreamFactory::registerFactory();
        httpFactoryRegistered = true;
    }

    // only the first thread creats the factory
    httpFactoryMutex.unlock();

    // opener figures out if we have local or remote path
    // and creates appropriate stream
    Poco::URIStreamOpener &opener = Poco::URIStreamOpener::defaultOpener();

    std::istream* stream = NULL;
    try {
        stream = opener.open(str);
    } catch(Poco::Exception& e) {
        std::string urischeme = Poco::URI(str).getScheme();
        std::transform(urischeme.begin(), urischeme.end(), urischeme.begin(), mytolower);
        if (urischeme == "https") {
            rrLog(Logger::LOG_ERROR) << "HTTPS transport not supported";
            throw Exception("Could not open stream: HTTPS transport not supported");
        } else {
            rrLog(Logger::LOG_ERROR) << "Could not open stream: " << e.what();
        }
        // stream should still be NULL
    }
    if (stream)
    {
        // read the entire http stream into a std::string and return it.
        std::istreambuf_iterator<char> eos;
        std::string s(std::istreambuf_iterator<char>(*stream), eos);
        delete stream;

        // read from a stream, only case where we can flatten, so
        // check
        if (has_comp(s))
        {
            return flatten_comp(s, str);
        }
        if (has_qual(s))
        {
            rrLog(Logger::LOG_ERROR) << "Qual model discovered, but not supported.";
            throw std::domain_error("This SBML model contains information from the 'qual' package (for qualitative or 'logical' modeling).  These models are not supported by roadrunner or tellurium.  A good source of software that supports qual is the COLOMOTO consortium.  See http://www.colomoto.org/software/ for a list of other software packages that might work for you.");
        }
        if (has_spatial(s))
        {
            rrLog(Logger::LOG_ERROR) << "Spatial model discovered, but not supported.";
            throw std::domain_error("This SBML model contains information from the 'spatial' package.  These models are not supported by roadrunner or tellurium.  A few software packages that do support spatial (as of 2023) include VCell (https://vcell.org/), XitoSBML (https://github.com/spatialsimulator/XitoSBML) and Spatial SBML (https://github.com/fbergmann/spatial-sbml).");
        }
        if (has_multi(s))
        {
            rrLog(Logger::LOG_ERROR) << "Multi model discovered, but not supported.";
            throw std::domain_error("This SBML model contains information from the 'multi' package for multistate, multicomponent and multicompartment models, or 'rule-based' models.  These models are not supported by roadrunner or tellurium.  The best software package that supports multi (as of 2023) is Simmune (https://bioinformatics.niaid.nih.gov/simmune/).");
        }
        return s;
    }
    else
    {
        std::string msg = std::string(__FUNC__) +  ", could not open " + str + " as a file or uri";
        throw Exception(msg);
    }

    return str;
}



} /* namespace rr */





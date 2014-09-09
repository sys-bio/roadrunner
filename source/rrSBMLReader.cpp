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
 * performs some basic checks to see if the string is an sbml string.
 *
 * This just checks the basic format and checks on characters that are
 * not valid file path characters. So if this string matches this function,
 * there should basically be no way that this is a valid path or uri.
 */
static bool is_sbml(const std::string& str)
{
    //<?xml version="1.0" encoding="UTF-8"?> <sbml level="3" version="1" xmlns="http://www.sbml.org/sbml/level3/version1/core">
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
    return false;
}

/**
 * extract the <sbml...> tag from an sbml document.
 * This regex should stop at the first match.
 */
static const Poco::RegularExpression sbml_re("<\\s*sbml\\s*.*?>",
        RegularExpression::RE_UNGREEDY);

/**
 * Check if the given sbml string uses the composite extension.
 *
 * We only need to scan inside the <sbml ... > tag, which is
 * as the start of the doc, so this should be a very fast search.
 *
 * <sbml xmlns="http://www.sbml.org/sbml/level3/version1/core"
 *     xmlns:comp="http://www.sbml.org/sbml/level3/version1/comp/version1"
 *     level="3" version="1" comp:required="false">
 */
static bool has_comp(const string& sbml) {
    string sbmlss; // sbml substring, avoid looking in entire document.
    sbml_re.extract(sbml, sbmlss);
    static const string compns =
            "http://www.sbml.org/sbml/level3/version1/comp/version1";
    return sbmlss.find(compns) != string::npos;
}

/**
 * flatten a comp model.
 */
static string flatten_comp(const string& sbml, const std::string fname)
{
    Log(Logger::LOG_WARNING) << "Flattening model " << fname << std::endl
                             << "LEAKING MEMORY! due to leaks in comp sbml extension.";

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
            stringstream msg;
            msg << "Errors durring model flattening, model *MIGHT* contain errors:"
                    << endl;

            for (int i = 0; i < log->getNumErrors(); ++i)
            {
                const libsbml::SBMLError* error = log->getError(i);
                if (error->getSeverity() >= libsbml::LIBSBML_SEV_ERROR)
                {
                    msg << "SBMLError(" << i << "): " << error->getMessage();
                }
            }
            Log(rr::Logger::LOG_WARNING) << msg.str();
        }
    }

    libsbml::SBMLWriter writer;

    std::stringstream stream;

    writer.writeSBML(doc, stream);

    delete converter;
    delete doc;

    return stream.str();
}


std::string SBMLReader::read(const std::string& str)
{
    if (is_sbml(str))
    {
        if(has_comp(str))
        {
            Log(Logger::LOG_WARNING) <<
                    "Unable to flatten odel read from string which has comp "
                    "extension, libSBML needs the full file path for flatening "
                    "to work, ";
        }
        return str;
    }

    // at this point, we have to assume that the string is to be interpreted as a
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

    std::istream* stream = opener.open(str);
    if (stream)
    {
        // read the entire http stream into a string and return it.
        std::istreambuf_iterator<char> eos;
        std::string s(std::istreambuf_iterator<char>(*stream), eos);
        delete stream;

        // read from a stream, only case where we can flatten, so
        // check
        if (has_comp(s))
        {
            return flatten_comp(s, str);
        }
        return s;
    }
    else
    {
        string msg = string(__FUNC__) +  ", could not open " + str + " as a file or uri";
        throw Exception(msg);
    }

    return str;
}

} /* namespace rr */





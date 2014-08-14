// == PREAMBLE ================================================

// * Licensed under the Apache License, Version 2.0; see README

/*
 * Backtrace.cpp
 *
 *  Created on: Aug 13, 2014
 *      Author: JKM
 */

// == INCLUDES ================================================

#include "GPUSimModel.h"
#include "GPUSimException.h"
#include "GPUSimModelGenerator.h"
#include "rrConfig.h"
// #include "nullptr.h"

#include <sbml/SBMLTypes.h>
// #include <sbml/SBMLReader.h>

// == CODE ====================================================

namespace rr
{

namespace rrgpu
{

static libsbml::SBMLDocument *checkedReadSBMLFromString(const char* xml) {
    libsbml::SBMLDocument *doc = libsbml::readSBMLFromString(xml);

    if (doc)
    {
        if (doc->getModel() == 0)
        {
            // fatal error
            libsbml::SBMLErrorLog *log = doc->getErrorLog();
            string errors = log ? log->toString() : " NULL SBML Error Log";
            delete doc;
            throw_gpusim_exception("Fatal SBML error, no model, errors in sbml document: " + errors);
        }
        else if (doc->getNumErrors() > 0)
        {
            libsbml::SBMLErrorLog *log = doc->getErrorLog();
            string errors = log ? log->toString() : " NULL SBML Error Log";
            Log(rr::Logger::LOG_WARNING) << "Warning, errors found in sbml document: " + errors;
        }
    }
    else
    {
        delete doc;
        throw_gpusim_exception("readSBMLFromString returned NULL, no further information available");
    }
    return doc;
}

GPUSimModel::GPUSimModel(std::string const &sbml, unsigned options) {
//     typedef std::unique_ptr<libsbml::SBMLDocument> SBMLDocumentPtr;
    libsbml::SBMLDocument* doc{checkedReadSBMLFromString(sbml.c_str())};

    if (options & ModelGenerator::CONSERVED_MOIETIES) {
        if ((Config::getInt(Config::ROADRUNNER_DISABLE_WARNINGS) &
                Config::ROADRUNNER_DISABLE_WARNINGS_CONSERVED_MOIETY) == 0)
        {
            Log(Logger::LOG_NOTICE) << "performing conserved moiety conversion";
        }

        // check if already conserved doc
        if (!conservation::ConservationExtension::isConservedMoietyDocument(doc)) {
            moietyConverter = ConservedMoietyConverterPtr(new conservation::ConservedMoietyConverter());

            if (moietyConverter->setDocument(doc) != libsbml::LIBSBML_OPERATION_SUCCESS)
            {
                throw_gpusim_exception("error setting conserved moiety converter document");
            }

            if (moietyConverter->convert() != libsbml::LIBSBML_OPERATION_SUCCESS)
            {
                throw_gpusim_exception("error converting document to conserved moieties");
            }

            doc = moietyConverter->getDocument();

            libsbml::SBMLWriter sw;
            char* convertedStr = sw.writeToString(doc);

            Log(Logger::LOG_INFORMATION) << "***************** Conserved Moiety Converted Document ***************";
            Log(Logger::LOG_INFORMATION) << convertedStr;
            Log(Logger::LOG_INFORMATION) << "*********************************************************************";

            free(convertedStr);
        }
    }
    if(!moietyConverter)
        ownedDoc = SBMLDocumentPtr(doc);
    // use getDocument from now on
}

libsbml::SBMLDocument* GPUSimModel::getDocument() {
    if(moietyConverter)
        return moietyConverter->getDocument();
    else if(ownedDoc)
        return ownedDoc.get();
    else
        throw_gpusim_exception("Missing SBML document");
}

} // namespace rrgpu

} // namespace rr

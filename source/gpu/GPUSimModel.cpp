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

//     assert(sizeof(modelDataFieldsNames) / sizeof(const char*)
//             == NotSafe_FloatingSpeciesAmounts + 1
//             && "wrong number of items in modelDataFieldsNames");

//     modelName = getDocument()->getModel()->getName();

    // add species initially without props so they can be referenced by rules
    for (uint i = 0; i < model->getListOfSpecies()->size(); ++i) {
        const Species *s = model->getListOfSpecies()->get(i);

        if (s->getBoundaryCondition())
            continue;

        const std::string& sid = s->getId();

        addSpecies(FloatingSpeciesPtr(sid));
    }

    // read SBML rules and add them to the model
    {
        const libsbml::ListOfRules * rules = getDocument()->getModel()->getListOfRules();
        for (unsigned i = 0; i < rules->size(); ++i)
        {
            const libsbml::Rule *rule = rules->get(i);

            if (dynamic_cast<const libsbml::AssignmentRule*>(rule))
            {
                assigmentRules.insert(rule->getVariable());
            }
            else if (dynamic_cast<const libsbml::RateRule*>(rule))
            {
//                 uint rri = rateRules.size();
//                 rateRules[rule->getId()] = rri;
            }
            else if (dynamic_cast<const libsbml::AlgebraicRule*>(rule))
            {
                char* formula = SBML_formulaToString(rule->getMath());
                Log(Logger::LOG_WARNING)
                    << "Unable to handle algebraic rules. Formula '0 = "
                    << formula << "' ignored.";
                free(formula);
            }
        }
    }

    // SBML initial assignment rules
    {
        const libsbml::ListOfInitialAssignments *initAssignmentList =
                getDocument()->getModel()->getListOfInitialAssignments();

        for (unsigned i = 0; i < initAssignmentList->size(); ++i) {
            const libsbml::InitialAssignment *ia = initAssignmentList->get(i);
            addRule(ModelRulePtr(new InitialAssignmentRule(ia->getSymbol())));
        }
    }

    // figure out 'fully' indendent floating species -- those without rules.
    for (uint i = 0; i < model->getListOfSpecies()->size(); ++i) {
        const Species *s = model->getListOfSpecies()->get(i);

        if (s->getBoundaryCondition())
            continue;

        const std::string& sid = s->getId();

        FloatingSpecies* species = findFloatingSpeciesById(sid);

        if (getRules().contains(species))
            species->setIsIndependent(false);

        // conserved moiety species assignment rules do not apply at
        // time t < 0
        species->setIsIndepInitFltSpecies(
          !hasInitialAssignmentRule(sid) &&
          (!hasAssignmentRule(sid) || conservedMoiety));

        if (conservedMoiety)
            species_->setIsConservedMoiety(ConservationExtension::getConservedMoiety(*s));
    }

    // get the compartments, need to reorder them to set the independent ones
    // first. Make sure compartments is called *before* float species.
//     initCompartments();


    // process the floating species
//     initFloatingSpecies(getDocument()->getModel(), options & rr::ModelGenerator::CONSERVED_MOIETIES);

    // display compartment info. We need to get the compartments before the
    // so we can get the species compartments. But the struct anal dumps
    // a bunch of stuff, so to keep things looking nice in the log, we
    // display the compartment info here.
//     displayCompartmentInfo();

//     initBoundarySpecies(getDocument()->getModel());

//     initGlobalParameters(getDocument()->getModel(), options & rr::ModelGenerator::CONSERVED_MOIETIES);

//     initReactions(getDocument()->getModel());

//     initEvents(getDocument()->getModel());
}

libsbml::SBMLDocument* GPUSimModel::getDocument() {
    if(moietyConverter)
        return moietyConverter->getDocument();
    else if(ownedDoc)
        return ownedDoc.get();
    else
        throw_gpusim_exception("Missing SBML document");
}

FloatingSpecies* GPUSimModel::findFloatingSpeciesById(const std::string& id) {
    for(FloatingSpecies* s : getFloatingSpecies())
        if(s->matchId(id))
            return s;
    throw_gpusim_exception("No such floating species for id \"" + id + "\"");
}

// from LLVMModelDataSymbols
void GPUSimModel::initFloatingSpecies(bool computeAndAssignConsevationLaws) {
    const ListOfSpecies *species = model->getListOfSpecies();


}

bool GPUSimModel::isIndependentElement(const FloatingSpecies& s) const
{
    return rateRules.find(id) == rateRules.end() &&
            assigmentRules.find(id) == assigmentRules.end();
}

bool GPUSimModel::isIndependentInitFloatingSpecies(const std::string& id) const
{
    return !rateRules.contains(id) &&
            !assigmentRules.contains(id);
}

} // namespace rrgpu

} // namespace rr

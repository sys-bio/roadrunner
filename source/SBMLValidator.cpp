/*
 * SBMLValidator.cpp
 *
 *  Created on: Aug 24, 2014
 *      Author: andy
 */

#include "SBMLValidator.h"
#include "rrSBMLReader.h"

#include <sbml/SBMLDocument.h>

#include "rrLogger.h"

#include <sstream>

using namespace libsbml;

using std::string;

namespace rr
{

std::string validateSBML(const std::string src, unsigned opt)
{
    std::stringstream errors;
    SBMLDocument *doc = NULL;

    try {
        string sbml = SBMLReader::read(src);
        doc =  readSBMLFromString (sbml.c_str());

        doc->setConsistencyChecks(LIBSBML_CAT_GENERAL_CONSISTENCY,
                opt & VALIDATE_GENERAL);

        doc->setConsistencyChecks(LIBSBML_CAT_UNITS_CONSISTENCY,
                opt & VALIDATE_UNITS);

        doc->setConsistencyChecks(LIBSBML_CAT_IDENTIFIER_CONSISTENCY,
                opt & VALIDATE_IDENTIFIER);

        doc->setConsistencyChecks(LIBSBML_CAT_MATHML_CONSISTENCY,
                opt & VALIDATE_MATHML);

        doc->setConsistencyChecks(LIBSBML_CAT_OVERDETERMINED_MODEL,
                opt & VALIDATE_OVERDETERMINED);

        doc->setConsistencyChecks(LIBSBML_CAT_MODELING_PRACTICE,
                opt & VALIDATE_MODELING_PRACTICE);

        doc->checkConsistency();

        SBMLErrorLog* errorLog = doc->getErrorLog();

        if(errorLog) {
            errorLog->printErrors(errors);
        }

    } catch(...) {
        delete doc;
        throw;
    }

    delete doc;
    return errors.str();
}

// Return true if this species reference has an initial assignment
static bool hasInitialAssignment(const SpeciesReference* s) {
    const Model* m = s->getModel();
    const ListOfInitialAssignments *assn = m->getListOfInitialAssignments();

    for (unsigned i = 0; i < assn->size(); ++i) {
        const InitialAssignment *a = assn->get(i);
        if (a->getSymbol() == s->getId())
            return true;
    }

    return false;
}

// Return true if this species reference has an assignment rule
static bool hasAssignmentRule(const SpeciesReference* s) {
    const Model* m = s->getModel();
    const ListOfRules* rules = m->getListOfRules();

    for (unsigned i = 0; i < rules->size(); ++i) {
        const Rule *rule = rules->get(i);
        if (const AssignmentRule* a = dynamic_cast<const AssignmentRule*>(rule))
            if (a->getVariable() ==  s->getId())
                return true;
    }

    return false;
}

// Return true if stoichiometry is defined for this species reference
static bool isStoichDefined(const SpeciesReference* s) {
    if (!s)
        // null ref
        return false;
    if (hasInitialAssignment(s) || hasAssignmentRule(s))
        // stoich can be set by assignments
        return true;
    return s->isSetStoichiometry();
}

// Return true if stoichiometry is defined for every reaction in the model
bool isStoichDefined(const std::string sbml) {
    SBMLDocument *doc = NULL;

    bool sbml_decl_okay = false;

    // check for <?xml
    size_t pos = sbml.find("<");
    if (pos != std::string::npos) {
      pos = sbml.find("?", pos+1);
      if (pos != std::string::npos) {
        pos = sbml.find("xml", pos+1);
        if (pos != std::string::npos)
          sbml_decl_okay = true;
      }
    }

    // check for <sbml
    pos = sbml.find("<");
    if (pos != std::string::npos) {
      pos = sbml.find("sbml", pos+1);
      if (pos != std::string::npos)
        sbml_decl_okay = true;
    }

    if (!sbml_decl_okay)
      throw std::runtime_error("SBML document must begin with an XML declaration or an SBML declaration");

    try {
        doc =  readSBMLFromString (sbml.c_str());

        if (!doc)
          throw std::runtime_error("Unable to read SBML");

        if (doc->getLevel() < 3) {
            delete doc;
            return true;                                    // stoichiometry has a default value in level 1 & 2
        }

        const Model *m = doc->getModel();

        if (!m)
          throw std::runtime_error("SBML string invalid or missing model");

        for (int j = 0; j<m->getNumReactions(); ++j) {
            const Reaction* r = m->getReaction(j);
            if (!r)
                throw std::runtime_error("No reaction");

            // check stoich defined on reactants / products
            for (int k = 0; k<r->getNumReactants(); ++k) {
                if (!isStoichDefined(r->getReactant(k))) {
                    delete doc;
                    return false;
                }
            }

            for (int k = 0; k<r->getNumProducts(); ++k) {
                if (!isStoichDefined(r->getProduct(k))) {
                    delete doc;
                    return false;
                }
            }

            // modifiers have no stoichiometry
        }

    } catch(...) {
        delete doc;
        throw;
    }

    delete doc;
    return true;
}

std::string fixMissingStoich(const std::string sbml) {
    SBMLDocument *doc = NULL;

    try {
        doc =  readSBMLFromString (sbml.c_str());

        Model *m = doc->getModel();

        for (int j = 0; j<m->getNumReactions(); ++j) {
            Reaction* r = m->getReaction(j);
            if (!r)
                throw std::runtime_error("No reaction");

            // check stoich defined on reactants / products
            for (int k = 0; k<r->getNumReactants(); ++k) {
                SpeciesReference* s = r->getReactant(k);
                if (!isStoichDefined(s))
                    if (s->setStoichiometry(1.) != LIBSBML_OPERATION_SUCCESS)
                        throw std::runtime_error("Unable to set stoichiometry");
            }

            for (int k = 0; k<r->getNumProducts(); ++k) {
                SpeciesReference* s = r->getProduct(k);
                if (!isStoichDefined(s))
                    if (s->setStoichiometry(1.) != LIBSBML_OPERATION_SUCCESS)
                        throw std::runtime_error("Unable to set stoichiometry");
            }

            // modifiers have no stoichiometry
        }

    } catch(...) {
        delete doc;
        throw;
    }

    SBMLWriter writer;

    char* sbml_cstr = writer.writeSBMLToString(doc);
    delete doc;
    std::string result(sbml_cstr);
    free(sbml_cstr);
    return result;
}

} /* namespace rr */

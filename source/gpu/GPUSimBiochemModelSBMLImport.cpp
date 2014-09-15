// == PREAMBLE ================================================

// * Licensed under the Apache License, Version 2.0; see README

/*
 * Backtrace.cpp
 *
 *  Created on: Aug 13, 2014
 *      Author: JKM
 */

// == INCLUDES ================================================

#include "GPUSimBiochemModel.h"
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

// --* Global *--

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

ModelASTNodePtr ConvertSbmlASTNode(const Reaction* r, const libsbml::ASTNode* node);

// because SBML sum nodes are implemented as n-ary ops instead of binary ops
ModelASTNodePtr ConvertSbmlPlusNode(const Reaction* r, const libsbml::ASTNode* node, int start) {
    assert(node->getType() == libsbml::AST_PLUS && "Sanity check");
    if (start+2 == node->getNumChildren())
        return ModelASTNodePtr(new SumASTNode(ConvertSbmlASTNode(r, node->getChild(start)), ConvertSbmlASTNode(r, node->getChild(start+1))));
    else
        return ModelASTNodePtr(new SumASTNode(ConvertSbmlASTNode(r, node->getChild(start)), ConvertSbmlPlusNode(r, node, start+1)));
}

// because SBML product nodes are implemented as n-ary ops instead of binary ops
ModelASTNodePtr ConvertSbmlTimesNode(const Reaction* r, const libsbml::ASTNode* node, int start) {
    assert(node->getType() == libsbml::AST_TIMES && "Sanity check");
    if (start+2 == node->getNumChildren())
        return ModelASTNodePtr(new ProductASTNode(ConvertSbmlASTNode(r, node->getChild(start)), ConvertSbmlASTNode(r, node->getChild(start+1))));
    else
        return ModelASTNodePtr(new ProductASTNode(ConvertSbmlASTNode(r, node->getChild(start)), ConvertSbmlTimesNode(r, node, start+1)));
}

// you get the idea
ModelASTNodePtr ConvertSbmlDivisionNode(const Reaction* r, const libsbml::ASTNode* node, int start) {
    assert(node->getType() == libsbml::AST_DIVIDE && "Sanity check");
    if (start+2 == node->getNumChildren())
        return ModelASTNodePtr(new QuotientASTNode(ConvertSbmlASTNode(r, node->getChild(start)), ConvertSbmlASTNode(r, node->getChild(start+1))));
    else
        return ModelASTNodePtr(new QuotientASTNode(ConvertSbmlASTNode(r, node->getChild(start)), ConvertSbmlDivisionNode(r, node, start+1)));
}

ModelASTNodePtr ConvertSbmlASTNode(const Reaction* r, const libsbml::ASTNode* node) {
    assert(node);
    switch (node->getType()) {
        case libsbml::AST_PLUS:
            if(node->getNumChildren() < 2)
                throw_gpusim_exception("Sum expression should have degree >= 2");
            if (2 == node->getNumChildren())
                return ModelASTNodePtr(new SumASTNode(ConvertSbmlASTNode(r, node->getChild(0)), ConvertSbmlASTNode(r, node->getChild(1))));
            else
                return ModelASTNodePtr(new SumASTNode(ConvertSbmlASTNode(r, node->getChild(0)), ConvertSbmlPlusNode(r, node, 1)));
        case libsbml::AST_TIMES:
            if(node->getNumChildren() < 2)
                throw_gpusim_exception("Product expression should have degree >= 2");
            if (2 == node->getNumChildren())
                return ModelASTNodePtr(new ProductASTNode(ConvertSbmlASTNode(r, node->getChild(0)), ConvertSbmlASTNode(r, node->getChild(1))));
            else
                return ModelASTNodePtr(new ProductASTNode(ConvertSbmlASTNode(r, node->getChild(0)), ConvertSbmlTimesNode(r, node, 1)));
        case libsbml::AST_DIVIDE:
            if(node->getNumChildren() < 2)
                throw_gpusim_exception("Quotient expression should have degree == 2");
            if (2 == node->getNumChildren())
                return ModelASTNodePtr(new QuotientASTNode(ConvertSbmlASTNode(r, node->getChild(0)), ConvertSbmlASTNode(r, node->getChild(1))));
            else
                return ModelASTNodePtr(new QuotientASTNode(ConvertSbmlASTNode(r, node->getChild(0)), ConvertSbmlDivisionNode(r, node, 1)));
        case libsbml::AST_NAME:
            if (r->isParameter(node->getName()))
                return ModelASTNodePtr(new ParameterRefASTNode(r, node->getName()));
            else if (r->isParticipant(node->getName()))
                return ModelASTNodePtr(new FloatingSpeciesRefASTNode(r->getFloatingSpecies(node->getName())));
            else
                throw_gpusim_exception(std::string("Unknown NAME node: ") + node->getName());
        case libsbml::AST_INTEGER:
            return ModelASTNodePtr(new IntegerLiteralASTNode(node->getInteger()));
        case libsbml::AST_POWER:
            assert((node->getNumChildren() == 2) && "Exponent expression should have degree == 2");
                return ModelASTNodePtr(new ExponentiationASTNode(ConvertSbmlASTNode(r, node->getChild(0)), ConvertSbmlASTNode(r, node->getChild(1))));
        default:
            throw_gpusim_exception("Unknown node type: " + std::to_string(node->getType()) + (node->getOperatorName() ? " (" + std::string(node->getOperatorName()) + ")" : std::string("")));
    }
}

// --* ModelElement *--

// --* Reaction *--

bool Reaction::isParameter(const String& p) const {
# if GPUSIM_MODEL_USE_SBML
    // Level 2
    assert(sbmlmodel_ && "Need to set sbmlmodel_");
    const libsbml::ListOfParameters* params = sbmlmodel_->getListOfParameters();
    for (int i=0; i<sbmlmodel_->getNumParameters(); ++i) {
        const libsbml::Parameter* x = params->get(i);
        if (x->getId() == p)
            return true;
    }
    return false;
# else
# error "Need SBML"
# endif
}

double Reaction::getParameterVal(const String& p) const {
# if GPUSIM_MODEL_USE_SBML
    assert(sbmlmodel_ && "Need to set sbmlmodel_");
    const libsbml::ListOfParameters* params = sbmlmodel_->getListOfParameters();
    for (int i=0; i<sbmlmodel_->getNumParameters(); ++i) {
        const libsbml::Parameter* x = params->get(i);
        if (x->getId() == p)
            return x->getValue();
    }
    throw_gpusim_exception("No such parameter \"" + p + "\"");
# else
# error "Need SBML"
# endif
}

// --* GPUSimModel *--

GPUSimModel::GPUSimModel(std::string const &sbml, unsigned options) {
//     typedef std::unique_ptr<libsbml::SBMLDocument> SBMLDocumentPtr;
    Log(Logger::LOG_DEBUG) << "libSBML: read SBML from string";
    libsbml::SBMLDocument* doc{checkedReadSBMLFromString(sbml.c_str())};

    Log(Logger::LOG_DEBUG) << "Begin loading model";
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

    // from LLVM component - should keep?
//     assert(sizeof(modelDataFieldsNames) / sizeof(const char*)
//             == NotSafe_FloatingSpeciesAmounts + 1
//             && "wrong number of items in modelDataFieldsNames");

    // add floating species initially without props so they can be referenced by rules
    for (uint i = 0; i < getDocument()->getModel()->getListOfSpecies()->size(); ++i) {
        const libsbml::Species *s = getDocument()->getModel()->getListOfSpecies()->get(i);

        const std::string& id = s->getId();

        // not a floating species
        if (s->getBoundaryCondition())
            addSpecies(BoundarySpeciesPtr(new BoundarySpecies(id)));
        else
            addSpecies(FloatingSpeciesPtr(new FloatingSpecies(id)));
    }

    // add reactions
    for (uint i = 0; i < getDocument()->getModel()->getListOfReactions()->size(); ++i) {
        const libsbml::Reaction *sbmlrxn = getDocument()->getModel()->getListOfReactions()->get(i);

        const std::string& id = sbmlrxn->getId();

        if (sbmlrxn->getFast())
            throw_gpusim_exception("Fast reactions not yet supported");

        Reaction* r = addReaction(ReactionPtr(new Reaction(id)));
        r->setSBMLReaction(sbmlrxn);
        r->setSBMLModel(getDocument()->getModel());

        // add reactants
        for (uint i_react=0; i_react < sbmlrxn->getNumReactants(); ++i_react) {
            r->addReactant(getSpeciesById(sbmlrxn->getReactant(i_react)->getSpecies()));
        }

        // add products
        for (uint i_prod=0; i_prod < sbmlrxn->getNumProducts(); ++i_prod) {
            r->addProduct(getSpeciesById(sbmlrxn->getProduct(i_prod)->getSpecies()));
        }

        // kinetic law
        const libsbml::ASTNode* root = sbmlrxn->getKineticLaw()->getMath();
        r->getKineticLaw().getAlgebra().setRoot(ConvertSbmlASTNode(r, root));
    }

    // read SBML rules and add them to the model
    {
        const libsbml::ListOfRules * rules = getDocument()->getModel()->getListOfRules();
        for (unsigned i = 0; i < rules->size(); ++i)
        {
            const libsbml::Rule *rule = rules->get(i);

            // assignment rule
            if (dynamic_cast<const libsbml::AssignmentRule*>(rule))
                addRule(ModelRulePtr(new AssignmentRule(rule->getVariable())));
            // rate rule
            else if (dynamic_cast<const libsbml::RateRule*>(rule))
                addRule(ModelRulePtr(new RateRule(rule->getId())));
            // algebraic rule
            else if (dynamic_cast<const libsbml::AlgebraicRule*>(rule)) {
                char* formula = SBML_formulaToString(rule->getMath());
                Log(Logger::LOG_WARNING)
                    << "Unable to handle algebraic rules. Formula '"
                    << formula << "' ignored.";
                addRule(ModelRulePtr(new AlgebraicRule(std::string(formula))));
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
    for (uint i = 0; i < getDocument()->getModel()->getListOfSpecies()->size(); ++i) {
        const libsbml::Species *s = getDocument()->getModel()->getListOfSpecies()->get(i);

        if (s->getBoundaryCondition())
            continue;

        const std::string& sid = s->getId();

        FloatingSpecies* species = getFloatingSpeciesById(sid);

        species->setIndex((int)i);

        if (getRules().contains(species))
            species->setIsIndependent(false);

        species->setIsConservedMoiety(conservation::ConservationExtension::getConservedMoiety(*s));

        // conserved moiety species assignment rules do not apply at
        // time t < 0
        species->setIsIndepInitFltSpecies(
          !hasInitialAssignmentRule(species) &&
          (!hasAssignmentRule(species) || species->getIsConservedMoiety()));

        if (s->isSetInitialConcentration())
            species->setInitialConcentration(s->getInitialConcentration());
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

# if GPUSIM_MODEL_USE_SBML

libsbml::SBMLDocument* GPUSimModel::getDocument() {
    if(moietyConverter)
        return moietyConverter->getDocument();
    else if(ownedDoc)
        return ownedDoc.get();
    else
        throw_gpusim_exception("Missing SBML document");
}

const libsbml::SBMLDocument* GPUSimModel::getDocument() const {
    if(moietyConverter)
        return moietyConverter->getDocument();
    else if(ownedDoc)
        return ownedDoc.get();
    else
        throw_gpusim_exception("Missing SBML document");
}

const libsbml::Model* GPUSimModel::getModel() {
    return getDocument()->getModel();
}

# endif

} // namespace rrgpu

} // namespace rr

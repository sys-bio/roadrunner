// Builds small SBML models with libsbml for the tests of the Jacobian.
#ifndef RR_TEST_SBML_MODEL_BUILDER_H
#define RR_TEST_SBML_MODEL_BUILDER_H

#include <sbml/SBMLTypes.h>

#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

/** A small SBML model built with libsbml, for patterns that can be checked by hand. */
class ModelBuilder {
public:
    ModelBuilder() : doc(new libsbml::SBMLDocument(3, 2)) {
        model = doc->createModel("m");
    }

    ModelBuilder &compartment(const std::string &id, bool constant = true) {
        auto *c = model->createCompartment();
        c->setId(id);
        c->setSize(1.0);
        c->setConstant(constant);
        c->setSpatialDimensions(3.0);
        return *this;
    }

    ModelBuilder &species(const std::string &id, const std::string &comp = "c", bool amounts = true) {
        auto *s = model->createSpecies();
        s->setId(id);
        s->setCompartment(comp);
        s->setInitialAmount(1.0);
        s->setHasOnlySubstanceUnits(amounts);
        s->setBoundaryCondition(false);
        s->setConstant(false);
        return *this;
    }

    ModelBuilder &parameter(const std::string &id, double value, bool constant = true) {
        auto *p = model->createParameter();
        p->setId(id);
        p->setValue(value);
        p->setConstant(constant);
        return *this;
    }

    ModelBuilder &reaction(const std::string &id, const std::vector<std::string> &reactants,
                           const std::vector<std::string> &products, const std::string &formula) {
        auto *r = model->createReaction();
        r->setId(id);
        r->setReversible(false);
        for (const auto &s: reactants) {
            auto *sr = r->createReactant();
            sr->setSpecies(s);
            sr->setStoichiometry(1.0);
            sr->setConstant(true);
        }
        for (const auto &s: products) {
            auto *sr = r->createProduct();
            sr->setSpecies(s);
            sr->setStoichiometry(1.0);
            sr->setConstant(true);
        }
        auto *kl = r->createKineticLaw();
        std::unique_ptr<libsbml::ASTNode> math(libsbml::SBML_parseL3Formula(formula.c_str()));
        kl->setMath(math.get());
        return *this;
    }

    ModelBuilder &assignmentRule(const std::string &variable, const std::string &formula) {
        auto *rule = model->createAssignmentRule();
        rule->setVariable(variable);
        std::unique_ptr<libsbml::ASTNode> math(libsbml::SBML_parseL3Formula(formula.c_str()));
        rule->setMath(math.get());
        return *this;
    }

    ModelBuilder &rateRule(const std::string &variable, const std::string &formula) {
        auto *rule = model->createRateRule();
        rule->setVariable(variable);
        std::unique_ptr<libsbml::ASTNode> math(libsbml::SBML_parseL3Formula(formula.c_str()));
        rule->setMath(math.get());
        return *this;
    }

    ModelBuilder &function(const std::string &id, const std::string &lambda) {
        auto *f = model->createFunctionDefinition();
        f->setId(id);
        std::unique_ptr<libsbml::ASTNode> math(libsbml::SBML_parseL3Formula(lambda.c_str()));
        f->setMath(math.get());
        return *this;
    }

    ModelBuilder &conversionFactor(const std::string &parameter) {
        model->setConversionFactor(parameter);
        return *this;
    }

    /** The SBML of the model. */
    std::string sbml() const {
        char *c = doc->toSBML();
        std::string s(c);
        free(c);
        return s;
    }

    std::unique_ptr<libsbml::SBMLDocument> doc;
    libsbml::Model *model;
};

#endif // RR_TEST_SBML_MODEL_BUILDER_H

// Tests of the conservative sparsity pattern and the column coloring of the Jacobian.

#include "gtest/gtest.h"
#include "llvm/JacobianSparsity.h"
#include "llvm/LLVMModelDataSymbols.h"

#include <sbml/SBMLTypes.h>

#include <algorithm>
#include <cstdlib>
#include <memory>
#include <set>
#include <string>
#include <vector>

using namespace rrllvm;

namespace {

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

        std::unique_ptr<libsbml::SBMLDocument> doc;
        libsbml::Model *model;
    };

    /** Entries (row, column) of a CSC pattern. */
    std::set<std::pair<int, int>> entries(const CscPattern &p) {
        std::set<std::pair<int, int>> e;
        for (int j = 0; j < p.nCols; ++j) {
            for (int k = p.colPtr[j]; k < p.colPtr[j + 1]; ++k) {
                e.insert({p.rowIdx[k], j});
            }
        }
        return e;
    }

    int stateColumn(const LLVMModelDataSymbols &sym, const std::string &id) {
        int rr = sym.getRateRuleIndex(id);
        if (rr >= 0) {
            return rr;
        }
        return (int) sym.getRateRuleSize() + sym.getFloatingSpeciesIndex(id);
    }
}

TEST(JacobianSparsityTests, MassActionPattern) {
    ModelBuilder b;
    b.compartment("c").species("S1").species("S2").parameter("k", 0.1)
            .reaction("R1", {"S1"}, {"S2"}, "k*S1");
    LLVMModelDataSymbols sym(b.model, 0);
    JacobianSparsity sp(*b.model, sym);

    int s1 = stateColumn(sym, "S1"), s2 = stateColumn(sym, "S2");
    int r1 = sym.getReactionIndex("R1");
    EXPECT_EQ(entries(sp.reactionRates()), (std::set<std::pair<int, int>>{{r1, s1}}));
    EXPECT_EQ(entries(sp.jacobian()), (std::set<std::pair<int, int>>{{s1, s1}, {s2, s1}, {s2, s2}}));
}

TEST(JacobianSparsityTests, AssignmentRuleChain) {
    ModelBuilder b;
    b.compartment("c").species("S1").species("S2").species("S3").parameter("p", 0.0, false)
            .assignmentRule("p", "2*S3")
            .reaction("R1", {"S1"}, {"S2"}, "p");
    LLVMModelDataSymbols sym(b.model, 0);
    JacobianSparsity sp(*b.model, sym);

    int r1 = sym.getReactionIndex("R1");
    EXPECT_EQ(entries(sp.reactionRates()), (std::set<std::pair<int, int>>{{r1, stateColumn(sym, "S3")}}));
}

TEST(JacobianSparsityTests, FunctionDefinitionArgs) {
    ModelBuilder b;
    b.compartment("c").species("S1").species("S2").parameter("k", 0.1)
            .function("f", "lambda(x, y, x*y)")
            .reaction("R1", {"S1"}, {"S2"}, "f(S1, k)");
    LLVMModelDataSymbols sym(b.model, 0);
    JacobianSparsity sp(*b.model, sym);

    int r1 = sym.getReactionIndex("R1");
    EXPECT_EQ(entries(sp.reactionRates()), (std::set<std::pair<int, int>>{{r1, stateColumn(sym, "S1")}}));
}

TEST(JacobianSparsityTests, PiecewiseInactiveBranchInPattern) {
    ModelBuilder b;
    b.compartment("c").species("S1").species("S2").parameter("k", 0.1)
            .reaction("R1", {"S2"}, {}, "piecewise(S1, time > 1000, 0)");
    LLVMModelDataSymbols sym(b.model, 0);
    JacobianSparsity sp(*b.model, sym);

    int r1 = sym.getReactionIndex("R1");
    EXPECT_TRUE(entries(sp.reactionRates()).count({r1, stateColumn(sym, "S1")}));
}

TEST(JacobianSparsityTests, VariableCompartmentColumn) {
    ModelBuilder b;
    b.compartment("c", false).species("S1", "c", false).species("S2").parameter("k", 0.1)
            .rateRule("c", "0.01")
            .reaction("R1", {"S1"}, {"S2"}, "k*S1");
    LLVMModelDataSymbols sym(b.model, 0);
    JacobianSparsity sp(*b.model, sym);

    int r1 = sym.getReactionIndex("R1");
    auto e = entries(sp.reactionRates());
    EXPECT_TRUE(e.count({r1, stateColumn(sym, "c")}));
    EXPECT_TRUE(e.count({r1, stateColumn(sym, "S1")}));
}

TEST(JacobianSparsityTests, ColoringIsValid) {
    const char *env = std::getenv("RR_ENZYME_MODELS");
    std::string path = (env ? std::string(env) : std::string(std::getenv("HOME")) + "/git/hctz-model/models")
                       + "/hctz_body_flat.xml";
    std::unique_ptr<libsbml::SBMLDocument> doc(libsbml::readSBMLFromFile(path.c_str()));
    ASSERT_EQ(doc->getNumErrors(libsbml::LIBSBML_SEV_FATAL) + doc->getNumErrors(libsbml::LIBSBML_SEV_ERROR), 0u);
    LLVMModelDataSymbols sym(doc->getModel(), 0);
    JacobianSparsity sp(*doc->getModel(), sym);

    const std::vector<int> &colors = sp.colors();
    int n = sp.reactionRates().nCols;
    ASSERT_EQ((int) colors.size(), n);

    // rows of the stacked [dv/dy; d(rate rule rates)/dy]
    std::vector<std::vector<int>> rows(sp.reactionRates().nRows + sp.rateRules().nRows);
    for (int j = 0; j < n; ++j) {
        for (int k = sp.reactionRates().colPtr[j]; k < sp.reactionRates().colPtr[j + 1]; ++k)
            rows[sp.reactionRates().rowIdx[k]].push_back(j);
        for (int k = sp.rateRules().colPtr[j]; k < sp.rateRules().colPtr[j + 1]; ++k)
            rows[sp.reactionRates().nRows + sp.rateRules().rowIdx[k]].push_back(j);
    }
    size_t maxRow = 0;
    for (const auto &row: rows) {
        std::set<int> seen;
        for (int j: row) {
            EXPECT_TRUE(seen.insert(colors[j]).second) << "two columns of color " << colors[j] << " share a row";
        }
        maxRow = std::max(maxRow, row.size());
    }
    EXPECT_GE(sp.numColors(), (int) maxRow);
    EXPECT_LT(sp.numColors(), n);
    std::cout << "hctz_body: " << n << " states, " << sp.numColors() << " colors, largest row " << maxRow
              << std::endl;
}

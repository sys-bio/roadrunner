#include "gtest/gtest.h"

#include "llvm/AlgebraicRuleConverter.h"
#include "llvm/LLVMException.h"
#include "rrRoadRunner.h"
#include "rrRoadRunnerOptions.h"

#include <sbml/SBMLTypes.h>

#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <vector>

using namespace libsbml;
using rr::RoadRunner;

namespace {
    const double NO_VALUE = -1234.5;

    /** A model made of parameters and rules, with no reactions. */
    class ModelBuilder {
    public:
        ModelBuilder() : doc(std::make_unique<SBMLDocument>(3, 2)), model(doc->createModel("m")) {}

        ModelBuilder& parameter(const std::string& id, double value = NO_VALUE, bool constant = false) {
            Parameter* p = model->createParameter();
            p->setId(id);
            p->setConstant(constant);
            if (value != NO_VALUE) {
                p->setValue(value);
            }
            return *this;
        }

        ModelBuilder& algebraic(const std::string& formula) {
            std::unique_ptr<ASTNode> math(SBML_parseL3Formula(formula.c_str()));
            model->createAlgebraicRule()->setMath(math.get());
            return *this;
        }

        ModelBuilder& assignment(const std::string& variable, const std::string& formula) {
            std::unique_ptr<ASTNode> math(SBML_parseL3Formula(formula.c_str()));
            AssignmentRule* rule = model->createAssignmentRule();
            rule->setVariable(variable);
            rule->setMath(math.get());
            return *this;
        }

        ModelBuilder& initial(const std::string& symbol, const std::string& formula) {
            std::unique_ptr<ASTNode> math(SBML_parseL3Formula(formula.c_str()));
            InitialAssignment* assignment = model->createInitialAssignment();
            assignment->setSymbol(symbol);
            assignment->setMath(math.get());
            return *this;
        }

        ModelBuilder& rate(const std::string& variable, const std::string& formula) {
            std::unique_ptr<ASTNode> math(SBML_parseL3Formula(formula.c_str()));
            RateRule* rule = model->createRateRule();
            rule->setVariable(variable);
            rule->setMath(math.get());
            return *this;
        }

        ModelBuilder& event(const std::string& trigger, const std::string& variable, const std::string& formula) {
            Event* e = model->createEvent();
            e->setUseValuesFromTriggerTime(true);
            Trigger* t = e->createTrigger();
            t->setInitialValue(false);
            t->setPersistent(true);
            std::unique_ptr<ASTNode> triggerMath(SBML_parseL3Formula(trigger.c_str()));
            t->setMath(triggerMath.get());
            EventAssignment* a = e->createEventAssignment();
            a->setVariable(variable);
            std::unique_ptr<ASTNode> math(SBML_parseL3Formula(formula.c_str()));
            a->setMath(math.get());
            return *this;
        }

        ModelBuilder& function(const std::string& id, const std::string& lambda) {
            FunctionDefinition* f = model->createFunctionDefinition();
            f->setId(id);
            std::unique_ptr<ASTNode> math(SBML_parseL3Formula(lambda.c_str()));
            f->setMath(math.get());
            return *this;
        }

        const SBMLDocument* document() const {
            return doc.get();
        }

        std::string sbml() const {
            return writeSBMLToStdString(doc.get());
        }

    private:
        std::unique_ptr<SBMLDocument> doc;
        Model* model;
    };

    std::vector<std::string> assignedVariables(const SBMLDocument* doc) {
        std::vector<std::string> variables;
        const Model* model = doc->getModel();
        for (unsigned i = 0; i < model->getNumRules(); ++i) {
            if (model->getRule(i)->isAssignment()) {
                variables.push_back(model->getRule(i)->getVariable());
            }
        }
        return variables;
    }

    /** The symbols assigned by the conversion, leaving out the model's own rules. */
    std::vector<std::string> solvedFor(const ModelBuilder& builder) {
        std::unique_ptr<SBMLDocument> converted(rrllvm::convertAlgebraicRules(builder.document()));
        EXPECT_TRUE(converted != nullptr);
        std::vector<std::string> variables = assignedVariables(converted.get());
        variables.erase(variables.begin(), variables.begin() + assignedVariables(builder.document()).size());
        return variables;
    }

    double valueOf(const ModelBuilder& builder, const std::string& id) {
        RoadRunner rr(builder.sbml());
        return rr.getValue(id);
    }

    std::string errorFrom(const ModelBuilder& builder) {
        try {
            std::unique_ptr<SBMLDocument> converted(rrllvm::convertAlgebraicRules(builder.document()));
        }
        catch (const rrllvm::LLVMException& e) {
            return e.what();
        }
        return "";
    }
}

TEST(AlgebraicRuleConverterTests, ReturnsNullWithoutAlgebraicRules) {
    ModelBuilder builder;
    builder.parameter("x", 1).assignment("x", "2");
    EXPECT_EQ(rrllvm::convertAlgebraicRules(builder.document()), nullptr);
}

TEST(AlgebraicRuleConverterTests, DoesNotChangeSourceDocument) {
    ModelBuilder builder;
    builder.parameter("x").algebraic("x - 2");
    std::unique_ptr<SBMLDocument> converted(rrllvm::convertAlgebraicRules(builder.document()));
    EXPECT_TRUE(builder.document()->getModel()->getRule(0u)->isAlgebraic());
    EXPECT_TRUE(converted->getModel()->getRule(0u)->isAssignment());
}

TEST(AlgebraicRuleConverterTests, DropsRuleWithoutMath) {
    ModelBuilder builder;
    builder.parameter("x", 3);
    const_cast<SBMLDocument*>(builder.document())->getModel()->createAlgebraicRule();
    std::unique_ptr<SBMLDocument> converted(rrllvm::convertAlgebraicRules(builder.document()));
    EXPECT_EQ(converted->getModel()->getNumRules(), 0u);
}

TEST(AlgebraicRuleConverterTests, SolvesRuleThatIsOnlyASymbol) {
    ModelBuilder builder;
    builder.parameter("x", 5).algebraic("x");
    EXPECT_DOUBLE_EQ(valueOf(builder, "x"), 0.0);
}

TEST(AlgebraicRuleConverterTests, SolvesWithCoefficient) {
    ModelBuilder builder;
    builder.parameter("k", 4, true).parameter("x").algebraic("2 * k * x / 8 - 3");
    EXPECT_DOUBLE_EQ(valueOf(builder, "x"), 3.0);
}

TEST(AlgebraicRuleConverterTests, SolvesNegatedTerms) {
    ModelBuilder builder;
    builder.parameter("x").algebraic("7 - (2 - x) * 3");
    EXPECT_NEAR(valueOf(builder, "x"), -1.0 / 3.0, 1e-12);
}

TEST(AlgebraicRuleConverterTests, KeepsCoefficientNextToLargeTerms) {
    ModelBuilder builder;
    builder.parameter("big", 1e20, true).parameter("x").algebraic("x + big - big - 1");
    EXPECT_DOUBLE_EQ(valueOf(builder, "x"), 1.0);
}

TEST(AlgebraicRuleConverterTests, ExpandsFunctionDefinitions) {
    ModelBuilder builder;
    builder.function("twice", "lambda(a, 2 * a)").parameter("x").algebraic("twice(x) - 8");
    EXPECT_DOUBLE_EQ(valueOf(builder, "x"), 4.0);
}

TEST(AlgebraicRuleConverterTests, ExpandsAssignmentRules) {
    ModelBuilder builder;
    builder.parameter("x").parameter("y").assignment("y", "x + 1").algebraic("y - 5");
    EXPECT_EQ(solvedFor(builder), std::vector<std::string>({ "x" }));
    EXPECT_DOUBLE_EQ(valueOf(builder, "x"), 4.0);
}

TEST(AlgebraicRuleConverterTests, SolvesChainedRules) {
    ModelBuilder builder;
    builder.parameter("x").parameter("y").algebraic("x + y - 10").algebraic("y - 4");
    EXPECT_DOUBLE_EQ(valueOf(builder, "x"), 6.0);
    EXPECT_DOUBLE_EQ(valueOf(builder, "y"), 4.0);
}

TEST(AlgebraicRuleConverterTests, FollowsRateRuleDuringSimulation) {
    ModelBuilder builder;
    builder.parameter("x", 1).parameter("y").rate("x", "1").algebraic("x + y - 10");
    RoadRunner rr(builder.sbml());
    rr.simulate(0, 4, 5);
    EXPECT_NEAR(rr.getValue("y"), 5.0, 1e-6);
}

TEST(AlgebraicRuleConverterTests, PrefersSymbolWithoutInitialValue) {
    ModelBuilder builder;
    builder.parameter("a", 1).parameter("b").parameter("c", 2).algebraic("a + b - c");
    EXPECT_EQ(solvedFor(builder), std::vector<std::string>({ "b" }));
}

TEST(AlgebraicRuleConverterTests, DoesNotSolveForEventTarget) {
    ModelBuilder builder;
    builder.parameter("a", 1).parameter("b", 2).event("time > 1", "a", "5").algebraic("a - b");
    EXPECT_EQ(solvedFor(builder), std::vector<std::string>({ "b" }));
}

TEST(AlgebraicRuleConverterTests, RejectsNonlinearRule) {
    ModelBuilder builder;
    builder.parameter("x").algebraic("x * x - 4");
    EXPECT_NE(errorFrom(builder).find("not supported"), std::string::npos);
}

TEST(AlgebraicRuleConverterTests, RejectsPowerOfUnknown) {
    ModelBuilder builder;
    builder.parameter("x").algebraic("x ^ 2 - 4");
    EXPECT_NE(errorFrom(builder).find("not supported"), std::string::npos);
}

TEST(AlgebraicRuleConverterTests, RejectsCoefficientThatChangesWithTime) {
    ModelBuilder builder;
    builder.parameter("x").algebraic("time * x - 1");
    EXPECT_NE(errorFrom(builder).find("constant coefficient"), std::string::npos);
}

TEST(AlgebraicRuleConverterTests, RejectsCoefficientFromRateRule) {
    ModelBuilder builder;
    builder.parameter("s", 1).parameter("x").rate("s", "-1").algebraic("s * x - 1");
    EXPECT_NE(errorFrom(builder).find("constant coefficient"), std::string::npos);
}

TEST(AlgebraicRuleConverterTests, RejectsCancelledSymbol) {
    ModelBuilder builder;
    builder.parameter("x").algebraic("x - x + 1");
    EXPECT_NE(errorFrom(builder).find("not supported"), std::string::npos);
}

TEST(AlgebraicRuleConverterTests, RejectsRuleWithNothingToDetermine) {
    ModelBuilder builder;
    builder.parameter("x", 1).rate("x", "1").algebraic("x - 1");
    EXPECT_NE(errorFrom(builder).find("not supported"), std::string::npos);
}

TEST(AlgebraicRuleConverterTests, DoesNotSolveForRateOfTarget) {
    // x has no initial value, so it would normally be preferred.
    ModelBuilder builder;
    builder.parameter("x").parameter("y", 1).parameter("w")
        .assignment("w", "rateOf(x)").algebraic("x + y - 3");
    EXPECT_EQ(solvedFor(builder), std::vector<std::string>{ "y" });
}

TEST(AlgebraicRuleConverterTests, SolvesSimultaneousRules) {
    ModelBuilder builder;
    builder.parameter("x").parameter("y").algebraic("x + y - 3").algebraic("x - y - 1");
    EXPECT_NEAR(valueOf(builder, "x"), 2, 1e-12);
    EXPECT_NEAR(valueOf(builder, "y"), 1, 1e-12);
}

TEST(AlgebraicRuleConverterTests, SolvesThreeSimultaneousRulesWithZeroLeadingCoefficient) {
    // x is missing from the first rule, so the rows have to be swapped.
    ModelBuilder builder;
    builder.parameter("x").parameter("y").parameter("z")
        .algebraic("y + z - 5").algebraic("x + y + z - 6").algebraic("x - y + 2 * z - 5");
    EXPECT_NEAR(valueOf(builder, "x"), 1, 1e-12);
    EXPECT_NEAR(valueOf(builder, "y"), 2, 1e-12);
    EXPECT_NEAR(valueOf(builder, "z"), 3, 1e-12);
}

TEST(AlgebraicRuleConverterTests, SolvesRingOfFiveSimultaneousRules) {
    ModelBuilder builder;
    builder.parameter("a").parameter("b").parameter("c").parameter("d").parameter("e")
        .parameter("r", 1).rate("r", "1")
        .algebraic("a + b - 3 * r").algebraic("b + c - 5 * r").algebraic("c + d - 7 * r")
        .algebraic("d + e - 9 * r").algebraic("e + a - 6 * r");
    RoadRunner rr(builder.sbml());
    EXPECT_NEAR(rr.getValue("a"), 1, 1e-12);
    EXPECT_NEAR(rr.getValue("b"), 2, 1e-12);
    EXPECT_NEAR(rr.getValue("c"), 3, 1e-12);
    EXPECT_NEAR(rr.getValue("d"), 4, 1e-12);
    EXPECT_NEAR(rr.getValue("e"), 5, 1e-12);
}

TEST(AlgebraicRuleConverterTests, SimultaneousRulesFollowRateRule) {
    ModelBuilder builder;
    builder.parameter("x").parameter("y").parameter("s", 1)
        .rate("s", "1").algebraic("x + y - s").algebraic("x - y");
    RoadRunner rr(builder.sbml());
    rr.simulate(0, 3, 4);
    EXPECT_NEAR(rr.getValue("s"), 4, 1e-6);
    EXPECT_NEAR(rr.getValue("x"), 2, 1e-6);
    EXPECT_NEAR(rr.getValue("y"), 2, 1e-6);
}

TEST(AlgebraicRuleConverterTests, SimultaneousRulesUseConstantCoefficients) {
    ModelBuilder builder;
    builder.parameter("x").parameter("y").parameter("a", 2, true).parameter("b", 4, true)
        .algebraic("a * x + y - 7").algebraic("x + b * y - 7");
    EXPECT_NEAR(valueOf(builder, "x"), 3, 1e-12);
    EXPECT_NEAR(valueOf(builder, "y"), 1, 1e-12);
}

TEST(AlgebraicRuleConverterTests, SimultaneousRulesHonourConstantChangedAfterLoading) {
    ModelBuilder builder;
    builder.parameter("x").parameter("y").parameter("k", 3, true)
        .algebraic("k * x + y - 7").algebraic("x - y - 1");
    RoadRunner rr(builder.sbml());
    EXPECT_NEAR(rr.getValue("x"), 2, 1e-12);
    rr.setValue("k", 7);
    EXPECT_NEAR(rr.getValue("x"), 1, 1e-12);
    EXPECT_NEAR(rr.getValue("y"), 0, 1e-12);
}

TEST(AlgebraicRuleConverterTests, SimultaneousRulesCombineWithChainedRule) {
    ModelBuilder builder;
    builder.parameter("w").parameter("x").parameter("y")
        .algebraic("w - x - y").algebraic("x + y - 3").algebraic("x - y - 1");
    EXPECT_NEAR(valueOf(builder, "w"), 3, 1e-12);
    EXPECT_NEAR(valueOf(builder, "x"), 2, 1e-12);
    EXPECT_NEAR(valueOf(builder, "y"), 1, 1e-12);
}

TEST(AlgebraicRuleConverterTests, SimultaneousRulesLeaveExtraSymbolFree) {
    ModelBuilder builder;
    builder.parameter("x").parameter("y").parameter("z", 4)
        .algebraic("x + y + z - 7").algebraic("x - y + z - 5");
    const std::vector<std::string> variables = solvedFor(builder);
    EXPECT_EQ(std::count(variables.begin(), variables.end(), "z"), 0);
    EXPECT_NEAR(valueOf(builder, "x"), 2, 1e-12);
    EXPECT_NEAR(valueOf(builder, "y"), 1, 1e-12);
}

TEST(AlgebraicRuleConverterTests, SimultaneousRulesUseInitialAssignmentForCoefficient) {
    ModelBuilder builder;
    builder.parameter("x").parameter("y").parameter("k", NO_VALUE, true)
        .initial("k", "1 + 1").algebraic("k * x + y - 7").algebraic("x + y - 4");
    EXPECT_NEAR(valueOf(builder, "x"), 3, 1e-12);
    EXPECT_NEAR(valueOf(builder, "y"), 1, 1e-12);
}

TEST(AlgebraicRuleConverterTests, RejectsDependentSimultaneousRules) {
    ModelBuilder builder;
    builder.parameter("x").parameter("y").algebraic("x + y - 1").algebraic("2 * x + 2 * y - 3");
    EXPECT_NE(errorFrom(builder).find("no unique"), std::string::npos);
}

TEST(AlgebraicRuleConverterTests, RejectsDependentRulesDespiteRoundOff) {
    ModelBuilder builder;
    builder.parameter("x").parameter("y").algebraic("0.1 * x + 0.2 * y - 1").algebraic("0.3 * x + 0.6 * y - 2");
    EXPECT_NE(errorFrom(builder).find("no unique"), std::string::npos);
}

TEST(AlgebraicRuleConverterTests, RejectsMoreRulesThanSymbols) {
    ModelBuilder builder;
    builder.parameter("x").parameter("y")
        .algebraic("x + y - 3").algebraic("x - y - 1").algebraic("x + 2 * y - 4");
    EXPECT_NE(errorFrom(builder).find("no unique"), std::string::npos);
}

TEST(AlgebraicRuleConverterTests, RejectsDependenceThatIsNotLinear) {
    ModelBuilder builder;
    builder.parameter("x").parameter("y").algebraic("x + y * y - 3").algebraic("y + x * x - 3");
    EXPECT_NE(errorFrom(builder).find("not linear"), std::string::npos);
}

TEST(AlgebraicRuleConverterTests, RoadRunnerReportsUnsupportedRule) {
    ModelBuilder builder;
    builder.parameter("x").algebraic("x * x - 4");
    EXPECT_THROW(RoadRunner rr(builder.sbml()), rrllvm::LLVMException);
}

TEST(AlgebraicRuleConverterTests, RoadRunnerKeepsOriginalRuleInSBML) {
    ModelBuilder builder;
    builder.parameter("x").algebraic("x - 2");
    RoadRunner rr(builder.sbml());
    EXPECT_NE(rr.getSBML().find("algebraicRule"), std::string::npos);
    EXPECT_NE(rr.getCurrentSBML().find("algebraicRule"), std::string::npos);
}

TEST(AlgebraicRuleConverterTests, CurrentSBMLCanBeReloaded) {
    ModelBuilder builder;
    builder.parameter("t0", 1).parameter("x").rate("t0", "1").algebraic("x - 2 * t0");
    RoadRunner rr(builder.sbml());
    rr.simulate(0, 2, 3);
    RoadRunner reloaded(rr.getCurrentSBML());
    EXPECT_NEAR(reloaded.getValue("x"), rr.getValue("x"), 1e-6);
}

TEST(AlgebraicRuleConverterTests, WorksWithConservedMoieties) {
    ModelBuilder builder;
    builder.parameter("x").algebraic("x - 2");
    RoadRunner rr(builder.sbml());
    rr.setConservedMoietyAnalysis(true);
    EXPECT_DOUBLE_EQ(rr.getValue("x"), 2.0);
}

TEST(AlgebraicRuleConverterTests, SurvivesSaveAndLoadState) {
    ModelBuilder builder;
    builder.parameter("t0", 1).parameter("x").rate("t0", "1").algebraic("x - 2 * t0");
    RoadRunner rr(builder.sbml());
    rr.simulate(0, 2, 3);
    const double before = rr.getValue("x");
    RoadRunner restored;
    restored.loadStateS(rr.saveStateS()); // takes ownership of the stream
    EXPECT_NEAR(restored.getValue("x"), before, 1e-9);
}

TEST(AlgebraicRuleConverterTests, SetValueOnSolvedSymbolIsRefused) {
    ModelBuilder builder;
    builder.parameter("x").algebraic("x - 2");
    RoadRunner rr(builder.sbml());
    EXPECT_ANY_THROW(rr.setValue("x", 9));
}

TEST(AlgebraicRuleConverterTests, ReloadsAfterAddingAParameter) {
    ModelBuilder builder;
    builder.parameter("x").algebraic("x - 2");
    RoadRunner rr(builder.sbml());
    rr.addParameter("extra", 1.0);
    EXPECT_DOUBLE_EQ(rr.getValue("x"), 2.0);
}

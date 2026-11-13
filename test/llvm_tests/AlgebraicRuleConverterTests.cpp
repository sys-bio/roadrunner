#include "gtest/gtest.h"

#include "llvm/AlgebraicRuleConverter.h"
#include "llvm/LLVMException.h"
#include "rrRoadRunner.h"
#include "rrRoadRunnerOptions.h"

#include <sbml/SBMLTypes.h>

#include <algorithm>
#include <cmath>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace libsbml;
using rr::RoadRunner;

namespace {
    const double NO_VALUE = -1234.5;

    /** A model made of parameters and rules, with no reactions. */
    class ModelBuilder {
    public:
        explicit ModelBuilder(unsigned level = 3, unsigned version = 2)
            : doc(std::make_unique<SBMLDocument>(level, version)), model(doc->createModel("m")) {}

        ModelBuilder& parameter(const std::string& id, double value = NO_VALUE, bool constant = false) {
            Parameter* p = model->createParameter();
            p->setId(id);
            p->setConstant(constant);
            if (value != NO_VALUE) {
                p->setValue(value);
            }
            return *this;
        }

        ModelBuilder& algebraic(const std::string& formula, const std::string& id = "") {
            std::unique_ptr<ASTNode> math(SBML_parseL3Formula(formula.c_str()));
            AlgebraicRule* rule = model->createAlgebraicRule();
            rule->setMath(math.get());
            if (!id.empty()) {
                rule->setIdAttribute(id);
            }
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

TEST(AlgebraicRuleConverterTests, SimultaneousRulesSurviveCoefficientBecomingZero) {
    // With k = 2 at load, elimination would divide by k. After k = 0 the
    // system still has the unique solution x = y = 1.
    ModelBuilder builder;
    builder.parameter("x").parameter("y").parameter("k", 2, true)
        .algebraic("k * x + y - 1").algebraic("x + y - 2");
    RoadRunner rr(builder.sbml());
    EXPECT_NEAR(rr.getValue("x"), -1, 1e-12);
    rr.setValue("k", 0);
    EXPECT_NEAR(rr.getValue("x"), 1, 1e-12);
    EXPECT_NEAR(rr.getValue("y"), 1, 1e-12);
    rr.setValue("k", 3);
    EXPECT_NEAR(rr.getValue("x"), -0.5, 1e-12);
    EXPECT_NEAR(rr.getValue("y"), 2.5, 1e-12);
}

TEST(AlgebraicRuleConverterTests, ThreeSimultaneousRulesSurviveCoefficientsChanging) {
    ModelBuilder builder;
    builder.parameter("x").parameter("y").parameter("z")
        .parameter("a", 5, true).parameter("b", 1, true)
        .algebraic("a * x + y + z - 6").algebraic("x + y - 2").algebraic("b * y + z - 3");
    RoadRunner rr(builder.sbml());
    // a = 5, b = 1: x = 0.6, y = 1.4, z = 1.6. Elimination pivots on a.
    EXPECT_NEAR(rr.getValue("x"), 0.6, 1e-12);
    EXPECT_NEAR(rr.getValue("y"), 1.4, 1e-12);
    EXPECT_NEAR(rr.getValue("z"), 1.6, 1e-12);
    // a = 0, b = 2: y + z = 6, x + y = 2, 2y + z = 3 -> x = 5, y = -3, z = 9
    rr.setValue("a", 0);
    rr.setValue("b", 2);
    EXPECT_NEAR(rr.getValue("x"), 5, 1e-12);
    EXPECT_NEAR(rr.getValue("y"), -3, 1e-12);
    EXPECT_NEAR(rr.getValue("z"), 9, 1e-12);
}

TEST(AlgebraicRuleConverterTests, LargeSimultaneousSystemWithConstantCoefficientLoads) {
    // Larger than Cramer's rule is used for, so elimination is kept.
    ModelBuilder builder;
    builder.parameter("k", 2, true);
    const int n = 7;
    for (int i = 0; i < n; ++i) {
        builder.parameter("x" + std::to_string(i));
    }
    for (int i = 0; i < n; ++i) {
        builder.algebraic("k * x" + std::to_string(i) + " + x" + std::to_string((i + 1) % n) + " - 3");
    }
    // Every x is 1.
    EXPECT_NEAR(valueOf(builder, "x0"), 1, 1e-12);
    EXPECT_NEAR(valueOf(builder, "x6"), 1, 1e-12);
}

TEST(AlgebraicRuleConverterTests, RuleSolvesForOtherSymbolWhenSolvedOneBecomesConstant) {
    ModelBuilder builder;
    builder.parameter("s1", 6).parameter("s2", 3).algebraic("s1 + 2 * s2 - 10");
    RoadRunner rr(builder.sbml());
    EXPECT_EQ(solvedFor(builder), std::vector<std::string>{ "s1" });
    EXPECT_NEAR(rr.getValue("s1"), 4, 1e-12);
    rr.setConstant("s1", true);
    rr.setValue("s1", 8);
    EXPECT_NEAR(rr.getValue("s2"), 1, 1e-12);
    EXPECT_ANY_THROW(rr.setValue("s2", 5));
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

// One rule, two symbols: which one it determines depends on which is constant.

TEST(AlgebraicRuleConverterTests, OneRuleTwoSymbolsDeterminesTheVariableOne) {
    ModelBuilder xConstant;
    xConstant.parameter("x", 1, true).parameter("y").algebraic("x + y + 3");
    EXPECT_EQ(solvedFor(xConstant), std::vector<std::string>{ "y" });
    EXPECT_NEAR(valueOf(xConstant, "y"), -4, 1e-12);

    ModelBuilder yConstant;
    yConstant.parameter("x").parameter("y", 2, true).algebraic("x + y + 3");
    EXPECT_EQ(solvedFor(yConstant), std::vector<std::string>{ "x" });
    EXPECT_NEAR(valueOf(yConstant, "x"), -5, 1e-12);
}

namespace {
    std::string setValueError(RoadRunner& rr, const std::string& id) {
        try {
            rr.setValue(id, 9);
        }
        catch (const std::exception& e) {
            return e.what();
        }
        return "";
    }
}

TEST(AlgebraicRuleConverterTests, SettingSolvedSymbolNamesTheAlgebraicRule) {
    ModelBuilder builder;
    builder.parameter("x", 1, true).parameter("y").parameter("z").parameter("w", 2)
        .algebraic("x + y + 3").assignment("z", "2 * x");
    RoadRunner rr(builder.sbml());

    const std::string solved = setValueError(rr, "y");
    EXPECT_NE(solved.find("algebraic rule"), std::string::npos) << solved;
    EXPECT_EQ(solved.find("assignment rule"), std::string::npos) << solved;

    const std::string assigned = setValueError(rr, "z");
    EXPECT_NE(assigned.find("assignment rule"), std::string::npos) << assigned;

    // A symbol the rule mentions but doesn't solve for can still be set.
    EXPECT_EQ(setValueError(rr, "x"), "");
    EXPECT_NEAR(rr.getValue("y"), -12, 1e-12);
    EXPECT_EQ(setValueError(rr, "w"), "");
}

TEST(AlgebraicRuleConverterTests, OneRuleTwoSymbolsRejectedWhenBothConstant) {
    ModelBuilder builder;
    builder.parameter("x", 1, true).parameter("y", 2, true).algebraic("x + y + 3");
    EXPECT_FALSE(errorFrom(builder).empty());
}

// Coupled rules over a time course. The SBML Test Suite has no case where
// algebraic rules must be solved together.

TEST(AlgebraicRuleConverterTests, CoupledRulesFollowSpeciesChangedByReaction) {
    const std::string sbml = R"(<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">
  <model id="m">
    <listOfCompartments><compartment id="C" size="1" constant="true"/></listOfCompartments>
    <listOfSpecies>
      <species id="S1" compartment="C" initialAmount="10" hasOnlySubstanceUnits="true"
               boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="k" value="0.5" constant="true"/>
      <parameter id="P" constant="false"/>
      <parameter id="Q" constant="false"/>
    </listOfParameters>
    <listOfRules>
      <algebraicRule><math xmlns="http://www.w3.org/1998/Math/MathML">
        <apply><minus/><apply><plus/><ci>P</ci><ci>Q</ci></apply><ci>S1</ci></apply></math></algebraicRule>
      <algebraicRule><math xmlns="http://www.w3.org/1998/Math/MathML">
        <apply><minus/><apply><minus/><ci>P</ci><ci>Q</ci></apply><cn>2</cn></apply></math></algebraicRule>
    </listOfRules>
    <listOfReactions>
      <reaction id="R" reversible="false">
        <listOfReactants><speciesReference species="S1" stoichiometry="1" constant="true"/></listOfReactants>
        <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply><times/><ci>k</ci><ci>S1</ci></apply></math></kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>)";
    RoadRunner rr(sbml);
    rr.simulate(0, 2, 5);
    const double s1 = 10 * std::exp(-1.0);
    EXPECT_NEAR(rr.getValue("S1"), s1, 1e-5);
    EXPECT_NEAR(rr.getValue("P"), (s1 + 2) / 2, 1e-5);
    EXPECT_NEAR(rr.getValue("Q"), (s1 - 2) / 2, 1e-5);
}

TEST(AlgebraicRuleConverterTests, CoupledRulesFollowRateRuleAndChangedCoefficient) {
    ModelBuilder builder;
    builder.parameter("A", 4).parameter("k", 1, true).parameter("x").parameter("y")
        .rate("A", "-A").algebraic("k * x + y - A").algebraic("x - y - 1");
    RoadRunner rr(builder.sbml());
    rr.simulate(0, 1, 3);
    double a = 4 * std::exp(-1.0);
    EXPECT_NEAR(rr.getValue("x"), (a + 1) / 2, 1e-5);
    rr.setValue("k", 3);
    rr.simulate(1, 1, 3); // start, duration, points
    a = 4 * std::exp(-2.0);
    EXPECT_NEAR(rr.getValue("x"), (a + 1) / 4, 1e-5);
    EXPECT_NEAR(rr.getValue("y"), (a + 1) / 4 - 1, 1e-5);
}

TEST(AlgebraicRuleConverterTests, ThreeCoupledRulesFollowTwoRateRules) {
    ModelBuilder builder;
    builder.parameter("u", 0).parameter("v", 1).parameter("x").parameter("y").parameter("z")
        .rate("u", "1").rate("v", "-v")
        .algebraic("x + y + z - u").algebraic("x - y - v").algebraic("y - z");
    RoadRunner rr(builder.sbml());
    rr.simulate(0, 3, 4);
    // y = z, x = y + v, 3y + v = u
    const double u = 3;
    const double v = std::exp(-3.0);
    const double y = (u - v) / 3;
    EXPECT_NEAR(rr.getValue("y"), y, 1e-5);
    EXPECT_NEAR(rr.getValue("z"), y, 1e-5);
    EXPECT_NEAR(rr.getValue("x"), y + v, 1e-5);
}

// Adding and removing algebraic rules on a loaded model.

TEST(AlgebraicRuleConverterTests, AddAlgebraicRuleDeterminesSymbol) {
    ModelBuilder builder;
    builder.parameter("a", 1, true).parameter("x", 0);
    RoadRunner rr(builder.sbml());
    rr.addAlgebraicRule("r", "x - 2 * a");
    EXPECT_EQ(rr.getAlgebraicRuleIds(), std::vector<std::string>{ "r" });
    EXPECT_NEAR(rr.getValue("x"), 2, 1e-12);
    rr.setValue("a", 3);
    EXPECT_NEAR(rr.getValue("x"), 6, 1e-12);
    EXPECT_ANY_THROW(rr.setValue("x", 9));
}

TEST(AlgebraicRuleConverterTests, AddCoupledAlgebraicRulesWithoutRegenerating) {
    ModelBuilder builder;
    builder.parameter("x", 0).parameter("y", 0);
    RoadRunner rr(builder.sbml());
    rr.addAlgebraicRule("r1", "x + y - 3", false);
    rr.addAlgebraicRule("r2", "x - y - 1");
    EXPECT_NEAR(rr.getValue("x"), 2, 1e-12);
    EXPECT_NEAR(rr.getValue("y"), 1, 1e-12);
}

TEST(AlgebraicRuleConverterTests, AddAlgebraicRuleIsUndoneWhenUnsupported) {
    ModelBuilder builder;
    builder.parameter("x", 0);
    RoadRunner rr(builder.sbml());
    EXPECT_ANY_THROW(rr.addAlgebraicRule("bad", "x * x - 4"));
    EXPECT_TRUE(rr.getAlgebraicRuleIds().empty());
    EXPECT_EQ(rr.getCurrentSBML().find("algebraicRule"), std::string::npos);
    rr.setValue("x", 1);
    EXPECT_NO_THROW(rr.simulate(0, 1, 2));
}

TEST(AlgebraicRuleConverterTests, AddAlgebraicRuleRejectsBadIdsAndFormulas) {
    ModelBuilder builder;
    builder.parameter("x", 0);
    RoadRunner rr(builder.sbml());
    EXPECT_THROW(rr.addAlgebraicRule("x", "x - 1"), std::invalid_argument);
    EXPECT_THROW(rr.addAlgebraicRule("", "x - 1"), std::invalid_argument);
    EXPECT_THROW(rr.addAlgebraicRule("1r", "x - 1"), std::invalid_argument);
    EXPECT_THROW(rr.addAlgebraicRule("r", "x - "), std::invalid_argument);
    rr.addAlgebraicRule("r", "x - 1");
    EXPECT_THROW(rr.addAlgebraicRule("r", "x - 2"), std::invalid_argument);
}

TEST(AlgebraicRuleConverterTests, RemoveAlgebraicRuleKeepsValueOfSymbolWithoutOne) {
    ModelBuilder builder;
    builder.parameter("x").algebraic("x - 2", "r");
    RoadRunner rr(builder.sbml());
    rr.removeAlgebraicRule("r");
    EXPECT_TRUE(rr.getAlgebraicRuleIds().empty());
    EXPECT_NEAR(rr.getValue("x"), 2, 1e-12);
    rr.setValue("x", 9);
    EXPECT_NEAR(rr.getValue("x"), 9, 1e-12);
}

TEST(AlgebraicRuleConverterTests, RemoveAlgebraicRuleFreesSolvedSymbol) {
    ModelBuilder builder;
    builder.parameter("x", 0);
    RoadRunner rr(builder.sbml());
    rr.addAlgebraicRule("r", "x - 5");
    rr.removeAlgebraicRule("r");
    EXPECT_NO_THROW(rr.setValue("x", 7));
    EXPECT_NEAR(rr.getValue("x"), 7, 1e-12);
    EXPECT_THROW(rr.removeAlgebraicRule("r"), std::invalid_argument);
}

TEST(AlgebraicRuleConverterTests, AlgebraicRuleEditingUsesMetaIdBeforeL3V2) {
    ModelBuilder builder(3, 1);
    builder.parameter("x", 0);
    RoadRunner rr(builder.sbml());
    rr.addAlgebraicRule("r", "x - 3");
    EXPECT_NEAR(rr.getValue("x"), 3, 1e-12);
    EXPECT_EQ(rr.getAlgebraicRuleIds(), std::vector<std::string>{ "r" });
    EXPECT_NE(rr.getCurrentSBML().find("metaid=\"r\""), std::string::npos);
    rr.removeAlgebraicRule("r");
    EXPECT_NO_THROW(rr.setValue("x", 1));
}

#include "gtest/gtest.h"

#include "llvm/AlgebraicRuleConverter.h"
#include "llvm/LLVMException.h"
#include "rrRoadRunner.h"
#include "rrRoadRunnerOptions.h"

#include <sbml/SBMLTypes.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
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

TEST(AlgebraicRuleConverterTests, AcceptsCoefficientThatChangesWithTime) {
    // time * x = 1 gives x = 1 / time, which is infinite at time zero, as 'x := 1 / time' would be.
    ModelBuilder builder;
    builder.parameter("x").algebraic("time * x - 1");
    RoadRunner rr(builder.sbml());
    EXPECT_TRUE(std::isinf(rr.getValue("x")));
    rr.simulate(0, 2, 3);
    EXPECT_NEAR(rr.getValue("x"), 0.5, 1e-6);
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

TEST(AlgebraicRuleConverterTests, MakingTheOnlySolvableSymbolConstantIsRefused) {
    // x is the only symbol the rule can determine. With x constant as well there is
    // nothing to solve for.
    ModelBuilder builder;
    builder.parameter("x", 1).parameter("y", 2, true).algebraic("x + y + 3");
    RoadRunner rr(builder.sbml());
    EXPECT_NEAR(rr.getValue("x"), -5, 1e-12);
    EXPECT_THROW(rr.setConstant("x", true), std::exception);
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

namespace {
    /** n unknowns in a ring, k * x_i + x_(i+1) = 3, with k a parameter that can change. */
    void addRingOfRules(ModelBuilder& builder, int n, double k) {
        builder.parameter("k", k, true);
        for (int i = 0; i < n; ++i) {
            builder.parameter("x" + std::to_string(i));
        }
        for (int i = 0; i < n; ++i) {
            builder.algebraic("k * x" + std::to_string(i) + " + x" + std::to_string((i + 1) % n) + " - 3");
        }
    }
}

TEST(AlgebraicRuleConverterTests, LargeSimultaneousSystemSurvivesCoefficientBecomingZero) {
    // Above the size Cramer's rule is used for, the pivots chosen at load are
    // kept. With k = 0 the rules say x(i+1) = 3, which still has a unique solution.
    const int n = 7;
    ModelBuilder builder;
    addRingOfRules(builder, n, 2);
    RoadRunner rr(builder.sbml());
    EXPECT_NEAR(rr.getValue("x0"), 1, 1e-12);
    rr.setValue("k", 0);
    for (int i = 0; i < n; ++i) {
        EXPECT_NEAR(rr.getValue("x" + std::to_string(i)), 3, 1e-12) << "x" << i;
    }
}

TEST(AlgebraicRuleConverterTests, CramersRuleSolvesRingsWhenCoefficientChanges) {
    // The ring has the solution x = 3 / (k + 1) whenever its determinant,
    // k^n - (-1)^n, is not zero, which holds for these values of k.
    for (int n : { 4, 6 }) {
        ModelBuilder builder;
        addRingOfRules(builder, n, 2);
        RoadRunner rr(builder.sbml());
        EXPECT_NEAR(rr.getValue("x0"), 1, 1e-12) << "n = " << n;
        for (double k : { 0.0, 3.0, -2.0 }) {
            rr.setValue("k", k);
            for (int i = 0; i < n; ++i) {
                EXPECT_NEAR(rr.getValue("x" + std::to_string(i)), 3 / (k + 1), 1e-9)
                    << "n = " << n << ", k = " << k << ", x" << i;
            }
        }
    }
}

TEST(AlgebraicRuleConverterTests, DISABLED_FailedRegenerationLeavesNoUnsupportedRuleBehind) {
    // The unsupported rule is either refused when it is added or refused by the
    // next regeneration. Either way the document has to match the model that is
    // loaded, so later edits are not blocked by it.
    ModelBuilder builder;
    builder.parameter("x", 0).parameter("y", 0);
    RoadRunner rr(builder.sbml());
    try {
        rr.addAlgebraicRule("bad", "x * x - 4", false);
    }
    catch (const std::exception&) {
    }
    try {
        rr.addAlgebraicRule("ok", "y - 1");
    }
    catch (const std::exception&) {
    }
    const std::vector<std::string> ids = rr.getAlgebraicRuleIds();
    EXPECT_EQ(std::count(ids.begin(), ids.end(), "bad"), 0);
    EXPECT_NO_THROW(rr.addAlgebraicRule("ok2", "y - 2"));
    EXPECT_NEAR(rr.getValue("y"), 2, 1e-12);
}

TEST(AlgebraicRuleConverterTests, MakingSymbolWithoutValueConstantReportsMissingValue) {
    // y has no value of its own; the rule gives it -4. Once y is constant nothing
    // computes it, and roadrunner does not guess a value: regenerating the model
    // reports it as missing, as for any parameter without a value.
    ModelBuilder builder;
    builder.parameter("x", 1, true).parameter("y").algebraic("x + y + 3");
    RoadRunner rr(builder.sbml());
    EXPECT_NEAR(rr.getValue("y"), -4, 1e-12);
    rr.setConstant("x", false);

    std::string error;
    try {
        rr.setConstant("y", true);
    }
    catch (const std::exception& e) {
        error = e.what();
    }
    EXPECT_NE(error.find("missing a value"), std::string::npos) << "error was: '" << error << "'";
}

namespace {
    /** x is held at 5 by an algebraic rule; @p valueAttribute is x's own value, if any. */
    std::string xHeldByAlgebraicRule(const std::string& valueAttribute) {
        return std::string(R"(<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">
  <model metaid="__main" id="__main">
    <listOfParameters>
      <parameter id="x")") + valueAttribute + R"( constant="false"/>
    </listOfParameters>
    <listOfRules>
      <algebraicRule id="_alg0">
        <math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply>
            <minus/>
            <ci> x </ci>
            <cn type="integer"> 5 </cn>
          </apply>
        </math>
      </algebraicRule>
    </listOfRules>
  </model>
</sbml>)";
    }
}

TEST(AlgebraicRuleConverterTests, RemoveAlgebraicRuleLeavesInitialValueOfSymbolThatHasOne) {
    // The rule holds x at 5, but x's own value is 3, and that is what stays.
    RoadRunner rr(xHeldByAlgebraicRule(R"( value="3")"));
    EXPECT_NEAR(rr.getValue("x"), 5, 1e-12);
    rr.removeAlgebraicRule("_alg0");
    EXPECT_NEAR(rr.getValue("init(x)"), 3, 1e-12);
    EXPECT_NO_THROW(rr.setValue("x", 7));
}

TEST(AlgebraicRuleConverterTests, RemoveAlgebraicRuleDoesNotInventValueForSymbolWithoutOne) {
    // x has no value of its own. Without the rule nothing would give it one, and
    // roadrunner does not guess, so the removal is refused and undone.
    RoadRunner rr(xHeldByAlgebraicRule(""));
    EXPECT_NEAR(rr.getValue("x"), 5, 1e-12);

    std::string error;
    try {
        rr.removeAlgebraicRule("_alg0");
    }
    catch (const std::exception& e) {
        error = e.what();
    }
    EXPECT_NE(error.find("missing a value"), std::string::npos) << "error was: '" << error << "'";

    // The rule is still there, and still determines x.
    const std::vector<std::string> ids = rr.getAlgebraicRuleIds();
    EXPECT_EQ(std::count(ids.begin(), ids.end(), "_alg0"), 1);
    EXPECT_NEAR(rr.getValue("x"), 5, 1e-12);
    EXPECT_ANY_THROW(rr.setValue("x", 9));
}

TEST(AlgebraicRuleConverterTests, RemoveAlgebraicRuleLeavesSymbolWithoutValueUnsetWhenNotRegenerating) {
    RoadRunner rr(xHeldByAlgebraicRule(""));
    rr.removeAlgebraicRule("_alg0", false);
    std::unique_ptr<SBMLDocument> doc(readSBMLFromString(rr.getSBML().c_str()));
    EXPECT_FALSE(doc->getModel()->getParameter("x")->isSetValue());
}

TEST(AlgebraicRuleConverterTests, SetValueIsRefusedForSymbolDeterminedByAlgebraicRule) {
    RoadRunner rr(xHeldByAlgebraicRule(R"( value="3")"));
    EXPECT_ANY_THROW(rr.setValue("x", 7));
    EXPECT_NEAR(rr.getValue("x"), 5, 1e-12);
}

TEST(AlgebraicRuleConverterTests, SetInitialValueIsRefusedForSymbolDeterminedByAlgebraicRule) {
    RoadRunner rr(xHeldByAlgebraicRule(R"( value="3")"));
    EXPECT_ANY_THROW(rr.setValue("init(x)", 9));
    EXPECT_NEAR(rr.getValue("x"), 5, 1e-12);
}

TEST(AlgebraicRuleConverterTests, RemoveAlgebraicRuleWithoutRegeneratingThenSetInitialValue) {
    // x has no value of its own, so removing the rule and giving it a value are two
    // steps, and the model can only be generated after both. Not regenerating after
    // the first leaves the loaded model alone until the value is set.
    RoadRunner rr(xHeldByAlgebraicRule(""));
    const rr::ExecutableModel* before = rr.getModel();
    ASSERT_NO_THROW(rr.removeAlgebraicRule("_alg0", false));
    EXPECT_EQ(rr.getModel(), before);
    EXPECT_TRUE(rr.getAlgebraicRuleIds().empty());

    ASSERT_NO_THROW(rr.setValue("init(x)", 3));
    EXPECT_NE(rr.getModel(), before);
    EXPECT_NEAR(rr.getValue("init(x)"), 3, 1e-12);
    EXPECT_NO_THROW(rr.setValue("x", 7));
    EXPECT_NEAR(rr.getValue("x"), 7, 1e-12);
}

TEST(AlgebraicRuleConverterTests, RemoveAlgebraicRuleWithoutRegeneratingThenRegenerate) {
    RoadRunner rr(xHeldByAlgebraicRule(R"( value="3")"));
    const rr::ExecutableModel* before = rr.getModel();
    ASSERT_NO_THROW(rr.removeAlgebraicRule("_alg0", false));
    EXPECT_EQ(rr.getModel(), before);

    ASSERT_NO_THROW(rr.regenerateModel());
    EXPECT_NE(rr.getModel(), before);
    EXPECT_NEAR(rr.getValue("init(x)"), 3, 1e-12);
    EXPECT_NO_THROW(rr.setValue("x", 7));
    EXPECT_NEAR(rr.getValue("x"), 7, 1e-12);
}

namespace {
    /** Solves a * x = b by Gaussian elimination with partial pivoting. */
    std::vector<double> solveDense(std::vector<std::vector<double>> a, std::vector<double> b) {
        const size_t n = b.size();
        for (size_t k = 0; k < n; ++k) {
            size_t pivot = k;
            for (size_t r = k + 1; r < n; ++r) {
                if (std::fabs(a[r][k]) > std::fabs(a[pivot][k])) {
                    pivot = r;
                }
            }
            std::swap(a[k], a[pivot]);
            std::swap(b[k], b[pivot]);
            for (size_t r = k + 1; r < n; ++r) {
                const double factor = a[r][k] / a[k][k];
                for (size_t c = k; c < n; ++c) {
                    a[r][c] -= factor * a[k][c];
                }
                b[r] -= factor * b[k];
            }
        }
        std::vector<double> x(n);
        for (size_t k = n; k-- > 0;) {
            double sum = b[k];
            for (size_t c = k + 1; c < n; ++c) {
                sum -= a[k][c] * x[c];
            }
            x[k] = sum / a[k][k];
        }
        return x;
    }

    std::string coefficientId(int row, int column) {
        return "a" + std::to_string(row) + "_" + std::to_string(column);
    }
}

TEST(AlgebraicRuleConverterTests, CramersRuleSolvesDenseSystemOfSixWhenCoefficientsChange) {
    // Six rules, sum_j a_ij * x_j = b_i, where all 36 coefficients are parameters.
    // This is the largest system Cramer's rule is used for, and the one with the
    // largest formulas: each determinant has hundreds of terms.
    const int n = 6;
    std::vector<double> b;
    for (int i = 0; i < n; ++i) {
        b.push_back(3.0 * i - 4);
    }

    // Diagonally dominant at load, so elimination pivots on the diagonal.
    std::vector<std::vector<double>> a(n, std::vector<double>(n));
    ModelBuilder builder;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            a[i][j] = (i == j) ? 20 : 1 + (2 * i + j) % 2;
            builder.parameter(coefficientId(i, j), a[i][j], true);
        }
    }
    for (int i = 0; i < n; ++i) {
        builder.parameter("x" + std::to_string(i));
    }
    for (int i = 0; i < n; ++i) {
        std::string formula;
        for (int j = 0; j < n; ++j) {
            formula += (j ? " + " : "") + coefficientId(i, j) + " * x" + std::to_string(j);
        }
        builder.algebraic(formula + " - (" + std::to_string(b[i]) + ")");
    }

    const auto start = std::chrono::steady_clock::now();
    RoadRunner rr(builder.sbml());
    const auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start).count();
    std::cout << "[  timing  ] dense system of six rules loaded in " << milliseconds << " ms" << std::endl;
    RecordProperty("load_milliseconds", static_cast<int>(milliseconds));

    auto expectSolution = [&](const std::string& when) {
        const std::vector<double> expected = solveDense(a, b);
        for (int i = 0; i < n; ++i) {
            EXPECT_NEAR(rr.getValue("x" + std::to_string(i)), expected[i], 1e-9) << when << ", x" << i;
        }
    };
    auto assignCoefficients = [&]() {
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                rr.setValue(coefficientId(i, j), a[i][j]);
            }
        }
    };

    expectSolution("at load");

    // Every coefficient that was a pivot at load is now zero.
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            a[i][j] = (j == (i + 1) % n) ? 20 : (i == j ? 0 : 1 + (i + j) % 2);
        }
    }
    assignCoefficients();
    expectSolution("with the diagonal zero");

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            a[i][j] = (j == (i + 2) % n) ? -15 : (i == j ? 3 : 1);
        }
    }
    assignCoefficients();
    expectSolution("with a different dominant pattern");
}

TEST(AlgebraicRuleConverterTests, MathlessAlgebraicRuleIsDroppedAtLoadLikeOtherMathlessRules) {
    // A rule with no math is ignored at load and does not stay in the document,
    // as for an assignment rule, so there is nothing to list or to remove.
    RoadRunner rr(R"(<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">
  <model id="m">
    <listOfParameters>
      <parameter id="x" value="3" constant="false"/>
    </listOfParameters>
    <listOfRules>
      <algebraicRule id="r"/>
    </listOfRules>
  </model>
</sbml>)");
    EXPECT_NEAR(rr.getValue("x"), 3, 1e-12);
    EXPECT_NO_THROW(rr.setValue("x", 7));
    EXPECT_EQ(rr.getSBML().find("algebraicRule"), std::string::npos);
    ASSERT_TRUE(rr.getAlgebraicRuleIds().empty());
    EXPECT_THROW(rr.removeAlgebraicRule("r"), std::invalid_argument);
}

TEST(AlgebraicRuleConverterTests, CramersRuleLeavesExtraSymbolFreeWhenCoefficientChanges) {
    // Two rules and three symbols: z keeps its value, and x and y follow k.
    // The rules have to hold for every k that leaves the system solvable.
    ModelBuilder builder;
    builder.parameter("x").parameter("y").parameter("z", 4).parameter("k", 2, true)
        .algebraic("k * x + y + z - 7").algebraic("x - y + z - 5");
    RoadRunner rr(builder.sbml());
    for (double k : { 2.0, 0.0, 3.0 }) {
        rr.setValue("k", k);
        const double x = rr.getValue("x");
        const double y = rr.getValue("y");
        const double z = rr.getValue("z");
        EXPECT_NEAR(k * x + y + z - 7, 0, 1e-9) << "k = " << k;
        EXPECT_NEAR(x - y + z - 5, 0, 1e-9) << "k = " << k;
        EXPECT_NEAR(z, 4, 1e-12) << "k = " << k;
    }
}

TEST(AlgebraicRuleConverterTests, SettingConstantSymbolInRuleWorksAndSolvedSymbolNamesTheRule) {
    // y is constant, so the rule x - y = 0 solves for x. y is mentioned but not
    // solved for: it can be set, and x follows. x cannot be set, and the error
    // names the algebraic rule.
    ModelBuilder builder;
    builder.parameter("x", 1).parameter("y", 2, true).algebraic("x - y");
    RoadRunner rr(builder.sbml());
    EXPECT_NEAR(rr.getValue("x"), 2, 1e-12);

    const std::string solved = setValueError(rr, "x");
    EXPECT_NE(solved.find("algebraic rule"), std::string::npos) << solved;
    EXPECT_EQ(solved.find("assignment rule"), std::string::npos) << solved;

    EXPECT_EQ(setValueError(rr, "y"), "");
    EXPECT_NEAR(rr.getValue("x"), 9, 1e-12);
}

TEST(AlgebraicRuleConverterTests, AcceptsCoefficientFromRateRuleWhenItStaysNonZero) {
    // The rule y / x = 1 has the coefficient 1 / x in y. x follows a rate rule,
    // so the coefficient changes, but it is not zero, and y follows x.
    ModelBuilder builder;
    builder.parameter("x", 1).parameter("y").rate("x", "1").algebraic("y / x - 1");
    RoadRunner rr(builder.sbml());
    EXPECT_NEAR(rr.getValue("y"), 1, 1e-12);
    rr.simulate(0, 2, 5);
    EXPECT_NEAR(rr.getValue("x"), 3, 1e-6);
    EXPECT_NEAR(rr.getValue("y"), 3, 1e-6);
    EXPECT_ANY_THROW(rr.setValue("y", 9));
}

namespace {
    /** An SBML Level 3 Version 2 model with the given lists as its body. */
    std::string sbmlModel(const std::string& body) {
        return R"(<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">
  <model id="m">)" + body + R"(
  </model>
</sbml>)";
    }

    /** The algebraic rule 0 = symbol - 2 * p. */
    std::string ruleSolvingForTwiceP(const std::string& symbol) {
        return R"(
    <listOfRules>
      <algebraicRule><math xmlns="http://www.w3.org/1998/Math/MathML">
        <apply><minus/><ci>)" + symbol + R"(</ci><apply><times/><cn>2</cn><ci>p</ci></apply></apply>
      </math></algebraicRule>
    </listOfRules>)";
    }
}

TEST(AlgebraicRuleConverterTests, SolvesForSpeciesConcentration) {
    // The species is a concentration in the rule, in a compartment of size 2.
    RoadRunner rr(sbmlModel(R"(
    <listOfCompartments><compartment id="C" size="2" constant="true"/></listOfCompartments>
    <listOfSpecies>
      <species id="S" compartment="C" hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters><parameter id="p" value="3" constant="true"/></listOfParameters>)"
        + ruleSolvingForTwiceP("S")));
    EXPECT_NEAR(rr.getValue("[S]"), 6, 1e-12);
    EXPECT_NEAR(rr.getValue("S"), 12, 1e-12);
    rr.setValue("p", 5);
    EXPECT_NEAR(rr.getValue("[S]"), 10, 1e-12);
    EXPECT_NEAR(rr.getValue("S"), 20, 1e-12);
    EXPECT_ANY_THROW(rr.setValue("S", 1));
    EXPECT_ANY_THROW(rr.setValue("[S]", 1));
}

TEST(AlgebraicRuleConverterTests, SolvesForSpeciesAmount) {
    // The species is an amount in the rule, in a compartment of size 2.
    RoadRunner rr(sbmlModel(R"(
    <listOfCompartments><compartment id="C" size="2" constant="true"/></listOfCompartments>
    <listOfSpecies>
      <species id="S" compartment="C" hasOnlySubstanceUnits="true" boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters><parameter id="p" value="3" constant="true"/></listOfParameters>)"
        + ruleSolvingForTwiceP("S")));
    EXPECT_NEAR(rr.getValue("S"), 6, 1e-12);
    EXPECT_NEAR(rr.getValue("[S]"), 3, 1e-12);
    rr.setValue("p", 5);
    EXPECT_NEAR(rr.getValue("S"), 10, 1e-12);
    EXPECT_NEAR(rr.getValue("[S]"), 5, 1e-12);
    EXPECT_ANY_THROW(rr.setValue("S", 1));
}

TEST(AlgebraicRuleConverterTests, SolvesForCompartmentSize) {
    // The compartment has no size of its own. The species amount stays at 4, so its
    // concentration follows the size.
    RoadRunner rr(sbmlModel(R"(
    <listOfCompartments><compartment id="C" constant="false"/></listOfCompartments>
    <listOfSpecies>
      <species id="S" compartment="C" initialAmount="4" hasOnlySubstanceUnits="true"
               boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters><parameter id="p" value="3" constant="true"/></listOfParameters>)"
        + ruleSolvingForTwiceP("C")));
    EXPECT_NEAR(rr.getValue("C"), 6, 1e-12);
    EXPECT_NEAR(rr.getValue("[S]"), 4.0 / 6, 1e-12);
    rr.setValue("p", 4);
    EXPECT_NEAR(rr.getValue("C"), 8, 1e-12);
    EXPECT_NEAR(rr.getValue("[S]"), 0.5, 1e-12);
    EXPECT_ANY_THROW(rr.setValue("C", 1));
}

TEST(AlgebraicRuleConverterTests, SolvesForBoundarySpeciesThatDrivesReaction) {
    // B is held at 3 by the rule, and makes S at the rate k * B = 1.5.
    RoadRunner rr(sbmlModel(R"(
    <listOfCompartments><compartment id="C" size="1" constant="true"/></listOfCompartments>
    <listOfSpecies>
      <species id="B" compartment="C" hasOnlySubstanceUnits="true" boundaryCondition="true" constant="false"/>
      <species id="S" compartment="C" initialAmount="0" hasOnlySubstanceUnits="true"
               boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters><parameter id="k" value="0.5" constant="true"/></listOfParameters>
    <listOfRules>
      <algebraicRule><math xmlns="http://www.w3.org/1998/Math/MathML">
        <apply><minus/><ci>B</ci><cn>3</cn></apply></math></algebraicRule>
    </listOfRules>
    <listOfReactions>
      <reaction id="R" reversible="false">
        <listOfReactants><speciesReference species="B" stoichiometry="1" constant="true"/></listOfReactants>
        <listOfProducts><speciesReference species="S" stoichiometry="1" constant="true"/></listOfProducts>
        <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply><times/><ci>k</ci><ci>B</ci></apply></math></kineticLaw>
      </reaction>
    </listOfReactions>)"));
    EXPECT_NEAR(rr.getValue("B"), 3, 1e-12);
    rr.simulate(0, 2, 5);
    EXPECT_NEAR(rr.getValue("B"), 3, 1e-12);
    EXPECT_NEAR(rr.getValue("S"), 3, 1e-6);
    EXPECT_ANY_THROW(rr.setValue("B", 1));
}

TEST(AlgebraicRuleConverterTests, SolvesForStoichiometry) {
    // The rule gives the reactant's stoichiometry the value 2, so S falls at 2 per unit time.
    RoadRunner rr(sbmlModel(R"(
    <listOfCompartments><compartment id="C" size="1" constant="true"/></listOfCompartments>
    <listOfSpecies>
      <species id="S" compartment="C" initialAmount="10" hasOnlySubstanceUnits="true"
               boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfRules>
      <algebraicRule><math xmlns="http://www.w3.org/1998/Math/MathML">
        <apply><minus/><ci>sr</ci><cn>2</cn></apply></math></algebraicRule>
    </listOfRules>
    <listOfReactions>
      <reaction id="R" reversible="false">
        <listOfReactants><speciesReference id="sr" species="S" constant="false"/></listOfReactants>
        <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML"><cn>1</cn></math></kineticLaw>
      </reaction>
    </listOfReactions>)"));
    rr.simulate(0, 1, 3);
    EXPECT_NEAR(rr.getValue("S"), 8, 1e-6);
}

TEST(AlgebraicRuleConverterTests, SingularSystemGivesNonFiniteValuesWithoutThrowing) {
    // A coefficient that reaches zero at run time makes the solved value
    // non-finite. That is the same as an assignment rule x := 1 / k.
    {
        ModelBuilder builder;
        builder.parameter("x").parameter("k", 2, true).algebraic("k * x - 1");
        RoadRunner rr(builder.sbml());
        EXPECT_NEAR(rr.getValue("x"), 0.5, 1e-12);
        EXPECT_NO_THROW(rr.setValue("k", 0));
        EXPECT_FALSE(std::isfinite(rr.getValue("x")));
        rr.setValue("k", 2);
        EXPECT_NEAR(rr.getValue("x"), 0.5, 1e-12);
    }

    // Two rules in two symbols are singular when k = 1: both say x + y = 3.
    {
        ModelBuilder builder;
        builder.parameter("x").parameter("y").parameter("k", 2, true)
            .parameter("a", 3, true).parameter("b", 3, true)
            .algebraic("x + y - a").algebraic("x + k * y - b");
        RoadRunner rr(builder.sbml());
        EXPECT_NEAR(rr.getValue("x"), 3, 1e-12);
        EXPECT_NEAR(rr.getValue("y"), 0, 1e-12);
        EXPECT_NO_THROW(rr.setValue("k", 1));
        EXPECT_FALSE(std::isfinite(rr.getValue("x")));
        EXPECT_FALSE(std::isfinite(rr.getValue("y")));
        rr.setValue("k", 2);
        EXPECT_NEAR(rr.getValue("x"), 3, 1e-12);
        EXPECT_NEAR(rr.getValue("y"), 0, 1e-12);
    }
}

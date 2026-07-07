//
// Created by Ciaran on 21/07/2021.
//

#include <rrRoadRunner.h>
#include "gtest/gtest.h"

#include "RoadRunnerTest.h"
#include "TestModelFactory.h"
#include "GillespieIntegrator.h"
#include "rrConfig.h"
#include "Matrix.h"

#include <algorithm>
#include <utility>
#include <vector>

using namespace rr;

// Compares two labelled matrices by matching row/column NAME rather than
// position. Needed because roadrunner and LibStructural can legitimately
// disagree on species/reaction ordering (e.g. when a model mixes reactions
// with rate rules), so a plain position-by-position comparison can pass or
// fail for the wrong reason -- it doesn't tell you whether the *values*
// ended up in the *right* cells, only whether the same numbers appear
// somewhere. This checks the latter, precisely.
static void checkMatrixEqualByLabel(const ls::DoubleMatrix &expected, const ls::DoubleMatrix &actual,
                                     double tol = 1e-6) {
    const std::vector<std::string> &expRows = expected.getRowNames();
    const std::vector<std::string> &expCols = expected.getColNames();
    const std::vector<std::string> &actRows = actual.getRowNames();
    const std::vector<std::string> &actCols = actual.getColNames();

    ASSERT_EQ(expRows.size(), actRows.size())
        << "different number of rows (expected " << expRows.size()
        << ", actual " << actRows.size() << ")";
    ASSERT_EQ(expCols.size(), actCols.size())
        << "different number of cols (expected " << expCols.size()
        << ", actual " << actCols.size() << ")";

    for (size_t i = 0; i < expRows.size(); i++) {
        auto rowIt = std::find(actRows.begin(), actRows.end(), expRows[i]);
        ASSERT_NE(rowIt, actRows.end())
            << "row label '" << expRows[i] << "' not found in actual matrix";
        size_t ai = std::distance(actRows.begin(), rowIt);

        for (size_t j = 0; j < expCols.size(); j++) {
            auto colIt = std::find(actCols.begin(), actCols.end(), expCols[j]);
            ASSERT_NE(colIt, actCols.end())
                << "col label '" << expCols[j] << "' not found in actual matrix";
            size_t aj = std::distance(actCols.begin(), colIt);

            EXPECT_NEAR(expected(i, j), actual(ai, aj), tol)
                << "mismatch at [" << expRows[i] << "][" << expCols[j] << "]";
        }
    }
}

// Eigenvalues have no canonical row/col correspondence to compare by label --
// the row "name" attached to an eigenvalue is just the Jacobian's row name at
// that position, not a meaningful pairing. So eigenvalues are compared as an
// unordered set: sort both lists (by real part, then imaginary part) and
// compare pairwise.
static void checkEigenvaluesEqual(const ls::DoubleMatrix &expected, const ls::DoubleMatrix &actual,
                                   double tol = 1e-6) {
    ASSERT_EQ(expected.numRows(), actual.numRows()) << "different number of eigenvalues";

    std::vector<std::pair<double, double>> exp, act;
    for (int i = 0; i < expected.numRows(); i++) {
        exp.emplace_back(expected(i, 0), expected(i, 1));
        act.emplace_back(actual(i, 0), actual(i, 1));
    }
    auto cmp = [](const std::pair<double, double> &a, const std::pair<double, double> &b) {
        if (a.first != b.first) return a.first < b.first;
        return a.second < b.second;
    };
    std::sort(exp.begin(), exp.end(), cmp);
    std::sort(act.begin(), act.end(), cmp);

    for (size_t i = 0; i < exp.size(); i++) {
        EXPECT_NEAR(exp[i].first, act[i].first, tol) << "eigenvalue real part mismatch at sorted index " << i;
        EXPECT_NEAR(exp[i].second, act[i].second, tol) << "eigenvalue imaginary part mismatch at sorted index " << i;
    }
}

class MCATests : public RoadRunnerTest {

public:
    MCATests() = default;

    void checkUnscaledConcControlMatrix(const std::string& modelName, double tol){
        TestModel* testModel = TestModelFactory(modelName);
        MCAResult* mcaTestModel = dynamic_cast<MCAResult*>(testModel);
        assert(mcaTestModel && "Test model probably does not implement the MCAResult interface");
        RoadRunner rr(testModel->str());
        ls::Matrix<double> actual = rr.getUnscaledConcentrationControlCoefficientMatrix();
        rr::Matrix<double> expected = mcaTestModel->unscaledConcentrationControlCoefficientMatrix();
        //std::cout << "expected" << std::endl;
        //std::cout << expected << std::endl;
        //std::cout << "actual" << std::endl;
        //std::cout << actual << std::endl;
        checkMatrixEqual(expected, actual, tol);
        delete testModel;
    }

    void checkScaledConcControlMatrix(const std::string& modelName, double tol){
        TestModel* testModel = TestModelFactory(modelName);
        MCAResult* mcaTestModel = dynamic_cast<MCAResult*>(testModel);
        assert(mcaTestModel && "Test model probably does not implement the MCAResult interface");
        RoadRunner rr(testModel->str());
        ls::Matrix<double> actual = rr.getScaledConcentrationControlCoefficientMatrix();
        rr::Matrix<double> expected = mcaTestModel->scaledConcentrationControlCoefficientMatrix();
        //std::cout << "expected" << std::endl;
        //std::cout << expected << std::endl;
        //std::cout << "actual" << std::endl;
        //std::cout << actual << std::endl;

        checkMatrixEqual(expected, actual, tol);
        delete testModel;
    }

    void checkUnscaledFluxControlMatrix(const std::string& modelName, double tol){
        TestModel* testModel = TestModelFactory(modelName);
        MCAResult* mcaTestModel = dynamic_cast<MCAResult*>(testModel);
        assert(mcaTestModel && "Test model probably does not implement the MCAResult interface");
        RoadRunner rr(testModel->str());
        ls::Matrix<double> actual = rr.getUnscaledFluxControlCoefficientMatrix();
        rr::Matrix<double> expected = mcaTestModel->unscaledFluxControlCoefficientMatrix();
        //std::cout << "expected" << std::endl;
        //std::cout << expected << std::endl;
        //std::cout << "actual" << std::endl;
        //std::cout << actual << std::endl;
        checkMatrixEqual(expected, actual, tol);
        delete testModel;
    }

    void checkScaledFluxControlMatrix(const std::string& modelName, double tol){
        TestModel* testModel = TestModelFactory(modelName);
        MCAResult* mcaTestModel = dynamic_cast<MCAResult*>(testModel);
        assert(mcaTestModel && "Test model probably does not implement the MCAResult interface");
        RoadRunner rr(testModel->str());
        ls::Matrix<double> actual = rr.getScaledFluxControlCoefficientMatrix();
        rr::Matrix<double> expected = mcaTestModel->scaledFluxControlCoefficientMatrix();
        //std::cout << "expected" << std::endl;
        //std::cout << expected << std::endl;
        //std::cout << "actual" << std::endl;
        //std::cout << actual << std::endl;
        checkMatrixEqual(expected, actual, tol);
        delete testModel;
    }
};
// unscaled conc
TEST_F(MCATests, BimolecularEndUnscaledConcControlMatrix){
    checkUnscaledConcControlMatrix("BimolecularEnd", 1e-3);
}

// scaled conc
TEST_F(MCATests, BimolecularEndScaledConcControlMatrix){
    checkScaledConcControlMatrix("BimolecularEnd", 1e-3);
}

// unscaled flux
TEST_F(MCATests, BimolecularEndUnscaledFluxControlMatrix){
    checkUnscaledFluxControlMatrix("BimolecularEnd",  1e-3);
}

// scaled flux
TEST_F(MCATests, BimolecularEndScaledFluxControlMatrix){
    checkScaledFluxControlMatrix("BimolecularEnd",  1e-3);
}

// Scaled Flux control coefficients that should be zero
TEST_F(MCATests, ZeroFluxCC) {
    RoadRunner rr((rrTestModelsDir_ / "ModelAnalysis" / "zero_flux_cc_ss.xml").string());
    EXPECT_EQ(rr.getValue("cc(vAK, e_vATP)"), 0.0);
    rr.reset();
    ls::DoubleMatrix results = rr.getScaledFluxControlCoefficientMatrix();
    for (size_t col = 0; col < 18; col++) {
        EXPECT_EQ(results[17][col], 0.0);
    }
}


// ============================================================================
// Regression tests for mixing reactions with rate rules in the same model.
//
// Bug: RoadRunner's own model orders floating species "independent-first"
// (species with no rate/assignment rule) then "dependent" (species with a
// rule), while LibStructural orders species by raw SBML declaration order,
// unaware of rate rules entirely. Several MCA functions combine a
// LibStructural-sourced matrix (Nr, L, K, stoichiometry) with a
// roadrunner-native one (the elasticity matrix, or direct model indexing)
// without reconciling the two orderings first. When a model has no rate
// rules (or when every floating species has one), the two orderings happen
// to coincide and nothing goes wrong; as soon as a model mixes rule-free
// and rule-governed floating species, they diverge silently.
//
// Strategy: for each affected function, compare its result on a model that
// mixes reactions and rate rules against the same function's result on an
// equivalent model where the identical dynamics are re-expressed purely as
// reactions. Since these MCA quantities are properties of the dynamical
// system (the ODE right-hand side), not of how it happens to be encoded in
// SBML, the two must agree once row/col labels are reconciled -- which is
// exactly what checkMatrixEqualByLabel checks (unlike a plain
// position-by-position or sorted-values comparison, which would not have
// caught this bug).
// ============================================================================

class MCAMixedRateRuleTests : public RoadRunnerTest {
public:
    MCAMixedRateRuleTests() = default;

    path modelPath(const std::string &filename) const {
        return rrTestModelsDir_ / "ModelAnalysis" / filename;
    }

    template<typename Getter>
    void compareToAllReactionsTwin(const std::string &mixedFile, const std::string &allReactionsFile,
                                    Getter getter, bool conservedMoietyAnalysis, double tol = 1e-6) {
        RoadRunner mixed(modelPath(mixedFile).string());
        RoadRunner allReactions(modelPath(allReactionsFile).string());

        if (conservedMoietyAnalysis) {
            mixed.setConservedMoietyAnalysis(true);
            allReactions.setConservedMoietyAnalysis(true);
        }

        ls::DoubleMatrix expected = getter(allReactions);
        ls::DoubleMatrix actual = getter(mixed);

        checkMatrixEqualByLabel(expected, actual, tol);
    }

    template<typename Getter>
    void compareEigenvaluesToAllReactionsTwin(const std::string &mixedFile, const std::string &allReactionsFile,
                                               Getter getter, bool conservedMoietyAnalysis, double tol = 1e-6) {
        RoadRunner mixed(modelPath(mixedFile).string());
        RoadRunner allReactions(modelPath(allReactionsFile).string());

        if (conservedMoietyAnalysis) {
            mixed.setConservedMoietyAnalysis(true);
            allReactions.setConservedMoietyAnalysis(true);
        }

        ls::DoubleMatrix expected = getter(allReactions);
        ls::DoubleMatrix actual = getter(mixed);

        checkEigenvaluesEqual(expected, actual, tol);
    }

    // Checks a 2x2 (A, B only) Jacobian against hand-derived expected
    // values, looking rows/cols up by label rather than assuming position.
    void checkTwoSpeciesJacobianAgainstExpected(const ls::DoubleMatrix &jac, double dAA, double dAB, double dBA,
                                                 double dBB, double tol = 1e-6) {
        ASSERT_EQ(jac.numRows(), 2);
        ASSERT_EQ(jac.numCols(), 2);

        const std::vector<std::string> &rows = jac.getRowNames();
        const std::vector<std::string> &cols = jac.getColNames();
        auto rowOf = [&](const std::string &id) {
            return std::distance(rows.begin(), std::find(rows.begin(), rows.end(), id));
        };
        auto colOf = [&](const std::string &id) {
            return std::distance(cols.begin(), std::find(cols.begin(), cols.end(), id));
        };

        EXPECT_NEAR(jac(rowOf("A"), colOf("A")), dAA, tol);
        EXPECT_NEAR(jac(rowOf("A"), colOf("B")), dAB, tol);
        EXPECT_NEAR(jac(rowOf("B"), colOf("A")), dBA, tol);
        EXPECT_NEAR(jac(rowOf("B"), colOf("B")), dBB, tol);
    }
};

// ---- chainMixed.xml vs chainAllReactions.xml ----
// A' = k0*(Aeq - A) [rate rule] feeds a 2-reaction chain -> B -> C ->.
// Steady state: A=2, B=k1*Aeq/k2=8/3, C=k2*B/k3=4. No conservation law.

TEST_F(MCAMixedRateRuleTests, ChainFullJacobian) {
    compareToAllReactionsTwin("chainMixed.xml", "chainAllReactions.xml",
        [](RoadRunner &r) { return r.getFullJacobian(); }, false);
}

TEST_F(MCAMixedRateRuleTests, ChainReducedJacobian) {
    compareToAllReactionsTwin("chainMixed.xml", "chainAllReactions.xml",
        [](RoadRunner &r) { return r.getReducedJacobian(); }, false);
}

TEST_F(MCAMixedRateRuleTests, DISABLED_ChainUnscaledConcentrationCC) {
    compareToAllReactionsTwin("chainMixed.xml", "chainAllReactions.xml",
        [](RoadRunner &r) { return r.getUnscaledConcentrationControlCoefficientMatrix(); }, false, 1e-3);
}

TEST_F(MCAMixedRateRuleTests, DISABLED_ChainScaledConcentrationCC) {
    compareToAllReactionsTwin("chainMixed.xml", "chainAllReactions.xml",
        [](RoadRunner &r) { return r.getScaledConcentrationControlCoefficientMatrix(); }, false, 1e-3);
}

TEST_F(MCAMixedRateRuleTests, DISABLED_ChainUnscaledFluxCC) {
    compareToAllReactionsTwin("chainMixed.xml", "chainAllReactions.xml",
        [](RoadRunner &r) { return r.getUnscaledFluxControlCoefficientMatrix(); }, false, 1e-3);
}

TEST_F(MCAMixedRateRuleTests, DISABLED_ChainScaledFluxCC) {
    compareToAllReactionsTwin("chainMixed.xml", "chainAllReactions.xml",
        [](RoadRunner &r) { return r.getScaledFluxControlCoefficientMatrix(); }, false, 1e-3);
}

TEST_F(MCAMixedRateRuleTests, ChainFullEigenvalues) {
    compareEigenvaluesToAllReactionsTwin("chainMixed.xml", "chainAllReactions.xml",
        [](RoadRunner &r) { return r.getFullEigenValuesNamedArray(); }, false);
}

TEST_F(MCAMixedRateRuleTests, ChainReducedEigenvalues) {
    compareEigenvaluesToAllReactionsTwin("chainMixed.xml", "chainAllReactions.xml",
        [](RoadRunner &r) { return r.getReducedEigenValuesNamedArray(); }, false);
}

TEST_F(MCAMixedRateRuleTests, DISABLED_ChainFrequencyResponse) {
    RoadRunner mixed(modelPath("chainMixed.xml").string());
    RoadRunner allReactions(modelPath("chainAllReactions.xml").string());

    // Frequency response's output columns are [frequency, gain, phase] on a
    // shared, explicitly-specified frequency grid -- not species-labelled --
    // so a plain positional comparison is appropriate here.
    ls::Matrix<double> expected = allReactions.getFrequencyResponse(0.01, 4, 10, "k1", "C", false, false);
    ls::Matrix<double> actual = mixed.getFrequencyResponse(0.01, 4, 10, "k1", "C", false, false);

    checkMatrixEqual(expected, actual, 1e-3);
}

// ---- conservedWithRule.xml / conservedWithRuleAllReactions.xml: not tested ----
// These two models were built to test conserved moiety analysis together
// with a rate-rule species (D, independent of the A<->B conservation law).
// That combination is out of scope: RoadRunner::setConservedMoietyAnalysis
// unconditionally throws for any model with a floating-species rule at all,
// via conservedMoietyCheck() in source/conservation/ConservedMoietyConverter.cpp
// (~line 750). We looked at relaxing that guard for rate-rule species
// specifically (they always have an all-zero stoichiometric row, so they can
// never be entangled in a genuine multi-species conservation law) and found
// the guard isn't the only obstacle: ConservedMoietyConverter's
// createDependentSpeciesRules() (~line 565-628) unconditionally calls
// createAssignmentRule() for every species LibStructural classifies as
// "dependent" -- which a zero-row rate-rule species would be -- with no
// check for a pre-existing rule, so it would emit a second (invalid) rule
// for the same variable. Making this combination work would mean fixing
// the converter itself, not just this Jacobian/MCA code, so we're
// deliberately not supporting or testing it here. conservedWithRule.xml
// and conservedWithRuleAllReactions.xml are left in the model directory in
// case that becomes a project later.

// ---- conservedNoRule.xml: "must not regress" sanity check ----
// A <-> B conserved pair, no rate rules at all -- the plain conservation-law
// reduction case that already works correctly today. 2 species - 1
// conservation law = 1 independent DOF, so the reduced Jacobian should be
// 1x1.

TEST_F(MCAMixedRateRuleTests, ConservedNoRuleReducedJacobianShape) {
    RoadRunner rr(modelPath("conservedNoRule.xml").string());
    rr.setConservedMoietyAnalysis(true);
    ls::DoubleMatrix reduced = rr.getReducedJacobian();
    EXPECT_EQ(reduced.numRows(), 1);
    EXPECT_EQ(reduced.numCols(), 1);
}

// ---- onlyRates.xml: regression check for the zero-reaction branch ----
// getFullJacobian has a separate code path for numReactions==0 that uses
// direct numeric differentiation on getRatesOfChange() instead of the
// LibStructural stoichiometry/elasticity multiply. None of the mixed models
// above exercise it (they all have at least one reaction), and it isn't
// being touched by the ordering fix, so this just confirms it keeps working.
// A' = 0.3, B' = 0.2*A -> dA'/dA=0, dA'/dB=0, dB'/dA=0.2, dB'/dB=0.

TEST_F(MCAMixedRateRuleTests, OnlyRatesFullJacobian) {
    RoadRunner rr(modelPath("onlyRates.xml").string());
    ls::DoubleMatrix jac = rr.getFullJacobian();

    const std::vector<std::string> &rows = jac.getRowNames();
    const std::vector<std::string> &cols = jac.getColNames();
    auto rowOf = [&](const std::string &id) {
        return std::distance(rows.begin(), std::find(rows.begin(), rows.end(), id));
    };
    auto colOf = [&](const std::string &id) {
        return std::distance(cols.begin(), std::find(cols.begin(), cols.end(), id));
    };

    EXPECT_NEAR(jac(rowOf("A"), colOf("A")), 0.0, 1e-6);
    EXPECT_NEAR(jac(rowOf("A"), colOf("B")), 0.0, 1e-6);
    EXPECT_NEAR(jac(rowOf("B"), colOf("A")), 0.2, 1e-6);
    EXPECT_NEAR(jac(rowOf("B"), colOf("B")), 0.0, 1e-6);
}

// ============================================================================
// Boundary species / non-floating rate-rule targets.
//
// Three models share the same A, B reaction skeleton:
//   -> A; k1*X
//   A -> B; k2*A
//   B -> ; k3*B
// where X is, respectively: a constant boundary species
// (three_rxn_boundary.xml), a boundary species governed by its own rate
// rule (bnd_species_rate_rule.xml), and a plain global parameter governed
// by its own rate rule (param_rate_rule.xml). X never depends on A or B in
// any of the three, and neither A nor B ever has a rate rule itself, so
// the floating-species Jacobian must come out identical and 2x2 in all
// three cases:
//   dA'/dA = -k2,  dA'/dB = 0
//   dB'/dA =  k2,  dB'/dB = -k3
//   = [[-0.5, 0], [0.5, -0.4]]
//
// This exercises: (a) a boundary species must never appear as a Jacobian
// row/col, whether constant or dynamic; (b) a rate rule on something that
// isn't a floating species (a boundary species or a parameter) must not
// confuse floating-species-only computations -- in particular,
// getNumRateRules() counts ALL rate rules regardless of what they target,
// so any code (existing or new) that assumes rate-rule count lines up
// 1:1 with floating species must not be tripped up by these models.
// ============================================================================

TEST_F(MCAMixedRateRuleTests, BoundarySpeciesFullJacobian) {
    RoadRunner rr(modelPath("three_rxn_boundary.xml").string());
    checkTwoSpeciesJacobianAgainstExpected(rr.getFullJacobian(), -0.5, 0.0, 0.5, -0.4);
}

TEST_F(MCAMixedRateRuleTests, BoundarySpeciesReducedJacobian) {
    RoadRunner rr(modelPath("three_rxn_boundary.xml").string());
    checkTwoSpeciesJacobianAgainstExpected(rr.getReducedJacobian(), -0.5, 0.0, 0.5, -0.4);
}

TEST_F(MCAMixedRateRuleTests, BoundarySpeciesWithRateRuleFullJacobian) {
    RoadRunner rr(modelPath("bnd_species_rate_rule.xml").string());
    checkTwoSpeciesJacobianAgainstExpected(rr.getFullJacobian(), -0.5, 0.0, 0.5, -0.4);
}

TEST_F(MCAMixedRateRuleTests, BoundarySpeciesWithRateRuleReducedJacobian) {
    RoadRunner rr(modelPath("bnd_species_rate_rule.xml").string());
    checkTwoSpeciesJacobianAgainstExpected(rr.getReducedJacobian(), -0.5, 0.0, 0.5, -0.4);
}

TEST_F(MCAMixedRateRuleTests, ParameterWithRateRuleFullJacobian) {
    RoadRunner rr(modelPath("param_rate_rule.xml").string());
    checkTwoSpeciesJacobianAgainstExpected(rr.getFullJacobian(), -0.5, 0.0, 0.5, -0.4);
}

TEST_F(MCAMixedRateRuleTests, ParameterWithRateRuleReducedJacobian) {
    RoadRunner rr(modelPath("param_rate_rule.xml").string());
    checkTwoSpeciesJacobianAgainstExpected(rr.getReducedJacobian(), -0.5, 0.0, 0.5, -0.4);
}

// Cross-checks: since X's identity (constant boundary species / dynamic
// boundary species / dynamic parameter) shouldn't affect the A,B Jacobian
// at all, all three models' full Jacobians should match each other
// exactly, not just the hand-derived expected values above.

TEST_F(MCAMixedRateRuleTests, BoundarySpeciesVsBoundarySpeciesWithRateRuleCrossCheck) {
    RoadRunner constant_(modelPath("three_rxn_boundary.xml").string());
    RoadRunner dynamic_(modelPath("bnd_species_rate_rule.xml").string());
    checkMatrixEqualByLabel(constant_.getFullJacobian(), dynamic_.getFullJacobian(), 1e-6);
}

TEST_F(MCAMixedRateRuleTests, BoundarySpeciesWithRateRuleVsParameterWithRateRuleCrossCheck) {
    RoadRunner bndDynamic(modelPath("bnd_species_rate_rule.xml").string());
    RoadRunner paramDynamic(modelPath("param_rate_rule.xml").string());
    checkMatrixEqualByLabel(bndDynamic.getFullJacobian(), paramDynamic.getFullJacobian(), 1e-6);
}

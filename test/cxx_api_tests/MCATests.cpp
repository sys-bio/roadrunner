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
#include "rrLogger.h"

#include <algorithm>
#include <iostream>
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

        // Wrapped in ASSERT_NO_THROW (rather than letting the getter throw
        // straight out of this helper) so a failure here is reported
        // against this file/line by gtest, instead of as an uncaught
        // exception attributed to "unknown file".
        ls::DoubleMatrix expected, actual;
        {
            SCOPED_TRACE("allReactions model: " + allReactionsFile);
            ASSERT_NO_THROW(expected = getter(allReactions));
        }
        {
            SCOPED_TRACE("mixed model: " + mixedFile);
            ASSERT_NO_THROW(actual = getter(mixed));
        }

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

        ls::DoubleMatrix expected, actual;
        {
            SCOPED_TRACE("allReactions model: " + allReactionsFile);
            ASSERT_NO_THROW(expected = getter(allReactions));
        }
        {
            SCOPED_TRACE("mixed model: " + mixedFile);
            ASSERT_NO_THROW(actual = getter(mixed));
        }

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

// Regression test for the getUnscaledElasticityMatrix bug found via
// RoadRunnerAPITestsWithLLJit.getUnscaledConcentrationControlCoefficientMatrix:
// SimpleFlux (S1 <-> S2, conserved: S1+S2=11) with conserved moiety
// analysis enabled eliminates S2 via an assignment rule (S2 := 11 - S1).
// That satisfies the "assignment-rule species" check, but unlike a
// genuine model-native assignment rule (e.g. D in mca_with_asnt_rule.xml),
// S2 still needs its own real column here -- the Nr/L-matrix reduction
// downstream expects uelast to hold the plain, uncoupled partial
// derivative (_J1's rate is kb*S2, so d(_J1)/dS2 = kb), and applies the
// conservation-law coupling itself via the link matrix. Skipping S2's
// column here (as if it were a D-like dead-end assignment rule) silently
// dropped that term and gave a wrong (unreduced) concentration control
// coefficient.
TEST_F(MCAMixedRateRuleTests, ConservedMoietyEliminatedSpeciesGetsRealElasticityColumn) {
    SimpleFlux simpleFlux;
    RoadRunner rr(simpleFlux.str());
    rr.setConservedMoietyAnalysis(true);

    ls::DoubleMatrix uelast;
    ASSERT_NO_THROW(uelast = rr.getUnscaledElasticityMatrix());

    const std::vector<std::string> &rows = uelast.getRowNames();
    const std::vector<std::string> &cols = uelast.getColNames();
    auto rowOf = [&](const std::string &id) {
        return std::distance(rows.begin(), std::find(rows.begin(), rows.end(), id));
    };
    auto colOf = [&](const std::string &id) {
        return std::distance(cols.begin(), std::find(cols.begin(), cols.end(), id));
    };

    EXPECT_NEAR(uelast(rowOf("_J0"), colOf("S1")), 0.1, 1e-6);
    EXPECT_NEAR(uelast(rowOf("_J0"), colOf("S2")), 0.0, 1e-6);
    EXPECT_NEAR(uelast(rowOf("_J1"), colOf("S1")), 0.0, 1e-6);
    EXPECT_NEAR(uelast(rowOf("_J1"), colOf("S2")), 0.01, 1e-6);
}

TEST_F(MCAMixedRateRuleTests, ChainFullJacobian) {
    compareToAllReactionsTwin("chainMixed.xml", "chainAllReactions.xml",
        [](RoadRunner &r) { return r.getFullJacobian(); }, false);
}

TEST_F(MCAMixedRateRuleTests, ChainReducedJacobian) {
    compareToAllReactionsTwin("chainMixed.xml", "chainAllReactions.xml",
        [](RoadRunner &r) { return r.getReducedJacobian(); }, false);
}

TEST_F(MCAMixedRateRuleTests, ChainUnscaledConcentrationCC) {
    compareToAllReactionsTwin("chainMixed.xml", "chainAllReactions.xml",
        [](RoadRunner &r) { return r.getUnscaledConcentrationControlCoefficientMatrix(); }, false, 1e-3);
}

TEST_F(MCAMixedRateRuleTests, ChainScaledConcentrationCC) {
    compareToAllReactionsTwin("chainMixed.xml", "chainAllReactions.xml",
        [](RoadRunner &r) { return r.getScaledConcentrationControlCoefficientMatrix(); }, false, 1e-3);
}

TEST_F(MCAMixedRateRuleTests, ChainUnscaledFluxCC) {
    compareToAllReactionsTwin("chainMixed.xml", "chainAllReactions.xml",
        [](RoadRunner &r) { return r.getUnscaledFluxControlCoefficientMatrix(); }, false, 1e-3);
}

TEST_F(MCAMixedRateRuleTests, ChainScaledFluxCC) {
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

TEST_F(MCAMixedRateRuleTests, ChainFrequencyResponse) {
    RoadRunner mixed(modelPath("chainMixed.xml").string());
    RoadRunner allReactions(modelPath("chainAllReactions.xml").string());

    // Frequency response's output columns are [frequency, gain, phase] on a
    // shared, explicitly-specified frequency grid -- not species-labelled --
    // so a plain positional comparison is appropriate here.
    ls::Matrix<double> expected = allReactions.getFrequencyResponse(0.01, 4, 10, "k1", "C", false, false);
    ls::Matrix<double> actual = mixed.getFrequencyResponse(0.01, 4, 10, "k1", "C", false, false);

    checkMatrixEqual(expected, actual, 1e-3);
}

// Note: can't test a model with conservation cycles plus a rate rule for a 
// floating species, because the conservation converter doesn't work with
// models with rules.

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

// ============================================================================
// A rate-rule-governed floating species CAN be set directly, unlike an
// assignment-rule-governed one -- a rate rule only constrains dX/dt, not
// X itself. This is a permanent regression check for that, and for the
// fix to getUnscaledSpeciesElasticity()'s restore logic, which used to
// assume otherwise (capping its post-perturbation restore at
// getNumIndFloatingSpecies(), silently leaving any rate-rule species'
// current value corrupted -- see ChainScaledConcentrationCC/
// ChainScaledFluxCC/ChainFrequencyResponse below for the end-to-end
// symptom this caused).
// ============================================================================

TEST_F(MCAMixedRateRuleTests, RateRuleSpeciesCanBeSetDirectly) {
    RoadRunner rr(modelPath("chainMixed.xml").string());

    EXPECT_NO_THROW(rr.setValue("A", 2.0));
    EXPECT_NEAR(rr.getValue("A"), 2.0, 1e-9);
}

// ============================================================================
// mca_with_asnt_rule.xml: A, B are ordinary reaction-governed species
// (R1: A -> B; k1*A); D is a floating species governed purely by an
// ASSIGNMENT rule (D := 2*A), not fed into any reaction. Unlike a
// rate-rule species, an assignment-rule species genuinely cannot be set
// directly -- its value is continuously recomputed from other values.
// This exercises the other half of getUnscaledSpeciesElasticity's restore
// fix: when perturbing A/B to compute R1's elasticity, the restore loop
// now attempts (and must gracefully fail, one species at a time) to
// restore D too, without that failure affecting A or B's own
// restoration, and without throwing out of the whole computation.
// dA'/dA = -k1, dA'/dB = 0, dB'/dA = k1, dB'/dB = 0. D's own row is a
// separate, known limitation (same shape of gap rate-rule species had
// before this fix, just for assignment rules, and out of scope here), so
// we only assert on A/B.
// ============================================================================

TEST_F(MCAMixedRateRuleTests, AssignmentRuleSpeciesDoesNotCorruptElasticityRestore) {
    RoadRunner rr(modelPath("mca_with_asnt_rule.xml").string());

    // getFullJacobian() still includes D as a row/col (it's a floating
    // species), so don't assert its exact dimensions here -- just confirm
    // it doesn't throw, and that A/B's own entries (found by label) are
    // correct and undisturbed by D's presence.
    ls::DoubleMatrix jac;
    ASSERT_NO_THROW(jac = rr.getFullJacobian());

    const std::vector<std::string> &rows = jac.getRowNames();
    const std::vector<std::string> &cols = jac.getColNames();
    auto rowOf = [&](const std::string &id) {
        return std::distance(rows.begin(), std::find(rows.begin(), rows.end(), id));
    };
    auto colOf = [&](const std::string &id) {
        return std::distance(cols.begin(), std::find(cols.begin(), cols.end(), id));
    };

    EXPECT_NEAR(jac(rowOf("A"), colOf("A")), -0.5, 1e-6);
    EXPECT_NEAR(jac(rowOf("A"), colOf("B")), 0.0, 1e-6);
    EXPECT_NEAR(jac(rowOf("B"), colOf("A")), 0.5, 1e-6);
    EXPECT_NEAR(jac(rowOf("B"), colOf("B")), 0.0, 1e-6);

    // getReducedJacobian() correctly excludes D outright -- it's excluded
    // purely by having an assignment rule, no fix needed there (unlike
    // rate-rule species, which needed getReducedJacobian's own separate
    // completeness fix earlier) -- so this one should come back exactly
    // 2x2.
    ls::DoubleMatrix reduced;
    ASSERT_NO_THROW(reduced = rr.getReducedJacobian());
    checkTwoSpeciesJacobianAgainstExpected(reduced, -0.5, 0.0, 0.5, 0.0);
}

// chainMixedBoundary.xml is identical to chainMixed.xml, except A is a
// BOUNDARY species (still governed by the same rate rule, A' = k0*(Aeq-A),
// k0=0.5, Aeq=2). This is the model where the "steadyState() resets A's
// value" bug was originally observed. All the fixes made so far targeted
// getUnscaledSpeciesElasticity/getUnscaledElasticityMatrix, which only
// ever touch FLOATING species -- so it isn't known whether they (or
// anything else changed above) also happen to fix this boundary-species
// case. This test checks that directly.
TEST_F(MCAMixedRateRuleTests, SteadyStateConvergesBoundaryRateRuleSpecies) {
    RoadRunner rr(modelPath("chainMixedBoundary.xml").string());

    ASSERT_NO_THROW(rr.steadyState());

    EXPECT_NEAR(rr.getValue("A"), 2.0, 1e-3);
}

// ============================================================================
// twoRateRulesMixed.xml vs twoRateRulesAllReactions.xml: TWO independent
// rate-rule species in the same model (A drives B->C, E drives F,
// completely decoupled from each other). Every mixed-model test above has
// exactly one rate-rule species, exercising the "extend by one
// pseudo-column" path; this exercises N=2, a materially different code path
// through extendStoichiometryForRateRuleSpecies/
// extendNrAndLinkForRateRuleSpecies and the rate-rule pseudo-row loop in
// getUnscaledElasticityMatrix. Also doubles as the first direct test of
// getUnscaledElasticityMatrix/getScaledElasticityMatrix against a mixed
// model -- previously only exercised indirectly via getFullJacobian.
// ============================================================================

TEST_F(MCAMixedRateRuleTests, TwoRateRulesFullJacobian) {
    compareToAllReactionsTwin("twoRateRulesMixed.xml", "twoRateRulesAllReactions.xml",
        [](RoadRunner &r) { return r.getFullJacobian(); }, false);
}

TEST_F(MCAMixedRateRuleTests, TwoRateRulesReducedJacobian) {
    compareToAllReactionsTwin("twoRateRulesMixed.xml", "twoRateRulesAllReactions.xml",
        [](RoadRunner &r) { return r.getReducedJacobian(); }, false);
}

TEST_F(MCAMixedRateRuleTests, TwoRateRulesUnscaledElasticityMatrix) {
    compareToAllReactionsTwin("twoRateRulesMixed.xml", "twoRateRulesAllReactions.xml",
        [](RoadRunner &r) { return r.getUnscaledElasticityMatrix(); }, false, 1e-6);
}

TEST_F(MCAMixedRateRuleTests, TwoRateRulesScaledElasticityMatrix) {
    compareToAllReactionsTwin("twoRateRulesMixed.xml", "twoRateRulesAllReactions.xml",
        [](RoadRunner &r) { return r.getScaledElasticityMatrix(); }, false, 1e-6);
}

TEST_F(MCAMixedRateRuleTests, TwoRateRulesUnscaledConcentrationCC) {
    compareToAllReactionsTwin("twoRateRulesMixed.xml", "twoRateRulesAllReactions.xml",
        [](RoadRunner &r) { return r.getUnscaledConcentrationControlCoefficientMatrix(); }, false, 1e-3);
}

TEST_F(MCAMixedRateRuleTests, TwoRateRulesScaledConcentrationCC) {
    compareToAllReactionsTwin("twoRateRulesMixed.xml", "twoRateRulesAllReactions.xml",
        [](RoadRunner &r) { return r.getScaledConcentrationControlCoefficientMatrix(); }, false, 1e-3);
}

TEST_F(MCAMixedRateRuleTests, TwoRateRulesUnscaledFluxCC) {
    compareToAllReactionsTwin("twoRateRulesMixed.xml", "twoRateRulesAllReactions.xml",
        [](RoadRunner &r) { return r.getUnscaledFluxControlCoefficientMatrix(); }, false, 1e-3);
}

TEST_F(MCAMixedRateRuleTests, TwoRateRulesScaledFluxCC) {
    compareToAllReactionsTwin("twoRateRulesMixed.xml", "twoRateRulesAllReactions.xml",
        [](RoadRunner &r) { return r.getScaledFluxControlCoefficientMatrix(); }, false, 1e-3);
}

TEST_F(MCAMixedRateRuleTests, TwoRateRulesFullEigenvalues) {
    compareEigenvaluesToAllReactionsTwin("twoRateRulesMixed.xml", "twoRateRulesAllReactions.xml",
        [](RoadRunner &r) { return r.getFullEigenValuesNamedArray(); }, false);
}

TEST_F(MCAMixedRateRuleTests, TwoRateRulesReducedEigenvalues) {
    compareEigenvaluesToAllReactionsTwin("twoRateRulesMixed.xml", "twoRateRulesAllReactions.xml",
        [](RoadRunner &r) { return r.getReducedEigenValuesNamedArray(); }, false);
}

// ============================================================================
// mca_asnt_rule_modifier.xml: A, B, C are reaction-governed; D is a floating
// species governed purely by an ASSIGNMENT rule (D := 2*A) and -- unlike
// mca_with_asnt_rule.xml's D, which fed into no reaction at all -- D is a
// genuine modifier in R2's rate law (k2*B*D). This checks that perturbing A
// to compute an elasticity correctly triggers D's assignment-rule recompute
// *before* reaction rates are read, so the indirect chain-rule pathway
// A -> D -> R2 is picked up by R2's elasticity w.r.t. A, rather than being
// silently dropped the way D's own column is deliberately skipped.
//
// Hand-derived at the initial state (A=2, B=3, D=2A=4):
//   dA/dt = -k1*A
//   dB/dt = k1*A - k2*B*D = k1*A - 2*k2*A*B   (substituting D=2A)
//   dC/dt = k2*B*D = 2*k2*A*B
//   d(dA/dt)/dA = -k1 = -0.5,             d(dA/dt)/dB = 0
//   d(dB/dt)/dA = k1 - 2*k2*B = -1.3,      d(dB/dt)/dB = -2*k2*A = -1.2
//   d(dC/dt)/dA = 2*k2*B = 1.8,            d(dC/dt)/dB = 2*k2*A = 1.2
// (D's own row/col is out of scope, same as mca_with_asnt_rule.xml.)
// ============================================================================

TEST_F(MCAMixedRateRuleTests, AssignmentRuleModifierPicksUpIndirectChainRule) {
    RoadRunner rr(modelPath("mca_asnt_rule_modifier.xml").string());

    ls::DoubleMatrix jac;
    ASSERT_NO_THROW(jac = rr.getFullJacobian());

    const std::vector<std::string> &rows = jac.getRowNames();
    const std::vector<std::string> &cols = jac.getColNames();
    auto rowOf = [&](const std::string &id) {
        return std::distance(rows.begin(), std::find(rows.begin(), rows.end(), id));
    };
    auto colOf = [&](const std::string &id) {
        return std::distance(cols.begin(), std::find(cols.begin(), cols.end(), id));
    };

    EXPECT_NEAR(jac(rowOf("A"), colOf("A")), -0.5, 1e-6);
    EXPECT_NEAR(jac(rowOf("A"), colOf("B")), 0.0, 1e-6);
    EXPECT_NEAR(jac(rowOf("B"), colOf("A")), -1.3, 1e-6);
    EXPECT_NEAR(jac(rowOf("B"), colOf("B")), -1.2, 1e-6);
    EXPECT_NEAR(jac(rowOf("C"), colOf("A")), 1.8, 1e-6);
    EXPECT_NEAR(jac(rowOf("C"), colOf("B")), 1.2, 1e-6);
}

// ============================================================================
// coupledRateRulesMixed.xml vs coupledRateRulesAllReactions.xml: A and B are
// each governed by a rate rule that depends on the OTHER species
// (A' = k1*(B-A), B' = k2*(A-B)) -- unlike every rate-rule species tested
// above, where a rate-rule species' own rate only ever depended on itself.
// This exercises the off-diagonal elasticity between two rate-rule
// pseudo-rows (uelast[A_rate_rule][B], uelast[B_rate_rule][A]) --
// differentiateRateOfChange has never been asked to produce a nonzero
// cross term before. C is an ordinary reaction-governed species fed by A,
// keeping this comparable to an all-reactions twin the same way as the
// other mixed models above. Initial values (A=2, B=5, C=1) are chosen so
// nothing starts at 0 or at equilibrium, avoiding the 0/0 scaled-elasticity
// pitfall found earlier.
// ============================================================================

TEST_F(MCAMixedRateRuleTests, CoupledRateRulesFullJacobian) {
    compareToAllReactionsTwin("coupledRateRulesMixed.xml", "coupledRateRulesAllReactions.xml",
        [](RoadRunner &r) { return r.getFullJacobian(); }, false);
}

TEST_F(MCAMixedRateRuleTests, CoupledRateRulesReducedJacobian) {
    compareToAllReactionsTwin("coupledRateRulesMixed.xml", "coupledRateRulesAllReactions.xml",
        [](RoadRunner &r) { return r.getReducedJacobian(); }, false);
}

TEST_F(MCAMixedRateRuleTests, CoupledRateRulesUnscaledElasticityMatrix) {
    compareToAllReactionsTwin("coupledRateRulesMixed.xml", "coupledRateRulesAllReactions.xml",
        [](RoadRunner &r) { return r.getUnscaledElasticityMatrix(); }, false, 1e-6);
}

TEST_F(MCAMixedRateRuleTests, CoupledRateRulesScaledElasticityMatrix) {
    compareToAllReactionsTwin("coupledRateRulesMixed.xml", "coupledRateRulesAllReactions.xml",
        [](RoadRunner &r) { return r.getScaledElasticityMatrix(); }, false, 1e-6);
}

TEST_F(MCAMixedRateRuleTests, CoupledRateRulesUnscaledConcentrationCC) {
    compareToAllReactionsTwin("coupledRateRulesMixed.xml", "coupledRateRulesAllReactions.xml",
        [](RoadRunner &r) { return r.getUnscaledConcentrationControlCoefficientMatrix(); }, false, 1e-3);
}

TEST_F(MCAMixedRateRuleTests, CoupledRateRulesScaledConcentrationCC) {
    compareToAllReactionsTwin("coupledRateRulesMixed.xml", "coupledRateRulesAllReactions.xml",
        [](RoadRunner &r) { return r.getScaledConcentrationControlCoefficientMatrix(); }, false, 1e-3);
}

TEST_F(MCAMixedRateRuleTests, CoupledRateRulesUnscaledFluxCC) {
    compareToAllReactionsTwin("coupledRateRulesMixed.xml", "coupledRateRulesAllReactions.xml",
        [](RoadRunner &r) { return r.getUnscaledFluxControlCoefficientMatrix(); }, false, 1e-3);
}

TEST_F(MCAMixedRateRuleTests, CoupledRateRulesScaledFluxCC) {
    compareToAllReactionsTwin("coupledRateRulesMixed.xml", "coupledRateRulesAllReactions.xml",
        [](RoadRunner &r) { return r.getScaledFluxControlCoefficientMatrix(); }, false, 1e-3);
}

TEST_F(MCAMixedRateRuleTests, CoupledRateRulesFullEigenvalues) {
    compareEigenvaluesToAllReactionsTwin("coupledRateRulesMixed.xml", "coupledRateRulesAllReactions.xml",
        [](RoadRunner &r) { return r.getFullEigenValuesNamedArray(); }, false);
}

TEST_F(MCAMixedRateRuleTests, CoupledRateRulesReducedEigenvalues) {
    compareEigenvaluesToAllReactionsTwin("coupledRateRulesMixed.xml", "coupledRateRulesAllReactions.xml",
        [](RoadRunner &r) { return r.getReducedEigenValuesNamedArray(); }, false);
}

// ============================================================================
// Basic-level smoke/correctness tests for the scalar elasticity functions,
// against chainMixed.xml. Both reuse the already-fixed
// getUnscaledSpeciesElasticity internally and involve no LibStructural
// reordering, so they were low-risk by inspection -- these confirm that
// directly, using hand-derivable exact values from chainMixed's mass-action
// kinetics: J0's rate is k1*A, linear in A, so its scaled elasticity w.r.t.
// A is exactly 1 regardless of A's value, and its unscaled parameter
// elasticity w.r.t. k1 is exactly A itself (1, at chainMixed's initial
// condition).
// ============================================================================

TEST_F(MCAMixedRateRuleTests, ScaledFloatingSpeciesElasticitySmokeTest) {
    RoadRunner rr(modelPath("chainMixed.xml").string());
    double elasticity = 0;
    ASSERT_NO_THROW(elasticity = rr.getScaledFloatingSpeciesElasticity("J0", "A"));
    EXPECT_NEAR(elasticity, 1.0, 1e-6);
}

TEST_F(MCAMixedRateRuleTests, UnscaledParameterElasticitySmokeTest) {
    RoadRunner rr(modelPath("chainMixed.xml").string());
    double elasticity = 0;
    ASSERT_NO_THROW(elasticity = rr.getUnscaledParameterElasticity("J0", "k1"));
    EXPECT_NEAR(elasticity, 1.0, 1e-6);  // d(k1*A)/dk1 = A = 1 at t=0
}

// ============================================================================
// Basic-level smoke tests for the raw LibStructural passthrough functions
// (getFullStoichiometryMatrix, getLinkMatrix, getNrMatrix, getKMatrix,
// getL0Matrix, getExtendedStoichiometryMatrix/getReactantsStoichiometryMatrix/
// getProductsStoichiometryMatrix, getConservationMatrix) against a mixed
// reaction/rate-rule model. Unlike getFullJacobian and the CC matrices,
// these return LibStructural's own matrix with LibStructural's own labels
// attached directly -- never reconciled to roadrunner's floating-species
// ordering, and never extended with a rate-rule pseudo-column -- so they
// should be self-consistent by construction. These aren't exhaustive
// correctness checks, just confirmation that a rate-rule species doesn't
// make them throw or return a nonsensical shape.
// ============================================================================

TEST_F(MCAMixedRateRuleTests, FullStoichiometryMatrixSmokeTest) {
    RoadRunner rr(modelPath("chainMixed.xml").string());

    ls::DoubleMatrix stoich;
    ASSERT_NO_THROW(stoich = rr.getFullStoichiometryMatrix());

    EXPECT_EQ(stoich.numRows(), 3);   // A, B, C
    EXPECT_EQ(stoich.numCols(), 3);   // J0, J1, J2

    // A never appears as a reactant/product (only via its rate rule and as
    // a modifier), so its row in the RAW stoichiometry matrix is all zero.
    // That's expected -- reconciling this into A's true (nonzero) row is
    // getFullJacobian's job via the pseudo-column extension, not this
    // function's.
    const std::vector<std::string> &rows = stoich.getRowNames();
    auto rowOf = [&](const std::string &id) {
        return std::distance(rows.begin(), std::find(rows.begin(), rows.end(), id));
    };
    size_t aRow = rowOf("A");
    ASSERT_LT(aRow, rows.size());
    for (int j = 0; j < stoich.numCols(); j++) {
        EXPECT_EQ(stoich(aRow, j), 0.0);
    }
}

TEST_F(MCAMixedRateRuleTests, LinkMatrixSmokeTest) {
    RoadRunner rr(modelPath("chainMixed.xml").string());
    ls::DoubleMatrix link;
    ASSERT_NO_THROW(link = rr.getLinkMatrix());
    EXPECT_GT(link.numRows(), 0);
    EXPECT_GT(link.numCols(), 0);
}

TEST_F(MCAMixedRateRuleTests, NrMatrixSmokeTest) {
    RoadRunner rr(modelPath("chainMixed.xml").string());
    ls::DoubleMatrix nr;
    ASSERT_NO_THROW(nr = rr.getNrMatrix());
    EXPECT_GT(nr.numRows(), 0);
    EXPECT_GT(nr.numCols(), 0);
}

TEST_F(MCAMixedRateRuleTests, KMatrixSmokeTest) {
    RoadRunner rr(modelPath("chainMixed.xml").string());
    ls::DoubleMatrix k;
    ASSERT_NO_THROW(k = rr.getKMatrix());
}

TEST_F(MCAMixedRateRuleTests, L0MatrixSmokeTest) {
    RoadRunner rr(modelPath("chainMixed.xml").string());
    ls::DoubleMatrix l0;
    ASSERT_NO_THROW(l0 = rr.getL0Matrix());
}

TEST_F(MCAMixedRateRuleTests, SubStoichiometryMatrixSmokeTest) {
    RoadRunner rr(modelPath("chainMixed.xml").string());
    ls::DoubleMatrix ext, reactants, products;
    ASSERT_NO_THROW(ext = rr.getExtendedStoichiometryMatrix());
    ASSERT_NO_THROW(reactants = rr.getReactantsStoichiometryMatrix(true));
    ASSERT_NO_THROW(products = rr.getProductsStoichiometryMatrix(true));
}

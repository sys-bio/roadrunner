#include "gtest/gtest.h"
#include "rrRoadRunner.h"
#include "rrException.h"
#include "rrUtils.h"
#include "rrTestSuiteModelSimulation.h"
#include "sbml/SBMLTypes.h"
#include "sbml/SBMLReader.h"
#include "../test_util.h"
#include <filesystem>
#include <algorithm>
#include "RoadRunnerTest.h"
#include "llvm/LLVMException.h"

using namespace testing;
using namespace rr;
using namespace std;
using std::filesystem::path;

class SBMLFeatures : public RoadRunnerTest {
public:
  path SBMLFeaturesDir = rrTestModelsDir_ / "SBMLFeatures";
  SBMLFeatures() = default;
};

// stoich_rr.xml:    N governed by a rate rule, dN/dt = 1, N(0) = 0.  dX/dt = N*0.5.
// stoich_ar.xml:    N governed by an assignment rule, N = time.      dX/dt = N*0.5.
// stoich_ar_const.xml: N governed by an assignment rule, N = 1.      dX/dt = N*0.5 = 0.5.
// stoich_event.xml: N unruled, starts at 0, event at time>1 sets N = 1.  dX/dt = N*0.5.

TEST_F(SBMLFeatures, variable_stoich_rr_simulate) {
  RoadRunner rr((SBMLFeaturesDir / "stoich_rr.xml").string());
  const ls::DoubleMatrix result = *rr.simulate(0.0, 2.0, 11);
  for (int i = 0; i < result.numRows(); i++) {
    EXPECT_NEAR(result(i, 1), 0.01 * i * i, 1e-6);
  }
}

TEST_F(SBMLFeatures, variable_stoich_ar_simulate) {
  RoadRunner rr((SBMLFeaturesDir / "stoich_ar.xml").string());
  const ls::DoubleMatrix result = *rr.simulate(0.0, 2.0, 11);
  for (int i = 0; i < result.numRows(); i++) {
    EXPECT_NEAR(result(i, 1), 0.01 * i * i, 1e-6);
  }
}

TEST_F(SBMLFeatures, variable_stoich_ar_const_simulate) {
  RoadRunner rr((SBMLFeaturesDir / "stoich_ar_const.xml").string());
  const ls::DoubleMatrix result = *rr.simulate(0.0, 2.0, 11);
  for (int i = 0; i < result.numRows(); i++) {
    EXPECT_NEAR(result(i, 1), 0.1 * i, 1e-6);
  }
}

TEST_F(SBMLFeatures, variable_stoich_event_simulate) {
  RoadRunner rr((SBMLFeaturesDir / "stoich_event.xml").string());
  const ls::DoubleMatrix result = *rr.simulate(0.0, 2.0, 11);
  for (int i = 0; i < result.numRows(); i++) {
    double expected = (i <= 5) ? 0.0 : 0.1 * (i - 5);
    EXPECT_NEAR(result(i, 1), expected, 1e-4);
  }
}


TEST_F(SBMLFeatures, variable_stoich_rr_set_initial) {
  RoadRunner rr((SBMLFeaturesDir / "stoich_rr.xml").string());
  rr.setValue("N", 2);
  const ls::DoubleMatrix result = *rr.simulate(0.0, 2.0, 11);
  for (int i = 0; i < result.numRows(); i++) {
    EXPECT_NEAR(result(i, 1), 0.2 * i + 0.01 * i * i, 1e-6);
  }
}

TEST_F(SBMLFeatures, variable_stoich_ar_set_throws) {
  RoadRunner rr((SBMLFeaturesDir / "stoich_ar.xml").string());
  EXPECT_THROW(rr.setValue("N", 2), rrllvm::LLVMException);
}

TEST_F(SBMLFeatures, variable_stoich_ar_const_set_throws) {
  RoadRunner rr((SBMLFeaturesDir / "stoich_ar_const.xml").string());
  EXPECT_THROW(rr.setValue("N", 2), rrllvm::LLVMException);
}

TEST_F(SBMLFeatures, variable_stoich_event_set_initial) {
  RoadRunner rr((SBMLFeaturesDir / "stoich_event.xml").string());
  rr.setValue("N", 2);
  const ls::DoubleMatrix result = *rr.simulate(0.0, 2.0, 11);
  for (int i = 0; i < result.numRows(); i++) {
    double expected = (i <= 5) ? 0.2 * i : 0.5 + 0.1 * i;
    EXPECT_NEAR(result(i, 1), expected, 1e-4);
  }
}


// A plain reset (TIME | RATE | FLOATING) mirrors global parameter semantics:
// it only restores a stoichiometry if a rate rule governs it. stoich_rr's N
// does have a rate rule, so plain reset() restores it.
TEST_F(SBMLFeatures, variable_stoich_rr_reset_restores_initial) {
  RoadRunner rr((SBMLFeaturesDir / "stoich_rr.xml").string());
  const ls::DoubleMatrix run1 = *rr.simulate(0.0, 2.0, 11);

  rr.reset();
  rr.setValue("init(N)", 2);
  const ls::DoubleMatrix run2 = *rr.simulate(0.0, 2.0, 11);

  rr.reset();
  const ls::DoubleMatrix run3 = *rr.simulate(0.0, 2.0, 11);

  for (int i = 0; i < run1.numRows(); i++) {
    EXPECT_NEAR(run1(i, 1), 0.01 * i * i, 1e-6);
    EXPECT_NEAR(run2(i, 1), 0.2 * i + 0.01 * i * i, 1e-6);
    EXPECT_NEAR(run3(i, 1), 0.2 * i + 0.01 * i * i, 1e-6);
  }
}

// An explicit full reset (SelectionRecord::ALL, which includes SBML_INITIALIZE)
// restores a stoichiometry to whatever init(N) is CURRENTLY configured to --
// not back to the original SBML-declared value. Since init(N) was changed to 2
// before the second reset, run3 should match run2 (N0=2), not run1 (N0=0).
TEST_F(SBMLFeatures, variable_stoich_rr_full_reset_restores_initial) {
  RoadRunner rr((SBMLFeaturesDir / "stoich_rr.xml").string());
  const ls::DoubleMatrix run1 = *rr.simulate(0.0, 2.0, 11);

  rr.reset(SelectionRecord::ALL);
  rr.setValue("init(N)", 2);
  const ls::DoubleMatrix run2 = *rr.simulate(0.0, 2.0, 11);

  rr.reset(SelectionRecord::ALL);
  const ls::DoubleMatrix run3 = *rr.simulate(0.0, 2.0, 11);

  for (int i = 0; i < run1.numRows(); i++) {
    EXPECT_NEAR(run3(i, 1), 0.2 * i + 0.01 * i * i, 1e-6);
  }
}

// stoich_event's N has no rate rule, so a plain reset() correctly leaves it
// alone (mirrors how a plain, non-rate-rule global parameter isn't restored
// by default reset() either). Only an explicit full reset restores it, to
// whatever init(N) is currently configured to -- see run2's formula below.
TEST_F(SBMLFeatures, variable_stoich_event_full_reset_restores_initial) {
  RoadRunner rr((SBMLFeaturesDir / "stoich_event.xml").string());
  const ls::DoubleMatrix run1 = *rr.simulate(0.0, 2.0, 11);

  rr.reset(SelectionRecord::ALL);
  rr.setValue("init(N)", 2);
  const ls::DoubleMatrix run2 = *rr.simulate(0.0, 2.0, 11);

  rr.reset(SelectionRecord::ALL);
  const ls::DoubleMatrix run3 = *rr.simulate(0.0, 2.0, 11);

  for (int i = 0; i < run1.numRows(); i++) {
    double expected2 = (i <= 5) ? 0.2 * i : 0.5 + 0.1 * i;
    EXPECT_NEAR(run3(i, 1), expected2, 1e-4);
  }
}


// dual_rate_rule_stoich.xml: S1_stoich (reactant of S1, rate rule 0.01)
// and S2_stoich (product of S2, rate rule 0.02) -- different species, no
// MultiSpeciesReference collision. The named-id form always returns the
// reference's own positive magnitude; the stoich(species, reaction) form
// reads the raw matrix cell (negative for a reactant).
TEST_F(SBMLFeatures, dual_rate_rule_stoich_selectors) {
  RoadRunner rr((SBMLFeaturesDir / "dual_rate_rule_stoich.xml").string());
  EXPECT_NEAR(rr.getValue("S1_stoich"), 1.0, 1e-9);
  EXPECT_NEAR(rr.getValue("S2_stoich"), 2.0, 1e-9);
  EXPECT_NEAR(rr.getValue("stoich(S1, J0)"), -1.0, 1e-9);
  EXPECT_NEAR(rr.getValue("stoich(S2, J0)"), 2.0, 1e-9);
}


// named_stoic_multi_reactant.xml: A is consumed via two colliding reactant
// references, r1=2 and r2=3, kinetic law v = k = 0.1 (constant, so dA/dt
// doesn't depend on A -- keeps the math polynomial in t).
// dA/dt = -(r1(t)+r2(t)) * 0.1

TEST_F(SBMLFeatures, multi_reactant_rate_rule_on_one) {
  // r1 gets a rate rule (dr1/dt = 0.5); r2 stays plain, untouched.
  // r1(t) = 2 + 0.5t, r2(t) = 3.
  // dA/dt = -(5 + 0.5t) * 0.1  ->  A(t) = 10 - (0.5t + 0.025t^2)
  RoadRunner rr((SBMLFeaturesDir / "named_stoic_multi_reactant.xml").string());
  EXPECT_NO_THROW(rr.addRateRule("r1", "0.5", true));

  rr.oneStep(0, 2.0);
  double r1;
  ASSERT_NO_THROW(r1 = rr.getValue("r1"));
  EXPECT_NEAR(r1, 3.0, 1e-4);
  EXPECT_NEAR(rr.getValue("r2"), 3.0, 1e-9);
  EXPECT_NEAR(rr.getValue("stoich(A, J0)"), -6.0, 1e-4);
  EXPECT_NEAR(rr.getValue("A"), 8.9, 1e-4);
}

TEST_F(SBMLFeatures, multi_reactant_rate_rule_on_both) {
  // r1: dr1/dt = 0.5, r1(0) = 2 -> r1(t) = 2 + 0.5t
  // r2: dr2/dt = 0.3, r2(0) = 3 -> r2(t) = 3 + 0.3t
  // dA/dt = -(5 + 0.8t) * 0.1  ->  A(t) = 10 - (0.5t + 0.04t^2)
  RoadRunner rr((SBMLFeaturesDir / "named_stoic_multi_reactant.xml").string());
  EXPECT_NO_THROW(rr.addRateRule("r1", "0.5", true));
  EXPECT_NO_THROW(rr.addRateRule("r2", "0.3", true));

  rr.oneStep(0, 2.0);
  double r1;
  ASSERT_NO_THROW(r1 = rr.getValue("r1"));
  EXPECT_NEAR(r1, 3.0, 1e-4);
  EXPECT_NEAR(rr.getValue("r2"), 3.6, 1e-4);
  EXPECT_NEAR(rr.getValue("stoich(A, J0)"), -6.6, 1e-4);
  EXPECT_NEAR(rr.getValue("A"), 8.84, 1e-4);
}

TEST_F(SBMLFeatures, multi_reactant_assignment_rule_on_one) {
  // r1 = 2 + 0.5*time (assignment rule); r2 stays plain.
  // Same shape as multi_reactant_rate_rule_on_one -> same A(t).
  RoadRunner rr((SBMLFeaturesDir / "named_stoic_multi_reactant.xml").string());
  EXPECT_NO_THROW(rr.addAssignmentRule("r1", "2 + 0.5*time", true));

  rr.oneStep(0, 2.0);
  double r1;
  ASSERT_NO_THROW(r1 = rr.getValue("r1"));
  EXPECT_NEAR(r1, 3.0, 1e-4);
  EXPECT_NEAR(rr.getValue("r2"), 3.0, 1e-9);
  EXPECT_NEAR(rr.getValue("stoich(A, J0)"), -6.0, 1e-4);
  EXPECT_NEAR(rr.getValue("A"), 8.9, 1e-4);
}

TEST_F(SBMLFeatures, multi_reactant_assignment_rule_on_both) {
  // r1 = 2 + 0.5*time, r2 = 3 + 0.3*time (both assignment rules).
  // Same shape as multi_reactant_rate_rule_on_both -> same A(t).
  RoadRunner rr((SBMLFeaturesDir / "named_stoic_multi_reactant.xml").string());
  EXPECT_NO_THROW(rr.addAssignmentRule("r1", "2 + 0.5*time", true));
  EXPECT_NO_THROW(rr.addAssignmentRule("r2", "3 + 0.3*time", true));

  rr.oneStep(0, 2.0);
  double r1;
  ASSERT_NO_THROW(r1 = rr.getValue("r1"));
  EXPECT_NEAR(r1, 3.0, 1e-4);
  EXPECT_NEAR(rr.getValue("r2"), 3.6, 1e-4);
  EXPECT_NEAR(rr.getValue("stoich(A, J0)"), -6.6, 1e-4);
  EXPECT_NEAR(rr.getValue("A"), 8.84, 1e-4);
}

TEST_F(SBMLFeatures, multi_reactant_mixed_rate_and_assignment_rule) {
  // r1 gets a rate rule (dr1/dt = 0.5), r2 gets an assignment rule
  // (3 + 0.3*time). Same shape as the "on_both" cases -> same A(t).
  RoadRunner rr((SBMLFeaturesDir / "named_stoic_multi_reactant.xml").string());
  EXPECT_NO_THROW(rr.addRateRule("r1", "0.5", true));
  EXPECT_NO_THROW(rr.addAssignmentRule("r2", "3 + 0.3*time", true));

  rr.oneStep(0, 2.0);
  double r1;
  ASSERT_NO_THROW(r1 = rr.getValue("r1"));
  EXPECT_NEAR(r1, 3.0, 1e-4);
  EXPECT_NEAR(rr.getValue("r2"), 3.6, 1e-4);
  EXPECT_NEAR(rr.getValue("stoich(A, J0)"), -6.6, 1e-4);
  EXPECT_NEAR(rr.getValue("A"), 8.84, 1e-4);
}


TEST_F(SBMLFeatures, multi_reactant_rate_rule_reset_semantics) {
  // r1 is rate-rule-governed (part of a colliding pair with r2, which
  // stays plain). Mirrors the single-reference reset semantics: plain
  // reset() restores a rate-rule-governed member (like a global parameter
  // would), and reset(ALL) re-syncs to whatever init(r1) is CURRENTLY
  // configured to, not the original declared value.
  RoadRunner rr((SBMLFeaturesDir / "named_stoic_multi_reactant.xml").string());
  EXPECT_NO_THROW(rr.addRateRule("r1", "0.5", true));

  rr.oneStep(0, 2.0);
  double r1;
  ASSERT_NO_THROW(r1 = rr.getValue("r1"));
  EXPECT_NEAR(r1, 3.0, 1e-4);
  EXPECT_NEAR(rr.getValue("r2"), 3.0, 1e-9);

  rr.reset();
  EXPECT_NEAR(rr.getValue("r1"), 2.0, 1e-9);
  EXPECT_NEAR(rr.getValue("r2"), 3.0, 1e-9);

  rr.reset(SelectionRecord::ALL);
  rr.setValue("init(r1)", 4);
  rr.oneStep(0, 2.0);
  EXPECT_NEAR(rr.getValue("r1"), 5.0, 1e-4);

  rr.reset(SelectionRecord::ALL);
  EXPECT_NEAR(rr.getValue("r1"), 4.0, 1e-9);
}


TEST_F(SBMLFeatures, variable_stoich_rr_selection_type) {
  RoadRunner rr((SBMLFeaturesDir / "stoich_rr.xml").string());
  SelectionRecord record = rr.createSelection("N");
  EXPECT_EQ(record.selectionType, SelectionRecord::STOICHIOMETRY);
  vector<string> ids = rr.getStoichiometryIds();
  EXPECT_NE(std::find(ids.begin(), ids.end(), "N"), ids.end());
}

TEST_F(SBMLFeatures, variable_stoich_ar_selection_type) {
  RoadRunner rr((SBMLFeaturesDir / "stoich_ar.xml").string());
  SelectionRecord record = rr.createSelection("N");
  EXPECT_EQ(record.selectionType, SelectionRecord::STOICHIOMETRY);
  vector<string> ids = rr.getStoichiometryIds();
  EXPECT_NE(std::find(ids.begin(), ids.end(), "N"), ids.end());
}

TEST_F(SBMLFeatures, variable_stoich_ar_const_selection_type) {
  RoadRunner rr((SBMLFeaturesDir / "stoich_ar_const.xml").string());
  SelectionRecord record = rr.createSelection("N");
  EXPECT_EQ(record.selectionType, SelectionRecord::STOICHIOMETRY);
  vector<string> ids = rr.getStoichiometryIds();
  EXPECT_NE(std::find(ids.begin(), ids.end(), "N"), ids.end());
}

TEST_F(SBMLFeatures, variable_stoich_event_selection_type) {
  RoadRunner rr((SBMLFeaturesDir / "stoich_event.xml").string());
  SelectionRecord record = rr.createSelection("N");
  EXPECT_EQ(record.selectionType, SelectionRecord::STOICHIOMETRY);
  vector<string> ids = rr.getStoichiometryIds();
  EXPECT_NE(std::find(ids.begin(), ids.end(), "N"), ids.end());
}


// Rate rules and assignment rules on named boundary-species
// stoichiometries. Since a boundary reference has no matrix cell, these
// mirror the plain (non-colliding) rate/assignment-rule tests above, just
// with a boundary species as the target instead of a floating one.

TEST_F(SBMLFeatures, named_boundary_stoich_assignment_rule) {
  // J2: S1 (floating reactant) -> m=X (boundary product). m = time.
  RoadRunner rr((SBMLFeaturesDir / "named_boundary_species_asnt_rule.xml").string());
  EXPECT_NEAR(rr.getValue("m"), 0.0, 1e-9);
  rr.oneStep(0, 2.0);
  EXPECT_NEAR(rr.getValue("m"), 2.0, 1e-9);
}

TEST_F(SBMLFeatures, named_boundary_stoich_rate_rule) {
  // J2: S1 (floating reactant) -> m=X (boundary product, declared 2).
  // dm/dt = 0.5.
  RoadRunner rr((SBMLFeaturesDir / "named_boundary_species_rate_rule.xml").string());
  EXPECT_NEAR(rr.getValue("m"), 2.0, 1e-9);
  rr.oneStep(0, 2.0);
  EXPECT_NEAR(rr.getValue("m"), 3.0, 1e-6);
}

TEST_F(SBMLFeatures, named_boundary_stoich_rate_rule_reset_semantics) {
  // Mirrors multi_reactant_rate_rule_reset_semantics: plain reset() restores
  // a rate-rule-governed boundary stoichiometry (like a global parameter
  // would), and reset(ALL) re-syncs to whatever init(m) is CURRENTLY
  // configured to, not the original declared value.
  RoadRunner rr((SBMLFeaturesDir / "named_boundary_species_rate_rule.xml").string());

  rr.oneStep(0, 2.0);
  EXPECT_NEAR(rr.getValue("m"), 3.0, 1e-6);

  rr.reset();
  EXPECT_NEAR(rr.getValue("m"), 2.0, 1e-9);

  rr.reset(SelectionRecord::ALL);
  rr.setValue("init(m)", 4);
  rr.oneStep(0, 2.0);
  EXPECT_NEAR(rr.getValue("m"), 5.0, 1e-4);

  rr.reset(SelectionRecord::ALL);
  EXPECT_NEAR(rr.getValue("m"), 4.0, 1e-9);
}

TEST_F(SBMLFeatures, named_boundary_stoich_referenced_by_other_stoich_assignment_rule) {
  // J2: m=X (boundary reactant, stoich 2, no rule) -> n=S1 (floating
  // product, assignment rule n = m). n itself is a plain, non-colliding
  // floating stoichiometry (already supported); what's new here is that its
  // assignment rule formula references "m", a boundary stoichiometry.
  RoadRunner rr((SBMLFeaturesDir / "named_boundary_species_in_kl_and_other_stoich.xml").string());
  EXPECT_EQ(rr.getValue("m"), 2.0);
  EXPECT_EQ(rr.getValue("n"), 2.0);

  rr.setValue("m", 5);
  EXPECT_EQ(rr.getValue("n"), 5.0);

  // stoich(S1, J2) reads the raw matrix cell for the floating species,
  // driven by n's assignment rule.
  EXPECT_NEAR(rr.getValue("stoich(S1, J2)"), 5.0, 1e-9);
}

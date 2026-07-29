#include "gtest/gtest.h"
#include "rrRoadRunner.h"
#include "rrException.h"
#include "rrUtils.h"
#include "rrTestSuiteModelSimulation.h"
#include "sbml/SBMLTypes.h"
#include "sbml/SBMLReader.h"
#include "../test_util.h"
#include <filesystem>
#include "RoadRunnerTest.h"
#include "llvm/LLVMException.h"
#include "llvm/LLVMExecutableModel.h"

using namespace testing;
using namespace rr;
using namespace std;
using std::filesystem::path;

class SBMLFeatures : public RoadRunnerTest {
public:
  path SBMLFeaturesDir = rrTestModelsDir_ / "SBMLFeatures";
  SBMLFeatures() = default;
};

TEST_F(SBMLFeatures, named_stoich_list) {
  rr::RoadRunner rr((SBMLFeaturesDir / "named_stoic_in_kinetic_law.xml").string());
  vector<string> stoichs = rr.getStoichiometryIds();
  EXPECT_STREQ(stoichs[0].c_str(), "stoich(A, J0)");
  EXPECT_STREQ(stoichs[1].c_str(), "n");
  EXPECT_STREQ(stoichs[2].c_str(), "m");
  EXPECT_STREQ(stoichs[3].c_str(), "q");
}


TEST_F(SBMLFeatures, issue1306_named_stoich_value) {
  rr::RoadRunner rr((SBMLFeaturesDir / "named_stoic_in_kinetic_law.xml").string());
  rr.setValue("n", 3);
  rr.setValue("m", 5);
  rr.setValue("q", 7);
  EXPECT_EQ(rr.getValue("n"), 3.0);
  EXPECT_EQ(rr.getValue("m"), 5.0);
  EXPECT_EQ(rr.getValue("q"), 7.0);
  // stoich(species, reaction) reads the raw stoichiometry-matrix cell:
  // negative for reactants (B, via "n"), positive for products (C via "m", D via "q").
  EXPECT_EQ(rr.getValue("stoich(B, J0)"), -3.0);
  EXPECT_EQ(rr.getValue("stoich(C, J0)"), 5.0);
  EXPECT_EQ(rr.getValue("stoich(D, J0)"), 7.0);
  EXPECT_EQ(rr.getValue("J0"), 15.0);
  rr.oneStep(0, 0.01);
  EXPECT_NEAR(rr.getValue("A"), 0.8607083324139845, 0.00001);
  EXPECT_NEAR(rr.getValue("B"), 6.582124997241953, 0.00001);
  EXPECT_NEAR(rr.getValue("C"), 0.6964583379300783, 0.00001);
  EXPECT_NEAR(rr.getValue("D"), 0.9750416731021092, 0.00001);
  string sbml = rr.getCurrentSBML();

}


TEST_F(SBMLFeatures, issue1306_named_stoich_steadyState) {
  rr::RoadRunner rr((SBMLFeaturesDir / "named_stoic_in_kinetic_law.xml").string());
  rr.steadyState();
  EXPECT_NEAR(rr.getValue("B"), 6.0, 0.00001);
  rr.reset();
  rr.setValue("n", 3);
  rr.setValue("m", 5);
  rr.setValue("q", 7);
  rr.setConservedMoietyAnalysis(true);
  EXPECT_EQ(rr.getValue("n"), 3.0);
  EXPECT_EQ(rr.getValue("m"), 5.0);
  //EXPECT_EQ(rr.getValue("q"), 7.0);
  //EXPECT_EQ(rr.getValue("J0"), 0.0);
  rr.steadyState();
  EXPECT_NEAR(rr.getValue("B"), 4.0, 0.00001);
}


TEST_F(SBMLFeatures, named_stoich_init_value) {
  rr::RoadRunner rr((SBMLFeaturesDir / "named_stoic_in_kinetic_law.xml").string());
  rr.setValue("init(n)", 3);
  rr.setValue("init(m)", 5);
  rr.setValue("init(q)", 7);
  EXPECT_EQ(rr.getValue("n"), 3.0);
  EXPECT_EQ(rr.getValue("m"), 5.0);
  EXPECT_EQ(rr.getValue("q"), 7.0);
  EXPECT_EQ(rr.getValue("init(n)"), 3.0);
  EXPECT_EQ(rr.getValue("init(m)"), 5.0);
  EXPECT_EQ(rr.getValue("init(q)"), 7.0);
}


TEST_F(SBMLFeatures, named_stoich_values) {
  rr::RoadRunner rr((SBMLFeaturesDir / "named_stoic_in_kinetic_law.xml").string());
  rr.setValue("n", 3);
  rr.setValue("m", 5);
  rr.setValue("q", 7);
  EXPECT_EQ(rr.getModel()->getValue("n"), 3.0);
  EXPECT_EQ(rr.getModel()->getValue("m"), 5.0);
  EXPECT_EQ(rr.getModel()->getValue("q"), 7.0);
  EXPECT_EQ(rr.getModel()->getValue("init(n)"), 3.0);
  EXPECT_EQ(rr.getModel()->getValue("init(m)"), 5.0);
  EXPECT_EQ(rr.getModel()->getValue("init(q)"), 7.0);
}


TEST_F(SBMLFeatures, add_rule_to_named_stoich) {
  // "n" is the reactant B's stoichiometry (declared 1); "m" (C, declared 2)
  // and "q" (D, declared 3) are unaffected. Kinetic law rate = (n+m+q)*A.
  rr::RoadRunner rr((SBMLFeaturesDir / "named_stoic_in_kinetic_law.xml").string());
  EXPECT_NO_THROW(rr.addAssignmentRule("n", "5", true));
  EXPECT_EQ(rr.getValue("n"), 5.0);
  // rate = (5 + 2 + 3) * A(1) = 10
  EXPECT_NEAR(rr.getValue("J0"), 10.0, 1e-9);

  // "q" is the product D's stoichiometry (declared 3).
  rr::RoadRunner rr2((SBMLFeaturesDir / "named_stoic_in_kinetic_law.xml").string());
  EXPECT_NO_THROW(rr2.addRateRule("q", "5", true));
  // rate rules haven't integrated yet at t=0; q should still be its declared value.
  EXPECT_NEAR(rr2.getValue("q"), 3.0, 1e-9);
  rr2.oneStep(0, 1.0);
  // dq/dt = 5, so after 1 time unit q should have grown by ~5.
  EXPECT_NEAR(rr2.getValue("q"), 8.0, 1e-6);
}


TEST_F(SBMLFeatures, add_rate_rule_to_reactant_stoich) {
  // "n" is reactant B's stoichiometry (declared 1). Mirrors the "q"
  // (product) rate-rule case in add_rule_to_named_stoich, but exercises the
  // reactant sign-handling path: createStoichiometryNode negates reactant
  // values for the CSR cell, and that same negated value must not leak into
  // the rate-rule integration slot.
  rr::RoadRunner rr((SBMLFeaturesDir / "named_stoic_in_kinetic_law.xml").string());
  EXPECT_NO_THROW(rr.addRateRule("n", "5", true));
  // rate rules haven't integrated yet at t=0; n should still be its declared value.
  EXPECT_NEAR(rr.getValue("n"), 1.0, 1e-9);
  rr.oneStep(0, 1.0);
  // dn/dt = 5, so after 1 time unit n should have grown by ~5, mirroring q's 3->8.
  EXPECT_NEAR(rr.getValue("n"), 6.0, 1e-6);
}


TEST_F(SBMLFeatures, add_assignment_rule_to_reactant_stoich_tracks_time) {
  // "n" is reactant B's stoichiometry. An assignment rule that depends on
  // time must be resynced into the stoichiometry matrix as the simulation
  // progresses, not just read once at t=0.
  rr::RoadRunner rr((SBMLFeaturesDir / "named_stoic_in_kinetic_law.xml").string());
  EXPECT_NO_THROW(rr.addAssignmentRule("n", "1 + time", true));
  EXPECT_NEAR(rr.getValue("n"), 1.0, 1e-9);
  rr.oneStep(0, 2.0);
  EXPECT_NEAR(rr.getValue("n"), 3.0, 1e-9);
}


TEST_F(SBMLFeatures, set_value_on_multi_reactant_stoich_before_simulate) {
  // A appears twice as a reactant in J0, via two independently named,
  // rule-free species references (r1=2, r2=3). Setting one before
  // simulating must not disturb the other, and the combined (net) cell used
  // in the reaction must reflect the sum of both.
  rr::RoadRunner rr((SBMLFeaturesDir / "named_stoic_multi_reactant.xml").string());
  double r1;
  ASSERT_NO_THROW(r1 = rr.getValue("r1"));
  EXPECT_EQ(r1, 2.0);
  EXPECT_EQ(rr.getValue("r2"), 3.0);

  EXPECT_NO_THROW(rr.setValue("r1", 5));
  EXPECT_EQ(rr.getValue("r1"), 5.0);
  EXPECT_EQ(rr.getValue("r2"), 3.0);

  // stoich(A, J0) reads the raw matrix cell: A is consumed here, so the
  // combined cell is negative: -(r1 + r2) = -8.
  EXPECT_NEAR(rr.getValue("stoich(A, J0)"), -8.0, 1e-9);

  // dA/dt = -(r1+r2)*k*A < 0, A should be decreasing
  rr.oneStep(0, 0.01);
  EXPECT_LT(rr.getValue("A"), 10.0);
}


TEST_F(SBMLFeatures, multi_product_stoich_collision_set_one_of_two) {
  // B is produced twice in J0, via two independently named, rule-free
  // species references (p1=2, p2=3). Setting one before simulating must
  // not disturb the other, and the combined (net) cell used in the
  // reaction must reflect the sum of both.
  rr::RoadRunner rr((SBMLFeaturesDir / "named_stoic_multi_product.xml").string());
  double p1;
  ASSERT_NO_THROW(p1 = rr.getValue("p1"));
  EXPECT_EQ(p1, 2.0);
  EXPECT_EQ(rr.getValue("p2"), 3.0);

  EXPECT_NO_THROW(rr.setValue("p1", 5));
  EXPECT_EQ(rr.getValue("p1"), 5.0);
  EXPECT_EQ(rr.getValue("p2"), 3.0);

  // combined cell should be p1 + p2 = 8 (products are positive, no sign flip)
  EXPECT_NEAR(rr.getValue("stoich(B, J0)"), 8.0, 1e-9);

  rr.oneStep(0, 0.01);
  EXPECT_GT(rr.getValue("B"), 0.0);
}


TEST_F(SBMLFeatures, multi_reactant_product_cross_collision_set_one_of_two) {
  // A appears once as a reactant (x1) and once as a product (x2) in the
  // SAME reaction -- no literal duplicate within either list, but both
  // occurrences still collide on the same (species, reaction) CSR cell
  // (LLVMModelDataSymbols shares one speciesMap across the reactant and
  // product loops). Setting one must not disturb the other. stoich(A, J0)
  // reads the raw matrix cell (-x1 + x2), unaffected by the ambiguity of
  // "reactant or product" that a per-reference read would have.
  rr::RoadRunner rr((SBMLFeaturesDir / "named_stoic_multi_mixed.xml").string());
  double x1;
  ASSERT_NO_THROW(x1 = rr.getValue("x1"));
  EXPECT_EQ(x1, 2.0);
  EXPECT_EQ(rr.getValue("x2"), 3.0);

  EXPECT_NO_THROW(rr.setValue("x1", 5));
  EXPECT_EQ(rr.getValue("x1"), 5.0);
  EXPECT_EQ(rr.getValue("x2"), 3.0);

  // net effect: -x1 + x2 = -5 + 3 = -2
  EXPECT_NEAR(rr.getValue("stoich(A, J0)"), -2.0, 1e-9);

  rr.oneStep(0, 0.01);
  EXPECT_LT(rr.getValue("A"), 10.0);
}


TEST_F(SBMLFeatures, multi_reactant_stoich_set_via_selector_throws) {
  // Unlike getValue(stoich(x,y)), which reads the role-agnostic matrix
  // cell, setValue(stoich(x,y), v) means "set the underlying
  // speciesReference" -- which is ambiguous when more than one reference
  // shares the cell, so it must throw rather than silently pick one.
  rr::RoadRunner rr((SBMLFeaturesDir / "named_stoic_multi_reactant.xml").string());
  EXPECT_THROW(rr.setValue("stoich(A, J0)", 5), rrllvm::LLVMException);
}


TEST_F(SBMLFeatures, named_stoich_selectors) {
  rr::RoadRunner rr((SBMLFeaturesDir / "named_stoic_in_kinetic_law.xml").string());
  SelectionRecord record = rr.createSelection("n");
  EXPECT_EQ(record.selectionType, SelectionRecord::STOICHIOMETRY);
  EXPECT_STREQ(record.to_string().c_str(), "n");
  EXPECT_EQ(record.index, 1);
  EXPECT_EQ(record.p1, "n");
  EXPECT_EQ(record.p2, "");

  record = rr.createSelection("stoich(A, J0)");
  EXPECT_EQ(record.selectionType, SelectionRecord::STOICHIOMETRY);
  EXPECT_STREQ(record.to_string().c_str(), "stoich(A, J0)");
  EXPECT_EQ(record.index, 0);
  EXPECT_EQ(record.p1, "A");
  EXPECT_EQ(record.p2, "J0");
}


TEST_F(SBMLFeatures, named_stoich_set_and_reset) {
  RoadRunner rr((SBMLFeaturesDir / "named_stoic.xml").string());
  rr.getSimulateOptions().setDuration(10);

  const ls::DoubleMatrix run1 = *rr.simulate();

  rr.reset(SelectionRecord::ALL);
  rr.setValue("N", 2);
  const ls::DoubleMatrix run2 = *rr.simulate();

  rr.reset(SelectionRecord::ALL);
  const ls::DoubleMatrix run3 = *rr.simulate();

  ASSERT_EQ(run1.numRows(), run2.numRows());
  ASSERT_EQ(run1.numRows(), run3.numRows());
  for (int i = 0; i < run1.numRows(); i++) {
    EXPECT_EQ(run1(i, 1), run3(i, 1));
    EXPECT_NEAR(run2(i, 1), 2 * run1(i, 1), 1e-12);
  }
}


TEST_F(SBMLFeatures, get_named_stoich_value_from_model) {
  rr::RoadRunner rr((SBMLFeaturesDir / "named_stoic_in_kinetic_law.xml").string());
  ExecutableModel* em = rr.getModel();
  rrllvm::LLVMExecutableModel* llem = static_cast<rrllvm::LLVMExecutableModel*>(em);
  
  EXPECT_EQ(llem->getValue("n"), 1);
  llem->setValue("n", 3);
  EXPECT_EQ(llem->getValue("n"), 3);

  // getValue(stoich(x,y)) reads the raw (role-agnostic) matrix cell, so A
  // being a reactant reads as -1. setValue(stoich(x,y), v), by contrast,
  // means "set the underlying speciesReference" -- it takes v as a
  // positive magnitude and sign-flips internally for a reactant, same as
  // the named-id form. Setting 5 here therefore stores -5.
  EXPECT_EQ(llem->getValue("stoich(A, J0)"), -1);
  llem->setValue("stoich(A, J0)", 5);
  EXPECT_EQ(llem->getValue("stoich(A, J0)"), -5);
}



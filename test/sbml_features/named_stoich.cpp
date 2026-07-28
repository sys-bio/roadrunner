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
  EXPECT_EQ(rr.getValue("stoich(B, J0)"), 3.0);
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

  EXPECT_EQ(llem->getValue("stoich(A, J0)"), 1);
  llem->setValue("stoich(A, J0)", 5);
  EXPECT_EQ(llem->getValue("stoich(A, J0)"), 5);
}



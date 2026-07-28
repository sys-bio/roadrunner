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
// does have a rate rule, so plain reset() is expected to restore it here --
// this is currently failing because LLVMExecutableModel::reset() has no
// resetOneType dispatch for SelectionRecord::STOICHIOMETRY at all yet.
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
    EXPECT_NEAR(run3(i, 1), run1(i, 1), 1e-6);
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

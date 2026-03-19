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
  string sbml = rr.getCurrentSBML();

}


TEST_F(SBMLFeatures, issue1306_named_stoich_steadyState) {
  rr::RoadRunner rr((SBMLFeaturesDir / "named_stoic_in_kinetic_law.xml").string());
  //rr.steadyState();
  //EXPECT_NEAR(rr.getValue("B"), 6.0, 0.00001);
  //rr.reset();
  rr.setValue("n", 3);
  rr.setValue("m", 5);
  rr.setValue("q", 7);
  rr.steadyState();
  EXPECT_NEAR(rr.getValue("B"), 4.0, 0.00001);
}



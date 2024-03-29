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

TEST_F(SBMLFeatures, PIECEWISE_TRIGGERS_1)
{
    RoadRunner rri((SBMLFeaturesDir / "need_piecewise_triggers.xml").string());
    const ls::DoubleMatrix* results = rri.simulate();
    EXPECT_NEAR(rri.getValue(rri.createSelection("X1")), 1.0310384266045054, 0.0001);
}

TEST_F(SBMLFeatures, PIECEWISE_TRIGGERS_2)
{
    RoadRunner rri((SBMLFeaturesDir / "need_piecewise_triggers_rev.xml").string());
    const ls::DoubleMatrix* results = rri.simulate();
    EXPECT_NEAR(rri.getValue(rri.createSelection("X1")), 1.0310384266045054, 0.0001);
}

TEST_F(SBMLFeatures, PIECEWISE_TRIGGERS_3)
{
    RoadRunner rri((SBMLFeaturesDir / "need_piecewise_triggers_3part.xml").string());
    const ls::DoubleMatrix* results = rri.simulate();
    EXPECT_NEAR(rri.getValue(rri.createSelection("X1")), 1.0310384266045054, 0.0001);
}

TEST_F(SBMLFeatures, PIECEWISE_AND_EVENT_TRIGGERS)
{
    //This worked in rr 2.5, but failed with the new version because piecewise triggers overrode the events triggers.
    RoadRunner rri((SBMLFeaturesDir / "need_piecewise_and_event_triggers.xml").string());
    const ls::DoubleMatrix* results = rri.simulate();
    EXPECT_NEAR(rri.getValue(rri.createSelection("X1")), 17.66586981819867, 0.0001);
}


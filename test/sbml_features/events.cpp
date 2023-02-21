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

TEST_F(SBMLFeatures, EVENT_T0_FIRING_L2)
{
    try
    {
        RoadRunner rri((SBMLFeaturesDir / "t0_firing_l2v1.xml").string());
        rri.simulate();
        EXPECT_EQ(rri.getValue(rri.createSelection("d")), 6.0);
    }
    catch (std::exception& ex)
    {
        std::cout << "Exception: " << ex.what() << std::endl;
        EXPECT_TRUE(false);
    }
}

TEST_F(SBMLFeatures, EVENT_T0_FIRING_L3)
{
    try
    {
        RoadRunner rri((SBMLFeaturesDir / "t0_firing_l3v1.xml").string());
        rri.simulate();
        EXPECT_EQ(rri.getValue(rri.createSelection("d")), 6.0);
    }
    catch (std::exception& ex)
    {
        std::cout << "Exception: " << ex.what() << std::endl;
        EXPECT_TRUE(false);
    }
}


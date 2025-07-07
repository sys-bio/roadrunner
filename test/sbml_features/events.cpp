#include "gtest/gtest.h"
#include "rrRoadRunner.h"
#include "rrException.h"
#include "rrUtils.h"
#include "rrConfig.h"
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

TEST_F(SBMLFeatures, EVENT_INFINITE_LOOP_CATCH)
{
    try
    {
        RoadRunner rri((SBMLFeaturesDir / "infinite_events.xml").string());
        rri.simulate();
        EXPECT_TRUE(false);
    }
    catch (std::exception& ex)
    {
        EXPECT_TRUE(string(ex.what()).find("Max number of cascaded events") != std::string::npos);
    }
}

TEST_F(SBMLFeatures, EVENT_NIGH_INFINITE_LOOP_CATCH)
{
    try
    {
        RoadRunner rri((SBMLFeaturesDir / "nigh_infinite_events.xml").string());
        rri.simulate();
        EXPECT_EQ(rri.getValue(rri.createSelection("y")), 5.0);
    }
    catch (std::exception& ex)
    {
        EXPECT_TRUE(string(ex.what()).find("Max number of cascaded events") != std::string::npos);
    }

    //Ensure the MAX_EVENT_CASCADE works when larger than number of loops
    Config::setValue(Config::MAX_EVENT_CASCADE, 100000);
    try
    {
        RoadRunner rri((SBMLFeaturesDir / "nigh_infinite_events.xml").string());
        rri.simulate();
        EXPECT_EQ(rri.getValue(rri.createSelection("y")), 5.0);
    }
    catch (std::exception& ex)
    {
        EXPECT_TRUE(false);
    }

    //Ensure the MAX_EVENT_CASCADE works when turned off
    Config::setValue(Config::MAX_EVENT_CASCADE, -1);
    try
    {
        RoadRunner rri((SBMLFeaturesDir / "nigh_infinite_events.xml").string());
        rri.simulate();
        EXPECT_EQ(rri.getValue(rri.createSelection("y")), 5.0);
    }
    catch (std::exception& ex)
    {
        EXPECT_TRUE(false);
    }

}

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


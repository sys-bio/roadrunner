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

TEST_F(SBMLFeatures, FAST_RXN)
{
    //Logger::enableConsoleLogging();
    //Logger::setLevel(Logger::LOG_DEBUG);

    try
    {
        RoadRunner rri((SBMLFeaturesDir / "fast_reaction.xml").string());
        rri.simulate();
        EXPECT_TRUE(false);
    }
    catch (std::exception& ex)
    {
        //std::cout << "Exception: " << ex.what() << std::endl;
        EXPECT_TRUE(string(ex.what()).find("Unable to support 'fast' reactions.") != string::npos);
    }
}

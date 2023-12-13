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

TEST_F(SBMLFeatures, SBML_qual)
{
    EXPECT_THROW(RoadRunner rri((SBMLFeaturesDir / "BIOMD0000000562_url.xml").string()), std::domain_error);
}

TEST_F(SBMLFeatures, SBML_spatial)
{
    EXPECT_THROW(RoadRunner rri((SBMLFeaturesDir / "organelles.xml").string()), std::domain_error);
}

TEST_F(SBMLFeatures, SBML_multi)
{
    EXPECT_THROW(RoadRunner rri((SBMLFeaturesDir / "simmune_Ecad.xml").string()), std::domain_error);
}

TEST_F(SBMLFeatures, SBML_fbc)
{
    RoadRunner rri((SBMLFeaturesDir / "fbc_example.xml").string());
    EXPECT_THROW(rri.simulate(), std::domain_error);
    RoadRunner rr2((SBMLFeaturesDir / "fbc_example.xml").string());
    rr2.addReaction("J1", { "S1" }, {  }, "5.4", true);
    rr2.simulate();
}


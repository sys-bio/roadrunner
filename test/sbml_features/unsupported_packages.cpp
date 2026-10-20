#include "gtest/gtest.h"
#include "rrRoadRunner.h"
#include "rrSBMLReader.h"
#include "rrException.h"
#include "rrUtils.h"
#include "rrTestSuiteModelSimulation.h"
#include "sbml/SBMLTypes.h"
#include "sbml/SBMLReader.h"
#include "../test_util.h"
#include <filesystem>
#include <fstream>
#include <sstream>
#include "RoadRunnerTest.h"

using namespace testing;
using namespace rr;
using namespace std;
using std::filesystem::path;

class SBMLFeatures : public RoadRunnerTest {
public:
    path SBMLFeaturesDir = rrTestModelsDir_ / "SBMLFeatures";
    SBMLFeatures() = default;

    static std::string readFileToString(const path& p) {
        std::ifstream in(p);
        std::stringstream ss;
        ss << in.rdbuf();
        return ss.str();
    }
};

TEST_F(SBMLFeatures, SBML_qual)
{
    RoadRunner rri;
    EXPECT_THROW(rri.load((SBMLFeaturesDir / "BIOMD0000000562_url.xml").string()), std::domain_error);
}

TEST_F(SBMLFeatures, SBML_spatial)
{
    RoadRunner rri;
    EXPECT_THROW(rri.load((SBMLFeaturesDir / "organelles.xml").string()), std::domain_error);
}

TEST_F(SBMLFeatures, SBML_multi)
{
    RoadRunner rri;
    EXPECT_THROW(rri.load((SBMLFeaturesDir / "simmune_Ecad.xml").string()), std::domain_error);
}

// A comp model read from a file path is flattened before being returned.
TEST_F(SBMLFeatures, SBML_comp)
{
    std::string flattened = SBMLReader::read((SBMLFeaturesDir / "comp_example.xml").string());
    EXPECT_EQ(flattened.find("level3/version1/comp"), std::string::npos);
    EXPECT_NE(flattened.find("sub1__S1"), std::string::npos);
}

// A comp model read directly from a std::string (not a file path) is also flattened.
TEST_F(SBMLFeatures, SBML_comp_fromString)
{
    std::string sbml = readFileToString(SBMLFeaturesDir / "comp_example.xml");
    std::string flattened = SBMLReader::read(sbml);
    EXPECT_EQ(flattened.find("level3/version1/comp"), std::string::npos);
    EXPECT_NE(flattened.find("sub1__S1"), std::string::npos);
}

// qual/spatial/multi models read directly from a std::string are rejected,
// just like when read from a file path (see SBML_qual/SBML_spatial/SBML_multi above).
TEST_F(SBMLFeatures, SBML_qual_fromString)
{
    std::string sbml = readFileToString(SBMLFeaturesDir / "BIOMD0000000562_url.xml");
    try {
        SBMLReader::read(sbml);
        FAIL() << "Expected std::domain_error to be thrown for a qual model read from a string";
    } catch (const std::domain_error& e) {
        std::string msg = e.what();
        EXPECT_NE(msg.find("qual"), std::string::npos);
        EXPECT_NE(msg.find("COLOMOTO"), std::string::npos);
    }
}

TEST_F(SBMLFeatures, SBML_spatial_fromString)
{
    std::string sbml = readFileToString(SBMLFeaturesDir / "organelles.xml");
    try {
        SBMLReader::read(sbml);
        FAIL() << "Expected std::domain_error to be thrown for a spatial model read from a string";
    } catch (const std::domain_error& e) {
        std::string msg = e.what();
        EXPECT_NE(msg.find("spatial"), std::string::npos);
        EXPECT_NE(msg.find("VCell"), std::string::npos);
    }
}

TEST_F(SBMLFeatures, SBML_multi_fromString)
{
    std::string sbml = readFileToString(SBMLFeaturesDir / "simmune_Ecad.xml");
    try {
        SBMLReader::read(sbml);
        FAIL() << "Expected std::domain_error to be thrown for a multi model read from a string";
    } catch (const std::domain_error& e) {
        std::string msg = e.what();
        EXPECT_NE(msg.find("multi"), std::string::npos);
        EXPECT_NE(msg.find("Simmune"), std::string::npos);
    }
}

TEST_F(SBMLFeatures, SBML_fbc)
{
    RoadRunner rri((SBMLFeaturesDir / "fbc_example.xml").string());
    EXPECT_THROW(rri.simulate(), std::domain_error);
    RoadRunner rr2((SBMLFeaturesDir / "fbc_example.xml").string());
    rr2.addReaction("J1", { "S1" }, {  }, "5.4", true);
    rr2.simulate();
}


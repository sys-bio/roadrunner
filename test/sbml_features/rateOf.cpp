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
#include "LLVMException.h"

using namespace testing;
using namespace rr;
using namespace std;
using std::filesystem::path;

class SBMLFeatures : public RoadRunnerTest {
public:
    path SBMLFeaturesDir = rrTestModelsDir_ / "SBMLFeatures";
    SBMLFeatures() = default;
};

TEST_F(SBMLFeatures, RATEOF_RNX1)
{
    //Logger::enableConsoleLogging();
    //Logger::setLevel(Logger::LOG_DEBUG);

    try
    {
        RoadRunner rri((SBMLFeaturesDir / "rateOf_1reaction.xml").string());
        rri.getSimulateOptions().setDuration(2);
        rri.simulate();
        EXPECT_EQ(rri.getValue(rri.createSelection("s2")), 1.2);
    }
    catch (std::exception& ex)
    {
        std::cout << "Exception: " << ex.what() << std::endl;
        EXPECT_TRUE(false);
    }
}

TEST_F(SBMLFeatures, RATEOF_AR1)
{
    try
    {
        RoadRunner rri((SBMLFeaturesDir / "rateOf_assignmentRule1.xml").string());
        rri.getSimulateOptions().setDuration(2);
        rri.simulate();
        EXPECT_EQ(rri.getValue(rri.createSelection("s2")), 0.0);
    }
    catch (std::exception& ex)
    {
        std::cout << "Exception: " << ex.what() << std::endl;
        EXPECT_TRUE(false);
    }
}

TEST_F(SBMLFeatures, RATEOF_AR2)
{
    try
    {
        RoadRunner rri((SBMLFeaturesDir / "rateOf_assignmentRule2.xml").string());
        EXPECT_EQ(rri.getValue(rri.createSelection("s2")), 1.0);
        EXPECT_EQ(rri.getValue(rri.createSelection("s1'")), 1.0);
        rri.getSimulateOptions().setDuration(1);
        rri.simulate();
        EXPECT_EQ(rri.getValue(rri.createSelection("s2")), 1.0);
        EXPECT_NEAR(rri.getValue(rri.createSelection("s1")), 6.0, 0.0001);
    }
    catch (std::exception& ex)
    {
        std::cout << "Exception: " << ex.what() << std::endl;
        EXPECT_TRUE(false);
    }
}

TEST_F(SBMLFeatures, RATEOF_AR_recursive_err)
{
  RoadRunner rri;
  EXPECT_THROW(rri.load((SBMLFeaturesDir / "rateOf_recursive_assignmentRule_invalid.xml").string()), rrllvm::LLVMException);
}

TEST_F(SBMLFeatures, RATEOF_AR_recursive_err2)
{
  RoadRunner rri;
  EXPECT_THROW(rri.load((SBMLFeaturesDir / "rateOf_recursive_assignmentRule_invalid2.xml").string()), rrllvm::LLVMException);
}

TEST_F(SBMLFeatures, RATEOF_RateRule_recursive_err)
{
  RoadRunner rri;
  EXPECT_THROW(rri.load((SBMLFeaturesDir / "rateOf_recursive_rateRule_invalid.xml").string()), rrllvm::LLVMException);
}

TEST_F(SBMLFeatures, RATEOF_RateRule_recursive_err2)
{
  RoadRunner rri;
  EXPECT_THROW(rri.load((SBMLFeaturesDir / "rateOf_recursive_rateRule_invalid2.xml").string()), rrllvm::LLVMException);
}

TEST_F(SBMLFeatures, RATEOF_Reaction_recursive_err)
{
  RoadRunner rri;
  EXPECT_THROW(rri.load((SBMLFeaturesDir / "rateOf_recursive_reaction_invalid.xml").string()), rrllvm::LLVMException);
}

TEST_F(SBMLFeatures, RATEOF_Reaction_recursive_err2)
{
  RoadRunner rri;
  EXPECT_THROW(rri.load((SBMLFeaturesDir / "rateOf_recursive_reaction_invalid2.xml").string()), rrllvm::LLVMException);
}

TEST_F(SBMLFeatures, RATEOF_NoRateOfAssignmentRule_err)
{
  RoadRunner rri;
  EXPECT_THROW(rri.load((SBMLFeaturesDir / "rateOf_noARrates_invalid.xml").string()), rrllvm::LLVMException);
}

TEST_F(SBMLFeatures, RATEOF_NoRateOfConcentrationWhenCompartmentHasAR_err)
{
  RoadRunner rri;
  EXPECT_THROW(rri.load((SBMLFeaturesDir / "rateOf_noSpeciesInARComp_conc_invalid.xml").string()), rrllvm::LLVMException);
}


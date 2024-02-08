#include <iostream>
#include <string>
#include <vector>
#include "Poco/File.h"
#include "gtest/gtest.h"
#include "C/rrc_api.h"
#include "rrUtils.h"
#include "rrRoadRunner.h"
#include "../test_util.h"
#include <filesystem>

#include "RoadRunnerTest.h"

using namespace std;
using namespace rr;
using namespace rrc;
using namespace Poco;
using std::filesystem::path;

using namespace std;

class RRTestFilesOtherTests : public RoadRunnerTest {
public:
    path rrTestFilesDir = rrTestDir_ / "rrtest_files";
    RRTestFilesOtherTests() = default;
};

TEST_F(RRTestFilesOtherTests, EMPTY_EVENT_ASSIGNMENTS) {
    // Event assignments in L3v2 can be empty
    RRHandle aRR = createRRInstance();
    path TestModelFileName = rrTestFilesDir / "eventassignment_nomath.xml";
    EXPECT_TRUE(std::filesystem::exists(TestModelFileName));

    EXPECT_TRUE(loadSBMLFromFileE(aRR, TestModelFileName.string().c_str(), true));
    RRCDataPtr results = simulate(aRR);
    EXPECT_NE(results, nullptr);

    freeRRCData(results);
    freeRRInstance(aRR);
}

TEST_F(RRTestFilesOtherTests, SAVED_SPECIES_AMOUNT) {
    path TestModelFileName = rrTestFilesDir / "species_conc.xml";
    EXPECT_TRUE(std::filesystem::exists(TestModelFileName));
    RoadRunner rr(TestModelFileName.string(), NULL);
    EXPECT_TRUE(rr.getFloatingSpeciesByIndex(0) == 5.0);
    string sbml = rr.getCurrentSBML();
    RoadRunner rr2(sbml, NULL);
    EXPECT_TRUE(rr2.getFloatingSpeciesByIndex(0) == 5.0);
    sbml = rr2.getCurrentSBML();
    RoadRunner rr3(sbml, NULL);
    EXPECT_TRUE(rr3.getFloatingSpeciesByIndex(0) == 5.0);
}

TEST_F(RRTestFilesOtherTests, SPECIES_UNITS) {
    path TestModelFileName = rrTestFilesDir / "species_hosu1.xml";
    EXPECT_TRUE(std::filesystem::exists(TestModelFileName));
    RoadRunner rr(TestModelFileName.string(), NULL);
    EXPECT_TRUE(rr.getFloatingSpeciesByIndex(0) == 3.0);
    string sbml = rr.getCurrentSBML();
    RoadRunner rr2(sbml, NULL);
    EXPECT_TRUE(rr2.getFloatingSpeciesByIndex(0) == 3.0);
    sbml = rr2.getCurrentSBML();
    RoadRunner rr3(sbml, NULL);
    EXPECT_TRUE(rr3.getFloatingSpeciesByIndex(0) == 3.0);
}

TEST_F(RRTestFilesOtherTests, SPECIES_UNITS2) {
    path TestModelFileName = rrTestFilesDir / "species_hosu2.xml";
    EXPECT_TRUE(std::filesystem::exists(TestModelFileName));
    RoadRunner rr(TestModelFileName.string(), NULL);
    EXPECT_TRUE(rr.getFloatingSpeciesByIndex(0) == 3.0);
    string sbml = rr.getCurrentSBML();
    RoadRunner rr2(sbml, NULL);
    EXPECT_TRUE(rr2.getFloatingSpeciesByIndex(0) == 3.0);
    sbml = rr2.getCurrentSBML();
    RoadRunner rr3(sbml, NULL);
    EXPECT_TRUE(rr3.getFloatingSpeciesByIndex(0) == 3.0);
}

TEST_F(RRTestFilesOtherTests, OUTPUT_FILE_VARIABLE_TIMESTEP) {
    path TestModelFileName = rrTestFilesDir / "output_testmodel.xml";
    path outputFileName =    rrTestFilesDir / "output_testmodel_variable.csv";
    path expectedFileName =  rrTestFilesDir / "expected_testmodel_variable.csv";
    EXPECT_TRUE(std::filesystem::exists(TestModelFileName));

    std::cout << TestModelFileName << std::endl;
    std::cout << outputFileName << std::endl;

    RoadRunner rr(TestModelFileName.string(), nullptr);
    rr.setIntegrator("gillespie");
    rr.getIntegrator()->setValue("seed", 123);

    SimulateOptions opt;
    opt.start = 0;
    opt.duration = 100;
    opt.steps = 0;
    opt.output_file = outputFileName.string();
    rr.setSimulateOptions(opt);

    vector<string> selections;
    selections.push_back("Time");
    selections.push_back("S1");
    rr.setSelections(selections);

    const ls::DoubleMatrix *result = rr.simulate();
    // the result returned should be empty
    EXPECT_TRUE(result->size() == 0);

    // confirm output files are the same
    EXPECT_TRUE(std::filesystem::exists(expectedFileName) && std::filesystem::exists(outputFileName));
    EXPECT_TRUE(filesAreEqual(outputFileName.string(), expectedFileName.string()));
    remove(outputFileName.string().c_str()); // clean up

}

TEST_F(RRTestFilesOtherTests, OUTPUT_FILE_FIXED_TIMESTEP) {
    path TestModelFileName = (rrTestFilesDir / "output_testmodel.xml");
    path outputFileName =    (rrTestFilesDir / "output_testmodel_fixed.csv");
    path expectedFileName =  (rrTestFilesDir / "expected_testmodel_fixed.csv");

    EXPECT_TRUE(std::filesystem::exists(TestModelFileName));
    RoadRunner rr(TestModelFileName.string(), NULL);
    SimulateOptions opt;
    rr.setIntegrator("gillespie");
    ASSERT_STREQ("gillespie", rr.getIntegrator()->getName().c_str());
    rr.getIntegrator()->setValue("seed", 123);
    rr.getIntegrator()->setValue("variable_step_size", false);
    opt.start = 0;
    opt.duration = 50;
    opt.steps = 100;
    opt.output_file = outputFileName.string();
    rr.setSimulateOptions(opt);

    vector<string> selections;
    selections.emplace_back("Time");
    selections.emplace_back("S1");
    rr.setSelections(selections);

    const ls::DoubleMatrix *result = rr.simulate();

    // the result returned should have size 0
    EXPECT_TRUE(result->size() == 0);

    // confirm output files are the same
    EXPECT_TRUE(std::filesystem::exists(expectedFileName) && std::filesystem::exists(outputFileName));
    EXPECT_TRUE(filesAreEqual(expectedFileName.string(), outputFileName.string()));
    remove(outputFileName.string().c_str()); // clean up

}

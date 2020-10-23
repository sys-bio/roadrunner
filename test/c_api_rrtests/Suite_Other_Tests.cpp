#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include "Poco/File.h"
#include "Poco/DOM/ProcessingInstruction.h"
#include "Poco/DOM/DOMParser.h"
#include "Poco/DOM/AutoPtr.h"
#include "Poco/DOM/Document.h"
#include "Poco/DOM/NodeIterator.h"
#include "Poco/DOM/NodeFilter.h"
#include "Poco/DOM/DOMWriter.h"
#include "Poco/DOM/TreeWalker.h"
#include "Poco/SAX/InputSource.h"
#include "Poco/MD5Engine.h"
#include "gtest/gtest.h"
#include "C/rrc_api.h"
#include "rrUtils.h"
#include "rrException.h"
#include "rrLogger.h"
#include "rrRoadRunner.h"
#include "../test_util.h"

using namespace std;
using namespace rr;
using namespace rrc;
using namespace Poco;
using namespace Poco::XML;

//using namespace Poco::XML::NodeFilter;

using namespace std;

extern string gRRTestDir;
extern string gRROutputDir;
extern RRHandle gRR;

string getListOfReactionsText(const string& fName);

TEST(OTHER_TESTS, EMPTY_EVENT_ASSIGNMENTS)
{
    // Event assignments in L3v2 can be empty
    RRHandle aRR = createRRInstance();
    string TestModelFileName = joinPath(gRRTestDir, "rrtest_files/eventassignment_nomath.xml");
    EXPECT_TRUE(fileExists(TestModelFileName));

    EXPECT_TRUE(loadSBMLFromFileE(aRR, TestModelFileName.c_str(), true));
    EXPECT_TRUE(simulate(aRR));


}

TEST(OTHER_TESTS, SAVED_SPECIES_AMOUNT)
{
    string TestModelFileName = joinPath(gRRTestDir, "rrtest_files/species_conc.xml");
    EXPECT_TRUE(fileExists(TestModelFileName));
    RoadRunner rr(TestModelFileName, NULL);
    EXPECT_TRUE(rr.getFloatingSpeciesByIndex(0) == 5.0);
    string sbml = rr.getCurrentSBML();
    RoadRunner rr2(sbml, NULL);
    EXPECT_TRUE(rr2.getFloatingSpeciesByIndex(0) == 5.0);
    sbml = rr2.getCurrentSBML();
    RoadRunner rr3(sbml, NULL);
    EXPECT_TRUE(rr3.getFloatingSpeciesByIndex(0) == 5.0);
}

TEST(OTHER_TESTS, SPECIES_UNITS)
{
    string TestModelFileName = joinPath(gRRTestDir, "rrtest_files/species_hosu1.xml");
    EXPECT_TRUE(fileExists(TestModelFileName));
    RoadRunner rr(TestModelFileName, NULL);
    EXPECT_TRUE(rr.getFloatingSpeciesByIndex(0) == 3.0);
    string sbml = rr.getCurrentSBML();
    RoadRunner rr2(sbml, NULL);
    EXPECT_TRUE(rr2.getFloatingSpeciesByIndex(0) == 3.0);
    sbml = rr2.getCurrentSBML();
    RoadRunner rr3(sbml, NULL);
    EXPECT_TRUE(rr3.getFloatingSpeciesByIndex(0) == 3.0);
}

TEST(OTHER_TESTS, SPECIES_UNITS2)
{
    string TestModelFileName = joinPath(gRRTestDir, "rrtest_files/species_hosu2.xml");
    EXPECT_TRUE(fileExists(TestModelFileName));
    RoadRunner rr(TestModelFileName, NULL);
    EXPECT_TRUE(rr.getFloatingSpeciesByIndex(0) == 3.0);
    string sbml = rr.getCurrentSBML();
    RoadRunner rr2(sbml, NULL);
    EXPECT_TRUE(rr2.getFloatingSpeciesByIndex(0) == 3.0);
    sbml = rr2.getCurrentSBML();
    RoadRunner rr3(sbml, NULL);
    EXPECT_TRUE(rr3.getFloatingSpeciesByIndex(0) == 3.0);
}

TEST(OTHER_TESTS, OUTPUT_FILE_VARIABLE_TIMESTEP) {
    string TestModelFileName = joinPath(gRRTestDir, "rrtest_files/output_testmodel.xml");
    string outputFileName = "output_testmodel_variable.csv";
    string expectedFileName = joinPath(gRRTestDir, "rrtest_files/expected_testmodel_variable.csv");
    EXPECT_TRUE(fileExists(TestModelFileName));

    RoadRunner rr(TestModelFileName, NULL);
    rr.setIntegrator("gillespie");
    rr.getIntegrator()->setValue("seed", 123);
    SimulateOptions opt;
    opt.start = 0;
    opt.duration = 100;
    opt.steps = 0;
    opt.output_file = outputFileName;
    rr.setSimulateOptions(opt);

    vector<string> selections;
    selections.push_back("Time");
    selections.push_back("S1");
    rr.setSelections(selections);

    const ls::DoubleMatrix* result = rr.simulate();
    // the result returned should be empty
    EXPECT_TRUE(result->size() == 0);

    // confirm output files are the same
    EXPECT_TRUE(fileExists(expectedFileName) && fileExists(outputFileName));
    EXPECT_TRUE(filesAreEqual(outputFileName, expectedFileName));

}

TEST(OTHER_TESTS, OUTPUT_FILE_FIXED_TIMESTEP) {
    string TestModelFileName = joinPath(gRRTestDir, "rrtest_files/output_testmodel.xml");
    string outputFileName = "output_testmodel_fixed.csv";
    string expectedFileName = joinPath(gRRTestDir, "rrtest_files/expected_testmodel_fixed.csv");

    EXPECT_TRUE(fileExists(TestModelFileName));
    RoadRunner rr(TestModelFileName, NULL);
    SimulateOptions opt;
    rr.setIntegrator("gillespie");
    rr.getIntegrator()->setValue("seed", 123);
    rr.getIntegrator()->setValue("variable_step_size", false);
    opt.start = 0;
    opt.duration = 50;
    opt.steps = 100;
    opt.output_file = outputFileName;
    rr.setSimulateOptions(opt);

    vector<string> selections;
    selections.push_back("Time");
    selections.push_back("S1");
    rr.setSelections(selections);

    const ls::DoubleMatrix* result = rr.simulate();

    // the result returned should have size 0
    EXPECT_TRUE(result->size() == 0);

    // confirm output files are the same
    EXPECT_TRUE(fileExists(expectedFileName) && fileExists(outputFileName));
    EXPECT_TRUE(filesAreEqual(expectedFileName, outputFileName));
}

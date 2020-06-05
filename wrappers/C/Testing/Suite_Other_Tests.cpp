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
#include "unit_test/UnitTest++.h"
#include "rrc_api.h"
#include "rrUtils.h"
#include "rrException.h"
#include "rrLogger.h"
#include "rrRoadRunner.h"
#include "src/TestUtils.h"

using namespace UnitTest;
using namespace std;
using namespace rr;
using namespace rrc;
using namespace Poco;
using namespace Poco::XML;

//using namespace Poco::XML::NodeFilter;

using namespace std;

extern string   gTempFolder;
extern string   gTestDataFolder;
extern string   gCompiler;


string getListOfReactionsText(const string& fName);

SUITE(OTHER_TESTS)
{
    TEST(EMPTY_EVENT_ASSIGNMENTS)
    {
        // Event assignments in L3v2 can be empty
        RRHandle aRR = createRRInstanceEx(gTempFolder.c_str(), gCompiler.c_str());
        string TestModelFileName = joinPath(gTestDataFolder, "eventassignment_nomath.xml");
        CHECK(fileExists(TestModelFileName));

        CHECK(loadSBMLFromFileE(aRR, TestModelFileName.c_str(), true));
        CHECK(simulate(aRR));


    }

    TEST(SAVED_SPECIES_AMOUNT)
    {
        string TestModelFileName = joinPath(gTestDataFolder, "species_conc.xml");
        CHECK(fileExists(TestModelFileName));
        RoadRunner rr(TestModelFileName, NULL);
        CHECK(rr.getFloatingSpeciesByIndex(0) == 5.0);
        string sbml = rr.getCurrentSBML();
        RoadRunner rr2(sbml, NULL);
        CHECK(rr2.getFloatingSpeciesByIndex(0) == 5.0);
        sbml = rr2.getCurrentSBML();
        RoadRunner rr3(sbml, NULL);
        CHECK(rr3.getFloatingSpeciesByIndex(0) == 5.0);
    }

    TEST(SPECIES_UNITS)
    {
        string TestModelFileName = joinPath(gTestDataFolder, "species_hosu1.xml");
        CHECK(fileExists(TestModelFileName));
        RoadRunner rr(TestModelFileName, NULL);
        CHECK(rr.getFloatingSpeciesByIndex(0) == 3.0);
        string sbml = rr.getCurrentSBML();
        RoadRunner rr2(sbml, NULL);
        CHECK(rr2.getFloatingSpeciesByIndex(0) == 3.0);
        sbml = rr2.getCurrentSBML();
        RoadRunner rr3(sbml, NULL);
        CHECK(rr3.getFloatingSpeciesByIndex(0) == 3.0);
    }

    TEST(SPECIES_UNITS2)
    {
        string TestModelFileName = joinPath(gTestDataFolder, "species_hosu2.xml");
        CHECK(fileExists(TestModelFileName));
        RoadRunner rr(TestModelFileName, NULL);
        CHECK(rr.getFloatingSpeciesByIndex(0) == 3.0);
        string sbml = rr.getCurrentSBML();
        RoadRunner rr2(sbml, NULL);
        CHECK(rr2.getFloatingSpeciesByIndex(0) == 3.0);
        sbml = rr2.getCurrentSBML();
        RoadRunner rr3(sbml, NULL);
        CHECK(rr3.getFloatingSpeciesByIndex(0) == 3.0);
    }

    TEST(OUTPUT_FILE_VARIABLE_TIMESTEP) {
        string TestModelFileName = joinPath(gTestDataFolder, "output_testmodel.xml");
        string outputFileName = joinPath(gTempFolder, "output_testmodel_variable.csv");
        string expectedFileName = joinPath(gTestDataFolder, "expected_testmodel_variable.csv");
        CHECK(fileExists(TestModelFileName));

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
        CHECK(result-> size() == 0);

        // confirm output files are the same
        CHECK(fileExists(expectedFileName) && fileExists(outputFileName));
        CHECK(filesAreEqual(outputFileName, expectedFileName));

    }

    TEST(OUTPUT_FILE_FIXED_TIMESTEP) {
        string TestModelFileName = joinPath(gTestDataFolder, "output_testmodel.xml");
        string outputFileName = joinPath(gTempFolder, "output_testmodel_fixed.csv");
        string expectedFileName = joinPath(gTestDataFolder, "expected_testmodel_fixed.csv");

        CHECK(fileExists(TestModelFileName));
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
        CHECK(result->size() == 0);

        // confirm output files are the same
        CHECK(fileExists(expectedFileName) && fileExists(outputFileName));
        CHECK(filesAreEqual(expectedFileName, outputFileName));
    }
}


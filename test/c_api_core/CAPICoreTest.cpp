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
#include "RoadRunnerTest.h"
#include <filesystem>

using namespace std;
using namespace rr;
using namespace rrc;
using namespace Poco;
using namespace Poco::XML;
using std::filesystem::path;

#include "CAPICoreTest.h"

CAPICoreTest::CAPICoreTest() {
    cAPICoreModelsDir = rrTestModelsDir_ / "CAPICore";
    if (!fs::exists(cAPICoreModelsDir)) {
        throw std::invalid_argument("CAPICoreTest::CAPICoreTest(): " + cAPICoreModelsDir.string());
    }
    gRR = (RRHandle) &r;

    // make a copy each time so the original doesn't get modified inplace
    testModelFilePath = cAPICoreModelsDir / path("Test_1.xml");
};

std::string CAPICoreTest::getListOfReactionsText(const string &fName) {
    ifstream in(testModelFilePath.string().c_str());
    InputSource src(in);
    DOMParser parser;
    AutoPtr<Document> pDoc = parser.parse(&src);
    TreeWalker it(pDoc, Poco::XML::NodeFilter::SHOW_ALL);

    Node *pNode = it.nextNode();
    string result;
    while (pNode) {
        clog << pNode->nodeName() << endl;
        string nodeID = "listOfReactions";
        if (toUpper(pNode->nodeName()) == toUpper(nodeID)) {
            DOMWriter aWriter;
            stringstream xml;
            aWriter.writeNode(xml, pNode);
            result = xml.str();
            break;
        }
        pNode = it.nextNode();
    }

    result.erase(std::remove_if(result.begin(), result.end(), ::isspace), result.end());
    return result;
}


TEST_F(CAPICoreTest, VERSION) {
    RoadRunner aRR;
    string versionInfo = aRR.getExtendedVersionInfo();
}


TEST_F(CAPICoreTest, LOGGING) {
    // save log level
    int logLevel = rr::Logger::getLevel();

    RRHandle aRR = createRRInstance();
    rr::Logger::setLevel(rr::Logger::LOG_INFORMATION);
    enableLoggingToFile();
    logMsg(CL_PRIO_INFORMATION, "A log message before closing the logger");
    disableLoggingToFile();
    logMsg(CL_PRIO_INFORMATION, "This message is not written to the logger");
    freeRRInstance(aRR);

    // restore log level
    rr::Logger::setLevel(logLevel);
}


TEST_F(CAPICoreTest, ReloadingModelModelRecompilation) {
    RRHandle aRR = createRRInstance();
    std::cout << testModelFilePath << std::endl;
    EXPECT_TRUE(std::filesystem::exists(testModelFilePath));

    EXPECT_TRUE(loadSBMLFromFileE(aRR, testModelFilePath.string().c_str(), true));

    //Load the same model again, but do not recompile the model DLL..
    EXPECT_TRUE(loadSBMLFromFileE(aRR, testModelFilePath.string().c_str(), true));
    freeRRInstance(aRR);
}

TEST_F(CAPICoreTest, ReloadingModelNoModelRecompilation) {
    RRHandle aRR = createRRInstance();
    EXPECT_TRUE(std::filesystem::exists(testModelFilePath));

    EXPECT_TRUE(loadSBMLFromFileE(aRR, testModelFilePath.string().c_str(), true));

    //Load the same model again, but do not recompile the model DLL..
    EXPECT_TRUE(loadSBMLFromFileE(aRR, testModelFilePath.string().c_str(), false));
    freeRRInstance(aRR);
}

TEST_F(CAPICoreTest, LoadingModelMultipleInstances) {
    RRHandle aRR1 = createRRInstance();
    RRHandle aRR2 = createRRInstance();

    EXPECT_TRUE(loadSBMLFromFileE(aRR1, testModelFilePath.string().c_str(), true));
    EXPECT_TRUE(loadSBMLFromFileE(aRR2, testModelFilePath.string().c_str(), true));

    //Load the same model again, but do not recompile the model DLL..
    EXPECT_TRUE(loadSBMLFromFileE(aRR1, testModelFilePath.string().c_str(), false));
    EXPECT_TRUE(loadSBMLFromFileE(aRR2, testModelFilePath.string().c_str(), false));

    freeRRInstance(aRR1);
    freeRRInstance(aRR2);
}

TEST_F(CAPICoreTest, ParsingModelXML) {

    string modelXML = getListOfReactionsText(testModelFilePath.string());
    EXPECT_TRUE(modelXML.size() > 0);
}

TEST_F(CAPICoreTest, GeneratingModelHash) {
    string content = getListOfReactionsText(testModelFilePath.string());
    MD5Engine md5;
    md5.update(content);
    string digestString(Poco::DigestEngine::digestToHex(md5.digest()));
    EXPECT_EQ("d996bae1bec8f6efb81c4571aa7fc10d", digestString);
}

TEST_F(CAPICoreTest, LoadModelFromString) {
    string xml = getFileContent((testModelFilePath.string()));
    RRHandle aRR1 = createRRInstance();
    RRHandle aRR2 = createRRInstance();
    EXPECT_TRUE(loadSBML(aRR1, xml.c_str()));
    EXPECT_TRUE(loadSBMLEx(aRR2, xml.c_str(), true));

    //Load the same model again, but do not recompile the model DLL..
    EXPECT_TRUE(loadSBMLEx(aRR1, xml.c_str(), false));
    EXPECT_TRUE(loadSBMLEx(aRR2, xml.c_str(), false));
    freeRRInstance(aRR1);
    freeRRInstance(aRR2);
}

TEST_F(CAPICoreTest, SimulateTimes) {
    string xml = getFileContent((testModelFilePath.string()));
    RRHandle aRR1 = createRRInstance();
    EXPECT_TRUE(loadSBML(aRR1, xml.c_str()));
    double times[4] = {0, 1, 5, 10};
    RRCDataPtr results = simulateTimes(aRR1, times, 4);
    EXPECT_EQ(results->RSize, 4);
    int csize = results->CSize;
    EXPECT_EQ(results->Data[0 * csize], 0);
    EXPECT_EQ(results->Data[1 * csize], 1);
    EXPECT_EQ(results->Data[2 * csize], 5);
    EXPECT_EQ(results->Data[3 * csize], 10);
    freeRRInstance(aRR1);
    freeRRCData(results);
}

#if !defined(__APPLE__)
TEST_F(CAPICoreTest, GetMicroSeconds) {
    // make sure that the time is essentially the same as sleep time in
    // milliseconds.
    int64_t millis = 123;
    int64_t start = rr::getMicroSeconds();
    // sleep for milliseconds
    rr::sleep((int) millis);
    int64_t end = rr::getMicroSeconds();
    int64_t diff = end - start;

    cout << "microseconds, start: " << start << ", end: " << end << ", diff: " << diff
         << ", expected sleep mu s: " << 1000 * millis
         << ", diff between expected and actual mu s: " << diff - (1000 * millis) << endl;

    // timer varies from system to system, but should be semi-close, like say 20%
    // tolerance changed to 50% because
    EXPECT_NEAR(diff, 1000 * millis, (1000 * millis) / 2);

    // make sure its increasing
    EXPECT_TRUE(diff > 0);

    // now make sure that the microseconds is monotinically increasing
    const int len = 5;
    int64_t prev = rr::getMicroSeconds();
    for (int i = 0; i < len; ++i) {
        // the timer on Windows appears to be significantaly less 
        // accurate than standard Unix machines, so have to sleep 
        // for the miminal amount to time to get the alleged 'microseconds'
        // to move. Timer is also not super accurate in virtualbox. 
        rr::sleep(1);
        int64_t curr = rr::getMicroSeconds();
        EXPECT_TRUE(curr > prev);
        prev = curr;
    }
}

#endif


TEST_F(CAPICoreTest, CheckRegisteredIntegrators) {
    auto *rr = (RoadRunner *) createRRInstance();
    auto integratorNames = rr->getRegisteredIntegratorNames();
    ASSERT_EQ(integratorNames, std::vector<std::string>({"cvode", "gillespie", "rk4", "rk45", "euler"}));
    delete rr;
}

TEST_F(CAPICoreTest, CheckRK4WorksFromC) {
    // This code is the contents of createRRInstance in the C api
    char *capiLocation = getRRCAPILocation(); // return type is dynamically allocated.
    string rrInstallFolder(getParentFolder(capiLocation));
    free(capiLocation);
    std::filesystem::path supportCodeDir = std::filesystem::path(rrInstallFolder) /= "rr_support";
    RoadRunner rr("", getTempDir(), supportCodeDir.string());
    // end

    // This code is from the top of checkRRTest
    path rrTestFileDir = rrTestDir_ / "rrtest_files";
    path rrTestFileName = rrTestFileDir / "Test_1.rrtest";

    // This code from check_LoadData
    // need to re-assign it, Load does not clear old data.;

    IniFile iniFile;
    iniFile.Clear();

    EXPECT_TRUE(std::filesystem::exists(rrTestFileName));
    EXPECT_TRUE(iniFile.Load(rrTestFileName.string()));

    IniSection *sbmlsec = iniFile.GetSection("SBML");
    if (!sbmlsec) {
        EXPECT_TRUE(false);
        return;
    }
    sbmlsec->mIsUsed = true;
    string sbml = sbmlsec->GetNonKeysAsString();
    if (sbml.find('<') == string::npos) {
        sbml = (rrTestDir_ / path("rrtest_files") / sbml).string();
        EXPECT_TRUE(std::filesystem::exists(sbml));
    }
    if (!loadSBMLEx(&rr, sbml.c_str(), true)) {
        EXPECT_TRUE(false);
    }

    SimulateOptions opt;
    opt.start = 0;
    opt.duration = 10;
    auto *cvode = rr.simulate(&opt);

    rr.setIntegrator("rk4");
    auto *rk4 = rr.simulate(&opt);
    for (int k = 0; k < cvode->CSize(); k++) {
        EXPECT_NEAR((*cvode)(cvode->RSize() - 1, k), (*rk4)(rk4->RSize() - 1, k), 1e-6);
    }
}






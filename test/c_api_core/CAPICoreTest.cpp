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

    ASSERT_TRUE(loadSBMLFromFileE(aRR, testModelFilePath.string().c_str(), true));

    //Load the same model again, but do not recompile the model DLL..
    ASSERT_TRUE(loadSBMLFromFileE(aRR, testModelFilePath.string().c_str(), true));
    freeRRInstance(aRR);
}

TEST_F(CAPICoreTest, ReloadingModelNoModelRecompilation) {
    RRHandle aRR = createRRInstance();
    EXPECT_TRUE(std::filesystem::exists(testModelFilePath));

    ASSERT_TRUE(loadSBMLFromFileE(aRR, testModelFilePath.string().c_str(), true));

    //Load the same model again, but do not recompile the model DLL..
    ASSERT_TRUE(loadSBMLFromFileE(aRR, testModelFilePath.string().c_str(), false));
    freeRRInstance(aRR);
}

TEST_F(CAPICoreTest, LoadingModelMultipleInstances) {
    RRHandle aRR1 = createRRInstance();
    RRHandle aRR2 = createRRInstance();

    ASSERT_TRUE(loadSBMLFromFileE(aRR1, testModelFilePath.string().c_str(), true));
    ASSERT_TRUE(loadSBMLFromFileE(aRR2, testModelFilePath.string().c_str(), true));

    //Load the same model again, but do not recompile the model DLL..
    ASSERT_TRUE(loadSBMLFromFileE(aRR1, testModelFilePath.string().c_str(), false));
    ASSERT_TRUE(loadSBMLFromFileE(aRR2, testModelFilePath.string().c_str(), false));

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
    ASSERT_TRUE(loadSBML(aRR1, xml.c_str()));
    ASSERT_TRUE(loadSBMLEx(aRR2, xml.c_str(), true));

    //Load the same model again, but do not recompile the model DLL..
    ASSERT_TRUE(loadSBMLEx(aRR1, xml.c_str(), false));
    ASSERT_TRUE(loadSBMLEx(aRR2, xml.c_str(), false));
    freeRRInstance(aRR1);
    freeRRInstance(aRR2);
}

TEST_F(CAPICoreTest, SimulateTimes) {
    string xml = getFileContent((testModelFilePath.string()));
    RRHandle aRR1 = createRRInstance();
    ASSERT_TRUE(loadSBML(aRR1, xml.c_str()));
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
    opt.setStartTime(0);
    opt.setDuration(10);
    auto *cvode = rr.simulate(&opt);

    rr.setIntegrator("rk4");
    auto *rk4 = rr.simulate(&opt);
    for (int k = 0; k < cvode->CSize(); k++) {
        EXPECT_NEAR((*cvode)(cvode->RSize() - 1, k), (*rk4)(rk4->RSize() - 1, k), 1e-6);
    }
}

TEST_F(CAPICoreTest, SetAndGetMDiffStepSize) {
    string xml = getFileContent((testModelFilePath.string()));
    RRHandle rr = createRRInstance();
    ASSERT_TRUE(loadSBML(rr, xml.c_str()));

    double mDiffStepSize;
    EXPECT_TRUE(getDiffStepSize(rr, &mDiffStepSize));
    EXPECT_EQ(mDiffStepSize, 0.02);

    EXPECT_TRUE(setDiffStepSize(rr, 0.06));

    EXPECT_TRUE(getDiffStepSize(rr, &mDiffStepSize));
    EXPECT_EQ(mDiffStepSize, 0.06);

    freeRRInstance(rr);
}

TEST_F(CAPICoreTest, CheckGetCC) {
    
    RRHandle rrH = createRRInstance();
    ASSERT_TRUE(loadSBMLFromFileE(rrH, (cAPICoreModelsDir / path("steadystate.xml")).string().c_str(), true));

    double K = 0.8;
    double ss_val;
    double prev_AP;
    double AP;
    EXPECT_TRUE(steadyState(rrH, &ss_val));
    EXPECT_TRUE(getCC(rrH, "AP", "K", &prev_AP));

    bool found_local_max = false;
    bool success = true;
    for (int i = 1; i < 20; i++) {
        K = K + 0.01;
        EXPECT_TRUE(setValue(rrH, "K", K));
        EXPECT_TRUE(steadyState(rrH, &ss_val));
        EXPECT_TRUE(getCC(rrH, "AP", "K", &AP));
        if (!found_local_max) {
            if (AP < prev_AP) {
                found_local_max = true;
            }
        }
        else {
            if (AP > prev_AP) {
                success = false;
                break;
            }
        }
        prev_AP = AP;
    }

    EXPECT_TRUE(success);

    freeRRInstance(rrH);

}

TEST_F(CAPICoreTest, CheckGetEC) {

    RRHandle rrH = createRRInstance();
    ASSERT_TRUE(loadSBMLFromFileE(rrH, (cAPICoreModelsDir / path("steadystate.xml")).string().c_str(), true));

    RRListPtr actual = getElasticityCoefficientIds(rrH);
    std::vector<std::string> expectedRxns(
        { "_J0", "_J1" });
    std::vector<std::string> expectedIds(
        { "ec(_J0, A)", "ec(_J0, AP)", "ec(_J0, K)", "ec(_J0, Vm1)", "ec(_J0, Km1)", "ec(_J1, A)", "ec(_J1, AP)", "ec(_J1, K)", "ec(_J1, Vm1)", "ec(_J1, Km1)"});

    for (int r = 0; r < actual->Count; r++) {
        RRListItem* element = actual->Items[r];
        RRList* sublist = element->data.lValue;
        char* id = sublist->Items[0]->data.sValue;
        EXPECT_STREQ(id, expectedRxns[r].c_str());
        RRList* subsublist = sublist->Items[1]->data.lValue;
        for (int sr=0; sr < subsublist->Count; sr++) {
            char* id = subsublist->Items[sr]->data.sValue;
            EXPECT_STREQ(id, expectedIds[r*subsublist->Count + sr].c_str());
        }
    }

    freeRRList(actual);
    freeRRInstance(rrH);
}

TEST_F(CAPICoreTest, CheckGetUEC) {

    RRHandle rrH = createRRInstance();
    ASSERT_TRUE(loadSBMLFromFileE(rrH, (cAPICoreModelsDir / path("steadystate.xml")).string().c_str(), true));

    RRListPtr actual = getUnscaledElasticityCoefficientIds(rrH);
    std::vector<std::string> expectedRxns(
        { "_J0", "_J1" });
    std::vector<std::string> expectedIds(
        { "uec(_J0, A)", "uec(_J0, AP)", "uec(_J0, K)", "uec(_J0, Vm1)", "uec(_J0, Km1)", "uec(_J1, A)", "uec(_J1, AP)", "uec(_J1, K)", "uec(_J1, Vm1)", "uec(_J1, Km1)"});

    for (int r = 0; r < actual->Count; r++) {
        RRListItem* element = actual->Items[r];
        RRList* sublist = element->data.lValue;
        char* id = sublist->Items[0]->data.sValue;
        EXPECT_STREQ(id, expectedRxns[r].c_str());
        RRList* subsublist = sublist->Items[1]->data.lValue;
        for (int sr=0; sr < subsublist->Count; sr++) {
            char* id = subsublist->Items[sr]->data.sValue;
            EXPECT_STREQ(id, expectedIds[r*subsublist->Count + sr].c_str());
        }
    }

    freeRRList(actual);
    freeRRInstance(rrH);
}

TEST_F(CAPICoreTest, CheckGetFluxControlCoefficientIds) {

    RRHandle rrH = createRRInstance();
    ASSERT_TRUE(loadSBMLFromFileE(rrH, (cAPICoreModelsDir / path("steadystate.xml")).string().c_str(), true));

    RRListPtr actual = getFluxControlCoefficientIds(rrH);
    std::vector<std::string> expectedRxns(
        { "_J0", "_J1" });
    std::vector<std::string> expectedIds(
        { "cc(_J0, K)", "cc(_J0, Vm1)", "cc(_J0, Km1)", "cc(_J1, K)", "cc(_J1, Vm1)", "cc(_J1, Km1)"});

    for (int r = 0; r < actual->Count; r++) {
        RRListItem* element = actual->Items[r];
        RRList* sublist = element->data.lValue;
        char* id = sublist->Items[0]->data.sValue;
        EXPECT_STREQ(id, expectedRxns[r].c_str());
        RRList* subsublist = sublist->Items[1]->data.lValue;
        for (int sr=0; sr < subsublist->Count; sr++) {
            char* id = subsublist->Items[sr]->data.sValue;
            EXPECT_STREQ(id, expectedIds[r*subsublist->Count + sr].c_str());
        }
    }

    freeRRList(actual);
    freeRRInstance(rrH);
}

TEST_F(CAPICoreTest, CheckGetUnscaledFluxControlCoefficientIds) {

    RRHandle rrH = createRRInstance();
    ASSERT_TRUE(loadSBMLFromFileE(rrH, (cAPICoreModelsDir / path("steadystate.xml")).string().c_str(), true));

    RRListPtr actual = getUnscaledFluxControlCoefficientIds(rrH);
    std::vector<std::string> expectedRxns(
        { "_J0", "_J1" });
    std::vector<std::string> expectedIds(
        { "ucc(_J0, K)", "ucc(_J0, Vm1)", "ucc(_J0, Km1)", "ucc(_J1, K)", "ucc(_J1, Vm1)", "ucc(_J1, Km1)"});

    for (int r = 0; r < actual->Count; r++) {
        RRListItem* element = actual->Items[r];
        RRList* sublist = element->data.lValue;
        char* id = sublist->Items[0]->data.sValue;
        EXPECT_STREQ(id, expectedRxns[r].c_str());
        RRList* subsublist = sublist->Items[1]->data.lValue;
        for (int sr=0; sr < subsublist->Count; sr++) {
            char* id = subsublist->Items[sr]->data.sValue;
            EXPECT_STREQ(id, expectedIds[r*subsublist->Count + sr].c_str());
        }
    }

    freeRRList(actual);
    freeRRInstance(rrH);
}

TEST_F(CAPICoreTest, CheckGetConcentrationControlCoefficientIds) {

    RRHandle rrH = createRRInstance();
    ASSERT_TRUE(loadSBMLFromFileE(rrH, (cAPICoreModelsDir / path("steadystate.xml")).string().c_str(), true));

    RRListPtr actual = getConcentrationControlCoefficientIds(rrH);
    std::vector<std::string> expectedSpecies(
        { "A", "AP" });
    std::vector<std::string> expectedIds(
        { "cc(A, K)", "cc(A, Vm1)", "cc(A, Km1)", "cc(AP, K)", "cc(AP, Vm1)", "cc(AP, Km1)"});

    for (int r = 0; r < actual->Count; r++) {
        RRListItem* element = actual->Items[r];
        RRList* sublist = element->data.lValue;
        char* id = sublist->Items[0]->data.sValue;
        EXPECT_STREQ(id, expectedSpecies[r].c_str());
        RRList* subsublist = sublist->Items[1]->data.lValue;
        for (int sr=0; sr < subsublist->Count; sr++) {
            char* id = subsublist->Items[sr]->data.sValue;
            EXPECT_STREQ(id, expectedIds[r*subsublist->Count + sr].c_str());
        }
    }

    freeRRList(actual);
    freeRRInstance(rrH);
}

TEST_F(CAPICoreTest, CheckGetUnscaledConcentrationControlCoefficientIds) {

    RRHandle rrH = createRRInstance();
    ASSERT_TRUE(loadSBMLFromFileE(rrH, (cAPICoreModelsDir / path("steadystate.xml")).string().c_str(), true));

    RRListPtr actual = getUnscaledConcentrationControlCoefficientIds(rrH);
    std::vector<std::string> expectedSpecies(
        { "A", "AP" });
    std::vector<std::string> expectedIds(
        { "ucc(A, K)", "ucc(A, Vm1)", "ucc(A, Km1)", "ucc(AP, K)", "ucc(AP, Vm1)", "ucc(AP, Km1)"});

    for (int r = 0; r < actual->Count; r++) {
        RRListItem* element = actual->Items[r];
        RRList* sublist = element->data.lValue;
        char* id = sublist->Items[0]->data.sValue;
        EXPECT_STREQ(id, expectedSpecies[r].c_str());
        RRList* subsublist = sublist->Items[1]->data.lValue;
        for (int sr=0; sr < subsublist->Count; sr++) {
            char* id = subsublist->Items[sr]->data.sValue;
            EXPECT_STREQ(id, expectedIds[r*subsublist->Count + sr].c_str());
        }
    }

    freeRRList(actual);
    freeRRInstance(rrH);
}

TEST_F(CAPICoreTest, CheckSetTimeCourseSelectionListEx) {

    RRHandle rrH = createRRInstance();
    ASSERT_TRUE(loadSBMLFromFileE(rrH, (cAPICoreModelsDir / path("steadystate.xml")).string().c_str(), true));

    char time[] = "time";
    char uec[] = "uec(_J0, A)";
    char ec[] = "ec(_J1, K)";
    char* sel_list[] = { time, uec, ec };

    setTimeCourseSelectionListEx(rrH, 3, (const char**)sel_list);
    RRStringArrayPtr sel_rt = getTimeCourseSelectionList(rrH);
    ASSERT_EQ(sel_rt->Count, 3);
    for (int i = 0; i < 3; i++) {
        EXPECT_STREQ(sel_list[i], sel_rt->String[i]);
    }
    RRCDataPtr res = simulate(rrH);
    char** rescols = res->ColumnHeaders;
    for (int i = 0; i < 3; i++) {
        EXPECT_STREQ(sel_list[i], rescols[i]);
    }

    freeStringArray(sel_rt);
    freeRRInstance(rrH);
    freeRRCData(res);
}

TEST_F(CAPICoreTest, CheckSetSteadyStateSelectionListEx) {

    RRHandle rrH = createRRInstance();
    ASSERT_TRUE(loadSBMLFromFileE(rrH, (cAPICoreModelsDir / path("steadystate.xml")).string().c_str(), true));

    char time[] = "A";
    char uec[] = "uec(_J0, A)";
    char ec[] = "ec(_J1, K)";
    char* sel_list[] = { time, uec, ec };

    setSteadyStateSelectionListEx(rrH, 3, (const char**)sel_list);
    RRStringArrayPtr sel_rt = getSteadyStateSelectionList(rrH);
    ASSERT_EQ(sel_rt->Count, 3);
    for (int i = 0; i < 3; i++) {
        EXPECT_STREQ(sel_list[i], sel_rt->String[i]);
    }
    RRVectorPtr res = computeSteadyStateValues(rrH);
    ASSERT_EQ(res->Count, 3);
    double* values = res->Data;
    for (int i = 0; i < 3; i++) {
        double modelval;
        bool check = getValue(rrH, sel_list[i], &modelval);
        ASSERT_TRUE(check);
        EXPECT_EQ(values[i], modelval);
    }

    freeStringArray(sel_rt);
    freeVector(res);
    freeRRInstance(rrH);
}

TEST_F(CAPICoreTest, CheckGetStoichiometryMatrix) {

    RRHandle rrH = createRRInstance();
    ASSERT_TRUE(loadSBMLFromFileE(rrH, (cAPICoreModelsDir / path("steadystate.xml")).string().c_str(), true));

    RRDoubleMatrixPtr stoichs = getStoichiometryMatrix(rrH);
    ASSERT_EQ(stoichs->CSize, 2);
    ASSERT_EQ(stoichs->RSize, 2);
    EXPECT_EQ(stoichs->Data[0], -1.0);
    EXPECT_EQ(stoichs->Data[1], 1.0);
    EXPECT_EQ(stoichs->Data[2], 1.0);
    EXPECT_EQ(stoichs->Data[3], -1.0);

    freeRRInstance(rrH);
    freeMatrix(stoichs);
}

TEST_F(CAPICoreTest, CheckRatesOfChangeFunctions) {

    RRHandle rrH = createRRInstance();
    ASSERT_TRUE(loadSBMLFromFileE(rrH, (cAPICoreModelsDir / path("steadystate.xml")).string().c_str(), true));
    bool ret = setComputeAndAssignConservationLaws(rrH, true);

    RRVectorPtr roc = getRatesOfChange(rrH);
    ASSERT_EQ(roc->Count, 1);
    EXPECT_NEAR(roc->Data[0], -0.496, 0.001);
    freeVector(roc);

    RRStringArrayPtr roc_ids = getRatesOfChangeIds(rrH);
    ASSERT_EQ(roc_ids->Count, 1);
    EXPECT_STREQ(roc_ids->String[0], "A'");
    freeStringArray(roc_ids);

    roc_ids = getIndependentFloatingSpeciesIds(rrH);
    ASSERT_EQ(roc_ids->Count, 1);
    EXPECT_STREQ(roc_ids->String[0], "A");
    freeStringArray(roc_ids);

    roc_ids = getDependentFloatingSpeciesIds(rrH);
    ASSERT_EQ(roc_ids->Count, 1);
    EXPECT_STREQ(roc_ids->String[0], "AP");
    freeStringArray(roc_ids);

    freeRRInstance(rrH);
}

TEST_F(CAPICoreTest, CheckGetIndependentAndDependentFloatingSpecies_conserved) {
    RRHandle rrH = createRRInstance();
    EXPECT_TRUE(loadSBMLFromFileE(rrH, (cAPICoreModelsDir / path("steadystate.xml")).string().c_str(), true));
    setComputeAndAssignConservationLaws(rrH, true);

    setValue(rrH, "default_compartment", 10.0);
    setValue(rrH, "AP", 50.0);

    RRVectorPtr indep_conc = getIndependentFloatingSpeciesConcentrations(rrH);
    ASSERT_EQ(indep_conc->Count, 1);
    EXPECT_EQ(indep_conc->Data[0], 1);

    RRVectorPtr dep_conc = getDependentFloatingSpeciesConcentrations(rrH);
    ASSERT_EQ(dep_conc->Count, 1);
    EXPECT_EQ(dep_conc->Data[0], 5);

    RRVectorPtr indep_amount = getIndependentFloatingSpeciesAmounts(rrH);
    ASSERT_EQ(indep_amount->Count, 1);
    EXPECT_EQ(indep_amount->Data[0], 10);

    RRVectorPtr dep_amount = getDependentFloatingSpeciesAmounts(rrH);
    ASSERT_EQ(dep_amount->Count, 1);
    EXPECT_EQ(dep_amount->Data[0], 50);

    freeVector(indep_conc);
    freeVector(dep_conc);
    freeVector(indep_amount);
    freeVector(dep_amount);
    freeRRInstance(rrH);
}

TEST_F(CAPICoreTest, CheckGetIndependentAndDependentFloatingSpecies_nonconserved) {
    RRHandle rrH = createRRInstance();
    EXPECT_TRUE(loadSBMLFromFileE(rrH, (cAPICoreModelsDir / path("steadystate.xml")).string().c_str(), true));

    setValue(rrH, "default_compartment", 10.0);
    setValue(rrH, "AP", 50.0);

    setValue(rrH, "default_compartment", 10.0);
    setValue(rrH, "AP", 50.0);

    RRVectorPtr indep_conc = getIndependentFloatingSpeciesConcentrations(rrH);
    ASSERT_EQ(indep_conc->Count, 1);
    EXPECT_EQ(indep_conc->Data[0], 1);

    RRVectorPtr dep_conc = getDependentFloatingSpeciesConcentrations(rrH);
    ASSERT_EQ(dep_conc->Count, 1);
    EXPECT_EQ(dep_conc->Data[0], 5);

    RRVectorPtr indep_amount = getIndependentFloatingSpeciesAmounts(rrH);
    ASSERT_EQ(indep_amount->Count, 1);
    EXPECT_EQ(indep_amount->Data[0], 10);

    RRVectorPtr dep_amount = getDependentFloatingSpeciesAmounts(rrH);
    ASSERT_EQ(dep_amount->Count, 1);
    EXPECT_EQ(dep_amount->Data[0], 50);

    freeVector(indep_conc);
    freeVector(dep_conc);
    freeVector(indep_amount);
    freeVector(dep_amount);
    freeRRInstance(rrH);
}

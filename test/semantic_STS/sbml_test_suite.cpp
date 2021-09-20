#include "gtest/gtest.h"
//#include "rrLogger.h"
#include "rrRoadRunner.h"
#include "rrException.h"
#include "rrUtils.h"
#include "rrTestSuiteModelSimulation.h"
#include "llvm/LLVMException.h"
#include <filesystem>
#include "RoadRunnerTest.h"
#include <execution>

using namespace testing;
using namespace rr;
using namespace std;
using std::filesystem::path;


//extern path gRRTestDir;


class SbmlTestSuite : public RoadRunnerTest {
public:

    SbmlTestSuite() = default;

    bool RunTest(int caseNumber) {
        std::cout << "running case: " << caseNumber << std::endl;
        //Run the first and last version of the file.
        string modelFileName, settingsFileName, descriptionFileName;
        vector<string> lvs;
        lvs.push_back("l1v2");
        lvs.push_back("l2v1");
        lvs.push_back("l2v2");
        lvs.push_back("l2v3");
        lvs.push_back("l2v4");
        lvs.push_back("l2v5");
        lvs.push_back("l3v1");
        lvs.push_back("l3v2");
        string testsuitedir = (rrTestSbmlTestSuiteDir_ / path("semantic")).string();
        string modelFilePath(testsuitedir);
        string first = "";
        string last = "";
        for (size_t n = 0; n < lvs.size(); n++) {
            string lv = lvs[n];
            modelFilePath = testsuitedir; //Reset, since the subdir is added.
            createTestSuiteFileNameParts(caseNumber, "-sbml-" + lv + ".xml", modelFilePath, modelFileName,
                                         settingsFileName, descriptionFileName);
            ifstream ftest((path(modelFilePath) / path(modelFileName)).string());
            if (ftest.good()) {
                if (first.empty()) {
                    first = lv;
                } else {
                    last = lv;
                }
            }
        }
        bool ret = true;
        if (hasUnimplementedTags(modelFilePath + "/" + descriptionFileName)) {
            if (!first.empty()) {
                ret = CheckLoad(first, caseNumber);
            } else {
                rrLog(Logger::LOG_ERROR) << "No models found for test case" << caseNumber << endl;
                ret = false;
            }
            if (!last.empty()) {
                ret = CheckLoad(last, caseNumber) && ret;
            }
        } else {
            if (!first.empty()) {
                ret = RunTest(first, caseNumber);
                if (!ret && isSemiStochasticTest((path(modelFilePath) / path(descriptionFileName)).string())) {
                    //semistochastic tests fail once in a great while, but very very rarely twice in a row.
                    rrLog(Logger::LOG_WARNING) << "Test " << caseNumber
                                               << " failed, but we expect it to fail every so often.  Trying again...";
                    ret = RunTest(first, caseNumber);
                }
            } else {
                rrLog(Logger::LOG_ERROR) << "No models found for test case" << caseNumber << endl;
                ret = false;
            }
            if (!last.empty()) {
                ret = RunTest(last, caseNumber) && ret;
            }
        }
        return ret;
    }


    bool RunTest(const string &version, int caseNumber) {
        //cerr << "Running Test:\t" << caseNumber << "\t" << version;

        RoadRunner rr;
        TestSuiteModelSimulation simulation;
        try {
            LoadAndSimulate(version, caseNumber, rr, simulation);

            //Write result
            if (!simulation.SaveResult()) {
                //Failed to save data
                rrLog(Logger::LOG_ERROR) << "Failed to save result";
                throw ("Failed running simulation: Failed to save result");
            }

            if (!simulation.LoadReferenceData()) {
                rrLog(Logger::LOG_ERROR) << "Failed loading reference data";
                throw ("Failed loading reference data");
            }

            simulation.CreateErrorData();
            bool result = simulation.Pass();
            simulation.SaveAllData();

            //simulation.SaveModelAsXML(dataOutputFolder);

            //cerr<<"\t"<< (result == true ? "PASS" : "FAIL")<<endl;
            return result;
        }
        catch (Exception &ex) {
            string error = ex.what();
            //cerr << "\tFAIL" << endl;
            //cerr<<"Case "<<caseNumber<<": Exception: "<<error<<endl;
            return false;
        }

    }

    bool CheckLoad(const string &version, int caseNumber) {
        //cerr << "Checking Test Loading:\t" << caseNumber << "\t" << version;

        RoadRunner rr;
        TestSuiteModelSimulation simulation;

        try {
            LoadAndSimulate(version, caseNumber, rr, simulation);

            //If we've gotten this far, rejoice!  roadrunner didn't crash, which is good enough.
            //cerr << "\tPASS" << endl;
            return true;
        }
        catch (rrllvm::LLVMException &ex) {
            //Sometimes, rr itself knows when it can't load a model.  This is also fine.
            return true;
        }
        catch (Exception &ex) {
            string error = ex.what();
            //cerr << "\tFAIL" << endl;
            //cerr << "Case " << caseNumber << ": Exception: " << error << endl;
            return false;
        }

    }

    void LoadAndSimulate(const string &version, int caseNumber, RoadRunner &rr, TestSuiteModelSimulation &simulation) {

        string dummy;
        string logFileName;

        rr.getIntegrator()->setValue("stiff", false);

        //Create log file name, e.g. 00001.log
        createTestSuiteFileNameParts(caseNumber, "_" + version + ".log", dummy, logFileName, dummy, dummy);

        //rr.reset();
        simulation.UseEngine(&rr);

        //Setup filenames and paths...
        string testsuitedir = (rrTestSbmlTestSuiteDir_ / path("semantic")).string();
        string modelFilePath(testsuitedir);
        string modelFileName;
        string settingsFileName;
        string descriptionFileName;
        createTestSuiteFileNameParts(caseNumber, "-sbml-" + version + ".xml", modelFilePath, modelFileName,
                                     settingsFileName, descriptionFileName);

        //The following will load and compile and simulate the sbml model in the file
        simulation.SetCaseNumber(caseNumber);
        simulation.SetModelFilePath(modelFilePath);
        simulation.SetModelFileName(modelFileName);
        simulation.ReCompileIfDllExists(true);
        simulation.CopyFilesToOutputFolder();

        rr.setConservedMoietyAnalysis(false);

        if (!simulation.LoadSBMLFromFile()) {
            rrLog(Logger::LOG_ERROR) << "Failed loading SBML model";
            throw ("Failed loading SBML model");
        }
        //Then read settings file if it exists..
        string settingsOveride("");
        if (!simulation.LoadSettings(settingsOveride)) {
            rrLog(Logger::LOG_ERROR) << "Failed loading SBML model settings";
            throw ("Failed loading SBML model settings");
        }


        if (!isFBCTest(modelFilePath + "/" + descriptionFileName)) {
            //Only try simulating non-FBC tests.
            if (!simulation.Simulate()) {
                rrLog(Logger::LOG_ERROR) << "Failed running simulation";
                throw ("Failed running simulation");
            }
        }

    }

};

TEST_F(SbmlTestSuite, RunTests) {
    // list of all disabled tests
    std::vector<int> exclusion({
                                       533, 534, 536, 537, 538, 569, 570, 575, 663, 664, 762, 777, 844, 983,
                                       993, 1044, 1108, 1165, 1167, 1168, 1198, 1208, 1234, 1235, 1236, 1238,
                                       1239, 1241, 1242, 1244, 1248, 1249, 1250, 1251, 1252, 1253, 1254, 1255,
                                       1256, 1257, 1258, 1259, 1260, 1261, 1262, 1263, 1264, 1265, 1266, 1267,
                                       1268, 1269, 1270, 1282, 1283, 1284, 1285, 1286, 1288, 1289, 1290, 1291,
                                       1292, 1293, 1294, 1295, 1296, 1297, 1298, 1299, 1321, 1322, 1350, 1359,
                                       1368, 1377, 1386, 1398, 1400, 1401, 1402, 1403, 1405, 1406, 1408, 1409,
                                       1416, 1419, 1444, 1445, 1446, 1447, 1448, 1449, 1450, 1451, 1452, 1453,
                                       1454, 1455, 1456, 1457, 1458, 1459, 1460, 1461, 1462, 1463, 1464, 1465,
                                       1471, 1472, 1473, 1475, 1476, 1477, 1478, 1479, 1482, 1483, 1484, 1488,
                                       1498, 1499, 1500, 1501, 1502, 1503, 1504, 1505, 1506, 1507, 1508, 1509,
                                       1510, 1511, 1512, 1513, 1514, 1517, 1525, 1526, 1527, 1528, 1529, 1540,
                                       1541, 1542, 1543, 1552, 1553, 1554, 1556, 1567, 1571, 1574, 1575, 1576,
                                       1577, 1579, 1589, 1626, 1627, 1628, 1629, 1630, 1635, 1636, 1657, 1727,
                                       1728, 1729, 1736, 1737, 1738, 1778, 1780});
    int lastTest = 1809;
    std::vector<int> cases;

    // exclude disabled tests
    for (int i = 1; i <= lastTest; i++) {
        if (std::find(exclusion.begin(), exclusion.end(), i) == exclusion.end()) {
            cases.push_back(i);
        }
    }
    // execute in parallel
    std::for_each(std::execution::par, cases.begin(), cases.end(), [&](int i) {
        EXPECT_TRUE(RunTest(i));
    });
}


TEST_F(SbmlTestSuite, DISABLED_test_single) {
    // Use when need to run one test.
    EXPECT_TRUE(RunTest(28));
}

 TEST_F(SbmlTestSuite, DISABLED_t533){
    EXPECT_TRUE(RunTest(533));
}

 TEST_F(SbmlTestSuite, DISABLED_t534){
    EXPECT_TRUE(RunTest(534));
}

 TEST_F(SbmlTestSuite, DISABLED_t536){
    EXPECT_TRUE(RunTest(536));
}

 TEST_F(SbmlTestSuite, DISABLED_t537){
    EXPECT_TRUE(RunTest(537));
}

 TEST_F(SbmlTestSuite, DISABLED_t538){
    EXPECT_TRUE(RunTest(538));
}

 TEST_F(SbmlTestSuite, DISABLED_t569){
    EXPECT_TRUE(RunTest(569));
}

 TEST_F(SbmlTestSuite, DISABLED_t570){
    EXPECT_TRUE(RunTest(570));
}

 TEST_F(SbmlTestSuite, DISABLED_t575){
    EXPECT_TRUE(RunTest(575));
}

 TEST_F(SbmlTestSuite, DISABLED_t663){
    EXPECT_TRUE(RunTest(663));
}

 TEST_F(SbmlTestSuite, DISABLED_t664){
    EXPECT_TRUE(RunTest(664));
}

 TEST_F(SbmlTestSuite, DISABLED_t762){
    EXPECT_TRUE(RunTest(762));
}

 TEST_F(SbmlTestSuite, DISABLED_t777){
    EXPECT_TRUE(RunTest(777));
}

 TEST_F(SbmlTestSuite, DISABLED_t844){
    EXPECT_TRUE(RunTest(844));
}

 TEST_F(SbmlTestSuite, DISABLED_t983){
    EXPECT_TRUE(RunTest(983));
}

 TEST_F(SbmlTestSuite, DISABLED_t993){
    EXPECT_TRUE(RunTest(993));
}

 TEST_F(SbmlTestSuite, DISABLED_t1044){
    EXPECT_TRUE(RunTest(1044));
}

 TEST_F(SbmlTestSuite, DISABLED_t1108){
    EXPECT_TRUE(RunTest(1108));
}

 TEST_F(SbmlTestSuite, DISABLED_t1165){
    EXPECT_TRUE(RunTest(1165));
}

 TEST_F(SbmlTestSuite, DISABLED_t1167){
    EXPECT_TRUE(RunTest(1167));
}

 TEST_F(SbmlTestSuite, DISABLED_t1168){
    EXPECT_TRUE(RunTest(1168));
}

 TEST_F(SbmlTestSuite, DISABLED_t1198){
    EXPECT_TRUE(RunTest(1198));
}

 TEST_F(SbmlTestSuite, DISABLED_t1208){
    EXPECT_TRUE(RunTest(1208));
}

 TEST_F(SbmlTestSuite, DISABLED_t1234){
    EXPECT_TRUE(RunTest(1234));
}

 TEST_F(SbmlTestSuite, DISABLED_t1235){
    EXPECT_TRUE(RunTest(1235));
}

 TEST_F(SbmlTestSuite, DISABLED_t1236){
    EXPECT_TRUE(RunTest(1236));
}

 TEST_F(SbmlTestSuite, DISABLED_t1238){
    EXPECT_TRUE(RunTest(1238));
}

 TEST_F(SbmlTestSuite, DISABLED_t1239){
    EXPECT_TRUE(RunTest(1239));
}

 TEST_F(SbmlTestSuite, DISABLED_t1241){
    EXPECT_TRUE(RunTest(1241));
}

 TEST_F(SbmlTestSuite, DISABLED_t1242){
    EXPECT_TRUE(RunTest(1242));
}

 TEST_F(SbmlTestSuite, DISABLED_t1244){
    EXPECT_TRUE(RunTest(1244));
}

 TEST_F(SbmlTestSuite, DISABLED_t1248){
    EXPECT_TRUE(RunTest(1248));
}

 TEST_F(SbmlTestSuite, DISABLED_t1249){
    EXPECT_TRUE(RunTest(1249));
}

 TEST_F(SbmlTestSuite, DISABLED_t1250){
    EXPECT_TRUE(RunTest(1250));
}

 TEST_F(SbmlTestSuite, DISABLED_t1251){
    EXPECT_TRUE(RunTest(1251));
}

 TEST_F(SbmlTestSuite, DISABLED_t1252){
    EXPECT_TRUE(RunTest(1252));
}

 TEST_F(SbmlTestSuite, DISABLED_t1253){
    EXPECT_TRUE(RunTest(1253));
}

 TEST_F(SbmlTestSuite, DISABLED_t1254){
    EXPECT_TRUE(RunTest(1254));
}

 TEST_F(SbmlTestSuite, DISABLED_t1255){
    EXPECT_TRUE(RunTest(1255));
}

 TEST_F(SbmlTestSuite, DISABLED_t1256){
    EXPECT_TRUE(RunTest(1256));
}

 TEST_F(SbmlTestSuite, DISABLED_t1257){
    EXPECT_TRUE(RunTest(1257));
}

 TEST_F(SbmlTestSuite, DISABLED_t1258){
    EXPECT_TRUE(RunTest(1258));
}

 TEST_F(SbmlTestSuite, DISABLED_t1259){
    EXPECT_TRUE(RunTest(1259));
}

 TEST_F(SbmlTestSuite, DISABLED_t1260){
    EXPECT_TRUE(RunTest(1260));
}

 TEST_F(SbmlTestSuite, DISABLED_t1261){
    EXPECT_TRUE(RunTest(1261));
}

 TEST_F(SbmlTestSuite, DISABLED_t1262){
    EXPECT_TRUE(RunTest(1262));
}

 TEST_F(SbmlTestSuite, DISABLED_t1263){
    EXPECT_TRUE(RunTest(1263));
}

 TEST_F(SbmlTestSuite, DISABLED_t1264){
    EXPECT_TRUE(RunTest(1264));
}

 TEST_F(SbmlTestSuite, DISABLED_t1265){
    EXPECT_TRUE(RunTest(1265));
}

 TEST_F(SbmlTestSuite, DISABLED_t1266){
    EXPECT_TRUE(RunTest(1266));
}

 TEST_F(SbmlTestSuite, DISABLED_t1267){
    EXPECT_TRUE(RunTest(1267));
}

 TEST_F(SbmlTestSuite, DISABLED_t1268){
    EXPECT_TRUE(RunTest(1268));
}

 TEST_F(SbmlTestSuite, DISABLED_t1269){
    EXPECT_TRUE(RunTest(1269));
}

 TEST_F(SbmlTestSuite, DISABLED_t1270){
    EXPECT_TRUE(RunTest(1270));
}

 TEST_F(SbmlTestSuite, DISABLED_t1282){
    EXPECT_TRUE(RunTest(1282));
}

 TEST_F(SbmlTestSuite, DISABLED_t1283){
    EXPECT_TRUE(RunTest(1283));
}

 TEST_F(SbmlTestSuite, DISABLED_t1284){
    EXPECT_TRUE(RunTest(1284));
}

 TEST_F(SbmlTestSuite, DISABLED_t1285){
    EXPECT_TRUE(RunTest(1285));
}

 TEST_F(SbmlTestSuite, DISABLED_t1286){
    EXPECT_TRUE(RunTest(1286));
}

 TEST_F(SbmlTestSuite, DISABLED_t1288){
    EXPECT_TRUE(RunTest(1288));
}

 TEST_F(SbmlTestSuite, DISABLED_t1289){
    EXPECT_TRUE(RunTest(1289));
}

 TEST_F(SbmlTestSuite, DISABLED_t1290){
    EXPECT_TRUE(RunTest(1290));
}

 TEST_F(SbmlTestSuite, DISABLED_t1291){
    EXPECT_TRUE(RunTest(1291));
}

 TEST_F(SbmlTestSuite, DISABLED_t1292){
    EXPECT_TRUE(RunTest(1292));
}

 TEST_F(SbmlTestSuite, DISABLED_t1293){
    EXPECT_TRUE(RunTest(1293));
}

 TEST_F(SbmlTestSuite, DISABLED_t1294){
    EXPECT_TRUE(RunTest(1294));
}

 TEST_F(SbmlTestSuite, DISABLED_t1295){
    EXPECT_TRUE(RunTest(1295));
}

 TEST_F(SbmlTestSuite, DISABLED_t1296){
    EXPECT_TRUE(RunTest(1296));
}

 TEST_F(SbmlTestSuite, DISABLED_t1297){
    EXPECT_TRUE(RunTest(1297));
}

 TEST_F(SbmlTestSuite, DISABLED_t1298){
    EXPECT_TRUE(RunTest(1298));
}

 TEST_F(SbmlTestSuite, DISABLED_t1299){
    EXPECT_TRUE(RunTest(1299));
}

 TEST_F(SbmlTestSuite, DISABLED_t1321){
    EXPECT_TRUE(RunTest(1321));
}

 TEST_F(SbmlTestSuite, DISABLED_t1322){
    EXPECT_TRUE(RunTest(1322));
}

 TEST_F(SbmlTestSuite, DISABLED_t1350){
    EXPECT_TRUE(RunTest(1350));
}

 TEST_F(SbmlTestSuite, DISABLED_t1359){
    EXPECT_TRUE(RunTest(1359));
}

 TEST_F(SbmlTestSuite, DISABLED_t1368){
    EXPECT_TRUE(RunTest(1368));
}

 TEST_F(SbmlTestSuite, DISABLED_t1377){
    EXPECT_TRUE(RunTest(1377));
}

 TEST_F(SbmlTestSuite, DISABLED_t1386){
    EXPECT_TRUE(RunTest(1386));
}

 TEST_F(SbmlTestSuite, DISABLED_t1398){
    EXPECT_TRUE(RunTest(1398));
}

 TEST_F(SbmlTestSuite, DISABLED_t1400){
    EXPECT_TRUE(RunTest(1400));
}

 TEST_F(SbmlTestSuite, DISABLED_t1401){
    EXPECT_TRUE(RunTest(1401));
}

 TEST_F(SbmlTestSuite, DISABLED_t1402){
    EXPECT_TRUE(RunTest(1402));
}

 TEST_F(SbmlTestSuite, DISABLED_t1403){
    EXPECT_TRUE(RunTest(1403));
}

 TEST_F(SbmlTestSuite, DISABLED_t1405){
    EXPECT_TRUE(RunTest(1405));
}

 TEST_F(SbmlTestSuite, DISABLED_t1406){
    EXPECT_TRUE(RunTest(1406));
}

 TEST_F(SbmlTestSuite, DISABLED_t1408){
    EXPECT_TRUE(RunTest(1408));
}

 TEST_F(SbmlTestSuite, DISABLED_t1409){
    EXPECT_TRUE(RunTest(1409));
}

 TEST_F(SbmlTestSuite, DISABLED_t1416){
    EXPECT_TRUE(RunTest(1416));
}

 TEST_F(SbmlTestSuite, DISABLED_t1419){
    EXPECT_TRUE(RunTest(1419));
}

 TEST_F(SbmlTestSuite, DISABLED_t1444){
    EXPECT_TRUE(RunTest(1444));
}

 TEST_F(SbmlTestSuite, DISABLED_t1445){
    EXPECT_TRUE(RunTest(1445));
}

 TEST_F(SbmlTestSuite, DISABLED_t1446){
    EXPECT_TRUE(RunTest(1446));
}

 TEST_F(SbmlTestSuite, DISABLED_t1447){
    EXPECT_TRUE(RunTest(1447));
}

 TEST_F(SbmlTestSuite, DISABLED_t1448){
    EXPECT_TRUE(RunTest(1448));
}

 TEST_F(SbmlTestSuite, DISABLED_t1449){
    EXPECT_TRUE(RunTest(1449));
}

 TEST_F(SbmlTestSuite, DISABLED_t1450){
    EXPECT_TRUE(RunTest(1450));
}

 TEST_F(SbmlTestSuite, DISABLED_t1451){
    EXPECT_TRUE(RunTest(1451));
}

 TEST_F(SbmlTestSuite, DISABLED_t1452){
    EXPECT_TRUE(RunTest(1452));
}

 TEST_F(SbmlTestSuite, DISABLED_t1453){
    EXPECT_TRUE(RunTest(1453));
}

 TEST_F(SbmlTestSuite, DISABLED_t1454){
    EXPECT_TRUE(RunTest(1454));
}

 TEST_F(SbmlTestSuite, DISABLED_t1455){
    EXPECT_TRUE(RunTest(1455));
}

 TEST_F(SbmlTestSuite, DISABLED_t1456){
    EXPECT_TRUE(RunTest(1456));
}

 TEST_F(SbmlTestSuite, DISABLED_t1457){
    EXPECT_TRUE(RunTest(1457));
}

 TEST_F(SbmlTestSuite, DISABLED_t1458){
    EXPECT_TRUE(RunTest(1458));
}

 TEST_F(SbmlTestSuite, DISABLED_t1459){
    EXPECT_TRUE(RunTest(1459));
}

 TEST_F(SbmlTestSuite, DISABLED_t1460){
    EXPECT_TRUE(RunTest(1460));
}

 TEST_F(SbmlTestSuite, DISABLED_t1461){
    EXPECT_TRUE(RunTest(1461));
}

 TEST_F(SbmlTestSuite, DISABLED_t1462){
    EXPECT_TRUE(RunTest(1462));
}

 TEST_F(SbmlTestSuite, DISABLED_t1463){
    EXPECT_TRUE(RunTest(1463));
}

 TEST_F(SbmlTestSuite, DISABLED_t1464){
    EXPECT_TRUE(RunTest(1464));
}

 TEST_F(SbmlTestSuite, DISABLED_t1465){
    EXPECT_TRUE(RunTest(1465));
}

 TEST_F(SbmlTestSuite, DISABLED_t1471){
    EXPECT_TRUE(RunTest(1471));
}

 TEST_F(SbmlTestSuite, DISABLED_t1472){
    EXPECT_TRUE(RunTest(1472));
}

 TEST_F(SbmlTestSuite, DISABLED_t1473){
    EXPECT_TRUE(RunTest(1473));
}

 TEST_F(SbmlTestSuite, DISABLED_t1475){
    EXPECT_TRUE(RunTest(1475));
}

 TEST_F(SbmlTestSuite, DISABLED_t1476){
    EXPECT_TRUE(RunTest(1476));
}

 TEST_F(SbmlTestSuite, DISABLED_t1477){
    EXPECT_TRUE(RunTest(1477));
}

 TEST_F(SbmlTestSuite, DISABLED_t1478){
    EXPECT_TRUE(RunTest(1478));
}

 TEST_F(SbmlTestSuite, DISABLED_t1479){
    EXPECT_TRUE(RunTest(1479));
}

 TEST_F(SbmlTestSuite, DISABLED_t1482){
    EXPECT_TRUE(RunTest(1482));
}

 TEST_F(SbmlTestSuite, DISABLED_t1483){
    EXPECT_TRUE(RunTest(1483));
}

 TEST_F(SbmlTestSuite, DISABLED_t1484){
    EXPECT_TRUE(RunTest(1484));
}

 TEST_F(SbmlTestSuite, DISABLED_t1488){
    EXPECT_TRUE(RunTest(1488));
}

 TEST_F(SbmlTestSuite, DISABLED_t1498){
    EXPECT_TRUE(RunTest(1498));
}

 TEST_F(SbmlTestSuite, DISABLED_t1499){
    EXPECT_TRUE(RunTest(1499));
}

 TEST_F(SbmlTestSuite, DISABLED_t1500){
    EXPECT_TRUE(RunTest(1500));
}

 TEST_F(SbmlTestSuite, DISABLED_t1501){
    EXPECT_TRUE(RunTest(1501));
}

 TEST_F(SbmlTestSuite, DISABLED_t1502){
    EXPECT_TRUE(RunTest(1502));
}

 TEST_F(SbmlTestSuite, DISABLED_t1503){
    EXPECT_TRUE(RunTest(1503));
}

 TEST_F(SbmlTestSuite, DISABLED_t1504){
    EXPECT_TRUE(RunTest(1504));
}

 TEST_F(SbmlTestSuite, DISABLED_t1505){
    EXPECT_TRUE(RunTest(1505));
}

 TEST_F(SbmlTestSuite, DISABLED_t1506){
    EXPECT_TRUE(RunTest(1506));
}

 TEST_F(SbmlTestSuite, DISABLED_t1507){
    EXPECT_TRUE(RunTest(1507));
}

 TEST_F(SbmlTestSuite, DISABLED_t1508){
    EXPECT_TRUE(RunTest(1508));
}

 TEST_F(SbmlTestSuite, DISABLED_t1509){
    EXPECT_TRUE(RunTest(1509));
}

 TEST_F(SbmlTestSuite, DISABLED_t1510){
    EXPECT_TRUE(RunTest(1510));
}

 TEST_F(SbmlTestSuite, DISABLED_t1511){
    EXPECT_TRUE(RunTest(1511));
}

 TEST_F(SbmlTestSuite, DISABLED_t1512){
    EXPECT_TRUE(RunTest(1512));
}

 TEST_F(SbmlTestSuite, DISABLED_t1513){
    EXPECT_TRUE(RunTest(1513));
}

 TEST_F(SbmlTestSuite, DISABLED_t1514){
    EXPECT_TRUE(RunTest(1514));
}

 TEST_F(SbmlTestSuite, DISABLED_t1517){
    EXPECT_TRUE(RunTest(1517));
}

 TEST_F(SbmlTestSuite, DISABLED_t1525){
    EXPECT_TRUE(RunTest(1525));
}

 TEST_F(SbmlTestSuite, DISABLED_t1526){
    EXPECT_TRUE(RunTest(1526));
}

 TEST_F(SbmlTestSuite, DISABLED_t1527){
    EXPECT_TRUE(RunTest(1527));
}

 TEST_F(SbmlTestSuite, DISABLED_t1528){
    EXPECT_TRUE(RunTest(1528));
}

 TEST_F(SbmlTestSuite, DISABLED_t1529){
    EXPECT_TRUE(RunTest(1529));
}

 TEST_F(SbmlTestSuite, DISABLED_t1540){
    EXPECT_TRUE(RunTest(1540));
}

 TEST_F(SbmlTestSuite, DISABLED_t1541){
    EXPECT_TRUE(RunTest(1541));
}

 TEST_F(SbmlTestSuite, DISABLED_t1542){
    EXPECT_TRUE(RunTest(1542));
}

 TEST_F(SbmlTestSuite, DISABLED_t1543){
    EXPECT_TRUE(RunTest(1543));
}

 TEST_F(SbmlTestSuite, DISABLED_t1552){
    EXPECT_TRUE(RunTest(1552));
}

 TEST_F(SbmlTestSuite, DISABLED_t1553){
    EXPECT_TRUE(RunTest(1553));
}

 TEST_F(SbmlTestSuite, DISABLED_t1554){
    EXPECT_TRUE(RunTest(1554));
}

 TEST_F(SbmlTestSuite, DISABLED_t1556){
    EXPECT_TRUE(RunTest(1556));
}

 TEST_F(SbmlTestSuite, DISABLED_t1567){
    EXPECT_TRUE(RunTest(1567));
}

 TEST_F(SbmlTestSuite, DISABLED_t1571){
    EXPECT_TRUE(RunTest(1571));
}

 TEST_F(SbmlTestSuite, DISABLED_t1574){
    EXPECT_TRUE(RunTest(1574));
}

 TEST_F(SbmlTestSuite, DISABLED_t1575){
    EXPECT_TRUE(RunTest(1575));
}

 TEST_F(SbmlTestSuite, DISABLED_t1576){
    EXPECT_TRUE(RunTest(1576));
}

 TEST_F(SbmlTestSuite, DISABLED_t1577){
    EXPECT_TRUE(RunTest(1577));
}

 TEST_F(SbmlTestSuite, DISABLED_t1579){
    EXPECT_TRUE(RunTest(1579));
}

 TEST_F(SbmlTestSuite, DISABLED_t1589){
    EXPECT_TRUE(RunTest(1589));
}

 TEST_F(SbmlTestSuite, DISABLED_t1626){
    EXPECT_TRUE(RunTest(1626));
}

 TEST_F(SbmlTestSuite, DISABLED_t1627){
    EXPECT_TRUE(RunTest(1627));
}

 TEST_F(SbmlTestSuite, DISABLED_t1628){
    EXPECT_TRUE(RunTest(1628));
}

 TEST_F(SbmlTestSuite, DISABLED_t1629){
    EXPECT_TRUE(RunTest(1629));
}

 TEST_F(SbmlTestSuite, DISABLED_t1630){
    EXPECT_TRUE(RunTest(1630));
}

 TEST_F(SbmlTestSuite, DISABLED_t1635){
    EXPECT_TRUE(RunTest(1635));
}

 TEST_F(SbmlTestSuite, DISABLED_t1636){
    EXPECT_TRUE(RunTest(1636));
}

 TEST_F(SbmlTestSuite, DISABLED_t1657){
    EXPECT_TRUE(RunTest(1657));
}

 TEST_F(SbmlTestSuite, DISABLED_t1727){
    EXPECT_TRUE(RunTest(1727));
}

 TEST_F(SbmlTestSuite, DISABLED_t1728){
    EXPECT_TRUE(RunTest(1728));
}

 TEST_F(SbmlTestSuite, DISABLED_t1729){
    EXPECT_TRUE(RunTest(1729));
}

 TEST_F(SbmlTestSuite, DISABLED_t1736){
    EXPECT_TRUE(RunTest(1736));
}

 TEST_F(SbmlTestSuite, DISABLED_t1737){
    EXPECT_TRUE(RunTest(1737));
}

 TEST_F(SbmlTestSuite, DISABLED_t1738){
    EXPECT_TRUE(RunTest(1738));
}

 TEST_F(SbmlTestSuite, DISABLED_t1778){
    EXPECT_TRUE(RunTest(1778));
}

 TEST_F(SbmlTestSuite, DISABLED_t1780){
    EXPECT_TRUE(RunTest(1780));
}

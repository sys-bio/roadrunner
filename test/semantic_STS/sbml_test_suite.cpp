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


class SbmlTestSuite : public RoadRunnerTest
{
public:

    SbmlTestSuite() = default;

    bool RunTest(int caseNumber)
    {
        std::cout << "running case: " << caseNumber<<std::endl;
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
            createTestSuiteFileNameParts(caseNumber, "-sbml-" + lv + ".xml", modelFilePath, modelFileName, settingsFileName, descriptionFileName);
            ifstream ftest((path(modelFilePath) / path(modelFileName)).string());
            if (ftest.good()) {
                if (first.empty()) {
                    first = lv;
                }
                else {
                    last = lv;
                }
            }
        }
        bool ret = true;
        if (hasUnimplementedTags(modelFilePath + "/" + descriptionFileName)) {
            if (!first.empty()) {
                ret = CheckLoad(first, caseNumber);
            }
            else {
                rrLog(Logger::LOG_ERROR) << "No models found for test case" << caseNumber << endl;
                ret = false;
            }
            if (!last.empty()) {
                ret = CheckLoad(last, caseNumber) && ret;
            }
        }

        else {
            if (!first.empty()) {
                ret = RunTest(first, caseNumber);
                if (!ret && isSemiStochasticTest((path(modelFilePath) / path(descriptionFileName)).string())) {
                    //semistochastic tests fail once in a great while, but very very rarely twice in a row.
                    rrLog(Logger::LOG_WARNING) << "Test " << caseNumber << " failed, but we expect it to fail every so often.  Trying again...";
                    ret = RunTest(first, caseNumber);
                }
            }
            else {
                rrLog(Logger::LOG_ERROR) << "No models found for test case" << caseNumber << endl;
                ret = false;
            }
            if (!last.empty()) {
                ret = RunTest(last, caseNumber) && ret;
            }
        }
        return ret;
    }


    bool RunTest(const string& version, int caseNumber)
    {
        //cerr << "Running Test:\t" << caseNumber << "\t" << version;

        RoadRunner rr;
        TestSuiteModelSimulation simulation;
        try
        {
            LoadAndSimulate(version, caseNumber, rr, simulation);

            //Write result
            if(!simulation.SaveResult())
            {
                //Failed to save data
                rrLog(Logger::LOG_ERROR)<<"Failed to save result";
                throw("Failed running simulation: Failed to save result");
            }

            if(!simulation.LoadReferenceData())
            {
                rrLog(Logger::LOG_ERROR)<<"Failed loading reference data";
                throw("Failed loading reference data");
            }

            simulation.CreateErrorData();
            bool result = simulation.Pass();
            simulation.SaveAllData();

            //simulation.SaveModelAsXML(dataOutputFolder);

            //cerr<<"\t"<< (result == true ? "PASS" : "FAIL")<<endl;
            return result;
         }
        catch(Exception& ex)
        {
            string error = ex.what();
            //cerr << "\tFAIL" << endl;
            //cerr<<"Case "<<caseNumber<<": Exception: "<<error<<endl;
            return false;
        }

    }

    bool CheckLoad(const string& version, int caseNumber)
    {
        //cerr << "Checking Test Loading:\t" << caseNumber << "\t" << version;

        RoadRunner rr;
        TestSuiteModelSimulation simulation;

        try
        {
            LoadAndSimulate(version, caseNumber, rr, simulation);

            //If we've gotten this far, rejoice!  roadrunner didn't crash, which is good enough.
            //cerr << "\tPASS" << endl;
            return true;
        }
        catch (rrllvm::LLVMException& ex)
        {
            //Sometimes, rr itself knows when it can't load a model.  This is also fine.
            return true;
        }
        catch (Exception& ex)
        {
            string error = ex.what();
            //cerr << "\tFAIL" << endl;
            //cerr << "Case " << caseNumber << ": Exception: " << error << endl;
            return false;
        }

    }

    void LoadAndSimulate(const string& version, int caseNumber, RoadRunner& rr, TestSuiteModelSimulation& simulation)
    {

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
        createTestSuiteFileNameParts(caseNumber, "-sbml-" + version + ".xml", modelFilePath, modelFileName, settingsFileName, descriptionFileName);

        //The following will load and compile and simulate the sbml model in the file
        simulation.SetCaseNumber(caseNumber);
        simulation.SetModelFilePath(modelFilePath);
        simulation.SetModelFileName(modelFileName);
        simulation.ReCompileIfDllExists(true);
        simulation.CopyFilesToOutputFolder();

        rr.setConservedMoietyAnalysis(false);

        if (!simulation.LoadSBMLFromFile())
        {
            rrLog(Logger::LOG_ERROR) << "Failed loading SBML model";
            throw("Failed loading SBML model");
        }
        //Then read settings file if it exists..
        string settingsOveride("");
        if (!simulation.LoadSettings(settingsOveride))
        {
            rrLog(Logger::LOG_ERROR) << "Failed loading SBML model settings";
            throw("Failed loading SBML model settings");
        }


        if (!isFBCTest(modelFilePath + "/" + descriptionFileName)) {
            //Only try simulating non-FBC tests.
            if (!simulation.Simulate())
            {
                rrLog(Logger::LOG_ERROR) << "Failed running simulation";
                throw("Failed running simulation");
            }
        }

    }

};

TEST_F(SbmlTestSuite, RunTests){
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
    for (int i=1; i<=lastTest; i++) {
        if (std::find(exclusion.begin(), exclusion.end(), i) == exclusion.end()){
            cases.push_back(i);
        }
    }
    // execute in parallel
    std::for_each(std::execution::par, cases.begin(), cases.end(), [&](int i){
        EXPECT_TRUE(RunTest(i));
    });
}


TEST_F(SbmlTestSuite, DISABLED_test_single)
{
    // Use when need to run one test.
    EXPECT_TRUE(RunTest(28));
}


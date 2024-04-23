#include <string>
#include "gtest/gtest.h"
#include "rrConfig.h"
#include "rrIniFile.h"
#include "rrLogger.h"
#include "rrRoadRunner.h"
#include "rrUtils.h"
#include "sbml/SBMLDocument.h"
#include "Matrix.h"
#include "sbml/ListOf.h"
#include "sbml/Model.h"
#include "rrExecutableModel.h"
#include "rrTestSuiteModelSimulation.h"
#include "SBMLValidator.h"
#include "sbml/validator/SBMLInternalValidator.h"
#include "sbml/math/FormulaFormatter.h"
#include "sbml/math/L3FormulaFormatter.h"
#include "sbml/math/FormulaParser.h"
#include "llvm/Config/llvm-config.h"
#include "TestModelFactory.h"

#include "test_util.h"
#include <filesystem>
#include "RoadRunnerTest.h"
#include <mutex>

using namespace testing;
using namespace std;
using namespace ls;
using namespace rr;
using std::filesystem::path;

static std::mutex SerializationMutex;

/**
 * Suggestions for improving this test suite:
 *  - Descriptive names for tests.
 *  - Test one thing at a time.
 *  - Unit tests work better when you can look at any particular test
 *    and figure out what its testing which I find hard with
 *    the "lambda injection" approach.
 *    - Also, when a test breaks you need to spend a lot of time with the
 *      debugger to figure out what went wrong.
 *
 * See StateSavingTestsModelData for examples.
 */
class StateSavingTests : public RoadRunnerTest {
public:
    path stateSavingModelsDir = rrTestModelsDir_ / "StateSavingTests";
    path stateSavingOutputDir = rrTestDir_ / "StateSavingOutput";
    path fname_ = stateSavingOutputDir / "state-saving-test.rr";
    std::string fname;
    StateSavingTests() {
        fname = fname_.string();
//        Logger::setLevel(Logger::LOG_DEBUG);
    }

    ~StateSavingTests() override {
        if (fs::exists(stateSavingOutputDir)) {
            fs::remove_all(stateSavingOutputDir);
        }

        // clean up fname between runs!!!
        if (fs::exists(fname)){
            fs::remove(fname);
        }

    }

    bool RunStateSavingTest(void(*modification)(RoadRunner *, std::string), std::string version = "l2v4");

    bool RunStateSavingTest(int caseNumber, void(*modification)(RoadRunner *, std::string), std::string version = "l2v4");

    bool StateRunTestModelFromScratch(void(*generate)(RoadRunner *, std::string), std::string version = "l2v4");

};



bool StateSavingTests::RunStateSavingTest(void(*modification)(RoadRunner *, std::string), std::string version) {
    bool result(false);
    RoadRunner *rr;

    //Create instance..
    rr = new RoadRunner();

    string testName(UnitTest::GetInstance()->current_test_info()->name());
    string suiteName(UnitTest::GetInstance()->current_test_info()->test_suite_name());

    try {
        path dataOutputFolder(stateSavingOutputDir / suiteName);

        rr->getIntegrator()->setValue("stiff", false);

        if (!createFolder(dataOutputFolder.string())) {
            string msg("Failed creating output folder for data output: " + dataOutputFolder.string());
            throw (rr::Exception(msg));
        }
        //Create subfolder for data output
        dataOutputFolder = (dataOutputFolder / testName).string();

        if (!createFolder(dataOutputFolder.string())) {
            string msg("Failed creating output folder for data output: " + dataOutputFolder.string());
            throw (rr::Exception(msg));
        }

        TestSuiteModelSimulation simulation(dataOutputFolder);

        simulation.UseEngine(rr);

        //Read SBML models.....
        simulation.SetCaseNumber(0);

        path modelFilePath = rrTestModelsDir_ / suiteName / testName;
        string modelFileName = testName + "-sbml-" + version + ".xml";
        string settingsFileName = testName + "-settings.txt";

        //The following will load and compile and simulate the sbml model in the file
        simulation.SetModelFilePath(modelFilePath);
        simulation.SetModelFileName(modelFileName);
        simulation.ReCompileIfDllExists(true);
        simulation.CopyFilesToOutputFolder();
        rr->setConservedMoietyAnalysis(false);

        libsbml::SBMLReader reader;
        path fullPath = modelFilePath / modelFileName;

        if (!simulation.LoadSBMLFromFile()) {
            throw (Exception("Failed loading sbml from file"));
        }


        //Check first if file exists first
        if (!std::filesystem::exists(fullPath)) {
            rrLog(Logger::LOG_ERROR) << "sbml file " << fullPath << " not found";
            throw (Exception("No such SBML file: " + fullPath.string()));
        }


        LoadSBMLOptions opt;

        // don't generate cache for models
        opt.modelGeneratorOpt = opt.modelGeneratorOpt | LoadSBMLOptions::RECOMPILE;

        opt.modelGeneratorOpt = opt.modelGeneratorOpt | LoadSBMLOptions::MUTABLE_INITIAL_CONDITIONS;

        opt.modelGeneratorOpt = opt.modelGeneratorOpt & ~LoadSBMLOptions::READ_ONLY;

        opt.modelGeneratorOpt = opt.modelGeneratorOpt | LoadSBMLOptions::OPTIMIZE_CFG_SIMPLIFICATION;

        opt.modelGeneratorOpt = opt.modelGeneratorOpt | LoadSBMLOptions::OPTIMIZE_GVN;


        rr->load(fullPath.string(), &opt);

        //Then read settings file if it exists..
        if (!simulation.LoadSettings((modelFilePath / settingsFileName).string())) {
            throw (Exception("Failed loading simulation settings"));
        }
        modification(rr, fname);
        //Then Simulate model
        if (!simulation.Simulate()) {
            throw (Exception("Failed running simulation"));
        }

        //Write result
        if (!simulation.SaveResult()) {
            //Failed to save data
            throw (Exception("Failed saving result"));
        }

        if (!simulation.LoadReferenceData((modelFilePath / (testName + "-results.csv")).string())) {
            throw (Exception("Failed Loading reference data"));
        }

        simulation.CreateErrorData();
        result = simulation.Pass();
        result = simulation.SaveAllData() && result;
        result = simulation.SaveModelAsXML(dataOutputFolder) && result;
    }
    catch (std::exception &ex) {
        string error = ex.what();
        cerr << "Case " << testName << ": Exception: " << error << endl;
        delete rr;
        return false;
    }

    delete rr;
    return result;
}


bool StateSavingTests::RunStateSavingTest(int caseNumber, void(*modification)(RoadRunner *, std::string), std::string version) {
//    Config::setValue(Config::LLVM_BACKEND, Config::LLJIT);
    bool result(false);
    std::unique_ptr<RoadRunner> rr, rr2, rr3;

    //Create instance..
    rr = std::make_unique< RoadRunner>();
    string testName(UnitTest::GetInstance()->current_test_info()->name());
    string suiteName(UnitTest::GetInstance()->current_test_info()->test_suite_name());

    //Setup environment
    try {
        path dataOutputFolder = (stateSavingOutputDir / testName);
        string dummy;
        string logFileName;
        string settingsFileName;

        rr->getIntegrator()->setValue("stiff", false);

        //Create a log file name
        createTestSuiteFileNameParts(caseNumber, ".log", dummy, logFileName, settingsFileName, dummy);
        if (!createFolder(dataOutputFolder.string())) {
            string msg("Failed creating output folder for data output: " + dataOutputFolder.string());
            throw (rr::Exception(msg));
        }


        if (!createFolder(dataOutputFolder.string())) {
            string msg("Failed creating output folder for data output: " + dataOutputFolder.string());
            throw (rr::Exception(msg));
        }

        TestSuiteModelSimulation simulation(dataOutputFolder);

        simulation.UseEngine(rr.get());

        //Read SBML models.....
        string modelFilePath((rrTestDir_ / path("sbml-test-suite") / path("semantic")).string());
        string modelFileName;

        simulation.SetCaseNumber(caseNumber);
        createTestSuiteFileNameParts(caseNumber, "-sbml-" + version + ".xml", modelFilePath, modelFileName,
                                     settingsFileName, dummy);

        //The following will load and compile and simulate the sbml model in the file
        simulation.SetModelFilePath(modelFilePath);
        simulation.SetModelFileName(modelFileName);
        simulation.ReCompileIfDllExists(true);
        simulation.CopyFilesToOutputFolder();
        //setTempFolder(gRR, simulation.GetDataOutputFolder().c_str());
        rr->setConservedMoietyAnalysis(false);

        libsbml::SBMLReader reader;
        std::string fullPath = modelFilePath + "/" + modelFileName;

        if (!simulation.LoadSBMLFromFile()) {
            throw (Exception("Failed loading sbml from file"));
        }


        //Check first if file exists first
        if (!std::filesystem::exists(fullPath)) {
            rrLog(Logger::LOG_ERROR) << "sbml file " << fullPath << " not found";
            throw (Exception("No such SBML file: " + fullPath));
        }

        LoadSBMLOptions opt;

        // don't generate cache for models
        opt.modelGeneratorOpt = opt.modelGeneratorOpt | LoadSBMLOptions::RECOMPILE;
        opt.modelGeneratorOpt = opt.modelGeneratorOpt | LoadSBMLOptions::MUTABLE_INITIAL_CONDITIONS;
        opt.modelGeneratorOpt = opt.modelGeneratorOpt & ~LoadSBMLOptions::READ_ONLY;
        opt.modelGeneratorOpt = opt.modelGeneratorOpt | LoadSBMLOptions::OPTIMIZE_CFG_SIMPLIFICATION;
        opt.modelGeneratorOpt = opt.modelGeneratorOpt | LoadSBMLOptions::OPTIMIZE_GVN;

        rr->load(fullPath, &opt);

        //Then read settings file if it exists..
        string settingsOveride("");
        if (!simulation.LoadSettings(settingsOveride)) {
            throw (Exception("Failed loading simulation settings"));
        }
        //Perform the model editing action
        rr2 = std::make_unique<RoadRunner>(*rr);
        int rr2id = rr2->getInstanceID();
        modification(rr2.get(), fname);
        EXPECT_EQ(rr2->getInstanceID(), rr2id);
        rr3 = std::make_unique<RoadRunner>(*rr2);

        EXPECT_NE(rr->getInstanceID(), rr2->getInstanceID());
        EXPECT_NE(rr->getInstanceID(), rr3->getInstanceID());
        EXPECT_NE(rr2->getInstanceID(), rr3->getInstanceID());
        simulation.UseEngine(rr3.get());
        //Then Simulate model
        if (!simulation.Simulate()) {
            throw (Exception("Failed running simulation"));
        }

        //Write result
        if (!simulation.SaveResult()) {
            //Failed to save data
            throw (Exception("Failed saving result"));
        }

        if (!simulation.LoadReferenceData()) {
            throw (Exception("Failed Loading reference data"));
        }

        simulation.CreateErrorData();

        if (simulation.NrOfFailingPoints() > 0) {
            std::cout << simulation.NrOfFailingPoints() << std::endl;
        }
        result = simulation.Pass();
        result = simulation.SaveAllData() && result;
        result = simulation.SaveModelAsXML(dataOutputFolder) && result;
    }
    catch (std::exception &ex) {
        string error = ex.what();
        cerr << "Case " << caseNumber << ": Exception: " << error << endl;
        result = false;
    }

    return result;
}

bool StateSavingTests::StateRunTestModelFromScratch(void(*generate)(RoadRunner *, std::string), std::string version) {
    bool result(false);
    int level = version.at(1) - '0';
    int versionNum = version.at(3) - '0';
    RoadRunner rr(level, versionNum);


    string testName(UnitTest::GetInstance()->current_test_info()->name());
    string suiteName(UnitTest::GetInstance()->current_test_info()->test_suite_name());

    try {
        path dataOutputFolder((stateSavingOutputDir / suiteName));

        rr.getIntegrator()->setValue("stiff", false);

        if (!createFolder(dataOutputFolder.string())) {
            string msg("Failed creating output folder for data output: " + dataOutputFolder.string());
            throw (rr::Exception(msg));
        }
        //Create subfolder for data output
        dataOutputFolder = (dataOutputFolder / testName).string();

        if (!createFolder(dataOutputFolder.string())) {
            string msg("Failed creating output folder for data output: " + dataOutputFolder.string());
            throw (rr::Exception(msg));
        }

        TestSuiteModelSimulation simulation(dataOutputFolder);

        simulation.UseEngine(&rr);

        //Read SBML models.....
        simulation.SetCaseNumber(0);

        path modelFilePath = rrTestModelsDir_ / suiteName / testName;
        string modelFileName = testName + "-sbml-" + version + ".xml";
        string settingsFileName = testName + "-settings.txt";

        //The following will load and compile and simulate the sbml model in the file
        simulation.SetModelFilePath(modelFilePath);
        simulation.SetModelFileName(modelFileName);
        simulation.ReCompileIfDllExists(true);
        simulation.CopyFilesToOutputFolder();
        rr.setConservedMoietyAnalysis(false);

        libsbml::SBMLReader reader;
        path fullPath = modelFilePath / modelFileName;

        //Check first if file exists first
        if (!std::filesystem::exists(fullPath)) {
            rrLog(Logger::LOG_ERROR) << "sbml file " << fullPath << " not found";
            throw (Exception("No such SBML file: " + fullPath.string()));
        }


        LoadSBMLOptions opt;

        // don't generate cache for models
        opt.modelGeneratorOpt = opt.modelGeneratorOpt | LoadSBMLOptions::RECOMPILE;

        opt.modelGeneratorOpt = opt.modelGeneratorOpt | LoadSBMLOptions::MUTABLE_INITIAL_CONDITIONS;

        opt.modelGeneratorOpt = opt.modelGeneratorOpt & ~LoadSBMLOptions::READ_ONLY;

        opt.modelGeneratorOpt = opt.modelGeneratorOpt | LoadSBMLOptions::OPTIMIZE_CFG_SIMPLIFICATION;

        opt.modelGeneratorOpt = opt.modelGeneratorOpt | LoadSBMLOptions::OPTIMIZE_GVN;



        //Then read settings file if it exists..
        if (!simulation.LoadSettings((modelFilePath / settingsFileName).string())) {
            throw (Exception("Failed loading simulation settings"));
        }

        SerializationMutex.lock();
        generate(&rr, fname);
        SerializationMutex.unlock();
        //Then Simulate model
        if (!simulation.Simulate()) {
            throw (Exception("Failed running simulation"));
        }

        //Write result
        if (!simulation.SaveResult()) {
            //Failed to save data
            throw (Exception("Failed saving result"));
        }

        if (!simulation.LoadReferenceData((modelFilePath / (testName + "-results.csv")).string())) {
            throw (Exception("Failed Loading reference data"));
        }

        simulation.CreateErrorData();
        result = simulation.Pass();
        result = simulation.SaveAllData() && result;
        result = simulation.SaveModelAsXML(dataOutputFolder) && result;
        result = testValidateSBML(rr.getCurrentSBML()) && result;
    }
    catch (std::exception &ex) {
        string error = ex.what();
        cerr << "Case " << testName << ": Exception: " << error << endl;
        return false;
    }

    return result;
}

// IN LLVM 6.0.1, this test can result, depending on the OS,
//  in llvm calling *exit* instead of throwing.  We changed this
//  in our own updated version of llvm, and updated the patch value
//  accordingly.  The reverse can't be tested, since it'll either
//  exit or throw, depending.
TEST_F(StateSavingTests, LOAD_INVALID_FILE) {
    RoadRunner rri;
    EXPECT_THROW(rri.loadState((stateSavingModelsDir / "wrong-save-state.rr").string()), std::exception);
}

TEST_F(StateSavingTests, LOAD_VALID_FILE) {
    RoadRunner rri;
#if defined(_WIN32)
    rri.loadState((stateSavingModelsDir / "savedState_windows.rr").string());
    EXPECT_EQ(rri.getNumberOfFloatingSpecies(), 2);
    rri.loadState((stateSavingModelsDir / "savedState_windows.rr").string());
    EXPECT_EQ(rri.getNumberOfFloatingSpecies(), 2);
#elif (defined(__APPLE__))
    //Try the linux one?
    rri.loadState((stateSavingModelsDir / "savedState_linux.rr").string());
    EXPECT_EQ(rri.getNumberOfFloatingSpecies(), 2);
    rri.loadState((stateSavingModelsDir / "savedState_linux.rr").string());
    EXPECT_EQ(rri.getNumberOfFloatingSpecies(), 2);
#else
    rri.loadState((stateSavingModelsDir / "savedState_linux.rr").string());
    EXPECT_EQ(rri.getNumberOfFloatingSpecies(), 2);
    rri.loadState((stateSavingModelsDir / "savedState_linux.rr").string());
    EXPECT_EQ(rri.getNumberOfFloatingSpecies(), 2);
#endif
}

TEST_F(StateSavingTests, LOAD_NONEXISTENT_FILE) {
    RoadRunner rri;
    EXPECT_THROW(rri.loadState((rrTestModelsDir_ / "nonexistent-save-state.rr").string()), std::invalid_argument);
}

TEST_F(StateSavingTests, COPY_RR) {
    RoadRunner rr(3, 2);
    RoadRunner rr2(rr);
    ASSERT_TRUE(rr.getInstanceID() != rr2.getInstanceID());
}

TEST_F(StateSavingTests, COPY_RR_WITH_ASSIGNMENT) {
    path f = rrTestSbmlTestSuiteDir_ / "semantic" / "00001" / "00001-sbml-l2v4.xml";
    path stateFname = fs::current_path() / "save-state-tests.rr";
    RoadRunner rri(f.string());
    rri.getIntegrator()->setIndividualTolerance("S1", 5.0);
    rri.getIntegrator()->setIndividualTolerance("S2", 3.0);
    rri.saveState(stateFname.string());
    RoadRunner rri2(3, 1);
    rri2 = rri;
    EXPECT_TRUE(rri.getInstanceID() != rri2.getInstanceID());
    EXPECT_EQ(rri2.getIntegrator()->getAbsoluteToleranceVector()[0], 5.0);
    EXPECT_EQ(rri2.getIntegrator()->getAbsoluteToleranceVector()[1], 3.0);
}

TEST_F(StateSavingTests, COPY_RR_TWICE) {
    RoadRunner rr(3, 2);
    RoadRunner rr2(rr);
    RoadRunner rr3(rr);
    ASSERT_TRUE(rr.getInstanceID() != rr2.getInstanceID());
    ASSERT_TRUE(rr.getInstanceID() != rr3.getInstanceID());
}

TEST_F(StateSavingTests, CopyModelLoadedConstructor) {
    RoadRunner rr(OpenLinearFlux().str());
    RoadRunner rr2(rr);
    // need to use constructor to convert ls::Matrix to rr::Matrix
    rr::Matrix<double> result = rr::Matrix<double>(*rr2.simulate(0, 10, 11));
    ASSERT_TRUE(OpenLinearFlux().timeSeriesResult().almostEquals(result, 1e-3));
}

TEST_F(StateSavingTests, CopyModelLoadedAssignment) {
    RoadRunner rr(OpenLinearFlux().str());
    RoadRunner rr2 = rr;
    // need to use constructor to convert ls::Matrix to rr::Matrix
    rr::Matrix<double> result = rr::Matrix<double>(*rr2.simulate(0, 10, 11));
    ASSERT_TRUE(OpenLinearFlux().timeSeriesResult().almostEquals(result, 1e-3));
}

TEST_F(StateSavingTests, CopyModelTwiceModelLoadedConstructor) {
    RoadRunner rr(OpenLinearFlux().str());
    RoadRunner rr2(rr);
    RoadRunner rr3(rr);
    // need to use constructor to convert ls::Matrix to rr::Matrix
    rr::Matrix<double> result2 = rr::Matrix<double>(*rr2.simulate(0, 10, 11));
    rr::Matrix<double> result3 = rr::Matrix<double>(*rr3.simulate(0, 10, 11));
    ASSERT_TRUE(OpenLinearFlux().timeSeriesResult().almostEquals(result2, 1e-3));
    ASSERT_TRUE(OpenLinearFlux().timeSeriesResult().almostEquals(result3, 1e-3));
}

TEST_F(StateSavingTests, CopyModelTwiceModelLoadedAssignment) {
    RoadRunner rr(OpenLinearFlux().str());
    RoadRunner rr2 = rr;
    RoadRunner rr3 = rr;
    // need to use constructor to convert ls::Matrix to rr::Matrix
    rr::Matrix<double> result2 = rr::Matrix<double>(*rr2.simulate(0, 10, 11));
    rr::Matrix<double> result3 = rr::Matrix<double>(*rr3.simulate(0, 10, 11));
    ASSERT_TRUE(OpenLinearFlux().timeSeriesResult().almostEquals(result2, 1e-3));
    ASSERT_TRUE(OpenLinearFlux().timeSeriesResult().almostEquals(result3, 1e-3));
}

// repeat with LLJit

TEST_F(StateSavingTests, CopyModelAndCopyBackMCJit) {
    RoadRunner rr(OpenLinearFlux().str());
    RoadRunner rr2;
    rr2 = rr;
    rr = rr2;
    rr::Matrix<double> result = rr::Matrix<double>(*rr.simulate(0, 10, 11));
    rr::Matrix<double> result2 = rr::Matrix<double>(*rr2.simulate(0, 10, 11));
    ASSERT_TRUE(OpenLinearFlux().timeSeriesResult().almostEquals(result, 1e-3));
    ASSERT_TRUE(OpenLinearFlux().timeSeriesResult().almostEquals(result2, 1e-3));
}

TEST_F(StateSavingTests, CopyModelAndCopyBackLLJit) {
    Config::setValue(Config::LLVM_BACKEND, Config::LLJIT);
    RoadRunner rr(OpenLinearFlux().str());
    RoadRunner rr2;
    rr2 = rr;
    rr = rr2;
    // need to use constructor to convert ls::Matrix to rr::Matrix
    rr::Matrix<double> result = rr::Matrix<double>(*rr.simulate(0, 10, 11));
    rr::Matrix<double> result2 = rr::Matrix<double>(*rr2.simulate(0, 10, 11));
    ASSERT_TRUE(OpenLinearFlux().timeSeriesResult().almostEquals(result, 1e-3));
    ASSERT_TRUE(OpenLinearFlux().timeSeriesResult().almostEquals(result2, 1e-3));
}


TEST_F(StateSavingTests, COPY_RR_TWICE2) {
    RoadRunner rr(3, 2);
    RoadRunner rr2(rr);
    RoadRunner rr3(rr2);
    ASSERT_TRUE(rr.getInstanceID() != rr2.getInstanceID());
    ASSERT_TRUE(rr.getInstanceID() != rr3.getInstanceID());
}


TEST_F(StateSavingTests, SaveThenLoad) {
    ASSERT_TRUE(RunStateSavingTest(1, [](RoadRunner *rri, std::string fname) {
        rri->saveState(fname);
        rri->loadState(fname);
    }));
}

TEST_F(StateSavingTests, SaveThenLoadTwice) {
    ASSERT_TRUE(RunStateSavingTest(1, [](RoadRunner *rri, std::string fname) {
        rri->saveState(fname);
        rri->loadState(fname);
        rri->saveState(fname);
        rri->loadState(fname);
    }));
}

TEST_F(StateSavingTests, SaveThenLoadWithADifferentModel) {
    ASSERT_TRUE(RunStateSavingTest(1121, [](RoadRunner *rri, std::string fname) {
        rri->saveState(fname);
        rri->loadState(fname);
    }, "l3v1"));
}

TEST_F(StateSavingTests, SaveThenLoadWithADiffentModelTwice) {
    ASSERT_TRUE(RunStateSavingTest(1121, [](RoadRunner *rri, std::string fname) {
        rri->saveState(fname);
        rri->loadState(fname);
        rri->saveState(fname);
        rri->loadState(fname);
    }, "l3v1"));
}

TEST_F(StateSavingTests, SimulateThenSaveLoadThenReset) {
    ASSERT_TRUE(RunStateSavingTest(1, [](RoadRunner *rri, std::string fname) {
        rri->simulate();
        rri->saveState(fname);
        rri->loadState(fname);
        rri->reset();
    }));
}

TEST_F(StateSavingTests, SimulateThenSaveLoadThenResetWithDifferentModel) {
    ASSERT_TRUE(RunStateSavingTest(1121, [](RoadRunner *rri, std::string fname) {
        rri->simulate();
        rri->saveState(fname);
        rri->loadState(fname);
        rri->reset((int)SelectionRecord::ALL);
    }, "l3v1"));
}

TEST_F(StateSavingTests, SaveThenLoadCase1120) {
    ASSERT_TRUE(RunStateSavingTest(1120, [](RoadRunner *rri, std::string fname) {
        rri->saveState(fname);
        rri->loadState(fname);
    }, "l3v1"));
}

TEST_F(StateSavingTests, SaveThenLoadTwiceCase1120) {
    ASSERT_TRUE(RunStateSavingTest(1120, [](RoadRunner *rri, std::string fname) {
        rri->saveState(fname);
        rri->loadState(fname);
        rri->saveState(fname);
        rri->loadState(fname);
    }, "l3v1"));
}

TEST_F(StateSavingTests, SaveThenLoadAndSimulateCase1120) {
    ASSERT_TRUE(RunStateSavingTest(1120, [](RoadRunner *rri, std::string fname) {
        rri->simulate();
        rri->saveState(fname);
        rri->loadState(fname);
        rri->reset((int)SelectionRecord::ALL);
    }, "l3v1"));
}

TEST_F(StateSavingTests, SAVE_STATE_19) {
    ASSERT_TRUE(RunStateSavingTest([](RoadRunner *rri, std::string fname) {
        rri->getSimulateOptions().duration /= 2;
        rri->getSimulateOptions().steps /= 2;
        rri->simulate();
        rri->saveState(fname);
        rri->loadState(fname);
        rri->getSimulateOptions().start = rri->getSimulateOptions().duration;
    }, "l3v1"));
}

TEST_F(StateSavingTests, SAVE_STATE_20) {
    ASSERT_TRUE(RunStateSavingTest([](RoadRunner *rri, std::string fname) {
        rri->getSimulateOptions().duration /= 2;
        rri->getSimulateOptions().steps /= 2;
        rri->simulate();
        rri->saveState(fname);
        rri->loadState(fname);
        rri->getSimulateOptions().start = rri->getSimulateOptions().duration;
    }));
}

TEST_F(StateSavingTests, SAVE_STATE_21) {
    ASSERT_TRUE(RunStateSavingTest([](RoadRunner *rri, std::string fname) {
        rri->getSimulateOptions().duration = 1.50492;
        rri->getSimulateOptions().steps /= 2;
        rri->simulate();
        rri->saveState(fname);
        rri->loadState(fname);
        rri->getSimulateOptions().start = rri->getSimulateOptions().duration;
        rri->getSimulateOptions().duration = 3.0 - rri->getSimulateOptions().duration;
    }, "l3v1"));
}

TEST_F(StateSavingTests, RETAIN_ABSOLUTE_TOLERANCES_1) {
    path f = rrTestSbmlTestSuiteDir_ / "semantic" / "00001" / "00001-sbml-l2v4.xml";
    path stateFname = fs::current_path() / "save-state-tests.rr";
    RoadRunner rri(f.string());
    rri.getIntegrator()->setIndividualTolerance("S1", 5.0);
    rri.getIntegrator()->setIndividualTolerance("S2", 3.0);
    rri.saveState(stateFname.string());
    RoadRunner rri2;
    rri2.loadState(stateFname.string());
    EXPECT_EQ(rri2.getIntegrator()->getAbsoluteToleranceVector()[0], 5.0);
    EXPECT_EQ(rri2.getIntegrator()->getAbsoluteToleranceVector()[1], 3.0);
    if (fs::exists(stateFname)){
        fs::remove(stateFname);
    }
}

TEST_F(StateSavingTests, FROM_SCRATCH_1) {
    ASSERT_TRUE(StateRunTestModelFromScratch([](RoadRunner *rri, std::string fname) {
        rri->addCompartment("compartment", 1);
        rri->saveState("test-save-state.rr");
        rri->loadState("test-save-state.rr");
        rri->addSpeciesConcentration("S1", "compartment", 0.00015, true);
        rri->addSpeciesConcentration("S2", "compartment", 0, true);
        rri->addParameter("k1", 1);
        rri->addReaction("reaction1", {"S1"}, {"S2"}, "compartment * k1 * S1");
    }));
}

TEST_F(StateSavingTests, FROM_SCRATCH_2) {
    ASSERT_TRUE(StateRunTestModelFromScratch([](RoadRunner *rri, std::string fname) {
        rri->addCompartment("compartment", 1);
        rri->addSpeciesConcentration("S1", "compartment", 1, true);
        rri->addSpeciesConcentration("S2", "compartment", 0, true);

        rri->saveState("test-save-state.rr");
        rri->loadState("test-save-state.rr");

        rri->addParameter("k1", 1);
        rri->addReaction("reaction1", {"S1"}, {"S2"}, "compartment * k1 * S1");
        rri->addEvent("event1", true, "S1 < 0.1");
        rri->addDelay("event1", "1");
        rri->addEventAssignment("event1", "S1", "1");
    }));
}

TEST_F(StateSavingTests, FROM_SCRATCH_3) {
    ASSERT_TRUE(StateRunTestModelFromScratch([](RoadRunner *rri, std::string fname) {
        rri->addCompartment("compartment", 1);
        rri->addSpeciesConcentration("S1", "compartment", 0, true, true);
        rri->addRateRule("S1", "7");

        rri->saveState("test-save-state.rr");
        rri->loadState("test-save-state.rr");
    }));
}

TEST_F(StateSavingTests, FROM_SCRATCH_4) {
    ASSERT_TRUE(StateRunTestModelFromScratch([](RoadRunner *rri, std::string fname) {
        rri->addCompartment("compartment", 1);

        rri->saveState("test-save-state.rr");

        rri->addSpeciesConcentration("S1", "compartment", 7, true);

        rri->loadState("test-save-state.rr");

        rri->addSpeciesConcentration("S1", "compartment", 7, true, true);
        rri->addAssignmentRule("S1", "7");
    }));
}

TEST_F(StateSavingTests, FROM_SCRATCH_5) {
    ASSERT_TRUE(StateRunTestModelFromScratch([](RoadRunner *rri, std::string fname) {
        rri->addCompartment("compartment", 1);
        rri->saveState("test-save-state.rr");
        rri->loadState("test-save-state.rr");
        rri->addSpeciesConcentration("S1", "compartment", 1, true);
        rri->addSpeciesConcentration("S2", "compartment", 1.5e-15, true);
        rri->addSpeciesConcentration("S3", "compartment", 1, true, true);
        rri->addParameter("k1", 0.75);
        rri->addParameter("k2", 50);
        rri->addAssignmentRule("S3", "k1*S2");
        rri->addReaction("reaction1", {"S1"}, {"S2"}, "compartment * k2 * S1");
    }));
}

TEST_F(StateSavingTests, FROM_SCRATCH_6) {
    ASSERT_TRUE(StateRunTestModelFromScratch([](RoadRunner *rri, std::string fname) {
        rri->addParameter("Q", 0);
        rri->addParameter("R", 0);
        rri->addParameter("reset", 0);
        rri->addParameter("Q2", 0);
        rri->addParameter("R2", 0);
        rri->addParameter("reset2", 0);

        rri->addEvent("Qinc", true, "(time - reset) >= 0.01", true);
        rri->saveState("test-save-state.rr");
        rri->loadState("test-save-state.rr");
        rri->addEventAssignment("Qinc", "reset", "time", false);
        rri->addEventAssignment("Qinc", "Q", "Q + 0.01", false);
        rri->addPriority("Qinc", "1");

        rri->addEvent("Rinc", true, "(time - reset) >= 0.01", false);
        rri->addEventAssignment("Rinc", "reset", "time", false);
        rri->addEventAssignment("Rinc", "R", "R + 0.01", false);
        rri->addPriority("Rinc", "-1", false);

        rri->addEvent("Qinc2", true, "(time - reset2) >= 0.01", false);
        rri->addEventAssignment("Qinc2", "reset2", "time", false);
        rri->addEventAssignment("Qinc2", "Q2", "Q2 + 0.01", false);
        rri->addPriority("Qinc2", "-1", false);

        rri->addEvent("Rinc2", true, "(time - reset2) >= 0.01", false);
        rri->addEventAssignment("Rinc2", "reset2", "time", false);
        rri->addEventAssignment("Rinc2", "R2", "R2 + 0.01", false);
        rri->addPriority("Rinc2", "1");
    }, "l3v1"));
}

TEST_F(StateSavingTests, FROM_SCRATCH_7) {
    ASSERT_TRUE(StateRunTestModelFromScratch([](RoadRunner *rri, std::string fname) {
        rri->addCompartment("compartment", 1);
        rri->saveState("test-save-state.rr");
        rri->clearModel();
        rri->loadState("test-save-state.rr");
        rri->addSpeciesConcentration("S1", "compartment", 0.00015, true);
        rri->addSpeciesConcentration("S2", "compartment", 0, true);
        rri->addParameter("k1", 1);
        rri->addReaction("reaction1", {"S1"}, {"S2"}, "compartment * k1 * S1");
    }));
}

TEST_F(StateSavingTests, FROM_SCRATCH_8) {
    ASSERT_TRUE(StateRunTestModelFromScratch([](RoadRunner *rri, std::string fname) {
        rri->addCompartment("compartment", 1);
        rri->addSpeciesConcentration("S1", "compartment", 1, true);
        rri->addSpeciesConcentration("S2", "compartment", 0, true);

        rri->saveState("test-save-state.rr");
        rri->clearModel();
        rri->loadState("test-save-state.rr");

        rri->addParameter("k1", 1);
        rri->addReaction("reaction1", {"S1"}, {"S2"}, "compartment * k1 * S1");
        rri->addEvent("event1", true, "S1 < 0.1");
        rri->addDelay("event1", "1");
        rri->addEventAssignment("event1", "S1", "1");
    }));
}

TEST_F(StateSavingTests, FROM_SCRATCH_9) {
    ASSERT_TRUE(StateRunTestModelFromScratch([](RoadRunner *rri, std::string fname) {
        rri->addCompartment("compartment", 1);
        rri->addSpeciesConcentration("S1", "compartment", 0, true, true);
        rri->addRateRule("S1", "7");

        rri->saveState("test-save-state.rr");
        rri->clearModel();
        rri->loadState("test-save-state.rr");
    }));
}

TEST_F(StateSavingTests, LOAD_ON_NEW_INSTANCE) {
    RoadRunner rri1(OpenLinearFlux().str());
    rri1.saveState("test-save-state.rr");
    RoadRunner rri2;
    rri2.loadState("test-save-state.rr");
    ASSERT_TRUE(rri2.getFloatingSpeciesByIndex(0) == 0);
}


TEST_F(StateSavingTests, FromString) {
    RoadRunner rr(OpenLinearFlux().str());
    std::stringstream *stateStream = rr.saveStateS('b');
    RoadRunner rr2;
    rr2.loadStateS(stateStream);
    auto actualDataLsMatrix = *rr2.simulate(0, 10, 11);
    auto actualData = rr::Matrix<double>(actualDataLsMatrix); // for almostEquals
    auto expectedData = OpenLinearFlux().timeSeriesResult();
    ASSERT_TRUE(expectedData.almostEquals(actualData, 1e-4));
}

TEST_F(StateSavingTests, FromFile) {
    std::filesystem::path p = std::filesystem::current_path() / "savedState.rr";
    //std::cout << "saved to " << p << std::endl;
    RoadRunner rr(OpenLinearFlux().str());
    rr.saveState(p.string());
    RoadRunner rr2;
    rr2.loadState(p.string());
    auto actualDataLsMatrix = *rr2.simulate(0, 10, 11);
    auto actualData = rr::Matrix<double>(actualDataLsMatrix); // for almostEquals
    auto expectedData = OpenLinearFlux().timeSeriesResult();
    ASSERT_TRUE(expectedData.almostEquals(actualData, 1e-4));
}


class StateSavingTestsModelData : public RoadRunnerTest {
public:
    StateSavingTestsModelData() {
//        Logger::setLevel(Logger::LOG_DEBUG);
    };

    bool compareTwoLsMatrices(ls::Matrix<double> *expected_, ls::Matrix<double> *actual_, double tol) {
        rr::Matrix<double> expected(expected_);
        rr::Matrix<double> actual(actual_);
        return expected.almostEquals(actual, tol);
    }

    std::unique_ptr<RoadRunner> saveAndLoadState(RoadRunner *rr) {
        auto state = rr->saveStateS();
        auto rr2 = std::make_unique<RoadRunner>();
        rr2->loadStateS(state);
        return std::move(rr2);
    }

    void compareNumBoundarySpecies(RoadRunner *rr) {
        int originalNum = rr->getNumberOfBoundarySpecies();
        auto rr2 = saveAndLoadState(rr);
        int afterNum = rr2->getNumberOfBoundarySpecies();
        ASSERT_EQ(originalNum, afterNum);
    }

    void compareNumCompartments(RoadRunner *rr) {
        int originalNum = rr->getNumberOfCompartments();
        auto rr2 = saveAndLoadState(rr);
        int afterNum = rr2->getNumberOfCompartments();
        ASSERT_EQ(originalNum, afterNum);
    }

    void compareNumDependentSpecies(RoadRunner *rr) {
        int originalNum = rr->getNumberOfDependentSpecies();
        auto rr2 = saveAndLoadState(rr);
        int afterNum = rr2->getNumberOfDependentSpecies();
        ASSERT_EQ(originalNum, afterNum);
    }

    void compareNumFloatingSpecies(RoadRunner *rr) {
        int originalNum = rr->getNumberOfFloatingSpecies();
        auto rr2 = saveAndLoadState(rr);
        int afterNum = rr2->getNumberOfFloatingSpecies();
        ASSERT_EQ(originalNum, afterNum);
    }

    void compareNumGlobalParameters(RoadRunner *rr) {
        int originalNum = rr->getNumberOfGlobalParameters();
        auto rr2 = saveAndLoadState(rr);
        int afterNum = rr2->getNumberOfGlobalParameters();
        ASSERT_EQ(originalNum, afterNum);
    }

    void compareNumIndependentSpecies(RoadRunner *rr) {
        int originalNum = rr->getNumberOfIndependentSpecies();
        auto rr2 = saveAndLoadState(rr);
        int afterNum = rr2->getNumberOfIndependentSpecies();
        ASSERT_EQ(originalNum, afterNum);
    }

    void compareNumOfReactions(RoadRunner *rr) {
        int originalNum = rr->getNumberOfReactions();
        auto rr2 = saveAndLoadState(rr);
        int afterNum = rr2->getNumberOfReactions();
        ASSERT_EQ(originalNum, afterNum);
    }

    void compareSimulationOutput(RoadRunner *rr) {
        ls::Matrix<double> originalSimulationResults = *rr->simulate(0, 10, 11);
        auto rr2 = saveAndLoadState(rr);
        ls::Matrix<double> afterResults = *rr2->simulate(0, 10, 11);
        rrLogDebug << "Original simulation results: \n" << originalSimulationResults;
        rrLogDebug << "Post save and load state simulation results: \n" << afterResults;
        compareTwoLsMatrices(&originalSimulationResults, &afterResults, 1e-5);
    }


    void compareValuesBoundarySpecies(RoadRunner *rr) {
        auto expected = rr->getBoundarySpeciesAmountsNamedArray();
        auto rr2 = saveAndLoadState(rr);
        auto actual = rr2->getBoundarySpeciesAmountsNamedArray();
        ASSERT_TRUE(compareTwoLsMatrices(&expected, &actual, 1e-5));
    }

    void compareValuesCompartments(RoadRunner *rr) {
        int N = rr->getNumberOfCompartments();
        std::vector<double> expected;
        for (int i = 0; i < N; i++) {
            expected.push_back(rr->getCompartmentByIndex(i));
        }
        auto rr2 = saveAndLoadState(rr);
        std::vector<double> actual;
        for (int i = 0; i < N; i++) {
            actual.push_back(rr2->getCompartmentByIndex(i));
        }
        for (int i = 0; i < N; i++) {
            ASSERT_NEAR(expected[i], actual[i], 1e-5);
        }
    }

    void compareValuesFloatingSpecies(RoadRunner *rr) {
        auto expected = rr->getFloatingSpeciesAmountsNamedArray();
        auto rr2 = saveAndLoadState(rr);
        auto actual = rr2->getFloatingSpeciesAmountsNamedArray();
        //std::cout << expected << std::endl;
        //std::cout << actual << std::endl;
        ASSERT_TRUE(compareTwoLsMatrices(&expected, &actual, 1e-5));
    }

    void compareValuesGlobalParameters(RoadRunner *rr) {
        auto expected = rr->getGlobalParameterValues();
        auto rr2 = saveAndLoadState(rr);
        auto actual = rr2->getGlobalParameterValues();
        ASSERT_EQ(expected.size(), actual.size());
        for (int i = 0; i < expected.size(); i++) {
            ASSERT_NEAR(expected[i], actual[i], 1e-5);
        }
    }

    void compareValuesReactionRates(RoadRunner *rr) {
        auto expected = rr->getReactionRates();
        auto rr2 = saveAndLoadState(rr);
        auto actual = rr2->getReactionRates();
        ASSERT_EQ(expected.size(), actual.size());
        for (int i = 0; i < expected.size(); i++) {
            ASSERT_NEAR(expected[i], actual[i], 1e-5);
        }
    }
};

class StateSavingTestsBrown2004 : public StateSavingTestsModelData{
public:
    std::string sbml;

    StateSavingTestsBrown2004():
        StateSavingTestsModelData(),
        sbml(Brown2004().str()){}
};

class StateSavingTestsBrown2004MCJit : public StateSavingTestsBrown2004{
public:
    StateSavingTestsBrown2004MCJit()
        : StateSavingTestsBrown2004(){
        Config::setValue(Config::LLVM_BACKEND, Config::MCJIT);
    }
};

TEST_F(StateSavingTestsBrown2004MCJit, Brown2004NumBoundarySpecies) {
    RoadRunner rr(sbml);
    compareNumBoundarySpecies(&rr);
}

TEST_F(StateSavingTestsBrown2004MCJit, Brown2004CompareNumBoundarySpecies) {
    RoadRunner rr(sbml);
    compareNumBoundarySpecies(&rr);
}

TEST_F(StateSavingTestsBrown2004MCJit, Brown2004CompareNumCompartments) {
    RoadRunner rr(sbml);
    compareNumCompartments(&rr);
}

TEST_F(StateSavingTestsBrown2004MCJit, Brown2004CompareNumDependentSpecies) {
    RoadRunner rr(sbml);
    compareNumDependentSpecies(&rr);
}

TEST_F(StateSavingTestsBrown2004MCJit, Brown2004CompareNumFloatingSpecies) {
    RoadRunner rr(sbml);
    compareNumFloatingSpecies(&rr);
}

TEST_F(StateSavingTestsBrown2004MCJit, Brown2004CompareNumGlobalParameters) {
    RoadRunner rr(sbml);
    compareNumGlobalParameters(&rr);
}

TEST_F(StateSavingTestsBrown2004MCJit, Brown2004CompareNumIndependentSpecies) {
    RoadRunner rr(sbml);
    compareNumIndependentSpecies(&rr);
}

TEST_F(StateSavingTestsBrown2004MCJit, Brown2004CompareNumOfReactions) {
    RoadRunner rr(sbml);
    compareNumOfReactions(&rr);
}

TEST_F(StateSavingTestsBrown2004MCJit, Brown2004CompareValuesBoundarySpecies) {
    RoadRunner rr(sbml);
    compareValuesBoundarySpecies(&rr);
}

TEST_F(StateSavingTestsBrown2004MCJit, Brown2004CompareValuesCompartments) {
    RoadRunner rr(sbml);
    compareValuesCompartments(&rr);
}

TEST_F(StateSavingTestsBrown2004MCJit, Brown2004CompareValuesFloatingSpecies) {
    RoadRunner rr(sbml);
    compareValuesFloatingSpecies(&rr);
}

TEST_F(StateSavingTestsBrown2004MCJit, Brown2004CompareValuesGlobalParameters) {
    RoadRunner rr(sbml);
    compareValuesGlobalParameters(&rr);
}

TEST_F(StateSavingTestsBrown2004MCJit, Brown2004CompareValuesReactionRates) {
    RoadRunner rr(sbml);
    compareValuesReactionRates(&rr);
}

TEST_F(StateSavingTestsBrown2004MCJit, Brown2004CompareSimulationOutput) {
    RoadRunner rr(sbml);
    compareSimulationOutput(&rr);
}




class StateSavingTestsBrown2004LLJit : public StateSavingTestsBrown2004{
public:
    StateSavingTestsBrown2004LLJit()
        : StateSavingTestsBrown2004(){
        Config::setValue(Config::LLVM_BACKEND, Config::LLJIT);
    }
};

TEST_F(StateSavingTestsBrown2004LLJit, Brown2004NumBoundarySpecies) {
    RoadRunner rr(sbml);
    compareNumBoundarySpecies(&rr);
}

TEST_F(StateSavingTestsBrown2004LLJit, Brown2004CompareNumBoundarySpecies) {
    RoadRunner rr(sbml);
    compareNumBoundarySpecies(&rr);
}

TEST_F(StateSavingTestsBrown2004LLJit, Brown2004CompareNumCompartments) {
    RoadRunner rr(sbml);
    compareNumCompartments(&rr);
}

TEST_F(StateSavingTestsBrown2004LLJit, Brown2004CompareNumDependentSpecies) {
    RoadRunner rr(sbml);
    compareNumDependentSpecies(&rr);
}

TEST_F(StateSavingTestsBrown2004LLJit, Brown2004CompareNumFloatingSpecies) {
    RoadRunner rr(sbml);
    compareNumFloatingSpecies(&rr);
}

TEST_F(StateSavingTestsBrown2004LLJit, Brown2004CompareNumGlobalParameters) {
    RoadRunner rr(sbml);
    compareNumGlobalParameters(&rr);
}

TEST_F(StateSavingTestsBrown2004LLJit, Brown2004CompareNumIndependentSpecies) {
    RoadRunner rr(sbml);
    compareNumIndependentSpecies(&rr);
}

TEST_F(StateSavingTestsBrown2004LLJit, Brown2004CompareNumOfReactions) {
    RoadRunner rr(sbml);
    compareNumOfReactions(&rr);
}

TEST_F(StateSavingTestsBrown2004LLJit, Brown2004CompareValuesBoundarySpecies) {
    RoadRunner rr(sbml);
    compareValuesBoundarySpecies(&rr);
}

TEST_F(StateSavingTestsBrown2004LLJit, Brown2004CompareValuesCompartments) {
    RoadRunner rr(sbml);
    compareValuesCompartments(&rr);
}

TEST_F(StateSavingTestsBrown2004LLJit, Brown2004CompareValuesFloatingSpecies) {
    RoadRunner rr(sbml);
    compareValuesFloatingSpecies(&rr);
}

TEST_F(StateSavingTestsBrown2004LLJit, Brown2004CompareValuesGlobalParameters) {
    RoadRunner rr(sbml);
    compareValuesGlobalParameters(&rr);
}

TEST_F(StateSavingTestsBrown2004LLJit, Brown2004CompareValuesReactionRates) {
    RoadRunner rr(sbml);
    compareValuesReactionRates(&rr);
}

TEST_F(StateSavingTestsBrown2004LLJit, Brown2004CompareSimulationOutput) {
    RoadRunner rr(sbml);
    compareSimulationOutput(&rr);
}
















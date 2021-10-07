#include <string>
#include "gtest/gtest.h"
#include "rrConfig.h"
#include "rrIniFile.h"
#include "rrLogger.h"
#include "rrRoadRunner.h"
#include "rrUtils.h"
#include "sbml/SBMLDocument.h"
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

using namespace testing;
using namespace std;
using namespace ls;
using namespace rr;
using std::filesystem::path;

class StateSavingTests : public RoadRunnerTest {
public:
    path stateSavingModelsDir = rrTestModelsDir_ / "StateSaving";
    path stateSavingOutputDir = rrTestDir_ / "StateSavingOutput";

    StateSavingTests() = default;

    ~StateSavingTests() override {
        if (fs::exists(stateSavingOutputDir)) {
            fs::remove_all(stateSavingOutputDir);
        }
    }

    bool RunStateSavingTest(void(*modification)(RoadRunner *), std::string version = "l2v4");

    bool RunStateSavingTest(int caseNumber, void(*modification)(RoadRunner *), std::string version = "l2v4");

    bool StateRunTestModelFromScratch(void(*generate)(RoadRunner *), std::string version = "l2v4");
};


bool StateSavingTests::RunStateSavingTest(void(*modification)(RoadRunner *), std::string version) {
    bool result(false);
    RoadRunner *rr;

    //Create instance..
    rr = new RoadRunner();

    string testName(UnitTest::GetInstance()->current_test_info()->name());
    string suiteName(UnitTest::GetInstance()->current_test_info()->test_suite_name());

    libsbml::SBMLDocument *doc;

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
        doc = reader.readSBML(fullPath.string());

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
        modification(rr);
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
        delete doc;
        return false;
    }

    delete rr;
    delete doc;
    return result;
}


bool StateSavingTests::RunStateSavingTest(int caseNumber, void(*modification)(RoadRunner *), std::string version) {
    bool result(false);
    RoadRunner *rr, *rr2, *rr3;

    //Create instance..
    rr = new RoadRunner();
    string testName(UnitTest::GetInstance()->current_test_info()->name());
    string suiteName(UnitTest::GetInstance()->current_test_info()->test_suite_name());

    //Setup environment
    libsbml::SBMLDocument *doc;

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

        simulation.UseEngine(rr);

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
        doc = reader.readSBML(fullPath);

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
        rr2 = new RoadRunner(*rr);
        int rr2id = rr2->getInstanceID();
        modification(rr2);
        EXPECT_EQ(rr2->getInstanceID(), rr2id);
        rr3 = new RoadRunner(*rr2);
        EXPECT_NE(rr->getInstanceID(), rr2->getInstanceID());
        EXPECT_NE(rr->getInstanceID(), rr3->getInstanceID());
        EXPECT_NE(rr2->getInstanceID(), rr3->getInstanceID());
        simulation.UseEngine(rr3);
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
        result = simulation.Pass();
        result = simulation.SaveAllData() && result;
        result = simulation.SaveModelAsXML(dataOutputFolder) && result;
    }
    catch (std::exception &ex) {
        string error = ex.what();
        cerr << "Case " << caseNumber << ": Exception: " << error << endl;
        result = false;
    }

    delete rr;
    delete rr2;
    delete rr3;
    delete doc;
    return result;
}

bool StateSavingTests::StateRunTestModelFromScratch(void(*generate)(RoadRunner *), std::string version) {
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
        generate(&rr);
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
// on mac systems this test causes abort signal. We disable on mac.
#if (!defined(__APPLE__))
#  if LLVM_VERSION_PATCH > 1
    RoadRunner rri;
    EXPECT_THROW(rri.loadState((rrTestModelsDir_ / "wrong-save-state.rr").string()), std::exception);
#  endif
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

TEST_F(StateSavingTests, SAVE_STATE_1) {
    ASSERT_TRUE(RunStateSavingTest(1, [](RoadRunner *rri) {
        rri->saveState("save-state-test.rr");
        rri->loadState("save-state-test.rr");
    }));
}

TEST_F(StateSavingTests, SAVE_STATE_2) {
    ASSERT_TRUE(RunStateSavingTest(1, [](RoadRunner *rri) {
        rri->saveState("save-state-test.rr");
        rri->loadState("save-state-test.rr");
        rri->saveState("save-state-test.rr");
        rri->loadState("save-state-test.rr");
    }));
}

TEST_F(StateSavingTests, SAVE_STATE_3) {
    ASSERT_TRUE(RunStateSavingTest(1, [](RoadRunner *rri) {
        rri->loadState("save-state-test.rr");
    }));
}

TEST_F(StateSavingTests, SAVE_STATE_4) {
    ASSERT_TRUE(RunStateSavingTest(1, [](RoadRunner *rri) {
        rri->loadState("save-state-test.rr");
        rri->loadState("save-state-test.rr");
    }));
}

TEST_F(StateSavingTests, SAVE_STATE_5) {
    ASSERT_TRUE(RunStateSavingTest(1, [](RoadRunner *rri) {
        rri->loadState("save-state-test.rr");
        rri->saveState("save-state-test.rr");
        rri->loadState("save-state-test.rr");
    }));
}

TEST_F(StateSavingTests, SAVE_STATE_6) {
    ASSERT_TRUE(RunStateSavingTest(1121, [](RoadRunner *rri) {
        rri->saveState("save-state-test.rr");
        rri->loadState("save-state-test.rr");
    }, "l3v1"));
}

TEST_F(StateSavingTests, SAVE_STATE_7) {
    ASSERT_TRUE(RunStateSavingTest(1121, [](RoadRunner *rri) {
        rri->saveState("save-state-test.rr");
        rri->loadState("save-state-test.rr");
        rri->saveState("save-state-test.rr");
        rri->loadState("save-state-test.rr");
    }, "l3v1"));
}

TEST_F(StateSavingTests, SAVE_STATE_8) {
    ASSERT_TRUE(RunStateSavingTest(1121, [](RoadRunner *rri) {
        rri->loadState("save-state-test.rr");
    }, "l3v1"));
}

TEST_F(StateSavingTests, SAVE_STATE_9) {
    ASSERT_TRUE(RunStateSavingTest(1121, [](RoadRunner *rri) {
        rri->loadState("save-state-test.rr");
        rri->loadState("save-state-test.rr");
    }, "l3v1"));
}

TEST_F(StateSavingTests, SAVE_STATE_10) {
    ASSERT_TRUE(RunStateSavingTest(1121, [](RoadRunner *rri) {
        rri->loadState("save-state-test.rr");
        rri->saveState("save-state-test.rr");
        rri->loadState("save-state-test.rr");
    }, "l3v1"));
}

TEST_F(StateSavingTests, SAVE_STATE_11) {
    ASSERT_TRUE(RunStateSavingTest(1, [](RoadRunner *rri) {
        rri->simulate();
        rri->saveState("save-state-test.rr");
        rri->loadState("save-state-test.rr");
        rri->reset();
    }));
}

TEST_F(StateSavingTests, SAVE_STATE_12) {
    ASSERT_TRUE(RunStateSavingTest(1121, [](RoadRunner *rri) {
        rri->simulate();
        rri->saveState("save-state-test.rr");
        rri->loadState("save-state-test.rr");
        rri->reset(SelectionRecord::ALL);
    }, "l3v1"));
}

TEST_F(StateSavingTests, SAVE_STATE_13) {
    ASSERT_TRUE(RunStateSavingTest(1120, [](RoadRunner *rri) {
        rri->saveState("save-state-test.rr");
        rri->loadState("save-state-test.rr");
    }, "l3v1"));
}

TEST_F(StateSavingTests, SAVE_STATE_14) {
    ASSERT_TRUE(RunStateSavingTest(1120, [](RoadRunner *rri) {
        rri->saveState("save-state-test.rr");
        rri->loadState("save-state-test.rr");
        rri->saveState("save-state-test.rr");
        rri->loadState("save-state-test.rr");
    }, "l3v1"));
}

TEST_F(StateSavingTests, SAVE_STATE_15) {
    ASSERT_TRUE(RunStateSavingTest(1120, [](RoadRunner *rri) {
        rri->loadState("save-state-test.rr");
    }, "l3v1"));
}

TEST_F(StateSavingTests, SAVE_STATE_16) {
    ASSERT_TRUE(RunStateSavingTest(1120, [](RoadRunner *rri) {
        rri->loadState("save-state-test.rr");
        rri->loadState("save-state-test.rr");
    }, "l3v1"));
}

TEST_F(StateSavingTests, SAVE_STATE_17) {
    ASSERT_TRUE(RunStateSavingTest(1120, [](RoadRunner *rri) {
        rri->loadState("save-state-test.rr");
        rri->saveState("save-state-test.rr");
        rri->loadState("save-state-test.rr");
    }, "l3v1"));
}

TEST_F(StateSavingTests, SAVE_STATE_18) {
    ASSERT_TRUE(RunStateSavingTest(1120, [](RoadRunner *rri) {
        rri->simulate();
        rri->saveState("save-state-test.rr");
        rri->loadState("save-state-test.rr");
        rri->reset(SelectionRecord::ALL);
    }, "l3v1"));
}

TEST_F(StateSavingTests, SAVE_STATE_19) {
    ASSERT_TRUE(RunStateSavingTest([](RoadRunner *rri) {
        rri->getSimulateOptions().duration /= 2;
        rri->getSimulateOptions().steps /= 2;
        rri->simulate();
        rri->saveState("save-state-test.rr");
        rri->loadState("save-state-test.rr");
        rri->getSimulateOptions().start = rri->getSimulateOptions().duration;
    }, "l3v1"));
}

TEST_F(StateSavingTests, SAVE_STATE_20) {
    ASSERT_TRUE(RunStateSavingTest([](RoadRunner *rri) {
        rri->getSimulateOptions().duration /= 2;
        rri->getSimulateOptions().steps /= 2;
        rri->simulate();
        rri->saveState("save-state-test.rr");
        rri->loadState("save-state-test.rr");
        rri->getSimulateOptions().start = rri->getSimulateOptions().duration;
    }));
}

TEST_F(StateSavingTests, SAVE_STATE_21) {
    ASSERT_TRUE(RunStateSavingTest([](RoadRunner *rri) {
        rri->getSimulateOptions().duration = 1.50492;
        rri->getSimulateOptions().steps /= 2;
        rri->simulate();
        rri->saveState("save-state-test.rr");
        rri->loadState("save-state-test.rr");
        rri->getSimulateOptions().start = rri->getSimulateOptions().duration;
        rri->getSimulateOptions().duration = 3.0 - rri->getSimulateOptions().duration;
    }, "l3v1"));
}

TEST_F(StateSavingTests, RETAIN_ABSOLUTE_TOLERANCES_1) {
    RoadRunner rri((rrTestSbmlTestSuiteDir_ / "semantic" / "00001" / "00001-sbml-l2v4.xml").string());
    rri.getIntegrator()->setIndividualTolerance("S1", 5.0);
    rri.getIntegrator()->setIndividualTolerance("S2", 3.0);
    rri.saveState("save-state-test.rr");
    RoadRunner rri2;
    rri2.loadState("save-state-test.rr");
    EXPECT_EQ(rri2.getIntegrator()->getConcentrationTolerance()[0], 5.0);
    EXPECT_EQ(rri2.getIntegrator()->getConcentrationTolerance()[1], 3.0);
}

TEST_F(StateSavingTests, FROM_SCRATCH_1) {
    ASSERT_TRUE(StateRunTestModelFromScratch([](RoadRunner *rri) {
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
    ASSERT_TRUE(StateRunTestModelFromScratch([](RoadRunner *rri) {
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
    ASSERT_TRUE(StateRunTestModelFromScratch([](RoadRunner *rri) {
        rri->addCompartment("compartment", 1);
        rri->addSpeciesConcentration("S1", "compartment", 0, true, true);
        rri->addRateRule("S1", "7");

        rri->saveState("test-save-state.rr");
        rri->loadState("test-save-state.rr");
    }));
}

TEST_F(StateSavingTests, FROM_SCRATCH_4) {
    ASSERT_TRUE(StateRunTestModelFromScratch([](RoadRunner *rri) {
        rri->addCompartment("compartment", 1);

        rri->saveState("test-save-state.rr");

        rri->addSpeciesConcentration("S1", "compartment", 7, true);

        rri->loadState("test-save-state.rr");

        rri->addSpeciesConcentration("S1", "compartment", 7, true, true);
        rri->addAssignmentRule("S1", "7");
    }));
}

TEST_F(StateSavingTests, FROM_SCRATCH_5) {
    ASSERT_TRUE(StateRunTestModelFromScratch([](RoadRunner *rri) {
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
    ASSERT_TRUE(StateRunTestModelFromScratch([](RoadRunner *rri) {
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
    ASSERT_TRUE(StateRunTestModelFromScratch([](RoadRunner *rri) {
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
    ASSERT_TRUE(StateRunTestModelFromScratch([](RoadRunner *rri) {
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
    ASSERT_TRUE(StateRunTestModelFromScratch([](RoadRunner *rri) {
        rri->addCompartment("compartment", 1);
        rri->addSpeciesConcentration("S1", "compartment", 0, true, true);
        rri->addRateRule("S1", "7");

        rri->saveState("test-save-state.rr");
        rri->clearModel();
        rri->loadState("test-save-state.rr");
    }));
}

TEST_F(StateSavingTests, LOAD_ON_NEW_INSTANCE) {
    RoadRunner rri;
    rri.loadState("test-save-state.rr");
    ASSERT_TRUE(rri.getBoundarySpeciesByIndex(0) == 0);
}


class ForSerializationAsBinary {
public:
    ForSerializationAsBinary() = default;

    explicit ForSerializationAsBinary(int number)
            : number_(number) {}

    std::stringstream toBinaryStream() {
        std::stringstream out(std::ios::binary);
        out.write((const char *) &number_, sizeof(int));
        return out;
    }

    static void fromBinaryStream(ForSerializationAsBinary &obj, std::stringstream &os) {
        int n;
        os.read((char *) &n, sizeof(int));
        std::cout << "n: " << n << std::endl;
        obj.number_ = n;
    }

    int number_;
};

TEST(Serialisation, SimpleSerialization) {
    ForSerializationAsBinary serializationAsBinary(4);
    auto o = serializationAsBinary.toBinaryStream();
    ForSerializationAsBinary loaded;
    ForSerializationAsBinary::fromBinaryStream(loaded, o);
    std::cout << "loaded.number_: " << loaded.number_ << std::endl;

}

TEST_F(StateSavingTests, FromString) {
    RoadRunner rr(OpenLinearFlux().str());
    std::stringstream* stateStream = rr.saveStateS('b');
    RoadRunner rr2;
    rr2.loadStateS(stateStream);
    auto actualDataLsMatrix = *rr2.simulate(0, 10, 11);
    auto actualData = rr::Matrix<double>(actualDataLsMatrix); // for almostEquals
    auto expectedData = OpenLinearFlux().timeSeriesResult();
    ASSERT_TRUE(expectedData.almostEquals(actualData, 1e-4));
}

TEST_F(StateSavingTests, FromFile) {
    std::filesystem::path p = std::filesystem::current_path() / "savedState.rr";
    std::cout << "saved to " << p << std::endl;
    RoadRunner rr(OpenLinearFlux().str());
    rr.saveState(p.string());
    RoadRunner rr2;
    rr2.loadState(p.string());
    auto actualDataLsMatrix = *rr2.simulate(0, 10, 11);
    auto actualData = rr::Matrix<double>(actualDataLsMatrix); // for almostEquals
    auto expectedData = OpenLinearFlux().timeSeriesResult();
    ASSERT_TRUE(expectedData.almostEquals(actualData, 1e-4));
}















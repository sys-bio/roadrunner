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

using namespace testing;
using namespace rr;
using namespace std;
using std::filesystem::path;


class ModelEditingTests : public RoadRunnerTest {
public:

    path modelEditingTestOutputDir;

    ModelEditingTests() {
        modelEditingTestOutputDir = rrTestDir_ / "modelEditingTestOutputDir";
    }

    ~ModelEditingTests() override {
        if (fs::exists(modelEditingTestOutputDir)) {
            fs::remove_all(modelEditingTestOutputDir);
        }
    }

    bool RunModelEditingTest(void(*modification)(RoadRunner *), std::string version = "l2v4");

    bool RunTestModelFromScratch(void(*generate)(RoadRunner *), std::string version = "l2v4");

    bool RunTestWithEdit(const string &version, int caseNumber, void(*edit)(RoadRunner *, libsbml::SBMLDocument *),
                         std::string editName);

    static void readdAllReactions(RoadRunner *rri, libsbml::SBMLDocument *doc);

    static void readdAllSpecies(RoadRunner *rri, libsbml::SBMLDocument *doc);

    static void readdAllEvents(RoadRunner *rri, libsbml::SBMLDocument *doc);

    static void removeAndReaddAllSpecies(RoadRunner *rri, libsbml::SBMLDocument *doc);

    static void removeAndReaddAllReactions(RoadRunner *rri, libsbml::SBMLDocument *doc);

    static void removeAndReaddAllCompartments(RoadRunner *rri, libsbml::SBMLDocument *doc);


};


bool ModelEditingTests::RunModelEditingTest(void(*modification)(RoadRunner *), std::string version) {
    bool result(false);
    RoadRunner rr;
    UnitTest::GetInstance()->current_test_info()->test_case_name(); // doing anything??

    string testName(UnitTest::GetInstance()->current_test_info()->name());
    string suiteName(UnitTest::GetInstance()->current_test_info()->test_suite_name());

    libsbml::SBMLDocument *doc = nullptr;

    try {
        rr.getIntegrator()->setValue("stiff", false);

        if (!createFolder(modelEditingTestOutputDir.string())) {
            string msg("Failed creating output folder for data output: " + modelEditingTestOutputDir.string());
            throw (rr::Exception(msg));
        }
        //Create subfolder for data output
        path dataOutputFolder = modelEditingTestOutputDir / testName;

        if (!createFolder(dataOutputFolder.string())) {
            string msg("Failed creating output folder for data output: " + dataOutputFolder.string());
            throw (rr::Exception(msg));
        }

        TestSuiteModelSimulation simulation(dataOutputFolder);

        simulation.UseEngine(&rr);

        //Read SBML models.....
        simulation.SetCaseNumber(0);

        // fs::absolute normalizes fwd and bwd slashes on windows
        path modelFilePath = fs::absolute(rrTestModelsDir_ / suiteName / testName);
        string modelFileName = testName + "-sbml-" + version + ".xml";
        path settingsFileName = testName + "-settings.txt";

        //The following will load and compile and simulate the sbml model in the file
        simulation.SetModelFilePath(modelFilePath);
        simulation.SetModelFileName(modelFileName);
        simulation.ReCompileIfDllExists(true);
        simulation.CopyFilesToOutputFolder();
        rr.setConservedMoietyAnalysis(false);

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


        rr.load(fullPath.string(), &opt);

        //Then read settings file if it exists..
        if (!simulation.LoadSettings(modelFilePath / settingsFileName)) {
            throw (Exception("Failed loading simulation settings"));
        }
        modification(&rr);
        //Then Simulate model
        if (!simulation.Simulate()) {
            throw (Exception("Failed running simulation"));
        }

        //Write result
        if (!simulation.SaveResult()) {
            //Failed to save data
            throw (Exception("Failed saving result"));
        }

        if (!simulation.LoadReferenceData(modelFilePath / (testName + "-results.csv"))) {
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
        if (doc) {
            delete doc;
        }
        return false;
    }

    delete doc;

    return result;
}

bool ModelEditingTests::RunTestModelFromScratch(void(*generate)(RoadRunner *), std::string version) {
    bool result(false);
    int level = version.at(1) - '0';
    int versionNum = version.at(3) - '0';
    RoadRunner rr(level, versionNum);


    string testName(UnitTest::GetInstance()->current_test_info()->name());
    string suiteName(UnitTest::GetInstance()->current_test_info()->test_suite_name());

    try {
        rr.getIntegrator()->setValue("stiff", false);

        if (!createFolder(modelEditingTestOutputDir.string())) {
            string msg("Failed creating output folder for data output: " + modelEditingTestOutputDir.string());
            throw (rr::Exception(msg));
        }
        //Create subfolder for data output
        path dataOutputFolder = modelEditingTestOutputDir / testName;

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
        path settingsFileName = testName + "-settings.txt";

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
        if (!simulation.LoadSettings(modelFilePath / settingsFileName)) {
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

        if (!simulation.LoadReferenceData(modelFilePath / (testName + "-results.csv"))) {
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

bool ModelEditingTests::RunTestWithEdit(const string &version, int caseNumber,
                                        void(*edit)(RoadRunner *, libsbml::SBMLDocument *),
                                        std::string editName) {
    //rrLog(Logger::LOG_NOTICE) << "Running Test: " << editName << ", " << caseNumber << endl;
    bool result(false);
    RoadRunner rr;

    //Setup environment
    //setTempFolder(gRR, gTempFolder.c_str());
    libsbml::SBMLDocument *doc;

    try {
        path dataOutputFolder = modelEditingTestOutputDir / editName;
        string dummy;
        string logFileName;
        string settingsFileName;

        rr.getIntegrator()->setValue("stiff", false);

        //Create a log file name
        createTestSuiteFileNameParts(caseNumber, ".log", dummy, logFileName, settingsFileName, dummy);
        if (!createFolder(dataOutputFolder.string())) {
            string msg("Failed creating output folder for data output: " + dataOutputFolder.string());
            throw (rr::Exception(msg));
        }
        //Create subfolder for data output
        dataOutputFolder = dataOutputFolder / getTestSuiteSubFolderName(caseNumber);

        if (!createFolder(dataOutputFolder.string())) {
            string msg("Failed creating output folder for data output: " + dataOutputFolder.string());
            throw (rr::Exception(msg));
        }

        TestSuiteModelSimulation simulation(dataOutputFolder);

        simulation.UseEngine(&rr);

        //Read SBML models.....
        string modelFilePath = (rrTestDir_ / "sbml-test-suite/semantic/").string();
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
        rr.setConservedMoietyAnalysis(false);

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


        rr.load(fullPath, &opt);

        //Then read settings file if it exists..
        string settingsOveride("");
        if (!simulation.LoadSettings(settingsOveride)) {
            throw (Exception("Failed loading simulation settings"));
        }

        //Perform the model editing action
        edit(&rr, doc);
        //Then Simulate model
        if (!simulation.Simulate()) {
            throw (Exception("xFailed running simulation"));
        }

        //Write result
        if (!simulation.SaveResult()) {
            //Failed to save data
            throw (Exception("Failed saving result"));
        }

        if (!simulation.LoadReferenceData()) {
            throw (Exception("Failed Loading reference data"));
        }

        result = true;

        simulation.CreateErrorData();
        result = simulation.Pass();
        result = simulation.SaveAllData() && result;
        result = simulation.SaveModelAsXML(dataOutputFolder) && result;
        result = testValidateSBML(rr.getCurrentSBML()) && result;

    }
    catch (std::exception &ex) {
        string error = ex.what();
        cerr << "Case " << caseNumber << ": Exception: " << error << endl;
        std::cout << "Case " << caseNumber << ": Exception: " << error << endl;
        delete doc;
        return false;
    }

    delete doc;

    return result;
}

void ModelEditingTests::readdAllReactions(RoadRunner *rri, libsbml::SBMLDocument *doc) {
    libsbml::ListOfReactions *reactionsToAdd = doc->getModel()->getListOfReactions();
    std::vector<std::string> currReactionIds = rri->getReactionIds();
    for (int i = 0; i < reactionsToAdd->size(); i++) {
        libsbml::Reaction *next = reactionsToAdd->get(i);
        if (std::find(currReactionIds.begin(), currReactionIds.end(), next->getId()) ==
            currReactionIds.end()) {
            char *nextSBML = next->toSBML();
            rri->addReaction(nextSBML);
            free(nextSBML);
        }
    }
}

void ModelEditingTests::readdAllSpecies(RoadRunner *rri, libsbml::SBMLDocument *doc) {
    libsbml::ListOfSpecies *speciesToAdd = doc->getModel()->getListOfSpecies();
    std::vector<std::string> currSpeciesIds = rri->getBoundarySpeciesIds();

    for (std::string s : rri->getFloatingSpeciesIds())
        currSpeciesIds.push_back(s);

    libsbml::Species *next;
    for (int i = 0; i < speciesToAdd->size(); i++) {
        next = speciesToAdd->get(i);
        if (std::find(currSpeciesIds.begin(), currSpeciesIds.end(), next->getId()) == currSpeciesIds.end()) {
            rri->addSpeciesAmount(next->getId(), next->getCompartment(), next->getInitialAmount(),
                                  next->getHasOnlySubstanceUnits(), next->getBoundaryCondition(),
                                  next->getSubstanceUnits());
        }
    }
}

void ModelEditingTests::readdAllEvents(RoadRunner *rri, libsbml::SBMLDocument *doc) {
    libsbml::ListOfEvents *eventsToAdd = doc->getModel()->getListOfEvents();

    libsbml::Event *next;
    for (int i = 0; i < eventsToAdd->size(); i++) {
        next = eventsToAdd->get(i);
        char *L3_str = libsbml::SBML_formulaToL3String(next->getTrigger()->getMath());
        rri->addEvent(next->getId(), next->getUseValuesFromTriggerTime(), L3_str);
        free(L3_str);
        if (next->isSetDelay()) {
            L3_str = libsbml::SBML_formulaToL3String(next->getDelay()->getMath());
            rri->addDelay(next->getId(), L3_str);
            free(L3_str);
        }
        libsbml::ListOfEventAssignments *nextAssignments = next->getListOfEventAssignments();
        for (int j = 0; j < nextAssignments->size(); j++) {
            libsbml::EventAssignment *nextEA = nextAssignments->get(j);
            L3_str = libsbml::SBML_formulaToL3String(nextEA->getMath());
            rri->addEventAssignment(next->getId(), nextEA->getVariable(), L3_str);
            free(L3_str);
        }
    }
}

/*
* Pre: the model must not contain any rules reliant on species values
*/
void ModelEditingTests::removeAndReaddAllSpecies(RoadRunner *rri, libsbml::SBMLDocument *doc) {
    //Remove all species
    std::vector<std::string> floatingSpeciesIds = rri->getFloatingSpeciesIds();
    for (std::string sid : floatingSpeciesIds) {
        rri->removeSpecies(sid);
    }

    std::vector<std::string> boundarySpeciesIds = rri->getBoundarySpeciesIds();
    for (std::string sid : boundarySpeciesIds) {
        rri->removeSpecies(sid);
    }

    readdAllSpecies(rri, doc);

    readdAllReactions(rri, doc);

    readdAllEvents(rri, doc);
}

void ModelEditingTests::removeAndReaddAllReactions(RoadRunner *rri, libsbml::SBMLDocument *doc) {
    std::vector<std::string> reactionIds = rri->getReactionIds();
    for (std::string rid : reactionIds) {
        rri->removeReaction(rid);
    }
    readdAllReactions(rri, doc);
}

void ModelEditingTests::removeAndReaddAllCompartments(RoadRunner *rri, libsbml::SBMLDocument *doc) {
    std::vector<std::string> compartmentIds = rri->getCompartmentIds();
    for (std::string cid : compartmentIds) {
        rri->removeCompartment(cid, false);
    }

    libsbml::ListOfCompartments *compartmentsToAdd = doc->getModel()->getListOfCompartments();
    for (int i = 0; i < compartmentsToAdd->size(); i++) {
        libsbml::Compartment *next = compartmentsToAdd->get(i);
        rri->addCompartment(next->getId(), next->getVolume(), true);
    }
    readdAllSpecies(rri, doc);
    readdAllReactions(rri, doc);
    readdAllEvents(rri, doc);
}


TEST_F(ModelEditingTests, SET_BOUNDARY_INIT) {
    RoadRunner rri;
    //Logger::setLevel(Logger::LOG_DEBUG);
    rri.addCompartment("compartment", 3.5, false);
    //Boundary species:
    rri.addSpeciesConcentration("S2", "compartment", 1.0, false, true);
    double x = rri.getValue("init(S2)");
    EXPECT_EQ(x, 3.5);
    x = rri.getValue("init([S2])");
    EXPECT_EQ(x, 1.0);

    //Now set the initial amount to something new:
    rri.setValue("init(S2)", 5.0);
    x = rri.getValue("init(S2)");
    EXPECT_EQ(x, 5.0);
    x = rri.getValue("init([S2])");
    EXPECT_EQ(x, 5.0/3.5);

    //Now set the initial concentration to something new:
    rri.setValue("init([S2])", 5.5);
    x = rri.getValue("init(S2)");
    EXPECT_EQ(x, 5.5*3.5);
    x = rri.getValue("init([S2])");
    EXPECT_EQ(x, 5.5);

    //Boundary species:
    rri.addSpeciesAmount("S3", "compartment", 1.0, true, true);
    rri.setValue("init(S3)", 6.0);
    x = rri.getValue("init(S3)");
    EXPECT_EQ(x, 6.0);
    //Floating species:
    rri.addSpeciesConcentration("S1", "compartment", 1.0, false, false);
    rri.setValue("init(S1)", 7.0);
    x = rri.getValue("init(S1)");
    EXPECT_EQ(x, 7.0);
}


TEST_F(ModelEditingTests, CLEAR_MODEL_1) {
    RoadRunner rri;
    rri.addCompartment("compartment", 3.14159);
    rri.addSpeciesConcentration("S1", "compartment", 1.0, false, false);
    ASSERT_TRUE(rri.isModelLoaded());
    rri.clearModel();
    ASSERT_TRUE(!rri.isModelLoaded());
    rri.addCompartment("compartment", 3.14159);
    rri.addSpeciesConcentration("S2", "compartment", 2.0, false, false);
    ASSERT_TRUE(rri.getNumberOfFloatingSpecies() == 1);
}

TEST_F(ModelEditingTests, ALLOW_EVENT_ASSIGNMENT_AND_RATE_LAW_1) {
    try {
        RoadRunner rri;
        rri.addParameter("k1", 0.0);
        rri.addRateRule("k1", "1");
        rri.addEvent("e1", true, "k1 > 1");
        rri.addEventAssignment("e1", "k1", "1");
    }
    catch (std::exception &ex) {
        string error = ex.what();
        cerr << "ALLOW_EVENT_ASSIGNMENT_AND_RATE_LAW_1 Exception: " << error << endl;
        ASSERT_TRUE(false);
    }
}

TEST_F(ModelEditingTests, ALLOW_EVENT_ASSIGNMENT_AND_RATE_LAW_2) {
    try {
        RoadRunner rri;
        rri.addParameter("k1", 0.0);
        rri.addEvent("e1", true, "k1 > 1");
        rri.addEventAssignment("e1", "k1", "1");
        rri.addRateRule("k1", "1");
    }
    catch (std::exception &ex) {
        string error = ex.what();
        cerr << "ALLOW_EVENT_ASSIGNMENT_AND_RATE_LAW_2 Exception: " << error << endl;
        ASSERT_TRUE(false);
    }
}

TEST_F(ModelEditingTests, ALLOW_EVENT_ASSIGNMENT_AND_INITIAL_ASSIGNMENT_1) {
    try {
        RoadRunner rri;
        rri.addParameter("k1", 0.0);
        rri.addRateRule("k1", "1");
        //rri.addInitialAssignment("k1", "1");
    }
    catch (std::exception &ex) {
        string error = ex.what();
        EXPECT_TRUE(false) << "ALLOW_EVENT_ASSIGNMENT_AND_RATE_LAW_1 Exception: " << error << endl;
        cerr << "ALLOW_EVENT_ASSIGNMENT_AND_RATE_LAW_1 Exception: " << error << endl;
        ASSERT_TRUE(false);
    }
}

TEST_F(ModelEditingTests, ALLOW_EVENT_ASSIGNMENT_AND_INITIAL_ASSIGNMENT_2) {
    try {
        RoadRunner rri;
        rri.addParameter("k1", 0.0);
        //rri.addInitialAssignment("k1", "1");
        rri.addRateRule("k1", "1");
    }
    catch (std::exception &ex) {
        string error = ex.what();
        cerr << "ALLOW_EVENT_ASSIGNMENT_AND_RATE_LAW_2 Exception: " << error << endl;
        ASSERT_TRUE(false);
    }
}

TEST_F(ModelEditingTests, SET_CONSERVED_MOIETY_ANALYSIS) {
    RoadRunner rri((rrTestDir_ / "sbml-test-suite/semantic/00001/00001-sbml-l2v4.xml").string());
    rri.setGlobalParameterByIndex(0, 100);
    rri.setConservedMoietyAnalysis(true);

    ASSERT_TRUE(rri.getGlobalParameterByIndex(0) == 100);
}

TEST_F(ModelEditingTests, REMOVE_ASSIGNMENT_RULE_2) {
    ASSERT_TRUE(RunModelEditingTest([](RoadRunner *rri) {
        rri->removeRules("k1", true);
    }));
}

TEST_F(ModelEditingTests, REMOVE_ASSIGNMENT_RULE_3) {
    ASSERT_TRUE(RunModelEditingTest([](RoadRunner *rri) {
        rri->removeRules("k1", false);
    }));
}

TEST_F(ModelEditingTests, REMOVE_RATE_RULE_1) {
    ASSERT_TRUE(RunModelEditingTest([](RoadRunner *rri) {
        rri->removeRules("k1", false);
    }));
}

TEST_F(ModelEditingTests, REMOVE_RATE_RULE_2) {
    ASSERT_TRUE(RunModelEditingTest([](RoadRunner *rri) {
        rri->removeRules("k1", true);
    }));
}

TEST_F(ModelEditingTests, REMOVE_RATE_RULE_3) {
    ASSERT_TRUE(RunModelEditingTest([](RoadRunner *rri) {
        rri->simulate();
        rri->removeRules("k1", false);
    }));
}

TEST_F(ModelEditingTests, REMOVE_RATE_RULE_4) {
    ASSERT_TRUE(RunModelEditingTest([](RoadRunner *rri) {
        rri->simulate();
        rri->removeRules("k1", true);
    }));
}

TEST_F(ModelEditingTests, REMOVE_INITIAL_ASSIGNMENT_RULE_1) {
    ASSERT_TRUE(RunModelEditingTest([](RoadRunner *rri) {
        rri->removeInitialAssignment("k1");
    }));
}

TEST_F(ModelEditingTests, REMOVE_INITIAL_ASSIGNMENT_RULE_2) {
    ASSERT_TRUE(RunModelEditingTest([](RoadRunner *rri) {
        rri->simulate();
        rri->removeInitialAssignment("k1");
    }));
}

TEST_F(ModelEditingTests, ADD_REACTION_1) {
    ASSERT_TRUE(RunModelEditingTest([](RoadRunner *rri) {
        rri->addReaction("reaction2", {"S2"}, {"S1"}, "k1*S2", true);
    }));
}

TEST_F(ModelEditingTests, REMOVE_REACTION_1) {
    ASSERT_TRUE(RunModelEditingTest([](RoadRunner *rri) {
        rri->removeReaction("reaction2");
    }));
}

TEST_F(ModelEditingTests, ADD_REACTION_2) {
    ASSERT_TRUE(RunModelEditingTest([](RoadRunner *rri) {
        rri->addReaction("reaction2", {"2S1", "S2"}, {"2S2"}, "compartment * k1 * S1 + compartment * k1 * S2", true);
    }));
}

TEST_F(ModelEditingTests, REMOVE_REACTION_2) {
    ASSERT_TRUE(RunModelEditingTest([](RoadRunner *rri) {
        rri->removeReaction("reaction2");
    }));
}

TEST_F(ModelEditingTests, REMOVE_REACTION_3) {
    ASSERT_TRUE(RunModelEditingTest([](RoadRunner *rri) {
        rri->removeSpecies("S2");
    }));
}

TEST_F(ModelEditingTests, ADD_SPECIES_1) {
    ASSERT_TRUE(RunModelEditingTest([](RoadRunner *rri) {
        rri->addSpeciesConcentration("S3", "compartment", 0.0015, true, false);
    }));
}

TEST_F(ModelEditingTests, ADD_REACTION_3) {
    ASSERT_TRUE(RunModelEditingTest([](RoadRunner *rri) {
        rri->addSpeciesConcentration("S3", "compartment", 0.015, false, false);
        rri->addReaction("reaction3", {"S2"}, {"S3"}, "k2*S2");
    }));
}

TEST_F(ModelEditingTests, ADD_REACTION_4) {
    ASSERT_TRUE(RunModelEditingTest([](RoadRunner *rri) {
        rri->addSpeciesConcentration("S3", "compartment", 0.015, false);
        rri->addReaction("reaction3", {"S2"}, {"S3"}, "k2*S2");
    }));
}

TEST_F(ModelEditingTests, ADD_REACTION_5) {
    ASSERT_TRUE(RunModelEditingTest([](RoadRunner *rri) {
        rri->addSpeciesConcentration("S3", "compartment", 0.15, false, false);
        rri->addReaction("reaction3", {"S3"}, {"S1"}, "k2*S3");
    }));
}

TEST_F(ModelEditingTests, ADD_REACTION_6) {
    ASSERT_TRUE(RunModelEditingTest([](RoadRunner *rri) {
        rri->addReaction("reaction3", {"S3"}, {"S4"}, "C * k3 * S3 * time");
    }));
}

TEST_F(ModelEditingTests, REMOVE_SPECIES_1) {
    ASSERT_TRUE(RunModelEditingTest([](RoadRunner *rri) {
        rri->removeSpecies("S2", false);
        rri->addSpeciesConcentration("S3", "compartment", 0.00030, false, false);
        rri->addReaction("reaction1", {"S1"}, {"S3"}, "k1*S1", false);
        rri->addReaction("reaction2", {"S3"}, {"S1"}, "k2*S3", true);
    }));
}

TEST_F(ModelEditingTests, REMOVE_SPECIES_2) {
    ASSERT_TRUE(RunModelEditingTest([](RoadRunner *rri) {
        rri->removeSpecies("S4");
    }));
}

TEST_F(ModelEditingTests, ADD_EVENT_1) {
    ASSERT_TRUE(RunModelEditingTest([](RoadRunner *rri) {
        rri->addEvent("event1", true, "S1 > 0.00015", false);
        rri->addEventAssignment("event1", "S1", "1", true);
    }));
}

TEST_F(ModelEditingTests, EVENT_PRIORITY_1) {
    ASSERT_TRUE(RunModelEditingTest([](RoadRunner *rri) {
        rri->addEvent("_E0", true, "time >= 0.99", false);
        rri->addPriority("_E0", "1", false);
        rri->addEventAssignment("_E0", "S1", "4", false);
        rri->addEventAssignment("_E0", "S2", "5", false);
        rri->addEventAssignment("_E0", "S3", "6", false);

        rri->addEvent("_E1", true, "time >= 0.99", false);
        rri->addPriority("_E1", "0", false);
        rri->addEventAssignment("_E1", "S1", "1", false);
        rri->addEventAssignment("_E1", "S2", "2", false);
        rri->addEventAssignment("_E1", "S3", "3", true);
    }, "l3v1"));
}

TEST_F(ModelEditingTests, EVENT_DELAY_1) {
    ASSERT_TRUE(RunModelEditingTest([](RoadRunner *rri) {
        rri->addDelay("event1", "1");
    }));
}

TEST_F(ModelEditingTests, REMOVE_EVENT_1) {
    ASSERT_TRUE(RunModelEditingTest([](RoadRunner *rri) {
        rri->removeEvent("event1");
    }));
}

TEST_F(ModelEditingTests, ADD_PARAMETER_1) {
    ASSERT_TRUE(RunModelEditingTest([](RoadRunner *rri) {
        rri->addParameter("k1", 0.75, false);
        rri->addReaction("reaction1", {"S1", "S2"}, {"S3"}, "k1*S1*S2");
    }));
}

TEST_F(ModelEditingTests, ADD_COMPARTMENT_1) {
    ASSERT_TRUE(RunModelEditingTest([](RoadRunner *rri) {
        rri->addCompartment("compartment", 1, false);
        rri->addReaction("reaction1", {"S1", "S2"}, {"2S2"}, "compartment * k1 * S1 * S2", true);
    }));
}

TEST_F(ModelEditingTests, ADD_COMPARTMENT_2) {
    ASSERT_TRUE(RunModelEditingTest([](RoadRunner *rri) {
        rri->addCompartment("compartment", 1, false);
        rri->addReaction("reaction1", {"S1"}, {"S3"}, "compartment * k1 * S1");
    }));
}

TEST_F(ModelEditingTests, ADD_ASSIGNMENT_RULE_1) {
    ASSERT_TRUE(RunModelEditingTest([](RoadRunner *rri) {
        rri->setBoundary("S1", true);
        rri->addAssignmentRule("S1", "7");
    }));
}

TEST_F(ModelEditingTests, ADD_ASSIGNMENT_RULE_2) {
    ASSERT_TRUE(RunModelEditingTest([](RoadRunner *rri) {
        rri->setBoundary("S1", true);
        rri->addAssignmentRule("S1", "7");
    }));
}

TEST_F(ModelEditingTests, ADD_ASSIGNMENT_RULE_3) {
    ASSERT_TRUE(RunModelEditingTest([](RoadRunner *rri) {
        rri->setBoundary("S3", true);
        rri->addAssignmentRule("S3", "k1 * S2");
    }));
}

TEST_F(ModelEditingTests, ADD_RATE_RULE_1) {
    ASSERT_TRUE(RunModelEditingTest([](RoadRunner *rri) {
        rri->setBoundary("S1", true);
        rri->addRateRule("S1", "7");
    }));
}


TEST_F(ModelEditingTests, ADD_RATE_RULE_2) {
    ASSERT_TRUE(RunModelEditingTest([](RoadRunner *rri) {
        rri->setBoundary("S1", true);
        rri->setBoundary("S2", true);
        rri->addRateRule("S1", "-1 * k1 * S1", false);
        rri->addRateRule("S2", "k1 * S1", true);
    }));
}

TEST_F(ModelEditingTests, ADD_RATE_RULE_3) {
    ASSERT_TRUE(RunModelEditingTest([](RoadRunner *rri) {
        rri->setBoundary("S1", true);
        rri->setBoundary("S2", true);
        rri->addRateRule("S1", "-1 * k1 * S1");
        rri->addRateRule("S2", "k1 * S1");
    }));
}


TEST_F(ModelEditingTests, SET_KINETIC_LAW_1) {
    ASSERT_TRUE(RunModelEditingTest([](RoadRunner *rri) {
        rri->setKineticLaw("reaction2", "compartment * k2 * S3");
        ASSERT_TRUE(rri->getKineticLaw("reaction2") == "compartment * k2 * S3");
    }));
}

TEST_F(ModelEditingTests, SET_KINETIC_LAW_2) {
    ASSERT_TRUE(RunModelEditingTest([](RoadRunner *rri) {
        rri->setKineticLaw("reaction1", "compartment * k1 * S1 * S2");
        rri->setKineticLaw("reaction2", "compartment * k2 * S3 * S4");
        ASSERT_TRUE(rri->getKineticLaw("reaction1") == "compartment * k1 * S1 * S2");
        ASSERT_TRUE(rri->getKineticLaw("reaction2") == "compartment * k2 * S3 * S4");
    }));
}

TEST_F(ModelEditingTests, SET_KINETIC_LAW_3) {
    ASSERT_TRUE(RunModelEditingTest([](RoadRunner *rri) {
        rri->setKineticLaw("reaction1", "compartment * k1 * S1 * S2", false);
        rri->setKineticLaw("reaction2", "compartment * k2 * S3 * S4", true);
        ASSERT_TRUE(rri->getKineticLaw("reaction1") == "compartment * k1 * S1 * S2");
        ASSERT_TRUE(rri->getKineticLaw("reaction2") == "compartment * k2 * S3 * S4");
    }));
}

TEST_F(ModelEditingTests, TRANSFORM_1) {
    ASSERT_TRUE(RunModelEditingTest([](RoadRunner *rri) {
        rri->removeSpecies("S1", true);
        rri->removeSpecies("S2", true);
        rri->addSpeciesConcentration("S1", "compartment", 0.001);
        rri->addSpeciesConcentration("S2", "compartment", 0.001);
        rri->addSpeciesConcentration("S3", "compartment", 0.002);
        rri->addSpeciesConcentration("S4", "compartment", 0.001);
        rri->removeParameter("k1", true);
        rri->addParameter("k1", 750, true);
        rri->addParameter("k2", 250, true);
        rri->addReaction("reaction1", {"S1", "S2"}, {"S3", "S4"}, "compartment * k1 * S1 * S2", true);
        rri->addReaction("reaction2", {"S3", "S4"}, {"S1", "S2"}, "compartment * k2 * S3 * S4", true);
    }));
}

TEST_F(ModelEditingTests, PAUSE_1) {
    ASSERT_TRUE(RunModelEditingTest([](RoadRunner *rri) {
        rri->simulate();
        rri->addReaction("reaction3", {"S3"}, {"S2"}, "compartment * k2 * S3 * S4");
    }));
}

TEST_F(ModelEditingTests, PAUSE_2) {
    ASSERT_TRUE(RunModelEditingTest([](RoadRunner *rri) {
        rri->simulate();
        rri->removeReaction("reaction2");
    }));
}

TEST_F(ModelEditingTests, PAUSE_3) {
    ASSERT_TRUE(RunModelEditingTest([](RoadRunner *rri) {
        rri->simulate();
        rri->addSpeciesConcentration("S5", "compartment", 0.001, true);
    }));
}

TEST_F(ModelEditingTests, PAUSE_4) {
    ASSERT_TRUE(RunModelEditingTest([](RoadRunner *rri) {
        rri->simulate();
        rri->removeSpecies("S4");
    }));
}

TEST_F(ModelEditingTests, PAUSE_5) {
    ASSERT_TRUE(RunModelEditingTest([](RoadRunner *rri) {
        rri->simulate();
        rri->setKineticLaw("reaction3", "sin(S2)");
    }));
}

TEST_F(ModelEditingTests, PAUSE_6) {
    ASSERT_TRUE(RunModelEditingTest([](RoadRunner *rri) {
        rri->simulate();
        rri->addEvent("event1", true, "S2 > 0.0004", false);
        rri->addEventAssignment("event1", "S1", "0.1", true);
    }));
}

TEST_F(ModelEditingTests, PAUSE_7) {
    ASSERT_TRUE(RunModelEditingTest([](RoadRunner *rri) {
        rri->simulate();
        rri->removeEvent("event1");
    }));
}

TEST_F(ModelEditingTests, PAUSE_8) {
    ASSERT_TRUE(RunModelEditingTest([](RoadRunner *rri) {
        rri->simulate();
        rri->addRateRule("k1", "7");
    }));
}

TEST_F(ModelEditingTests, PAUSE_9) {
    ASSERT_TRUE(RunModelEditingTest([](RoadRunner *rri) {
        rri->simulate();
        rri->removeRules("k1");
    }));
}

TEST_F(ModelEditingTests, PAUSE_10) {
    ASSERT_TRUE(RunModelEditingTest([](RoadRunner *rri) {
        rri->simulate();
        rri->addDelay("event1", "0.2");
    }));
}


TEST_F(ModelEditingTests, ADD_EVENT_ASSIGNMENT_1) {
    ASSERT_TRUE(RunModelEditingTest([](RoadRunner *rri) {
        rri->addEventAssignment("event2", "S3", "1");
    }));
}

TEST_F(ModelEditingTests, REMOVE_RULES_2) {
    ASSERT_TRUE(RunModelEditingTest([](RoadRunner *rri) {
        rri->simulate();
        rri->removeRules("S4");
    }));
}

TEST_F(ModelEditingTests, ADD_EVENT_ASSIGNMENT_2) {
    ASSERT_TRUE(RunModelEditingTest([](RoadRunner *rri) {
        rri->simulate();
        rri->addEventAssignment("event1", "S3", "1");
    }));
}

TEST_F(ModelEditingTests, ADD_RATE_RULE_PARAMETER_1) {
    ASSERT_TRUE(RunModelEditingTest([](RoadRunner *rri) {
        rri->addRateRule("k1", "1000000");
    }));
}

TEST_F(ModelEditingTests, ADD_RATE_RULE_PARAMETER_2) {
    ASSERT_TRUE(RunModelEditingTest([](RoadRunner *rri) {
        rri->addParameter("k1", 1000000);
        rri->addRateRule("k1", "1000000");
        rri->addReaction("reaction1", {"S1", "S2"}, {"S3", "S4"}, "compartment * k1 * S1 * S2");
    }));
}

TEST_F(ModelEditingTests, ADD_TRIGGER_1) {
    ASSERT_TRUE(RunModelEditingTest([](RoadRunner *rri) {
        rri->addTrigger("event1", "S1 < 0.75");
    }));
}

TEST_F(ModelEditingTests, ADD_TRIGGER_2) {
    ASSERT_TRUE(RunModelEditingTest([](RoadRunner *rri) {
        rri->addTrigger("event1", "S1 < 0.75");
    }));
}

TEST_F(ModelEditingTests, REMOVE_COMPARTMENT_1) {
    ASSERT_TRUE(RunModelEditingTest([](RoadRunner *rri) {
        rri->removeCompartment("compartment");
    }));
}

TEST_F(ModelEditingTests, REMOVE_COMPARTMENT_2) {
    ASSERT_TRUE(RunModelEditingTest([](RoadRunner *rri) {
        rri->removeCompartment("compartment1");
    }));
}

TEST_F(ModelEditingTests, REMOVE_EVENT_ASSIGNMENT_1) {
    ASSERT_TRUE(RunModelEditingTest([](RoadRunner *rri) {
        rri->removeEventAssignments("event1", "S2");
    }));
}

TEST_F(ModelEditingTests, REMOVE_EVENT_ASSIGNMENT_2) {
    ASSERT_TRUE(RunModelEditingTest([](RoadRunner *rri) {
        rri->removeEventAssignments("event1", "S1");
    }));
}

TEST_F(ModelEditingTests, REMOVE_PARAMETER_1) {
    ASSERT_TRUE(RunModelEditingTest([](RoadRunner *rri) {
        rri->removeParameter("k4");
    }));
}

TEST_F(ModelEditingTests, ADD_DELAY_2) {
    ASSERT_TRUE(RunModelEditingTest([](RoadRunner *rri) {
        rri->addDelay("event1", "1", false);
        rri->addDelay("event2", "0.5");
    }));
}

TEST_F(ModelEditingTests, ADD_PRIORITY_2) {
    ASSERT_TRUE(RunModelEditingTest([](RoadRunner *rri) {
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

        rri->setSelections({"R", "Q", "reset", "R2", "Q2", "reset2"});
    }, "l3v1"));
}

TEST_F(ModelEditingTests, REMOVE_PARAM_RECURSE_1) {
    ASSERT_TRUE(RunModelEditingTest([](RoadRunner *rri) {
        rri->removeParameter("k2");
    }));
}


TEST_F(ModelEditingTests, REMOVE_PARAM_RECURSE_2) {
    ASSERT_TRUE(RunModelEditingTest([](RoadRunner *rri) {
        rri->removeParameter("k4");
    }));
}

TEST_F(ModelEditingTests, REMOVE_ELEMENT_ASNT_RULE) {
    RoadRunner rri((rrTestDir_ / path("models/ModelEditingTests/assignment_rule_species.xml")).string());
    rri.removeSpecies("S1");
    vector<string> arids = rri.getAssignmentRuleIds();
    EXPECT_EQ(arids.size(), 0);
    const ls::DoubleMatrix* results = rri.simulate();
    EXPECT_EQ(results->numCols(), 2);
}

TEST_F(ModelEditingTests, REMOVE_ELEMENT_RATE_RULE) {
    RoadRunner rri((rrTestDir_ / path("models/ModelEditingTests/assignment_rule_species.xml")).string());
    rri.removeSpecies("S2");
    vector<string> rrids = rri.getRateRuleIds();
    EXPECT_EQ(rrids.size(), 0);
    const ls::DoubleMatrix* results = rri.simulate();
    EXPECT_EQ(results->numCols(), 2);
}

TEST_F(ModelEditingTests, FROM_SCRATCH_1) {
    ASSERT_TRUE(RunTestModelFromScratch([](RoadRunner *rri) {
        rri->addCompartment("compartment", 1);
        rri->addSpeciesConcentration("S1", "compartment", 0.00015, true);
        rri->addSpeciesConcentration("S2", "compartment", 0);
        rri->addParameter("k1", 1);
        rri->addReaction("reaction1", {"S1"}, {"S2"}, "compartment * k1 * S1");
    }));
}

TEST_F(ModelEditingTests, FROM_SCRATCH_2) {
    ASSERT_TRUE(RunTestModelFromScratch([](RoadRunner *rri) {
        rri->addCompartment("compartment", 1);
        rri->addSpeciesConcentration("S1", "compartment", 1, true);
        rri->addSpeciesConcentration("S2", "compartment", 0, true);
        rri->addParameter("k1", 1);
        rri->addReaction("reaction1", {"S1"}, {"S2"}, "compartment * k1 * S1");
        rri->addEvent("event1", true, "S1 < 0.1");
        rri->addDelay("event1", "1");
        rri->addEventAssignment("event1", "S1", "1");
    }));
}

TEST_F(ModelEditingTests, FROM_SCRATCH_3) {
    ASSERT_TRUE(RunTestModelFromScratch([](RoadRunner *rri) {
        rri->addCompartment("compartment", 1);
        rri->addSpeciesConcentration("S1", "compartment", 0, true, true);
        rri->addRateRule("S1", "7");
    }));
}

TEST_F(ModelEditingTests, FROM_SCRATCH_4) {
    ASSERT_TRUE(RunTestModelFromScratch([](RoadRunner *rri) {
        rri->addCompartment("compartment", 1);
        rri->addSpeciesConcentration("S1", "compartment", 7, true, true);
        rri->addAssignmentRule("S1", "7");
    }));
}

TEST_F(ModelEditingTests, FROM_SCRATCH_5) {
    ASSERT_TRUE(RunTestModelFromScratch([](RoadRunner *rri) {
        rri->addCompartment("compartment", 1);
        rri->addSpeciesConcentration("S1", "compartment", 1, true);
        rri->addSpeciesConcentration("S2", "compartment", 1.5e-15, true);
        rri->addSpeciesConcentration("S3", "compartment", 1, true, true);
        rri->addParameter("k1", 0.75);
        rri->addParameter("k2", 50);
        rri->addAssignmentRule("S3", "k1*S2");
        rri->addReaction("reaction1", {"S1"}, {"S2"}, "compartment * k2 * S1");
    }));
}

TEST_F(ModelEditingTests, FROM_SCRATCH_1_L2V5) {
    RoadRunner rri(2, 5);
    rri.addCompartment("compartment", 1);
    rri.addSpeciesConcentration("S1", "compartment", 0.00015, true);
    rri.addSpeciesConcentration("S2", "compartment", 0, true);
    rri.addParameter("k1", 1);
    rri.addReaction("reaction1", {"S1"}, {"S2"}, "compartment * k1 * S1");
    ASSERT_TRUE(testValidateSBML(rri.getCurrentSBML()));
}

TEST_F(ModelEditingTests, FROM_SCRATCH_1_L3V2) {
    RoadRunner rri;
    rri.addCompartment("compartment", 1);
    rri.addSpeciesConcentration("S1", "compartment", 0.00015);
    rri.addSpeciesConcentration("S2", "compartment", 0);
    rri.addParameter("k1", 1);
    rri.addReaction("reaction1", {"S1"}, {"S2"}, "compartment * k1 * S1");
    ASSERT_TRUE(testValidateSBML(rri.getCurrentSBML()));
}

TEST_F(ModelEditingTests, FROM_SCRATCH_2_L3V2) {
    RoadRunner rri;
    rri.addCompartment("compartment", 1);
    rri.addSpeciesConcentration("S1", "compartment", 0.00015);
    rri.addSpeciesConcentration("S2", "compartment", 0);
    rri.addSpeciesConcentration("S3", "compartment", 0);
    rri.addParameter("k1", 1);
    rri.addReaction("reaction1", {"S1"}, {"S2"}, "compartment * k1 * S1 * S3");
    ASSERT_TRUE(testValidateSBML(rri.getCurrentSBML()));
}

TEST_F(ModelEditingTests, FROM_SCRATCH_3_L3V2) {
    RoadRunner rri;
    rri.addCompartment("compartment", 1);
    rri.addSpeciesConcentration("S1", "compartment", 0.00015);
    rri.addSpeciesConcentration("S2", "compartment", 0);
    rri.addSpeciesConcentration("S3", "compartment", 0);
    rri.addParameter("k1", 1);
    rri.addReaction("reaction1", {"S1"}, {"S2"}, "S3");
    ASSERT_TRUE(testValidateSBML(rri.getCurrentSBML()));
}

TEST_F(ModelEditingTests, FROM_SCRATCH_2_VALID_L3V1) {
    RoadRunner rri(3, 1);
    rri.addCompartment("compartment", 1);
    rri.addSpeciesConcentration("S1", "compartment", 0.00015);
    rri.addSpeciesConcentration("S2", "compartment", 0);
    rri.addSpeciesConcentration("S3", "compartment", 0);
    rri.addParameter("k1", 1);
    rri.addReaction("reaction1", {"S1"}, {"S2"}, "compartment * k1 * S1 * S3");
    ASSERT_TRUE(testValidateSBML(rri.getCurrentSBML()));
}

TEST_F(ModelEditingTests, FROM_SCRATCH_3_VALID_L3V1) {
    RoadRunner rri(3, 1);
    rri.addCompartment("compartment", 1);
    rri.addSpeciesConcentration("S1", "compartment", 0.00015);
    rri.addSpeciesConcentration("S2", "compartment", 0);
    rri.addSpeciesConcentration("S3", "compartment", 0);
    rri.addParameter("k1", 1);
    rri.addReaction("reaction1", {"S1"}, {"S2"}, "S3");
    ASSERT_TRUE(testValidateSBML(rri.getCurrentSBML()));
}


TEST_F(ModelEditingTests, FROM_SCRATCH_1_L3V1) {
    ASSERT_TRUE(RunTestModelFromScratch([](RoadRunner *rri) {
        rri->addCompartment("compartment", 1);
        rri->addSpeciesConcentration("S1", "compartment", 0.00015);
        rri->addSpeciesConcentration("S2", "compartment", 0);
        rri->addParameter("k1", 1);
        rri->addReaction("reaction1", {"S1"}, {"S2"}, "compartment * k1 * S1");
    }, "l3v1"));
}

TEST_F(ModelEditingTests, FROM_SCRATCH_2_L3V1) {
    ASSERT_TRUE(RunTestModelFromScratch([](RoadRunner *rri) {
        rri->addCompartment("compartment", 1);
        rri->addSpeciesConcentration("S1", "compartment", 1);
        rri->addSpeciesConcentration("S2", "compartment", 0);
        rri->addParameter("k1", 1);
        rri->addReaction("reaction1", {"S1"}, {"S2"}, "compartment * k1 * S1");
        rri->addEvent("event1", true, "S1 < 0.1");
        rri->addDelay("event1", "1");
        rri->addEventAssignment("event1", "S1", "1");
    }, "l3v1"));
}

TEST_F(ModelEditingTests, FROM_SCRATCH_3_L3V1) {
    ASSERT_TRUE(RunTestModelFromScratch([](RoadRunner *rri) {
        rri->addCompartment("compartment", 1);
        rri->addSpeciesConcentration("S1", "compartment", 0, false, true);
        rri->addRateRule("S1", "7");
    }, "l3v1"));
}

TEST_F(ModelEditingTests, FROM_SCRATCH_4_L3V1) {
    ASSERT_TRUE(RunTestModelFromScratch([](RoadRunner *rri) {
        rri->addCompartment("compartment", 1);
        rri->addSpeciesConcentration("S1", "compartment", 7, false, true);
        rri->addAssignmentRule("S1", "7");
    }, "l3v1"));
}

TEST_F(ModelEditingTests, FROM_SCRATCH_5_L3V1) {
    ASSERT_TRUE(RunTestModelFromScratch([](RoadRunner *rri) {
        rri->addCompartment("compartment", 1);
        rri->addSpeciesConcentration("S1", "compartment", 1);
        rri->addSpeciesConcentration("S2", "compartment", 1.5e-15);
        rri->addSpeciesConcentration("S3", "compartment", 1, false, true);
        rri->addParameter("k1", 0.75);
        rri->addParameter("k2", 50);
        rri->addAssignmentRule("S3", "k1*S2");
        rri->addReaction("reaction1", {"S1"}, {"S2"}, "compartment * k2 * S1");
    }, "l3v1"));
}

TEST_F(ModelEditingTests, FROM_SCRATCH_6) {
    ASSERT_TRUE(RunTestModelFromScratch([](RoadRunner *rri) {
        rri->addParameter("Q", 0);
        rri->addParameter("R", 0);
        rri->addParameter("reset", 0);
        rri->addParameter("Q2", 0);
        rri->addParameter("R2", 0);
        rri->addParameter("reset2", 0);

        rri->addEvent("Qinc", true, "(time - reset) >= 0.01", false);
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

TEST_F(ModelEditingTests, SET_PERSISTENT_1) {
    ASSERT_TRUE(RunModelEditingTest([](RoadRunner *rri) {
        rri->setPersistent("event1", true, false);
        rri->setPersistent("event2", true);
    }, "l3v1"));
}

TEST_F(ModelEditingTests, SET_PERSISTENT_2) {
    ASSERT_TRUE(RunModelEditingTest([](RoadRunner *rri) {
        rri->setPersistent("event1", false);
    }, "l3v1"));
}

TEST_F(ModelEditingTests, SET_CONSTANT_1) {
    ASSERT_TRUE(RunModelEditingTest([](RoadRunner *rri) {
        rri->setConstant("k1", false, false);
        rri->addRateRule("k1", "0.5");
    }));
}

TEST_F(ModelEditingTests, SET_CONSTANT_2) {
    ASSERT_TRUE(RunModelEditingTest([](RoadRunner *rri) {
        rri->setConstant("S1", false, false);
        rri->setBoundary("S1", true);
        rri->addRateRule("S1", "7");
    }));
}

TEST_F(ModelEditingTests, SET_HAS_ONLY_SUBSTANCE_UNITS_1) {
    ASSERT_TRUE(RunModelEditingTest([](RoadRunner *rri) {
        rri->setHasOnlySubstanceUnits("S1", true, false);
        rri->setHasOnlySubstanceUnits("S2", true);
    }));
}

TEST_F(ModelEditingTests, SET_HAS_ONLY_SUBSTANCE_UNITS_2) {
    ASSERT_TRUE(RunModelEditingTest([](RoadRunner *rri) {
        rri->setHasOnlySubstanceUnits("S1", true, false);
        rri->setHasOnlySubstanceUnits("S2", true, false);
        rri->setHasOnlySubstanceUnits("S3", true, false);
        rri->setHasOnlySubstanceUnits("S4", true);
    }));
}

TEST_F(ModelEditingTests, SET_HAS_ONLY_SUBSTANCE_UNITS_3) {
    ASSERT_TRUE(RunModelEditingTest([](RoadRunner *rri) {
        rri->setHasOnlySubstanceUnits("S1", false, false);
        rri->setHasOnlySubstanceUnits("S2", false);
    }));
}

TEST_F(ModelEditingTests, SET_INITIAL_CONCENTRATION_1) {
    ASSERT_TRUE(RunModelEditingTest([](RoadRunner *rri) {
        rri->addSpeciesConcentration("S1", "C", 0.0);
        rri->setInitConcentration("S1", 0.0004, false);
        rri->addSpeciesConcentration("S2", "C", 0.0);
        rri->setInitConcentration("S2", 0.00048, false);
        rri->addSpeciesConcentration("S3", "C", 0.0);
        rri->setInitConcentration("S3", 0.0008, false);
        rri->addSpeciesConcentration("S4", "C", 0.0);
        rri->setInitConcentration("S4", 0.0004);

        rri->addReaction("reaction1", {"S1", "S2"}, {"S3", "S4"}, "C * k1 * S1 * S2", false);
        rri->addReaction("reaction2", {"S3", "S4"}, {"S1", "S2"}, "C * k2 * S3 * S4");

        rri->addEvent("event1", true, "S4 > S2", false);
        rri->addEventAssignment("event1", "S1", "1/5000", false);
        rri->addEventAssignment("event1", "S4", "1/5000", true);
    }));
}

TEST_F(ModelEditingTests, SET_INITIAL_CONCENTRATION_2) {
    ASSERT_TRUE(RunModelEditingTest([](RoadRunner *rri) {
        rri->setInitConcentration("S1", 0.0004, false);
        rri->setInitConcentration("S2", 0.00048, false);
        rri->setInitConcentration("S3", 0.0008, false);
        rri->setInitConcentration("S4", 0.0004);
    }));
}

TEST_F(ModelEditingTests, SET_INITIAL_AMOUNT_1) {
    ASSERT_TRUE(RunModelEditingTest([](RoadRunner *rri) {
        rri->setInitAmount("S1", 0.00015, false);
        rri->setInitAmount("S2", 0);
    }));
}

TEST_F(ModelEditingTests, SET_INITIAL_AMOUNT_2) {
    ASSERT_TRUE(RunModelEditingTest([](RoadRunner *rri) {
        rri->addSpeciesConcentration("S1", "compartment", 1.0, false, false);
        rri->setInitAmount("S1", 0.00015, false);
        rri->addSpeciesConcentration("S2", "compartment", 1.0, false, false);
        rri->setInitAmount("S2", 0);

        rri->addReaction("reaction1", {"S1"}, {"S2"}, "compartment * k1 * S1");
        rri->reset();
    }));
}

TEST_F(ModelEditingTests, SET_REVERSIBLE_1) {
    ASSERT_TRUE(RunModelEditingTest([](RoadRunner *rri) {
        rri->setReversible("reaction2", true);
    }));
}

TEST_F(ModelEditingTests, SET_REVERSIBLE_2) {
    ASSERT_TRUE(RunModelEditingTest([](RoadRunner *rri) {
        rri->setReversible("reaction1", false);
    }));
}

TEST_F(ModelEditingTests, SET_TRIGGER_INITIAL_VALUE_1) {
    ASSERT_TRUE(RunModelEditingTest([](RoadRunner *rri) {
        rri->setTriggerInitialValue("event1", false);
    }, "l3v1"));
}

TEST_F(ModelEditingTests, SET_TRIGGER_INITIAL_VALUE_2) {
    ASSERT_TRUE(RunModelEditingTest([](RoadRunner *rri) {
        rri->setTriggerInitialValue("event1", true);
    }, "l3v1"));
}

TEST_F(ModelEditingTests, SET_TRIGGER_INITIAL_VALUE_3) {
    ASSERT_TRUE(RunModelEditingTest([](RoadRunner *rri) {
        rri->setTriggerInitialValue("event1", true);
    }, "l3v1"));
}

TEST_F(ModelEditingTests, RETAIN_ABSOLUTE_TOLERANCES_1) {
    RoadRunner *rri = new RoadRunner(
            (rrTestDir_ / path("sbml-test-suite") / path("semantic") / path("00001") / "00001-sbml-l2v4.xml").string());
    rri->getIntegrator()->setIndividualTolerance("S1", 5.0);
    rri->getIntegrator()->setIndividualTolerance("S2", 3.0);
    rri->removeSpecies("S2");
    EXPECT_TRUE(rri->getIntegrator()->getAbsoluteToleranceVector().size() == 1)
                        << "RETAIN_ABSOLUTE_TOLERANCES_1 failed, rri->getIntegrator()->getAbsoluteToleranceVector().size() != 1"
                        << endl;

    EXPECT_TRUE(rri->getIntegrator()->getAbsoluteToleranceVector()[0] == 5.0)
                        << "RETAIN_ABSOLUTE_TOLERANCES_1 failed, rri->getIntegrator()->getAbsoluteToleranceVector()[0] == 5.0"
                        << endl;
    delete rri;
}

TEST_F(ModelEditingTests, RETAIN_ABSOLUTE_TOLERANCES_2) {
    RoadRunner *rri = new RoadRunner((rrTestDir_ / "sbml-test-suite/semantic/00001/00001-sbml-l2v4.xml").string());
    rri->getIntegrator()->setIndividualTolerance("S1", 5.0);
    rri->getIntegrator()->setIndividualTolerance("S2", 3.0);
    rri->removeSpecies("S1");
    EXPECT_TRUE(rri->getIntegrator()->getAbsoluteToleranceVector().size() == 1)
                        << "RETAIN_ABSOLUTE_TOLERANCES_2 failed, rri->getIntegrator()->getAbsoluteToleranceVector().size() != 1"
                        << endl;

    EXPECT_TRUE(rri->getIntegrator()->getAbsoluteToleranceVector()[0] == 3.0)
                        << "RETAIN_ABSOLUTE_TOLERANCES_2 failed, rri->getIntegrator()->getAbsoluteToleranceVector()[0] == 3.0"
                        << endl;
    delete rri;
}

TEST_F(ModelEditingTests, READD_SPECIES) {
    //Remove and readd all the species from some SBML models that have no rate rules
    for (int i : {21})
        //for (int i : {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22
        //        , 23, 24, 25, 26, 27, 28, 29, 33, 34, 35})
    {
        EXPECT_TRUE(RunTestWithEdit("l2v4", i, removeAndReaddAllSpecies, "removeAndReaddAllSpecies"))
                            << "Remove and re-add all species for SBML Test " << i << " failed";
    }
}

TEST_F(ModelEditingTests, READD_REACTION) {
    for (int i = 1; i <= 38; i++) {
        EXPECT_TRUE(RunTestWithEdit("l2v4", i, removeAndReaddAllReactions, "removeAndReaddAllReactions"))
                            << "SBML Test " << i << " failed";
    }
}

TEST_F(ModelEditingTests, READD_COMPARTMENTS) {
    for (int i : {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
                  29, 33, 34, 35}) {
        EXPECT_TRUE(RunTestWithEdit("l2v4", i, removeAndReaddAllCompartments, "removeAndReaddAllCompartments"))
                            << "SBML Test " << i << " failed";
    }
}


TEST_F(ModelEditingTests, ONE_ASSIGNMENT_RULE) {
    ASSERT_TRUE(RunModelEditingTest([](RoadRunner *rri) {
        // Do nothing. This is here for a convenient test of issue #610, where
        // reaction rates should be updated due to assignmentRules but are not
        // due to optimization.
    }));
}

TEST_F(ModelEditingTests, ADD_INVALID_COMPARTMENT) {
    RoadRunner rri;
    EXPECT_THROW(rri.addCompartment("$compartment", 3.14159), std::invalid_argument);
}


TEST_F(ModelEditingTests, ADD_INVALID_SPECIES) {
    RoadRunner rri;
    rri.addCompartment("compartment", 3.14159);
    EXPECT_THROW(rri.addSpeciesConcentration("$S1", "compartment", 1.0, false, false), std::invalid_argument);
}

TEST_F(ModelEditingTests, ADD_INVALID_PARAM) {
    RoadRunner rri;
    EXPECT_THROW(rri.addParameter("$pid", 3.14159), std::invalid_argument);
}

TEST_F(ModelEditingTests, ADD_INVALID_RXN) {
    RoadRunner rri;
    rri.addCompartment("compartment", 3.14159);
    rri.addSpeciesConcentration("S1", "compartment", 1.0, false, false);
    rri.addSpeciesConcentration("S2", "compartment", 1.0, false, false);
    rri.addParameter("k1", 1.0);
    EXPECT_THROW(rri.addReaction("%rxn", {"S2"}, {"S1"}, "k1*S2", true), std::invalid_argument);
}


TEST_F(ModelEditingTests, CHECK_REGENERATE) {
    RoadRunner rri;
    rri.addCompartment("compartment", 1, false);
    rri.addSpeciesConcentration("S1", "compartment", 1.0, false, false, "", false);
    rri.addSpeciesConcentration("S2", "compartment", 1.0, false, false, "", false);
    rri.addParameter("k1", 1.0, false);
    rri.addReaction("reaction2", {"S2"}, {"S1"}, "k1*S2", false);
    rri.regenerateModel();
    try {
        int n = rri.getNumberOfFloatingSpecies();
    }
    catch (std::exception &ex) {
        string error = ex.what();
        cerr << "CHECK_REGENERATE FAIL: " << error << endl;
        ASSERT_TRUE (false);
    }

}


TEST_F(ModelEditingTests, GET_SET_MODELNAME1) {
    RoadRunner rri;
    EXPECT_STREQ(rri.getModelName().c_str(), "");
    rri.setModelName("test");
    EXPECT_STREQ(rri.getModelName().c_str(), "test");
}

TEST_F(ModelEditingTests, GET_SET_MODELNAME2) {
    RoadRunner rri("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<sbml xmlns=\"http://www.sbml.org/sbml/level3/version2/core\" level=\"3\" version=\"2\">\n  <model name=\"foo\"/>\n</sbml>\n");
    EXPECT_STREQ(rri.getModelName().c_str(), "foo");
    rri.setModelName("test");
    EXPECT_STREQ(rri.getModelName().c_str(), "test");
    EXPECT_STREQ(rri.getSBML().c_str(), "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<sbml xmlns=\"http://www.sbml.org/sbml/level3/version2/core\" level=\"3\" version=\"2\">\n  <model name=\"test\"/>\n</sbml>\n");
}

TEST_F(ModelEditingTests, GET_SET_MODELID1) {
    RoadRunner rri;
    EXPECT_STREQ(rri.getModelId().c_str(), "");
    rri.setModelId("test");
    EXPECT_STREQ(rri.getModelId().c_str(), "test");
}

TEST_F(ModelEditingTests, GET_SET_MODELID2) {
    RoadRunner rri("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<sbml xmlns=\"http://www.sbml.org/sbml/level3/version2/core\" level=\"3\" version=\"2\">\n  <model id=\"foo\"/>\n</sbml>\n");
    EXPECT_STREQ(rri.getModelId().c_str(), "foo");
    rri.setModelId("test");
    EXPECT_STREQ(rri.getModelId().c_str(), "test");
    EXPECT_STREQ(rri.getSBML().c_str(), "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<sbml xmlns=\"http://www.sbml.org/sbml/level3/version2/core\" level=\"3\" version=\"2\">\n  <model id=\"test\"/>\n</sbml>\n");
}


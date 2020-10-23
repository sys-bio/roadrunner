#include <string>
#include "gtest/gtest.h"
#include "rrConfig.h"
#include "rrLogger.h"
#include "rrRoadRunner.h"
#include "rrUtils.h"
#include "C/rrc_api.h"
#include "C/rrc_cpp_support.h"
#include "sbml/SBMLDocument.h"
#include "rrExecutableModel.h"

#include "rrSBMLTestSuiteSimulation_CAPI.h"

using namespace std;
using namespace ls;
using namespace rrc;
using namespace rr;
using namespace testing;

extern string gRRTestDir;
extern string gRROutputDir;
extern RRHandle gRR;

bool validateModifiedSBML(std::string sbml)
{
	libsbml::SBMLDocument *doc = libsbml::readSBMLFromString(sbml.c_str());
	bool result = true;

	
	if (doc->getNumErrors() != 0)
	{
		for (int i = 0; i < doc->getNumErrors(); i++)
		{ 
			std::cout << doc->getError(i)->getMessage() << std::endl;
		}
		result = false;
	}

	doc->setConsistencyChecks(libsbml::LIBSBML_CAT_MODELING_PRACTICE, false);
	doc->setConsistencyChecks(libsbml::LIBSBML_CAT_UNITS_CONSISTENCY, false);

	if (doc->validateSBML() != 0)
	{
		for (int i = 0; i < doc->getNumErrors(); i++)
		{
			std::cout << doc->getError(i)->getMessage() << std::endl;
		}
		result = false;
	}
	delete doc;
	return result;
}

/*
* Loads <prefix>/source/roadrunner/models/sbml-test-suite/cases/semantic/<suite-name>/<test-name>/<test-name>-sbml-*VERSION*.xml
* applies modification to the resulting roadrunner instance and compares the result to <test-name>-results.csv in the same folder,
* The method obtains test-name and suite-name from UnitTest++ so this method must be run within a UnitTest++ test
* Returns true if the results are close enough, false otherwise
*/
bool RunTestWithModification(void(*modification)(RRHandle), std::string version = "l2v4")
{
	bool result(false);
	RRHandle gRR;

    string testName(UnitTest::GetInstance()->current_test_info()->name());
    string suiteName(UnitTest::GetInstance()->current_test_info()->test_suite_name());

	//Create instance..
	gRR = createRRInstance();

	//Setup environment
	libsbml::SBMLDocument doc;

	if (!gRR)
	{
		return false;
	}

	try
	{
		setCurrentIntegratorParameterBoolean(gRR, "stiff", 0);

		//Create a log file name

		SBMLTestSuiteSimulation_CAPI simulation;

		simulation.UseHandle(gRR);

		//Read SBML models.....
		simulation.SetCaseNumber(0);

		string modelFilePath = gRRTestDir + "models/" + suiteName + "/" + testName + "/";
		string modelFileName = testName + "-sbml-" + version + ".xml";
		string settingsFileName = testName + "-settings.txt";

		//The following will load and compile and simulate the sbml model in the file
		simulation.SetModelFilePath(modelFilePath);
		simulation.SetModelFileName(modelFileName);
		simulation.ReCompileIfDllExists(true);
		simulation.CopyFilesToOutputFolder();
		setTempFolder(gRR, simulation.GetDataOutputFolder().c_str());
		setComputeAndAssignConservationLaws(gRR, false);

		libsbml::SBMLReader reader;
		std::string fullPath = modelFilePath + "/" + modelFileName;
		doc = *reader.readSBML(fullPath);

		if (!simulation.LoadSBMLFromFile())
		{
			throw(Exception("Failed loading sbml from file"));
		}


		//Check first if file exists first
		if (!fileExists(fullPath))
		{
			Log(Logger::LOG_ERROR) << "sbml file " << fullPath << " not found";
			throw(Exception("No such SBML file: " + fullPath));
		}

		RoadRunner* rri = (RoadRunner*)gRR;

		LoadSBMLOptions opt;

		// don't generate cache for models
		opt.modelGeneratorOpt = opt.modelGeneratorOpt | LoadSBMLOptions::RECOMPILE;

		opt.modelGeneratorOpt = opt.modelGeneratorOpt | LoadSBMLOptions::MUTABLE_INITIAL_CONDITIONS;

		opt.modelGeneratorOpt = opt.modelGeneratorOpt & ~LoadSBMLOptions::READ_ONLY;

		opt.modelGeneratorOpt = opt.modelGeneratorOpt | LoadSBMLOptions::OPTIMIZE_CFG_SIMPLIFICATION;

		opt.modelGeneratorOpt = opt.modelGeneratorOpt | LoadSBMLOptions::OPTIMIZE_GVN;


		rri->load(fullPath, &opt);

		//Then read settings file if it exists..
		if (!simulation.LoadSettings(joinPath(modelFilePath, settingsFileName)))
		{
			throw(Exception("Failed loading simulation settings"));
		}
		modification(gRR);
		//Then Simulate model
		if (!simulation.Simulate())
		{
			throw(Exception("Failed running simulation"));
		}

		//Write result
		if (!simulation.SaveResult())
		{
			//Failed to save data
			throw(Exception("Failed saving result"));
		}

		if (!simulation.LoadReferenceData(modelFilePath + "/" + testName + "-results.csv"))
		{
			throw(Exception("Failed Loading reference data"));
		}

		simulation.CreateErrorData();
		result = simulation.Pass();
		result = simulation.SaveAllData() && result;
		//result = simulation.SaveModelAsXML(dataOutputFolder) && result;
		if (!result)
		{
			Log(Logger::LOG_WARNING) << "\t\t =============== Test " << testName << " failed =============\n";
		}
	}
	catch (std::exception& ex)
	{
		string error = ex.what();
		cerr << "Case " << testName << ": Exception: " << error << endl;
		freeRRInstance(gRR);
		return false;
	}

	freeRRInstance(gRR);
	return result;
}

bool RunTestModelFromScratch(void(*generate)(RRHandle),std::string version = "l2v4")
{
	bool result(false);
	int level = version.at(1) - '0';
	int versionNum = version.at(3) - '0';
	RRHandle rrh = createRRInstance();
	RoadRunner &rr = *castToRoadRunner(rrh);


    string testName(UnitTest::GetInstance()->current_test_info()->name());
    string suiteName(UnitTest::GetInstance()->current_test_info()->test_suite_name());

	try
	{
		//Log(Logger::LOG_NOTICE) << "Running Test: " << testName << endl;

		rr.getIntegrator()->setValue("stiff", false);

		TestSuiteModelSimulation simulation;

		simulation.UseEngine(&rr);

		//Read SBML models.....
		simulation.SetCaseNumber(0);

		string modelFilePath = gRRTestDir + "models/" + suiteName + "/" + testName + "/";
		string modelFileName = testName + "-sbml-" + version + ".xml";
		string settingsFileName = testName + "-settings.txt";

		//The following will load and compile and simulate the sbml model in the file
		simulation.SetModelFilePath(modelFilePath);
		simulation.SetModelFileName(modelFileName);
		simulation.ReCompileIfDllExists(true);
		simulation.CopyFilesToOutputFolder();
		rr.setConservedMoietyAnalysis(false);

		libsbml::SBMLReader reader;
		std::string fullPath = modelFilePath + "/" + modelFileName;

		//Check first if file exists first
		if (!fileExists(fullPath))
		{
			Log(Logger::LOG_ERROR) << "sbml file " << fullPath << " not found";
			throw(Exception("No such SBML file: " + fullPath));
		}


		LoadSBMLOptions opt;

		// don't generate cache for models
		opt.modelGeneratorOpt = opt.modelGeneratorOpt | LoadSBMLOptions::RECOMPILE;

		opt.modelGeneratorOpt = opt.modelGeneratorOpt | LoadSBMLOptions::MUTABLE_INITIAL_CONDITIONS;

		opt.modelGeneratorOpt = opt.modelGeneratorOpt & ~LoadSBMLOptions::READ_ONLY;

		opt.modelGeneratorOpt = opt.modelGeneratorOpt | LoadSBMLOptions::OPTIMIZE_CFG_SIMPLIFICATION;

		opt.modelGeneratorOpt = opt.modelGeneratorOpt | LoadSBMLOptions::OPTIMIZE_GVN;



		//Then read settings file if it exists..
		if (!simulation.LoadSettings(joinPath(modelFilePath, settingsFileName)))
		{
			throw(Exception("Failed loading simulation settings"));
		}
		generate(&rr);
		//Then Simulate model
		if (!simulation.Simulate())
		{
			throw(Exception("Failed running simulation"));
		}

		//Write result
		if (!simulation.SaveResult())
		{
			//Failed to save data
			throw(Exception("Failed saving result"));
		}

		if (!simulation.LoadReferenceData(modelFilePath + "/" + testName + "-results.csv"))
		{
			throw(Exception("Failed Loading reference data"));
		}

		simulation.CreateErrorData();
		result = simulation.Pass();
		result = simulation.SaveAllData() && result;
		//result = simulation.SaveModelAsXML(dataOutputFolder) && result;
		result = validateModifiedSBML(rr.getCurrentSBML()) && result;
		if (!result)
		{
			Log(Logger::LOG_WARNING) << "\t\t =============== Test " << testName << " failed =============\n";
		}
	}
	catch (std::exception& ex)
	{
		string error = ex.what();
		cerr << "Case " << testName << ": Exception: " << error << endl;
		return false;
	}

	return result;
}

TEST(MODEL_EDITING_TEST_SUITE, CLEAR_MODEL_1)
{
    RoadRunner rri;
    rri.addCompartment("compartment", 3.14159);
    rri.addSpecies("S1", "compartment", 1.0, false);
    EXPECT_TRUE(rri.isModelLoaded());
    rri.clearModel();
    EXPECT_TRUE(!rri.isModelLoaded());
    rri.addCompartment("compartment", 3.14159);
    rri.addSpecies("S2", "compartment", 2.0, false);
    std::cout << rri.getNumberOfFloatingSpecies() << std::endl;
    EXPECT_TRUE(rri.getNumberOfFloatingSpecies() == 1);
}
TEST(MODEL_EDITING_TEST_SUITE, ADD_REACTION_1)
{
    EXPECT_TRUE(RunTestWithModification([](RRHandle rri) {
        const char* reactants[] = { "S2" };
        const char* products[] = { "S1" };
        addReaction(rri, "reaction2", reactants, 1, products, 1, "k1*S2");
        }));
}

TEST(MODEL_EDITING_TEST_SUITE, REMOVE_REACTION_1)
{
    EXPECT_TRUE(RunTestWithModification([](RRHandle rri) {
        removeReaction(rri, "reaction2");
        }));
}

TEST(MODEL_EDITING_TEST_SUITE, ADD_SPECIES_1)
{
    EXPECT_TRUE(RunTestWithModification([](RRHandle rri)
        {
            addSpecies(rri, "S3", "compartment", 0.0015, true, false);
        }));
}

TEST(MODEL_EDITING_TEST_SUITE, REMOVE_SPECIES_1)
{
    EXPECT_TRUE(RunTestWithModification([](RRHandle rri)
        {
            removeSpeciesNoRegen(rri, "S2");
            addSpeciesNoRegen(rri, "S3", "compartment", 0.00030, true, false);
            const char* reactants1[] = { "S1" };
            const char* products1[] = { "S3" };
            addReactionNoRegen(rri, "reaction1", reactants1, 1, products1, 1, "k1*S1");
            const char* reactants2[] = { "S3" };
            const char* products2[] = { "S1" };
            addReaction(rri, "reaction2", reactants2, 1, products2, 1, "k2*S3");
        }));
}

TEST(MODEL_EDITING_TEST_SUITE, ADD_EVENT_1)
{
    EXPECT_TRUE(RunTestWithModification([](RRHandle rri)
        {
            addEventNoRegen(rri, "event1", true, "S1 > 0.00015");
            addEventAssignment(rri, "event1", "S1", "1");
        }));
}

TEST(MODEL_EDITING_TEST_SUITE, EVENT_PRIORITY_1)
{
    EXPECT_TRUE(RunTestWithModification([](RRHandle rri)
        {
            addEvent(rri, "_E0", true, "time >= 0.99");
            addPriority(rri, "_E0", "1");
            addEventAssignment(rri, "_E0", "S1", "4");
            addEventAssignment(rri, "_E0", "S2", "5");
            addEventAssignment(rri, "_E0", "S3", "6");

            addEvent(rri, "_E1", true, "time >= 0.99");
            addPriority(rri, "_E1", "0");
            addEventAssignment(rri, "_E1", "S1", "1");
            addEventAssignment(rri, "_E1", "S2", "2");
            addEventAssignment(rri, "_E1", "S3", "3");
        }, "l3v1"));
}

TEST(MODEL_EDITING_TEST_SUITE, EVENT_DELAY_1)
{
    EXPECT_TRUE(RunTestWithModification([](RRHandle rri)
        {
            addDelay(rri, "event1", "1");
        }));
}

TEST(MODEL_EDITING_TEST_SUITE, REMOVE_EVENT_1)
{
    EXPECT_TRUE(RunTestWithModification([](RRHandle rri)
        {
            removeEvent(rri, "event1");
        }));
}

TEST(MODEL_EDITING_TEST_SUITE, ADD_PARAMETER_1)
{
    EXPECT_TRUE(RunTestWithModification([](RRHandle rri)
        {
            addParameterNoRegen(rri, "k1", 0.75);
            const char* reactants[] = { "S1", "S2" };
            const char* products[] = { "S3" };
            addReaction(rri, "reaction1", reactants, 2, products, 1, "k1*S1*S2");
        }));
}

TEST(MODEL_EDITING_TEST_SUITE, ADD_COMPARTMENT_1)
{
    EXPECT_TRUE(RunTestWithModification([](RRHandle rri)
        {
            addCompartment(rri, "compartment", 1);
            const char* reactants[] = { "S1", "S2" };
            const char* products[] = { "2S2" };
            addReaction(rri, "reaction1", reactants, 2, products, 1, "compartment * k1 * S1 * S2");
        }));
}

TEST(MODEL_EDITING_TEST_SUITE, ADD_ASSIGNMENT_RULE_1)
{
    EXPECT_TRUE(RunTestWithModification([](RRHandle rri)
        {
            setBoundary(rri, "S1", true);
            addAssignmentRule(rri, "S1", "7");
        }));
}

TEST(MODEL_EDITING_TEST_SUITE, ADD_RATE_RULE_1)
{
    EXPECT_TRUE(RunTestWithModification([](RRHandle rri)
        {
            setBoundary(rri, "S1", true);
            addRateRule(rri, "S1", "7");
        }));
}

TEST(MODEL_EDITING_TEST_SUITE, SET_KINETIC_LAW_1)
{
    EXPECT_TRUE(RunTestWithModification([](RRHandle rri)
        {
            setKineticLaw(rri, "reaction2", "compartment * k2 * S3");
        }));
}

TEST(MODEL_EDITING_TEST_SUITE, ADD_EVENT_ASSIGNMENT_1)
{
    EXPECT_TRUE(RunTestWithModification([](RRHandle rri)
        {
            addEventAssignment(rri, "event2", "S3", "1");
        }));
}

TEST(MODEL_EDITING_TEST_SUITE, ADD_TRIGGER_1)
{
    EXPECT_TRUE(RunTestWithModification([](RRHandle rri)
        {
            addTrigger(rri, "event1", "S1 < 0.75");
        }));
}

TEST(MODEL_EDITING_TEST_SUITE, PAUSE_10)
{
    EXPECT_TRUE(RunTestWithModification([](RRHandle rri)
        {
            simulate(rri);
            addDelay(rri, "event1", "0.2");
        }));
}

TEST(MODEL_EDITING_TEST_SUITE, REMOVE_COMPARTMENT_1)
{
    EXPECT_TRUE(RunTestWithModification([](RRHandle rri)
        {
            removeCompartment(rri, "compartment");
        }));
}

TEST(MODEL_EDITING_TEST_SUITE, REMOVE_PARAM_RECURSE_1)
{
    EXPECT_TRUE(RunTestWithModification([](RRHandle rri)
        {
            removeParameter(rri, "k2");
        }));
}

TEST(MODEL_EDITING_TEST_SUITE, REMOVE_EVENT_ASSIGNMENT_1)
{
    EXPECT_TRUE(RunTestWithModification([](RRHandle rri)
        {
            removeEventAssignments(rri, "event1", "S2");
        }));
}

TEST(MODEL_EDITING_TEST_SUITE, SET_PERSISTENT_1)
{
    EXPECT_TRUE(RunTestWithModification([](RRHandle rri)
        {
            setPersistentNoRegen(rri, "event1", true);
            setPersistent(rri, "event2", true);
        }, "l3v1"));
}

TEST(MODEL_EDITING_TEST_SUITE, SET_PERSISTENT_2)
{
    EXPECT_TRUE(RunTestWithModification([](RRHandle rri)
        {
            setPersistent(rri, "event1", false);
        }, "l3v1"));
}

TEST(MODEL_EDITING_TEST_SUITE, SET_CONSTANT_1)
{
    EXPECT_TRUE(RunTestWithModification([](RRHandle rri)
        {
            setConstantNoRegen(rri, "k1", false);
            addRateRule(rri, "k1", "0.5");
        }));
}

TEST(MODEL_EDITING_TEST_SUITE, SET_CONSTANT_2)
{
    EXPECT_TRUE(RunTestWithModification([](RRHandle rri)
        {
            setConstantNoRegen(rri, "S1", false);
            setBoundary(rri, "S1", true);
            addRateRule(rri, "S1", "7");
        }));
}

TEST(MODEL_EDITING_TEST_SUITE, SET_HAS_ONLY_SUBSTANCE_UNITS_1)
{
    EXPECT_TRUE(RunTestWithModification([](RRHandle rri)
        {
            setHasOnlySubstanceUnitsNoRegen(rri, "S1", true);
            setHasOnlySubstanceUnits(rri, "S2", true);
        }));
}

TEST(MODEL_EDITING_TEST_SUITE, SET_HAS_ONLY_SUBSTANCE_UNITS_2)
{
    EXPECT_TRUE(RunTestWithModification([](RRHandle rri)
        {
            setHasOnlySubstanceUnitsNoRegen(rri, "S1", true);
            setHasOnlySubstanceUnitsNoRegen(rri, "S2", true);
            setHasOnlySubstanceUnitsNoRegen(rri, "S3", true);
            setHasOnlySubstanceUnits(rri, "S4", true);
        }));
}

TEST(MODEL_EDITING_TEST_SUITE, SET_HAS_ONLY_SUBSTANCE_UNITS_3)
{
    EXPECT_TRUE(RunTestWithModification([](RRHandle rri)
        {
            setHasOnlySubstanceUnitsNoRegen(rri, "S1", false);
            setHasOnlySubstanceUnits(rri, "S2", false);
        }));
}

TEST(MODEL_EDITING_TEST_SUITE, SET_INITIAL_CONCENTRATION_1)
{
    EXPECT_TRUE(RunTestWithModification([](RRHandle rri)
        {
            addSpeciesNoRegen(rri, "S1", "C", 0.0, false, false);
            setInitConcentrationNoRegen(rri, "S1", 0.0004);
            addSpeciesNoRegen(rri, "S2", "C", 0.0, false, false);
            setInitConcentrationNoRegen(rri, "S2", 0.00048);
            addSpeciesNoRegen(rri, "S3", "C", 0.0, false, false);
            setInitConcentrationNoRegen(rri, "S3", 0.0008);
            addSpeciesNoRegen(rri, "S4", "C", 0.0, false, false);
            setInitConcentration(rri, "S4", 0.0004);

            const char* reactants1[] = { "S1", "S2" };
            const char* products1[] = { "S3", "S4" };

            addReactionNoRegen(rri, "reaction1", reactants1, 2, products1, 2, "C * k1 * S1 * S2");

            addReaction(rri, "reaction2", products1, 2, reactants1, 2, "C * k2 * S3 * S4");

            addEventNoRegen(rri, "event1", true, "S4 > S2");
            addEventAssignmentNoRegen(rri, "event1", "S1", "1/5000");
            addEventAssignment(rri, "event1", "S4", "1/5000");
        }));
}

TEST(MODEL_EDITING_TEST_SUITE, SET_INITIAL_CONCENTRATION_2)
{
    EXPECT_TRUE(RunTestWithModification([](RRHandle rri)
        {
            setInitConcentrationNoRegen(rri, "S1", 0.0004);
            setInitConcentrationNoRegen(rri, "S2", 0.00048);
            setInitConcentrationNoRegen(rri, "S3", 0.0008);
            setInitConcentration(rri, "S4", 0.0004);
        }));
}

TEST(MODEL_EDITING_TEST_SUITE, SET_INITIAL_AMOUNT_1)
{
    EXPECT_TRUE(RunTestWithModification([](RRHandle rri)
        {
            setInitAmountNoRegen(rri, "S1", 0.00015);
            setInitAmount(rri, "S2", 0);
        }));
}

TEST(MODEL_EDITING_TEST_SUITE, SET_INITIAL_AMOUNT_2)
{
    EXPECT_TRUE(RunTestWithModification([](RRHandle rri)
        {
            addSpecies(rri, "S1", "compartment", 0.0, false, false);
            setInitAmountNoRegen(rri, "S1", 0.00015);
            addSpecies(rri, "S2", "compartment", 0.0, false, false);
            setInitAmount(rri, "S2", 0);

            const char* reactants[] = { "S1" };
            const char* products[] = { "S2" };
            addReaction(rri, "reaction1", reactants, 1, products, 1, "compartment * k1 * S1");
            reset(rri);
        }));
}

TEST(MODEL_EDITING_TEST_SUITE, SET_REVERSIBLE_1)
{
    EXPECT_TRUE(RunTestWithModification([](RRHandle rri)
        {
            setReversible(rri, "reaction2", true);
        }));
}

TEST(MODEL_EDITING_TEST_SUITE, SET_REVERSIBLE_2)
{
    EXPECT_TRUE(RunTestWithModification([](RRHandle rri)
        {
            setReversible(rri, "reaction1", false);
        }));
}

TEST(MODEL_EDITING_TEST_SUITE, SET_TRIGGER_INITIAL_VALUE_1)
{
    EXPECT_TRUE(RunTestWithModification([](RRHandle rri)
        {
            setTriggerInitialValue(rri, "event1", false);
            reset(rri);
        }, "l3v1"));
}

TEST(MODEL_EDITING_TEST_SUITE, SET_TRIGGER_INITIAL_VALUE_2)
{
    EXPECT_TRUE(RunTestWithModification([](RRHandle rri)
        {
            setTriggerInitialValue(rri, "event1", true);
            reset(rri);
        }, "l3v1"));
}

TEST(MODEL_EDITING_TEST_SUITE, SET_TRIGGER_INITIAL_VALUE_3)
{
    EXPECT_TRUE(RunTestWithModification([](RRHandle rri)
        {
            setTriggerInitialValue(rri, "event1", true);
            reset(rri);
        }, "l3v1"));
}

TEST(MODEL_EDITING_TEST_SUITE, FROM_SCRATCH_1)
{
    EXPECT_TRUE(RunTestModelFromScratch([](RRHandle rri)
        {
            addCompartment(rri, "compartment", 1);
            addSpecies(rri, "S1", "compartment", 0.00015, false, false);
            addSpecies(rri, "S2", "compartment", 0, false, false);
            addParameter(rri, "k1", 1);
            const char* reactants[] = { "S1" };
            const char* products[] = { "S2" };
            addReaction(rri, "reaction1", reactants, 1, products, 1, "compartment * k1 * S1");
        }));
}

TEST(MODEL_EDITING_TEST_SUITE, FROM_SCRATCH_2)
{
    EXPECT_TRUE(RunTestModelFromScratch([](RRHandle rri)
        {
            addCompartment(rri, "compartment", 1);
            addSpecies(rri, "S1", "compartment", 1, false, false);
            addSpecies(rri, "S2", "compartment", 0, false, false);
            addParameter(rri, "k1", 1);
            const char* reactants[] = { "S1" };
            const char* products[] = { "S2" };
            addReaction(rri, "reaction1", reactants, 1, products, 1, "compartment * k1 * S1");
            addEvent(rri, "event1", true, "S1 < 0.1");
            addDelay(rri, "event1", "1");
            addEventAssignment(rri, "event1", "S1", "1");
        }));
}

TEST(MODEL_EDITING_TEST_SUITE, FROM_SCRATCH_3)
{
    EXPECT_TRUE(RunTestModelFromScratch([](RRHandle rri)
        {
            addCompartment(rri, "compartment", 1);
            addSpecies(rri, "S1", "compartment", 0, false, true);
            addRateRule(rri, "S1", "7");
        }));
}

TEST(MODEL_EDITING_TEST_SUITE, FROM_SCRATCH_4)
{
    EXPECT_TRUE(RunTestModelFromScratch([](RRHandle rri)
        {
            addCompartment(rri, "compartment", 1);
            addSpecies(rri, "S1", "compartment", 7, false, false);
            addAssignmentRule(rri, "S1", "7");
        }));
}

TEST(MODEL_EDITING_TEST_SUITE, FROM_SCRATCH_5)
{
    EXPECT_TRUE(RunTestModelFromScratch([](RRHandle rri)
        {
            addCompartment(rri, "compartment", 1);
            addSpecies(rri, "S1", "compartment", 1, false, false);
            addSpecies(rri, "S2", "compartment", 1.5e-15, false, false);
            addSpecies(rri, "S3", "compartment", 1, false, true);
            addParameter(rri, "k1", 0.75);
            addParameter(rri, "k2", 50);
            addAssignmentRule(rri, "S3", "k1*S2");
            const char* reactants[] = { "S1" };
            const char* products[] = { "S2" };
            addReaction(rri, "reaction1", reactants, 1, products, 1, "compartment * k2 * S1");
        }));
}

TEST(MODEL_EDITING_TEST_SUITE, FROM_SCRATCH_6)
{
    EXPECT_TRUE(RunTestModelFromScratch([](RRHandle rri)
        {
            addParameter(rri, "Q", 0);
            addParameter(rri, "R", 0);
            addParameter(rri, "reset", 0);
            addParameter(rri, "Q2", 0);
            addParameter(rri, "R2", 0);
            addParameter(rri, "reset2", 0);

            addEventNoRegen(rri, "Qinc", true, "(time - reset) >= 0.01");
            addEventAssignmentNoRegen(rri, "Qinc", "reset", "time");
            addEventAssignmentNoRegen(rri, "Qinc", "Q", "Q + 0.01");
            addPriority(rri, "Qinc", "1");

            addEventNoRegen(rri, "Rinc", true, "(time - reset) >= 0.01");
            addEventAssignmentNoRegen(rri, "Rinc", "reset", "time");
            addEventAssignmentNoRegen(rri, "Rinc", "R", "R + 0.01");
            addPriority(rri, "Rinc", "-1");

            addEvent(rri, "Qinc2", true, "(time - reset2) >= 0.01");
            addEventAssignment(rri, "Qinc2", "reset2", "time");
            addEventAssignment(rri, "Qinc2", "Q2", "Q2 + 0.01");
            addPriority(rri, "Qinc2", "-1");

            addEventNoRegen(rri, "Rinc2", true, "(time - reset2) >= 0.01");
            addEventAssignmentNoRegen(rri, "Rinc2", "reset2", "time");
            addEventAssignmentNoRegen(rri, "Rinc2", "R2", "R2 + 0.01");
            addPriority(rri, "Rinc2", "1");
        }, "l3v1"));
}

TEST(MODEL_EDITING_TEST_SUITE, FROM_SCRATCH_7)
{
    string modelFilePath(joinPath(gRRTestDir, "models/MODEL_EDITING_TEST_SUITE"));
    RRHandle rr = createRRInstance();
    loadSBML(rr, (modelFilePath + std::string("/tiny_example_1.xml")).c_str());
    addCompartmentNoRegen(rr, "c1", 3.0);
    addSpeciesNoRegen(rr, "S1", "c1", 0.0005, false, false);
    addSpeciesNoRegen(rr, "S2", "c1", 0.3, false, false);
    const char* reactants[] = { "S1" };
    const char* products[] = { "S1" };
    addReaction(rr, "reaction1", reactants, 1, products, 1, "c1 * S1 * S2");
    validateModifiedSBML(std::string(getSBML(rr)));
}

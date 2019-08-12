#include <string>
#include "Suite_TestModel.h"
#include "unit_test/UnitTest++.h"
#include "rrConfig.h"
#include "rrIniFile.h"
#include "rrLogger.h"
#include "rrRoadRunner.h"
#include "rrUtils.h"
#include "rrc_api.h"
#include "rrc_cpp_support.h"
#include "src/TestUtils.h"
#include "src/rrSBMLTestSuiteSimulation_CAPI.h"
#include "sbml/SBMLDocument.h"
#include "sbml/ListOf.h"
#include "sbml/Model.h"
#include "rrExecutableModel.h"

using namespace std;
using namespace UnitTest;
using namespace ls;
using namespace rrc;
using namespace rr;

extern string gTempFolder;
extern string gTSModelsPath;
extern string gCompiler;

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

	string testName(UnitTest::CurrentTest::Details()->testName);
	string suiteName(UnitTest::CurrentTest::Details()->suiteName);

	//Create instance..
	gRR = createRRInstanceEx(gTempFolder.c_str(), gCompiler.c_str());

	//Setup environment
	setTempFolder(gRR, gTempFolder.c_str());
	libsbml::SBMLDocument doc;

	if (!gRR)
	{
		return false;
	}

	try
	{
		Log(Logger::LOG_NOTICE) << "Running Test: " << testName << endl;
		string dataOutputFolder(joinPath(gTempFolder, suiteName));
		string dummy;
		string settingsFileName;

		setCurrentIntegratorParameterBoolean(gRR, "stiff", 0);

		//Create a log file name
		if (!createFolder(dataOutputFolder))
		{
			string msg("Failed creating output folder for data output: " + dataOutputFolder);
			throw(rr::Exception(msg));
		}
		//Create subfolder for data output
		dataOutputFolder = joinPath(dataOutputFolder, testName);

		if (!createFolder(dataOutputFolder))
		{
			string msg("Failed creating output folder for data output: " + dataOutputFolder);
			throw(rr::Exception(msg));
		}

		SBMLTestSuiteSimulation_CAPI simulation(dataOutputFolder);

		simulation.UseHandle(gRR);

		//Read SBML models.....
		string modelFilePath(joinPath(getParentFolder(getParentFolder(getParentFolder(gTSModelsPath))), suiteName));
		string modelFileName;

		simulation.SetCaseNumber(0);

		modelFilePath = joinPath(modelFilePath, testName);
		modelFileName = testName + "-sbml-" + version + ".xml";
		settingsFileName = testName + "-settings.txt";

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
		result = simulation.SaveModelAsXML(dataOutputFolder) && result;
		if (!result)
		{
			Log(Logger::LOG_WARNING) << "\t\t =============== Test " << testName << " failed =============\n";
		}
		else
		{
			Log(Logger::LOG_NOTICE) << "\t\tTest passed.\n";
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

SUITE(MODEL_EDITING_TEST_SUITE)
{
	TEST(ADD_REACTION_1)
	{
		CHECK(RunTestWithModification([](RRHandle rri) {
			const char* reactants[] = {"S2"};
			const char* products[] = {"S1"};
			addReaction(rri, "reaction2", reactants, 1, products, 1, "k1*S2");
		}));
	}

	TEST(REMOVE_REACTION_1)
	{
		CHECK(RunTestWithModification([](RRHandle rri) {
			removeReaction(rri, "reaction2");
		}));
	}

	TEST(ADD_SPECIES_1)
	{
		CHECK(RunTestWithModification([](RRHandle rri)
		{
			addSpecies(rri, "S3", "compartment", 0.0015, "substance");
		}));
	}

	TEST(REMOVE_SPECIES_1)
	{
		CHECK(RunTestWithModification([](RRHandle rri)
		{
			removeSpeciesNoRegen(rri, "S2");
			addSpeciesNoRegen(rri, "S3", "compartment", 0.00030, "substance");
			const char* reactants1[] = {"S1"};
			const char* products1[] = {"S3"};
			addReactionNoRegen(rri, "reaction1", reactants1, 1, products1, 1, "k1*S1");
			const char* reactants2[] = {"S3"};
			const char* products2[] = {"S1"};
			addReaction(rri, "reaction2", reactants2, 1, products2, 1, "k2*S3");
		}));
	}

	TEST(ADD_EVENT_1)
	{
		CHECK(RunTestWithModification([](RRHandle rri)
		{
			addEventNoRegen(rri, "event1", true, "S1 > 0.00015");
			addEventAssignment(rri, "event1", "S1", "1");
		}));
	}

	TEST(EVENT_PRIORITY_1)
	{
		CHECK(RunTestWithModification([](RRHandle rri)
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

	TEST(EVENT_DELAY_1)
	{
		CHECK(RunTestWithModification([](RRHandle rri)
		{
			addDelay(rri, "event1", "1");
		}));
	}

	TEST(REMOVE_EVENT_1)
	{
		CHECK(RunTestWithModification([](RRHandle rri)
		{
			removeEvent(rri, "event1");
		}));
	}
	
	TEST(ADD_PARAMETER_1)
	{
		CHECK(RunTestWithModification([](RRHandle rri)
		{
			addParameterNoRegen(rri, "k1", 0.75);
			const char* reactants[] = { "S1", "S2" };
			const char* products[] = {"S3"};
			addReaction(rri, "reaction1", reactants, 2, products, 1, "k1*S1*S2");
		}));
	}

	TEST(ADD_COMPARTMENT_1)
	{
		CHECK(RunTestWithModification([](RRHandle rri)
		{
			addCompartment(rri, "compartment", 1);
			const char* reactants[] = { "S1", "S2" };
			const char* products[] = {"2S2"};
			addReaction(rri, "reaction1", reactants, 2, products, 1, "compartment * k1 * S1 * S2");
		}));
	}

	TEST(ADD_ASSIGNMENT_RULE_1)
	{
		CHECK(RunTestWithModification([](RRHandle rri)
		{
			addAssignmentRule(rri, "S1", "7");
		}));
	}
	
	TEST(ADD_RATE_RULE_1)
	{
		CHECK(RunTestWithModification([](RRHandle rri)
		{
			addRateRule(rri, "S1", "7");
		}));
	}

	TEST(SET_KINETIC_LAW_1)
	{
		CHECK(RunTestWithModification([](RRHandle rri)
		{
			setKineticLaw(rri, "reaction2", "compartment * k2 * S3");
		}));
	}

	TEST(ADD_EVENT_ASSIGNMENT_1)
	{
		CHECK(RunTestWithModification([](RRHandle rri)
		{
			addEventAssignment(rri, "event2", "S3", "1");
		}));
	}

	TEST(ADD_TRIGGER_1)
	{
		CHECK(RunTestWithModification([](RRHandle rri)
		{
			addTrigger(rri, "event1", "S1 < 0.75");
		}));
	}

	TEST(PAUSE_10)
	{
		CHECK(RunTestWithModification([](RRHandle rri)
		{
			simulate(rri);
			addDelay(rri, "event1", "0.2");
		}));
	}

	TEST(REMOVE_COMPARTMENT_1)
	{
		CHECK(RunTestWithModification([](RRHandle rri)
		{
			removeCompartment(rri, "compartment");
		}));
	}

	TEST(REMOVE_PARAM_RECURSE_1)
	{
		CHECK(RunTestWithModification([](RRHandle rri)
		{
			removeParameter(rri, "k2");
		}));
	}

	TEST(REMOVE_EVENT_ASSIGNMENT_1)
	{
		CHECK(RunTestWithModification([](RRHandle rri)
		{
			removeEventAssignments(rri, "event1", "S2");
		}));
	}

	TEST(SET_PERSISTENT_1)
	{
		CHECK(RunTestWithModification([](RRHandle rri)
		{
			setPersistentNoRegen(rri, "event1", true);
			setPersistent(rri, "event2", true);
		}, "l3v1"));
	}

	TEST(SET_PERSISTENT_2)
	{
		CHECK(RunTestWithModification([](RRHandle rri)
		{
			setPersistent(rri, "event1", false);
		}, "l3v1"));
	}

	TEST(SET_CONSTANT_1)
	{
		CHECK(RunTestWithModification([](RRHandle rri)
		{
			setConstantNoRegen(rri, "k1", false);
			addRateRule(rri, "k1", "0.5");
		}));
	}

	TEST(SET_CONSTANT_2)
	{
		CHECK(RunTestWithModification([](RRHandle rri)
		{
			setConstantNoRegen(rri, "S1", false);
			addRateRule(rri, "S1", "7");
		}));
	}
	
	TEST(SET_HAS_ONLY_SUBSTANCE_UNITS_1)
	{
		CHECK(RunTestWithModification([](RRHandle rri)
		{
			setHasOnlySubstanceUnitsNoRegen(rri, "S1", true);
			setHasOnlySubstanceUnits(rri, "S2", true);
		}));
	}

	TEST(SET_HAS_ONLY_SUBSTANCE_UNITS_2)
	{
		CHECK(RunTestWithModification([](RRHandle rri)
		{
			setHasOnlySubstanceUnitsNoRegen(rri, "S1", true);
			setHasOnlySubstanceUnitsNoRegen(rri, "S2", true);
			setHasOnlySubstanceUnitsNoRegen(rri, "S3", true);
			setHasOnlySubstanceUnits(rri, "S4", true);
		}));
	}
	
	TEST(SET_HAS_ONLY_SUBSTANCE_UNITS_3)
	{
		CHECK(RunTestWithModification([](RRHandle rri)
		{
			setHasOnlySubstanceUnitsNoRegen(rri, "S1", false);
			setHasOnlySubstanceUnits(rri, "S2", false);
		}));
	}

	TEST(SET_INITIAL_CONCENTRATION_1)
	{
		CHECK(RunTestWithModification([](RRHandle rri)
		{
			addSpeciesNoRegen(rri, "S1", "C", 0.0, "");
			setInitConcentrationNoRegen(rri, "S1", 0.0004);
			addSpeciesNoRegen(rri, "S2", "C", 0.0, "");
			setInitConcentrationNoRegen(rri, "S2", 0.00048);
			addSpeciesNoRegen(rri, "S3", "C", 0.0, "");
			setInitConcentrationNoRegen(rri, "S3", 0.0008);
			addSpeciesNoRegen(rri, "S4", "C", 0.0, "");
			setInitConcentration(rri, "S4", 0.0004);

			const char* reactants1[] = {"S1", "S2"};
			const char* products1[] = {"S3", "S4"};

			addReactionNoRegen(rri, "reaction1", reactants1, 2, products1, 2, "C * k1 * S1 * S2");

			addReaction(rri, "reaction2", products1, 2, reactants1, 2, "C * k2 * S3 * S4");

			addEventNoRegen(rri, "event1", true, "S4 > S2");
			addEventAssignmentNoRegen(rri, "event1", "S1", "1/5000");
			addEventAssignment(rri, "event1", "S4", "1/5000");
		}));
	}

	TEST(SET_INITIAL_CONCENTRATION_2)
	{
		CHECK(RunTestWithModification([](RRHandle rri)
		{
			setInitConcentrationNoRegen(rri, "S1", 0.0004);
			setInitConcentrationNoRegen(rri, "S2", 0.00048);
			setInitConcentrationNoRegen(rri, "S3", 0.0008);
			setInitConcentration(rri, "S4", 0.0004);
		}));
	}

	TEST(SET_INITIAL_AMOUNT_1)
	{
		CHECK(RunTestWithModification([](RRHandle rri)
		{
			setInitAmountNoRegen(rri, "S1", 0.00015);
			setInitAmount(rri, "S2", 0);
		}));
	}

	TEST(SET_INITIAL_AMOUNT_2)
	{
		CHECK(RunTestWithModification([](RRHandle rri)
		{
			addSpecies(rri, "S1", "compartment", 0.0, "");
			setInitAmountNoRegen(rri, "S1", 0.00015);
			addSpecies(rri, "S2", "compartment", 0.0, "");
			setInitAmount(rri, "S2", 0);

			const char* reactants[] = {"S1"};
			const char* products[] = {"S2"};
			addReaction(rri, "reaction1", reactants, 1, products, 1, "compartment * k1 * S1");
			reset(rri);
		}));
	}

	TEST(SET_REVERSIBLE_1)
	{
		CHECK(RunTestWithModification([](RRHandle rri)
		{
			setReversible(rri, "reaction2", true);
		}));
	}

	TEST(SET_REVERSIBLE_2)
	{
		CHECK(RunTestWithModification([](RRHandle rri)
		{
			setReversible(rri, "reaction1", false);
		}));
	}

	TEST(SET_TRIGGER_INITIAL_VALUE_1)
	{
		CHECK(RunTestWithModification([](RRHandle rri)
		{
			setTriggerInitialValue(rri, "event1", false);
			reset(rri);
		}, "l3v1"));
	}

	TEST(SET_TRIGGER_INITIAL_VALUE_2)
	{
		CHECK(RunTestWithModification([](RRHandle rri)
		{
			setTriggerInitialValue(rri, "event1", true);
			reset(rri);
		}, "l3v1"));
	}

	TEST(SET_TRIGGER_INITIAL_VALUE_3)
	{
		CHECK(RunTestWithModification([](RRHandle rri)
		{
			setTriggerInitialValue(rri, "event1", true);
			reset(rri);
		}, "l3v1"));
	}
}
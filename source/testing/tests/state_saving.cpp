#include <string>
#include "unit_test/UnitTest++.h"
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

using namespace std;
using namespace UnitTest;
using namespace ls;
using namespace rr;

extern string gTempFolder;
extern string gTSModelsPath;
extern string gCompiler;

bool stateValidateModifiedSBML(std::string sbml)
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
bool RunStateSavingTest(void(*modification)(RoadRunner*), std::string version = "l2v4")
{
	bool result(false);
	RoadRunner *rr;

	//Create instance..
	rr = new RoadRunner();

	string testName(UnitTest::CurrentTest::Details()->testName);
	string suiteName(UnitTest::CurrentTest::Details()->suiteName);

	libsbml::SBMLDocument *doc;

	try
	{
		Log(Logger::LOG_NOTICE) << "Running Test: " << testName << endl;
		string dataOutputFolder(joinPath(gTempFolder, suiteName));
		string dummy;
		string logFileName;
		string settingsFileName;

		rr->getIntegrator()->setValue("stiff", false);

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

		TestSuiteModelSimulation simulation(dataOutputFolder);

		simulation.UseEngine(rr);

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
		rr->setConservedMoietyAnalysis(false);

		libsbml::SBMLReader reader;
		std::string fullPath = modelFilePath + "/" + modelFileName;
		doc = reader.readSBML(fullPath);

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


		LoadSBMLOptions opt;

		// don't generate cache for models
		opt.modelGeneratorOpt = opt.modelGeneratorOpt | LoadSBMLOptions::RECOMPILE;

		opt.modelGeneratorOpt = opt.modelGeneratorOpt | LoadSBMLOptions::MUTABLE_INITIAL_CONDITIONS;

		opt.modelGeneratorOpt = opt.modelGeneratorOpt & ~LoadSBMLOptions::READ_ONLY;

		opt.modelGeneratorOpt = opt.modelGeneratorOpt | LoadSBMLOptions::OPTIMIZE_CFG_SIMPLIFICATION;

		opt.modelGeneratorOpt = opt.modelGeneratorOpt | LoadSBMLOptions::OPTIMIZE_GVN;


		rr->load(fullPath, &opt);

		//Then read settings file if it exists..
		if (!simulation.LoadSettings(joinPath(modelFilePath, settingsFileName)))
		{
			throw(Exception("Failed loading simulation settings"));
		}
		modification(rr);
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
		delete rr;
		delete doc;
		return false;
	}

	delete rr;
	delete doc;
	return result;
}


bool RunStateSavingTest(int caseNumber, void(*modification)(RoadRunner*), std::string version = "l2v4")
{
	bool result(false);
	RoadRunner *rr;

	//Create instance..
	rr = new RoadRunner();
	string testName(UnitTest::CurrentTest::Details()->testName);
	string suiteName(UnitTest::CurrentTest::Details()->suiteName);

	//Setup environment
	//setTempFolder(gRR, gTempFolder.c_str());
	libsbml::SBMLDocument *doc;

	try
	{
		Log(Logger::LOG_NOTICE) << "Running Test: " << testName << endl;
		string dataOutputFolder(gTempFolder + "/" + testName);
		string dummy;
		string logFileName;
		string settingsFileName;

		rr->getIntegrator()->setValue("stiff", false);

		//Create a log file name
		createTestSuiteFileNameParts(caseNumber, ".log", dummy, logFileName, settingsFileName);
		if (!createFolder(dataOutputFolder))
		{
			string msg("Failed creating output folder for data output: " + dataOutputFolder);
			throw(rr::Exception(msg));
		}

		if (!createFolder(dataOutputFolder))
		{
			string msg("Failed creating output folder for data output: " + dataOutputFolder);
			throw(rr::Exception(msg));
		}

		TestSuiteModelSimulation simulation(dataOutputFolder);

		simulation.UseEngine(rr);

		//Read SBML models.....
		string modelFilePath(gTSModelsPath);
		string modelFileName;

		simulation.SetCaseNumber(caseNumber);
		createTestSuiteFileNameParts(caseNumber, "-sbml-" + version + ".xml", modelFilePath, modelFileName, settingsFileName);

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
		if (!simulation.LoadSettings(settingsOveride))
		{
			throw(Exception("Failed loading simulation settings"));
		}
		//Perform the model editing action
		modification(rr);
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

		if (!simulation.LoadReferenceData())
		{
			throw(Exception("Failed Loading reference data"));
		}

		simulation.CreateErrorData();
		result = simulation.Pass();
		result = simulation.SaveAllData() && result;
		result = simulation.SaveModelAsXML(dataOutputFolder) && result;
		if (!result)
		{
			Log(Logger::LOG_WARNING) << "\t\t =============== Test " << caseNumber << " failed =============\n";
		}
		else
		{
			Log(Logger::LOG_NOTICE) << "\t\tTest passed.\n";
		}
	}
	catch (std::exception& ex)
	{
		string error = ex.what();
		cerr << "Case " << caseNumber << ": Exception: " << error << endl;
		delete rr;
		delete doc;
		return false;
	}

	delete rr;
	delete doc;
	return result;
}

bool StateRunTestModelFromScratch(void(*generate)(RoadRunner*),std::string version = "l2v4")
{
	bool result(false);
	int level = version.at(1) - '0';
	int versionNum = version.at(3) - '0';
	RoadRunner rr(level, versionNum);


	string testName(UnitTest::CurrentTest::Details()->testName);
	string suiteName(UnitTest::CurrentTest::Details()->suiteName);

	libsbml::SBMLDocument *doc;

	try
	{
		Log(Logger::LOG_NOTICE) << "Running Test: " << testName << endl;
		string dataOutputFolder(joinPath(gTempFolder, suiteName));
		string dummy;
		string logFileName;
		string settingsFileName;

		rr.getIntegrator()->setValue("stiff", false);

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

		TestSuiteModelSimulation simulation(dataOutputFolder);

		simulation.UseEngine(&rr);

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
		result = simulation.SaveModelAsXML(dataOutputFolder) && result;
		result = stateValidateModifiedSBML(rr.getCurrentSBML()) && result;
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
		return false;
	}

	return result;
}

SUITE(STATE_SAVING_TEST_SUITE)
{
	TEST(SAVE_STATE_1)
	{
		CHECK(RunStateSavingTest(1, [](RoadRunner *rri)
		{
			rri->saveState("save-state-test.rr");
			rri->loadState("save-state-test.rr");
		}));
	}

	TEST(SAVE_STATE_2)
	{
		CHECK(RunStateSavingTest(1, [](RoadRunner *rri)
		{
			rri->saveState("save-state-test.rr");
			rri->loadState("save-state-test.rr");
			rri->saveState("save-state-test.rr");
			rri->loadState("save-state-test.rr");
		}));
	}

	TEST(SAVE_STATE_3)
	{
		CHECK(RunStateSavingTest(1, [](RoadRunner *rri)
		{
			rri->loadState("save-state-test.rr");
		}));
	}

	TEST(SAVE_STATE_4)
	{
		CHECK(RunStateSavingTest(1, [](RoadRunner *rri)
		{
			rri->loadState("save-state-test.rr");
			rri->loadState("save-state-test.rr");
		}));
	}

	TEST(SAVE_STATE_5)
	{
		CHECK(RunStateSavingTest(1, [](RoadRunner *rri)
		{
			rri->loadState("save-state-test.rr");
			rri->saveState("save-state-test.rr");
			rri->loadState("save-state-test.rr");
		}));
	}

	TEST(SAVE_STATE_6)
	{
		CHECK(RunStateSavingTest(1121, [](RoadRunner *rri)
		{
			rri->saveState("save-state-test.rr");
			rri->loadState("save-state-test.rr");
		}, "l3v1"));
	}

	TEST(SAVE_STATE_7)
	{
		CHECK(RunStateSavingTest(1121, [](RoadRunner *rri)
		{
			rri->saveState("save-state-test.rr");
			rri->loadState("save-state-test.rr");
			rri->saveState("save-state-test.rr");
			rri->loadState("save-state-test.rr");
		}, "l3v1"));
	}

	TEST(SAVE_STATE_8)
	{
		CHECK(RunStateSavingTest(1121, [](RoadRunner *rri)
		{
			rri->loadState("save-state-test.rr");
		}, "l3v1"));
	}

	TEST(SAVE_STATE_9)
	{
		CHECK(RunStateSavingTest(1121, [](RoadRunner *rri)
		{
			rri->loadState("save-state-test.rr");
			rri->loadState("save-state-test.rr");
		}, "l3v1"));
	}

	TEST(SAVE_STATE_10)
	{
		CHECK(RunStateSavingTest(1121, [](RoadRunner *rri)
		{
			rri->loadState("save-state-test.rr");
			rri->saveState("save-state-test.rr");
			rri->loadState("save-state-test.rr");
		}, "l3v1"));
	}

	TEST(SAVE_STATE_11)
	{
		CHECK(RunStateSavingTest(1, [](RoadRunner *rri)
		{
			rri->simulate();
			rri->saveState("save-state-test.rr");
			rri->loadState("save-state-test.rr");
			rri->reset();
		}));
	}

	TEST(SAVE_STATE_12)
	{
		CHECK(RunStateSavingTest(1121, [](RoadRunner *rri)
		{
			rri->simulate();
			rri->saveState("save-state-test.rr");
			rri->loadState("save-state-test.rr");
			rri->reset(SelectionRecord::ALL);
		}, "l3v1"));
	}

	TEST(SAVE_STATE_13)
	{
		CHECK(RunStateSavingTest(1120, [](RoadRunner *rri)
		{
			rri->saveState("save-state-test.rr");
			rri->loadState("save-state-test.rr");
		}, "l3v1"));
	}

	TEST(SAVE_STATE_14)
	{
		CHECK(RunStateSavingTest(1120, [](RoadRunner *rri)
		{
			rri->saveState("save-state-test.rr");
			rri->loadState("save-state-test.rr");
			rri->saveState("save-state-test.rr");
			rri->loadState("save-state-test.rr");
		}, "l3v1"));
	}

	TEST(SAVE_STATE_15)
	{
		CHECK(RunStateSavingTest(1120, [](RoadRunner *rri)
		{
			rri->loadState("save-state-test.rr");
		}, "l3v1"));
	}

	TEST(SAVE_STATE_16)
	{
		CHECK(RunStateSavingTest(1120, [](RoadRunner *rri)
		{
			rri->loadState("save-state-test.rr");
			rri->loadState("save-state-test.rr");
		}, "l3v1"));
	}

	TEST(SAVE_STATE_17)
	{
		CHECK(RunStateSavingTest(1120, [](RoadRunner *rri)
		{
			rri->loadState("save-state-test.rr");
			rri->saveState("save-state-test.rr");
			rri->loadState("save-state-test.rr");
		}, "l3v1"));
	}

	TEST(SAVE_STATE_18)
	{
		CHECK(RunStateSavingTest(1120, [](RoadRunner *rri)
		{
			rri->simulate();
			rri->saveState("save-state-test.rr");
			rri->loadState("save-state-test.rr");
			rri->reset(SelectionRecord::ALL);
		}, "l3v1"));
	}

	TEST(SAVE_STATE_19)
	{
		CHECK(RunStateSavingTest([](RoadRunner *rri)
		{
			rri->getSimulateOptions().duration /= 2;
			rri->getSimulateOptions().steps /= 2;
			rri->simulate();
			rri->saveState("save-state-test.rr");
			rri->loadState("save-state-test.rr");
			rri->getSimulateOptions().start = rri->getSimulateOptions().duration;
		}, "l3v1"));
	}

	TEST(SAVE_STATE_20)
	{
		CHECK(RunStateSavingTest([](RoadRunner *rri)
		{
			rri->getSimulateOptions().duration /= 2;
			rri->getSimulateOptions().steps /= 2;
			rri->simulate();
			rri->saveState("save-state-test.rr");
			rri->loadState("save-state-test.rr");
			rri->getSimulateOptions().start = rri->getSimulateOptions().duration;
		}));
	}

	TEST(SAVE_STATE_21)
	{
		CHECK(RunStateSavingTest([](RoadRunner *rri)
		{
			rri->getSimulateOptions().duration = 1.50492;
			rri->getSimulateOptions().steps /= 2;
			rri->simulate();
			rri->saveState("save-state-test.rr");
			rri->loadState("save-state-test.rr");
			rri->getSimulateOptions().start = rri->getSimulateOptions().duration;
			rri->getSimulateOptions().duration = 3.0 - rri->getSimulateOptions().duration;
		}, "l3v1"));
	}
	
	TEST(RETAIN_ABSOLUTE_TOLERANCES_1)
	{
		clog << "RETAIN_ABSOLUTE_TOLERANCES_1" << endl;
		RoadRunner rri(gTSModelsPath + "/00001/00001-sbml-l2v4.xml");
		rri.getIntegrator()->setIndividualTolerance("S1", 5.0);
		rri.getIntegrator()->setIndividualTolerance("S2", 3.0);
		rri.saveState("save-state-test.rr");
		RoadRunner rri2;
		rri2.loadState("save-state-test.rr");
		clog << rri2.getIntegrator()->getConcentrationTolerance()[0] << endl;
		clog << rri2.getIntegrator()->getConcentrationTolerance()[1] << endl;
		CHECK(rri2.getIntegrator()->getConcentrationTolerance()[0] == 5.0);
		CHECK(rri2.getIntegrator()->getConcentrationTolerance()[1] == 3.0);
	}

	TEST(FROM_SCRATCH_1)
	{
		CHECK(StateRunTestModelFromScratch([](RoadRunner *rri)
		{
			rri->addCompartment("compartment", 1);
			rri->saveState("test-save-state.rr");
			rri->loadState("test-save-state.rr");
			rri->addSpecies("S1", "compartment", 0.00015, "substance");
			rri->addSpecies("S2", "compartment", 0, "substance");
			rri->addParameter("k1", 1);
			rri->addReaction("reaction1", {"S1"}, {"S2"}, "compartment * k1 * S1");
		}));
	}

	TEST(FROM_SCRATCH_2)
	{
		CHECK(StateRunTestModelFromScratch([](RoadRunner *rri)
		{
			rri->addCompartment("compartment", 1);
			rri->addSpecies("S1", "compartment", 1, "substance");
			rri->addSpecies("S2", "compartment", 0, "substance");

			rri->saveState("test-save-state.rr");
			rri->loadState("test-save-state.rr");

			rri->addParameter("k1", 1);
			rri->addReaction("reaction1", {"S1"}, {"S2"}, "compartment * k1 * S1");
			rri->addEvent("event1", true, "S1 < 0.1");
			rri->addDelay("event1", "1");
			rri->addEventAssignment("event1", "S1", "1");
		}));
	}

	TEST(FROM_SCRATCH_3)
	{
		CHECK(StateRunTestModelFromScratch([](RoadRunner *rri)
		{
			rri->addCompartment("compartment", 1);
			rri->addSpecies("S1", "compartment", 0, "substance");
			rri->addRateRule("S1", "7");
			
			rri->saveState("test-save-state.rr");
			rri->loadState("test-save-state.rr");
		}));
	}
	
	TEST(FROM_SCRATCH_4)
	{
		CHECK(StateRunTestModelFromScratch([](RoadRunner *rri)
		{
			rri->addCompartment("compartment", 1);

			rri->saveState("test-save-state.rr");

			rri->addSpecies("S1", "compartment", 7, "substance");

			rri->loadState("test-save-state.rr");

			rri->addSpecies("S1", "compartment", 7, "substance");
			rri->addAssignmentRule("S1", "7");
		}));
	}

	TEST(FROM_SCRATCH_5)
	{
		CHECK(StateRunTestModelFromScratch([](RoadRunner *rri)
		{
			rri->addCompartment("compartment", 1);
			rri->saveState("test-save-state.rr");
			rri->loadState("test-save-state.rr");
			rri->addSpecies("S1", "compartment", 1, "substance");
			rri->addSpecies("S2", "compartment", 1.5e-15, "substance");
			rri->addSpecies("S3", "compartment", 1, "substance");
			rri->addParameter("k1", 0.75);
			rri->addParameter("k2", 50);
			rri->addAssignmentRule("S3", "k1*S2");
			rri->addReaction("reaction1", {"S1"}, {"S2"}, "compartment * k2 * S1");
		}));
	}

	TEST(FROM_SCRATCH_6)
	{
		CHECK(StateRunTestModelFromScratch([](RoadRunner *rri)
		{
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

	TEST(FROM_SCRATCH_7)
	{
		CHECK(StateRunTestModelFromScratch([](RoadRunner *rri)
		{
			rri->addCompartment("compartment", 1);
			rri->saveState("test-save-state.rr");
			rri->clearModel();
			rri->loadState("test-save-state.rr");
			rri->addSpecies("S1", "compartment", 0.00015, "substance");
			rri->addSpecies("S2", "compartment", 0, "substance");
			rri->addParameter("k1", 1);
			rri->addReaction("reaction1", {"S1"}, {"S2"}, "compartment * k1 * S1");
		}));
	}

	TEST(FROM_SCRATCH_8)
	{
		CHECK(StateRunTestModelFromScratch([](RoadRunner *rri)
		{
			rri->addCompartment("compartment", 1);
			rri->addSpecies("S1", "compartment", 1, "substance");
			rri->addSpecies("S2", "compartment", 0, "substance");

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

	TEST(FROM_SCRATCH_9)
	{
		CHECK(StateRunTestModelFromScratch([](RoadRunner *rri)
		{
			rri->addCompartment("compartment", 1);
			rri->addSpecies("S1", "compartment", 0, "substance");
			rri->addRateRule("S1", "7");
			
			rri->saveState("test-save-state.rr");
			rri->clearModel();
			rri->loadState("test-save-state.rr");
		}));
	}
}

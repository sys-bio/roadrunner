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
bool RunStateSavingTest(void(*modification)(RRHandle), std::string version = "l2v4")
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

bool RunStateSavingTest(int caseNumber, void(*modification)(RRHandle), std::string version = "l2v4")
{
	bool result(false);
	RRHandle gRR;

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
		Log(Logger::LOG_NOTICE) << "Running Test: " << caseNumber << endl;
		string dataOutputFolder(gTempFolder + "/state_saving");
		string dummy;
		string logFileName;
		string settingsFileName;

		setCurrentIntegratorParameterBoolean(gRR, "stiff", 0);

		//Create a log file name
		createTestSuiteFileNameParts(caseNumber, ".log", dummy, logFileName, settingsFileName);
		if (!createFolder(dataOutputFolder))
		{
			string msg("Failed creating output folder for data output: " + dataOutputFolder);
			throw(rr::Exception(msg));
		}
		//Create subfolder for data output
		dataOutputFolder = joinPath(dataOutputFolder, UnitTest::CurrentTest::Details()->testName);

		if (!createFolder(dataOutputFolder))
		{
			string msg("Failed creating output folder for data output: " + dataOutputFolder);
			throw(rr::Exception(msg));
		}

		SBMLTestSuiteSimulation_CAPI simulation(dataOutputFolder);

		simulation.UseHandle(gRR);

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
		string settingsOveride("");
		if (!simulation.LoadSettings(settingsOveride))
		{
			throw(Exception("Failed loading simulation settings"));
		}
		modification(rri);
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
		freeRRInstance(gRR);
		return false;
	}

	freeRRInstance(gRR);
	return result;
}

SUITE(STATE_SAVING_TEST_SUITE)
{
	TEST(SAVE_STATE_1)
	{
		CHECK(RunStateSavingTest(1, [](RRHandle rri)
		{
			saveState(rri, "save-state-test.rr");
			loadState(rri, "save-state-test.rr");
		}));
	}

	TEST(SAVE_STATE_2)
	{
		CHECK(RunStateSavingTest(1, [](RRHandle rri)
		{
			saveState(rri, "save-state-test.rr");
			loadState(rri, "save-state-test.rr");
			saveState(rri, "save-state-test.rr");
			loadState(rri, "save-state-test.rr");
		}));
	}

	TEST(SAVE_STATE_3)
	{
		CHECK(RunStateSavingTest(1, [](RRHandle rri)
		{
			loadState(rri, "save-state-test.rr");
		}));
	}

	TEST(SAVE_STATE_4)
	{
		CHECK(RunStateSavingTest(1, [](RRHandle rri)
		{
			loadState(rri, "save-state-test.rr");
			loadState(rri, "save-state-test.rr");
		}));
	}

	TEST(SAVE_STATE_5)
	{
		CHECK(RunStateSavingTest(1, [](RRHandle rri)
		{
			loadState(rri, "save-state-test.rr");
			saveState(rri, "save-state-test.rr");
			loadState(rri, "save-state-test.rr");
		}));
	}

	TEST(SAVE_STATE_6)
	{
		CHECK(RunStateSavingTest(1121, [](RRHandle rri)
		{
			saveState(rri, "save-state-test.rr");
			loadState(rri, "save-state-test.rr");
		}, "l3v1"));
	}

	TEST(SAVE_STATE_7)
	{
		CHECK(RunStateSavingTest(1121, [](RRHandle rri)
		{
			saveState(rri, "save-state-test.rr");
			loadState(rri, "save-state-test.rr");
			saveState(rri, "save-state-test.rr");
			loadState(rri, "save-state-test.rr");
		}, "l3v1"));
	}

	TEST(SAVE_STATE_8)
	{
		CHECK(RunStateSavingTest(1121, [](RRHandle rri)
		{
			loadState(rri, "save-state-test.rr");
		}, "l3v1"));
	}

	TEST(SAVE_STATE_9)
	{
		CHECK(RunStateSavingTest(1121, [](RRHandle rri)
		{
			loadState(rri, "save-state-test.rr");
			loadState(rri, "save-state-test.rr");
		}, "l3v1"));
	}

	TEST(SAVE_STATE_10)
	{
		CHECK(RunStateSavingTest(1121, [](RRHandle rri)
		{
			loadState(rri, "save-state-test.rr");
			saveState(rri, "save-state-test.rr");
			loadState(rri, "save-state-test.rr");
		}, "l3v1"));
	}

	TEST(SAVE_STATE_11)
	{
		CHECK(RunStateSavingTest(1, [](RRHandle rri)
		{
			simulate(rri);
			saveState(rri, "save-state-test.rr");
			loadState(rri, "save-state-test.rr");
			reset(rri);
		}));
	}

	TEST(SAVE_STATE_13)
	{
		CHECK(RunStateSavingTest(1120, [](RRHandle rri)
		{
			saveState(rri, "save-state-test.rr");
			loadState(rri, "save-state-test.rr");
		}, "l3v1"));
	}

	TEST(SAVE_STATE_14)
	{
		CHECK(RunStateSavingTest(1120, [](RRHandle rri)
		{
			saveState(rri, "save-state-test.rr");
			loadState(rri, "save-state-test.rr");
			saveState(rri, "save-state-test.rr");
			loadState(rri, "save-state-test.rr");
		}, "l3v1"));
	}

	TEST(SAVE_STATE_15)
	{
		CHECK(RunStateSavingTest(1120, [](RRHandle rri)
		{
			loadState(rri, "save-state-test.rr");
		}, "l3v1"));
	}

	TEST(SAVE_STATE_16)
	{
		CHECK(RunStateSavingTest(1120, [](RRHandle rri)
		{
			loadState(rri, "save-state-test.rr");
			loadState(rri, "save-state-test.rr");
		}, "l3v1"));
	}

	TEST(SAVE_STATE_17)
	{
		CHECK(RunStateSavingTest(1120, [](RRHandle rri)
		{
			loadState(rri, "save-state-test.rr");
			saveState(rri, "save-state-test.rr");
			loadState(rri, "save-state-test.rr");
		}, "l3v1"));
	}
}
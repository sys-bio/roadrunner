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
* Loads <prefix>/source/roadrunner/models/sbml-test-suite/cases/semantic/model_editing/NNNNN/NNNNN-sbml-*VERSION*.xml
* where NNNNN is case number
* applies modification to the resulting roadrunner instance and compares the result to NNNNN-results.csv in the same folder, 
* which should be the result of running the model NNNNN-sbml-*VERSION*-mod.xml, which should be the model expected after applying
* modification to the original model
* Returns true if the results are close enough, false otherwise
*/
bool RunModelEditingTest(int caseNumber, void(*modification)(RoadRunner*), std::string version = "l2v4")
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
		string dataOutputFolder(gTempFolder + "/model_editing");
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
		dataOutputFolder = joinPath(dataOutputFolder, getTestSuiteSubFolderName(caseNumber));

		if (!createFolder(dataOutputFolder))
		{
			string msg("Failed creating output folder for data output: " + dataOutputFolder);
			throw(rr::Exception(msg));
		}

		SBMLTestSuiteSimulation_CAPI simulation(dataOutputFolder);

		simulation.UseHandle(gRR);

		//Read SBML models.....
		string modelFilePath(gTSModelsPath + "/model_editing");
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

bool RunTestWithEdit(const string& version, int caseNumber, void(*edit)(RoadRunner*, libsbml::SBMLDocument*), std::string editName)
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
		string dataOutputFolder(gTempFolder + "/" + editName);
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
		dataOutputFolder = joinPath(dataOutputFolder, getTestSuiteSubFolderName(caseNumber));

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
		//Perform the model editing action
		edit(castToRoadRunner(gRR), &doc);
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

void readdAllReactions(RoadRunner *rri, libsbml::SBMLDocument *doc)
{
	libsbml::ListOfReactions *reactionsToAdd = doc->getModel()->getListOfReactions();
	std::vector<std::string> currReactionIds = rri->getReactionIds();
	for (int i = 0; i < reactionsToAdd->size(); i++)
	{
		libsbml::Reaction *next = reactionsToAdd->get(i);
		if (std::find(currReactionIds.begin(), currReactionIds.end(), next->getId()) ==
			currReactionIds.end())
			rri->addReaction(next->toSBML());
	}
}

void readdAllSpecies(RoadRunner *rri, libsbml::SBMLDocument *doc)
{
	libsbml::ListOfSpecies *speciesToAdd = doc->getModel()->getListOfSpecies();
	std::vector<std::string> currSpeciesIds = rri->getBoundarySpeciesIds();
	for (std::string s : rri->getFloatingSpeciesIds())
		currSpeciesIds.push_back(s);
	if (speciesToAdd->size() > 0)
	{
		libsbml::Species *next;
		for (int i = 0; i < speciesToAdd->size(); i++)
		{
			next = speciesToAdd->get(i);
			if (std::find(currSpeciesIds.begin(), currSpeciesIds.end(), next->getId()) == currSpeciesIds.end())
				rri->addSpecies(next->getId(), next->getCompartment(), next->getInitialConcentration(), "concentration", false);
		}
	}
}

void removeAndReaddAllSpecies(RoadRunner *rri, libsbml::SBMLDocument *doc)
{
	//Remove all species
	std::vector<std::string> floatingSpeciesIds = rri->getFloatingSpeciesIds();
	for (std::string sid : floatingSpeciesIds)
	{
		rri->removeSpecies(sid, false);
	}

	std::vector<std::string> boundarySpeciesIds = rri->getBoundarySpeciesIds();
	for (std::string sid : boundarySpeciesIds)
	{
		rri->removeSpecies(sid, false);
	}

	//Readd all species
	//libsbml::ListOfSpecies *speciesToAdd = doc->getModel()->getListOfSpecies();
	//if (speciesToAdd->size() > 0)
	//{
	//	libsbml::Species *next;
	//	for (int i = 0; i < speciesToAdd->size(); i++)
	//	{
	//		next = speciesToAdd->get(i);
	//		rri->addSpecies(next->getId(), next->getCompartment(), next->getInitialConcentration(), "concentration", false);
	//	}
	//}
	readdAllSpecies(rri, doc);

	readdAllReactions(rri, doc);
}

void removeAndReaddAllReactions(RoadRunner *rri, libsbml::SBMLDocument *doc)
{
	std::vector<std::string> reactionIds = rri->getReactionIds();
	for (std::string rid : reactionIds)
	{
		rri->removeReaction(rid, false);
	}
	readdAllReactions(rri, doc);
}

void removeAndReaddAllCompartments(RoadRunner *rri, libsbml::SBMLDocument *doc)
{
	std::vector<std::string> compartmentIds = rri->getCompartmentIds();
	for (std::string cid : compartmentIds)
	{
		rri->removeCompartment(cid, false);
	}

	libsbml::ListOfCompartments *compartmentsToAdd = doc->getModel()->getListOfCompartments();
	for (int i = 0; i < compartmentsToAdd->size(); i++)
	{
		libsbml::Compartment * next = compartmentsToAdd->get(i);
		rri->addCompartment(next->getId(), next->getVolume(), false);
	}
	readdAllSpecies(rri, doc);
	readdAllReactions(rri, doc);
}

SUITE(MODEL_EDITING_TEST_SUITE)
{
	/*TEST(ADD_REACTION_1)
	{
		CHECK(RunModelEditingTest(1, [](RoadRunner* rri) {
			rri->addReaction("reaction2", {"S2"}, {"S1"}, "k1*S2", true);
		}));
	}
	TEST(REMOVE_REACTION_1)
	{
		CHECK(RunModelEditingTest(2, [](RoadRunner* rri) {
			rri->removeReaction("reaction2");
		}));
	}
	TEST(ADD_REACTION_2)
	{
		CHECK(RunModelEditingTest(3, [](RoadRunner* rri) {
			rri->addReaction("reaction2", {"2S2", "S1"}, {"S1"}, "k1*S1 + k1*S2", true);
		}));
	}
	TEST(REMOVE_REACTION_2)
	{
		CHECK(RunModelEditingTest(4, [](RoadRunner* rri) {
			rri->removeReaction("reaction2");
		}));
	}
	TEST(REMOVE_REACTION_3)
	{
		CHECK(RunModelEditingTest(5, [](RoadRunner* rri) {
			rri->removeSpecies("S2");
		}));
	}*/
	/*TEST(ADD_SPECIES_1)
	{
		CHECK(RunModelEditingTest(6, [](RoadRunner *rri)
		{
			rri->addSpecies("S3", "compartment", 0.0015, "substance");
		}));
	}
	TEST(ADD_REACTION_3)
	{
		CHECK(RunModelEditingTest(7, [](RoadRunner *rri)
		{
			rri->addSpecies("S3", "compartment", 0.015, "substance", false);
			rri->addReaction("reaction3", { "S2" }, { "S3" }, "k2*S2");
		}));
	}
	TEST(ADD_REACTION_4)
	{
		CHECK(RunModelEditingTest(7, [](RoadRunner *rri)
		{
			rri->addSpecies("S3", "compartment", 0.015, "substance");
			rri->addReaction("reaction3", { "S2" }, { "S3" }, "k2*S2");
		}));
	}
	TEST(ADD_REACTION_5)
	{
		CHECK(RunModelEditingTest(8, [](RoadRunner *rri)
		{
			rri->addSpecies("S3", "compartment", 0.15, "substance", false);
			rri->addReaction("reaction3", { "S3" }, { "S1" }, "k2*S3");
		}));
	}
	TEST(REMOVE_SPECIES_1)
	{
		CHECK(RunModelEditingTest(9, [](RoadRunner *rri)
		{
			rri->removeSpecies("S2", false);
			rri->addSpecies("S3", "compartment", 0.00030, "substance", false);
			rri->addReaction("reaction1", { "S1" }, { "S3" }, "k1*S1", false);
			rri->addReaction("reaction2", { "S3" }, { "S1" }, "k2*S3", true);
		}));
	}

	TEST(REMOVE_SPECIES_2)
	{
		CHECK(RunModelEditingTest(14, [](RoadRunner *rri)
		{
			rri->removeSpecies("S4");
		}));
	}

	TEST(ADD_EVENT_1)
	{
		CHECK(RunModelEditingTest(10, [](RoadRunner *rri)
		{
			rri->addEvent("event1", true, "S1 > 0.00015", false);
			rri->addEventAssignment("event1", "S1", "1", true);
		}));
	}

	TEST(EVENT_PRIORITY_1)
	{
		CHECK(RunModelEditingTest(930, [](RoadRunner *rri)
		{
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

	TEST(EVENT_DELAY_1)
	{
		CHECK(RunModelEditingTest(71, [](RoadRunner *rri)
		{
			rri->addDelay("event1", "1");
		}));
	}

	TEST(REMOVE_EVENT_1)
	{
		CHECK(RunModelEditingTest(11, [](RoadRunner *rri)
		{
			rri->removeEvent("event1");
		}));
	}

	TEST(ADD_PARAMETER_1)
	{
		CHECK(RunModelEditingTest(12, [](RoadRunner *rri)
		{
			rri->addParameter("k1", 0.75, false);
			rri->addReaction("reaction1", { "S1", "S2" }, { "S3" }, "k1*S1*S2");
		}));
	}*/

	//Todo: Simulating after this removal *should* cause an error, find a way to check that
	/*
	TEST(REMOVE_PARAMETER_1)
	{
		CHECK(
			RunModelEditingTest(12, [](RoadRunner* rri)
			{
				rri->addParameter("k1", 0.75, false);
				rri->addReaction("reaction1", { "S1", "S2" }, { "S3" }, "k1*S1*S2");
				rri->removeParameter("k1");
			})
		);
	}*/

	/*TEST(ADD_COMPARTMENT_1)
	{
		CHECK(RunModelEditingTest(54, [](RoadRunner *rri)
		{
			rri->addCompartment("compartment", 1, false);
			rri->addReaction("reaction1", { "S1", "S2" }, { "2S2" }, "compartment * k1 * S1 * S2", true);
		}));
	}

	TEST(ADD_COMPARTMENT_2)
	{
		CHECK(RunModelEditingTest(56, [](RoadRunner *rri)
		{
			rri->addCompartment("compartment", 1, false);
			rri->addReaction("reaction1", { "S1" }, { "S3" }, "compartment * k1 * S1");
		}));
	}

	TEST(ADD_ASSIGNMENT_RULE_1)
	{
		CHECK(RunModelEditingTest(30, [](RoadRunner *rri)
		{
			rri->addAssignmentRule("S1", "7");
		}));
	}

	TEST(ADD_ASSIGNMENT_RULE_2)
	{
		CHECK(RunModelEditingTest(30, [](RoadRunner *rri)
		{
			rri->addAssignmentRule("S1", "7");
		}));
	}

	TEST(ADD_ASSIGNMENT_RULE_3)
	{
		CHECK(RunModelEditingTest(38, [](RoadRunner *rri)
		{
			rri->addAssignmentRule("S3", "k1 * S2");
		}));
	}

	TEST(ADD_RATE_RULE_1)
	{
		CHECK(RunModelEditingTest(31, [](RoadRunner *rri)
		{
			rri->addRateRule("S1", "7");
		}));
	}

	TEST(ADD_RATE_RULE_2)
	{
		CHECK(RunModelEditingTest(32, [](RoadRunner *rri)
		{
			rri->addRateRule("S1", "-1 * k1 * S1", false);
			rri->addRateRule("S2", "k1 * S1", true);
		}));
	}

	TEST(ADD_RATE_RULE_3)
	{
		CHECK(RunModelEditingTest(32, [](RoadRunner *rri)
		{
			rri->addRateRule("S1", "-1 * k1 * S1");
			rri->addRateRule("S2", "k1 * S1");
		}));
	}

	TEST(SET_KINETIC_LAW_1)
	{
		CHECK(RunModelEditingTest(13, [](RoadRunner *rri)
		{
			rri->setKineticLaw("reaction2", "compartment * k2 * S3");
		}));
	}

	TEST(SET_KINETIC_LAW_2)
	{
		CHECK(RunModelEditingTest(16, [](RoadRunner *rri)
		{
			rri->setKineticLaw("reaction1", "compartment * k1 * S1 * S2");
			rri->setKineticLaw("reaction2", "compartment * k2 * S3 * S4");
		}));
	}

	TEST(SET_KINETIC_LAW_3)
	{
		CHECK(RunModelEditingTest(16, [](RoadRunner *rri)
		{
			rri->setKineticLaw("reaction1", "compartment * k1 * S1 * S2", false);
			rri->setKineticLaw("reaction2", "compartment * k2 * S3 * S4", true);
		}));
	}

	TEST(TRANSFORM_1)
	{
		CHECK(RunModelEditingTest(47, [](RoadRunner *rri)
		{
			rri->removeSpecies("S1", true);
			rri->removeSpecies("S2", true);
			rri->addSpecies("S1", "compartment", 0.001, "substance", true);
			rri->addSpecies("S2", "compartment", 0.001, "substance", true);
			rri->addSpecies("S3", "compartment", 0.002, "substance", true);
			rri->addSpecies("S4", "compartment", 0.001, "substance", true);
			rri->removeParameter("k1", true);
			rri->addParameter("k1", 750, true);
			rri->addParameter("k2", 250, true);
			rri->addReaction("reaction1", {"S1", "S2"}, {"S3","S4"}, "compartment * k1 * S1 * S2", true);
			rri->addReaction("reaction2", {"S3", "S4"}, {"S1", "S2"}, "compartment * k2 * S3 * S4", true);
		}));
	}

	TEST(PAUSE_1)
	{
		CHECK(RunModelEditingTest(15, [](RoadRunner *rri)
		{
			rri->simulate();
			rri->addReaction("reaction3", {"S3"}, {"S2"}, "compartment * k2 * S3 * S4");
		}));
	}

	TEST(PAUSE_2)
	{
		CHECK(RunModelEditingTest(17, [](RoadRunner *rri)
		{
			rri->simulate();
			rri->removeReaction("reaction2");
		}));
	}

	TEST(PAUSE_3)
	{
		CHECK(RunModelEditingTest(18, [](RoadRunner *rri)
		{
			rri->simulate();
			rri->addSpecies("S5", "compartment", 0.001, "substance");
		}));
	}

	TEST(PAUSE_4)
	{
		CHECK(RunModelEditingTest(19, [](RoadRunner *rri)
		{
			rri->simulate();
			rri->removeSpecies("S4");
		}));
	}

	TEST(PAUSE_5)
	{
		CHECK(RunModelEditingTest(20, [](RoadRunner *rri)
		{
			rri->simulate();
			rri->setKineticLaw("reaction3", "sin(S2)");
		}));
	}

	TEST(PAUSE_6)
	{
		CHECK(RunModelEditingTest(21, [](RoadRunner *rri)
		{
			rri->simulate();
			rri->addEvent("event1", false, "S2 > 0.0004", false);
			rri->addEventAssignment("event1", "S1", "0.1", true);
		}));
	}*/

	/*TEST(PAUSE_7)
	{
		CHECK(RunModelEditingTest(348, [](RoadRunner *rri)
		{
			rri->simulate();
			rri->removeEvent("event1");
		}));
	}*/

	TEST(PAUSE_8)
	{
		CHECK(RunModelEditingTest(22, [](RoadRunner *rri)
		{
			rri->simulate();
			std::cout << rri->getFloatingSpeciesAmountsNamedArray() << std::endl;
			rri->addRateRule("S1", "7");
			std::cout << rri->getFloatingSpeciesAmountsNamedArray() << std::endl;
		}));
	}

	TEST(PAUSE_9)
	{
		CHECK(RunModelEditingTest(33, [](RoadRunner *rri)
		{
			rri->simulate();
			rri->removeRules("k1");
		}));
	}

	TEST(PAUSE_10)
	{
		CHECK(RunModelEditingTest(26, [](RoadRunner *rri)
		{
			rri->simulate();
			rri->addDelay("event1", "0.2");
		}));
	}

	TEST(ADD_EVENT_ASSIGNMENT_1)
	{
		CHECK(RunModelEditingTest(349, [](RoadRunner *rri)
		{
			rri->addEventAssignment("event2", "S3", "1");
		}));
	}
	/*TEST(READD_SPECIES)
	{
		clog << endl << "==== CHECK_READD_SPECIES ====" << endl << endl;
		for (int i = 30; i <= 40; i++)
		{
			if (!RunTestWithEdit("l2v4", i, removeAndReaddAllSpecies, "removeAndReaddAllSpecies"))
			{
				std::string failureMessage = "SBML Test " + to_string(i) + " failed";
				UnitTest::CurrentTest::Results()->OnTestFailure(*UnitTest::CurrentTest::Details(), failureMessage.c_str());
			}
		}
	}
	/*TEST(READD_REACTION)
	{
		clog << endl << "==== CHECK_READD_REACTION ====" << endl << endl;
		for (int i = 40; i <= 40; i++)
		{
			if (!RunTestWithEdit("l2v4", i, removeAndReaddAllReactions, "removeAndReaddAllReactions"))
			{
				std::string failureMessage = "SBML Test " + to_string(i) + " failed";
				UnitTest::CurrentTest::Results()->OnTestFailure(*UnitTest::CurrentTest::Details(), failureMessage.c_str());
			}
		}
	}
	TEST(READD_COMPARTMENTS)
	{
		clog << endl << "==== CHECK_READD_COMPARTMENTS ====" << endl << endl;
		for (int i = 30; i <= 40; i++)
		{
			if (!RunTestWithEdit("l2v4", i, removeAndReaddAllCompartments, "removeAndReaddAllCompartments"))
			{
				std::string failureMessage = "SBML Test " + to_string(i) + " failed";
				UnitTest::CurrentTest::Results()->OnTestFailure(*UnitTest::CurrentTest::Details(), failureMessage.c_str());
			}
		}
	}*/
}
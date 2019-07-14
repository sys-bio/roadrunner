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

using namespace std;
using namespace UnitTest;
using namespace ls;
using namespace rrc;
using namespace rr;

extern string gTempFolder;
extern string gTSModelsPath;
extern string gCompiler;

bool RunTestWithEdit(const string& version, int caseNumber, void (*edit)(RoadRunner*, libsbml::SBMLDocument*))
{
    bool result(false);
    RRHandle gRR;

    //Create instance..
    gRR = createRRInstanceEx(gTempFolder.c_str(), gCompiler.c_str());


    //Setup environment
    setTempFolder(gRR, gTempFolder.c_str());
	libsbml::SBMLDocument *doc;

    if(!gRR)
    {
        return false;
    }

    try
    {
        Log(Logger::LOG_NOTICE) << "Running Test: "<< caseNumber << endl;
        string dataOutputFolder(gTempFolder);
        string dummy;
        string logFileName;
        string settingsFileName;

        setCurrentIntegratorParameterBoolean(gRR, "stiff", 0);

        //Create a log file name
        createTestSuiteFileNameParts(caseNumber, ".log", dummy, logFileName, settingsFileName);

        //Create subfolder for data output
        dataOutputFolder = joinPath(dataOutputFolder, getTestSuiteSubFolderName(caseNumber));

        if(!createFolder(dataOutputFolder))
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
		doc = reader.readSBML(fullPath);

        if(!simulation.LoadSBMLFromFile())
        {
            throw(Exception("Failed loading sbml from file"));
        }


        //Check first if file exists first
        if(!fileExists(fullPath))
        {
            Log(Logger::LOG_ERROR) << "sbml file " << fullPath << " not found";
			throw(Exception("No such SBML file: " + fullPath));
        }

        RoadRunner* rri = (RoadRunner*) gRR;

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
        if(!simulation.LoadSettings(settingsOveride))
        {
            throw(Exception("Failed loading simulation settings"));
        }
        //Perform the model editing action
		edit(castToRoadRunner(gRR), doc);
        //Then Simulate model
        if(!simulation.Simulate())
        {
            throw(Exception("Failed running simulation"));
        }

        //Write result
        if(!simulation.SaveResult())
        {
            //Failed to save data
            throw(Exception("Failed saving result"));
        }

        if(!simulation.LoadReferenceData())
        {
            throw(Exception("Failed Loading reference data"));
        }

        simulation.CreateErrorData();
        result = simulation.Pass();
        result = simulation.SaveAllData() && result;
        result = simulation.SaveModelAsXML(dataOutputFolder) && result;
        if(!result)
        {
            Log(Logger::LOG_WARNING)<<"\t\t =============== Test "<<caseNumber<<" failed =============\n";
        }
        else
        {
            Log(Logger::LOG_NOTICE)<<"\t\tTest passed.\n";
        }
    }
    catch(std::exception& ex)
    {
        string error = ex.what();
        cerr<<"Case "<<caseNumber<<": Exception: "<<error<<endl;
        freeRRInstance(gRR);
		delete doc;
        return false;
    }

    freeRRInstance(gRR);
	delete doc;
    return result;
}

void readdAllReactions(RoadRunner *rri, libsbml::SBMLDocument *doc)
{
	libsbml::ListOfReactions *reactionsToAdd = doc->getModel()->getListOfReactions();
	std::vector<std::string> currReactionIds = rri->getReactionIds();
	if (reactionsToAdd->size() > 0) 
	{
		for (int i = 0; i < reactionsToAdd->size() - 1; i++)
		{
			libsbml::Reaction *next = reactionsToAdd->get(i);
			if(std::find(currReactionIds.begin(), currReactionIds.end(), next->getId()) != 
				   currReactionIds.end())
				rri->addReaction(next->toSBML(), false);
		}
		if(std::find(currReactionIds.begin(), currReactionIds.end(), reactionsToAdd->get(reactionsToAdd->size() - 1)->getId()) != 
			   currReactionIds.end())
			rri->addReaction(reactionsToAdd->get(reactionsToAdd->size() - 1)->toSBML(), true);
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
	libsbml::ListOfSpecies *speciesToAdd = doc->getModel()->getListOfSpecies();
	if (speciesToAdd->size() > 0)
	{
		libsbml::Species *next;
		for (int i = 0; i < speciesToAdd->size() - 1; i++)
		{
			next = speciesToAdd->get(i);
			rri->addSpecies(next->getId(), next->getCompartment(), next->getInitialConcentration(), "concentration", false);
		}
		next = speciesToAdd->get(speciesToAdd->size() - 1);
		rri->addSpecies(next->getId(), next->getCompartment(), next->getInitialConcentration(), "concentration", true);
	}

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

SUITE(MODEL_EDITING_TEST_SUITE)
{
	TEST(READD_FLOATING_SPECIES)
	{
		clog << endl << "==== CHECK_READD_FLOATING_SPECIES ====" << endl << endl;
		for (int i = 1; i < 530; i++)
		{
			CHECK(RunTestWithEdit("l2v4", i, removeAndReaddAllSpecies) && ("SBML Test " + to_string(i)).c_str());
		}
	}
	TEST(READD_REACTION)
	{
        clog << endl << "==== CHECK_READD_REACTION ====" << endl << endl;
		for (int i = 1; i < 530; i++)
		{
			CHECK(RunTestWithEdit("l2v4", i, removeAndReaddAllReactions) && ("SBML Test " + to_string(i)).c_str());
		}
	}
}
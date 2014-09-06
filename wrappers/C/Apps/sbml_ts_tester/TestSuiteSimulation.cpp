#pragma hdrstop
#include "rrException.h"
#include "rrUtils.h"
#include "rrRoadRunnerData.h"
#include "TestSuiteSimulation.h"
#include "rrLogger.h"

extern string gTSModelsPath;
extern string gTempFolder;
using namespace rr;
using namespace std;

RoadRunnerData convertCAPIResultData(RRCDataPtr        resultsHandle);

TestSuiteSimulation::TestSuiteSimulation(const string& dataOutputFolder, const string& modelFilePath, const string& modelFileName)
:
        rr::TestSuiteModelSimulation(dataOutputFolder, modelFilePath, modelFileName),
        mRRHandle(0)
{
}

TestSuiteSimulation::~TestSuiteSimulation()
{
}

void TestSuiteSimulation::UseHandle(RRHandle handle)
{
    mRRHandle = handle;
    if(mRRHandle)
    {
        this->UseEngine((RoadRunner*) mRRHandle);
    }
}


bool TestSuiteSimulation::LoadSBMLFromFile()
{
    if(!mRRHandle)
    {
        return false;
    }

    return loadSBMLFromFileE(mRRHandle, GetModelsFullFilePath().c_str(), true);
}

bool TestSuiteSimulation::LoadSettings(const string& settingsFName)
{
    mModelSettingsFileName = (settingsFName);

    if(!mModelSettingsFileName.size())
    {
        mModelSettingsFileName = joinPath(mModelFilePath, GetSettingsFileNameForCase(mCurrentCaseNumber));
    }

    SBMLModelSimulation::LoadSettings(mModelSettingsFileName);

    return true;
}

//Todo: bisarre to copy stuff back and forth..!
bool TestSuiteSimulation::Simulate()
{
    if(!mRRHandle)
    {
        return false;
    }

    RRCDataPtr resultData = simulate(mRRHandle);

    if (!resultData )
    {
        Log(Logger::LOG_ERROR) << "Failed simulate in test " << mCurrentCaseNumber
                << ", error: " << rrc::getLastError();
        return false;
    }

    mResultData = convertCAPIResultData(resultData);

    freeRRCData(resultData);
    return true;
}

RoadRunnerData TestSuiteSimulation::GetResult()
{
    return mResultData; //Not that pretty.
}

bool TestSuiteSimulation::SaveResult()
{
    string resultFileName(joinPath(mDataOutputFolder, "rrCAPI_" + mModelFileName));
    resultFileName = changeFileExtensionTo(resultFileName, ".csv");


    ofstream fs(resultFileName.c_str());
    fs << mResultData;
    fs.close();
    return true;
}

RoadRunnerData convertCAPIResultData(RRCDataPtr    result)
{
    RoadRunnerData resultData;

    std::vector<std::string> colNames;
    //Copy column names
    for(int i = 0; i < result->CSize; i++)
    {
        colNames.push_back(result->ColumnHeaders[i]);
    }

    resultData.setColumnNames(colNames);

    //Then the data
    int index = 0;
    resultData.allocate(result->RSize, result->CSize);

    for(int j = 0; j < result->RSize; j++)
    {
        for(int i = 0; i < result->CSize; i++)
        {
            resultData(j,i) = result->Data[index++];
        }
    }

    return resultData;
}


bool RunTest(const string& version, int caseNumber, const std::string& compiler)
{
    bool result = false;
    RRHandle gRR = 0;

    //Create instance..
    gRR = createRRInstanceEx(gTempFolder.c_str(), compiler.c_str());


    //Setup environment
    setTempFolder(gRR, gTempFolder.c_str());

    if(!gRR)
    {
        return false;
    }

    try
    {
        Log(Logger::LOG_NOTICE) << "Running Test: "<<caseNumber<<endl;
        string dataOutputFolder(gTempFolder);
        string dummy;
        string logFileName;
        string settingsFileName;

        //Create a log file name
        createTestSuiteFileNameParts(caseNumber, ".log", dummy, logFileName, settingsFileName);

        //Create subfolder for data output
        dataOutputFolder = joinPath(dataOutputFolder, getTestSuiteSubFolderName(caseNumber));

        if(!createFolder(dataOutputFolder))
        {
            string msg("Failed creating output folder for data output: " + dataOutputFolder);
            throw(rr::Exception(msg));
        }

        TestSuiteSimulation simulation(dataOutputFolder);

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

        if(!simulation.LoadSBMLFromFile())
        {
            throw Exception("Failed loading sbml from file");
        }

        //Then read settings file if it exists..
        string settingsOveride("");
        if(!simulation.LoadSettings(settingsOveride))
        {
            throw Exception("Failed loading simulation settings");
        }

        //Then Simulate model
        if(!simulation.Simulate())
        {
            throw Exception ("Failed running simulation");
        }

        //Write result
        if(!simulation.SaveResult())
        {
            //Failed to save data
            throw Exception("Failed saving result");
        }

        if(!simulation.LoadReferenceData())
        {
            throw Exception("Failed Loading reference data");
        }

        simulation.CreateErrorData();
        result = simulation.Pass();
        simulation.SaveAllData();
        simulation.SaveModelAsXML(dataOutputFolder);
        if(!result)
        {
            Log(Logger::LOG_NOTICE) << "\t\tTest failed using compiler \'" << getCompiler(gRR) << "\'" << endl;
        }
        else
        {
            Log(Logger::LOG_NOTICE) <<"\t\tTest passed using compiler \'" << getCompiler(gRR) << "\'" << endl;
        }
    }
    catch(rr::Exception& ex)
    {
        string error = ex.what();
        Log(Logger::LOG_NOTICE) << "Case "<<caseNumber<<": Exception: "<<error<<endl;
        result = false;;
    }

    // done with rr
    freeRRInstance(gRR);
    return result;
}

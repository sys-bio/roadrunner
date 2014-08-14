#pragma hdrstop
#include <iomanip>
#include <map>
#include "rrLogger.h"
#include "rrSBMLModelSimulation.h"
#include "rrUtils.h"
#include "rrRoadRunner.h"
//---------------------------------------------------------------------------

using namespace std;

namespace rr
{

SBMLModelSimulation::SBMLModelSimulation(const string& dataOutputFolder, const string& tempDataFilePath)
:
mModelFilePath(""),
mModelFileName(""),
mDataOutputFolder(dataOutputFolder),
mCompileIfDllExists(true),
mTempDataFolder(tempDataFilePath),
mEngine(NULL)
{
    mSettings.absolute    = 1.e-7;
    mSettings.relative    = 1.e-4;
}

SBMLModelSimulation::~SBMLModelSimulation()
{}

string SBMLModelSimulation::GetTempDataFolder()
{
    return mTempDataFolder;
}

void SBMLModelSimulation::ReCompileIfDllExists(const bool& doIt)
{
    mCompileIfDllExists = doIt;
}

bool SBMLModelSimulation::SetModelFilePath(const string& path)
{
    mModelFilePath = path;
    return true;
}

RoadRunnerData SBMLModelSimulation::GetResult()
{
    if(mEngine)
    {
        return RoadRunnerData(mEngine);
    }
    else
    {
        return RoadRunnerData();
    }
}

bool SBMLModelSimulation::SetModelFileName(const string& name)
{
    if(getFilePath(name).size() > 0)
    {
        mModelFilePath = getFilePath(name);
    }

    mModelFileName = getFileName(name);

    if(!fileExists(joinPath(mModelFilePath, mModelFileName)))
    {
        Log(Logger::LOG_ERROR)<<"The file: "<<joinPath(mModelFilePath, mModelFileName)<<" doesn't exist.";
        return false;
    }

    return true;
}

bool SBMLModelSimulation::SetDataOutputFolder(const string& name)
{
    mDataOutputFolder = name;
    return true;
}

string  SBMLModelSimulation::GetModelsFullFilePath()
{
    return joinPath(mModelFilePath, mModelFileName);
}

string  SBMLModelSimulation::GetDataOutputFolder()
{
    return mDataOutputFolder;
}

bool SBMLModelSimulation::DoCompileIfDllExists()
{
    return mCompileIfDllExists;
}

bool SBMLModelSimulation::UseEngine(RoadRunner* engine)
{
    mEngine = engine;
    return true;
}

bool SBMLModelSimulation::GenerateModelCode()
{
    return true;
}

bool SBMLModelSimulation::CompileModel()
{
    return true;
}

bool SBMLModelSimulation::LoadSettings(const string& settingsFName)
{
    string fName(settingsFName);

    if(!fName.size())
    {
        Log(Logger::LOG_ERROR)<<"Empty file name for setings file";
        return false;
    }

    mSettings = SimulateOptions(fName);

    if(mEngine)
    {
        // make a copy and tweak tolerances for integrator
        SimulateOptions opt = mSettings;
        opt.tweakTolerances();
        mEngine->setSimulateOptions(opt);
    }

    return true;
}

bool SBMLModelSimulation::SetTimeStart(const double& startTime)
{
    mSettings.start   = startTime;
    return true;
}

bool SBMLModelSimulation::SetTimeEnd(const double& endTime)
{
    mSettings.duration = endTime - mSettings.start;
    return true;
}

bool SBMLModelSimulation::SetNumberOfPoints(const int& steps)
{
    mSettings.steps       = steps;
    return true;
}

bool SBMLModelSimulation::SetSelectionList(const string& selectionList)
{
    vector<string> vars = splitString(selectionList, ", ");
    for(u_int i = 0; i < vars.size(); i++)
    {
        mSettings.variables.push_back(trim(vars[i]));
    }

    mEngine->setSimulateOptions(mSettings);

    return true;
}


bool SBMLModelSimulation::LoadSBMLFromFile()                    //Use current file information to load sbml from file
{
    if(!mEngine)
    {
        return false;
    }

    LoadSBMLOptions opt;
    opt.modelGeneratorOpt = mCompileIfDllExists ?
            opt.modelGeneratorOpt | LoadSBMLOptions::RECOMPILE :
            opt.modelGeneratorOpt & ~LoadSBMLOptions::RECOMPILE;

    mEngine->load(GetModelsFullFilePath(), &opt);
    return true;
}

bool SBMLModelSimulation::SaveModelAsXML(const string& folder)
{
    if(!mEngine)
    {
        return false;
    }
    string fName = joinPath(folder, mModelFileName);
    fName = changeFileExtensionTo(fName, "xml");

    fstream fs(fName.c_str(), fstream::out);

    if(!fs)
    {
        Log(Logger::LOG_ERROR)<<"Failed writing sbml to file "<< fName;
        return false;
    }
    fs<<mEngine->getSBML();
    fs.close();
    return true;
}

bool SBMLModelSimulation::CreateModel()
{
    if(!mEngine)
    {
        return false;
    }

    return true;
}


bool SBMLModelSimulation::GenerateAndCompileModel()
{
    return true;
}

bool SBMLModelSimulation::Simulate()
{
    if(!mEngine)
    {
        return false;
    }

    return mEngine->simulate(0) != 0;
}

bool SBMLModelSimulation::SaveResult()
{
    string resultFileName(joinPath(mDataOutputFolder, "rr_" + mModelFileName));
    resultFileName = changeFileExtensionTo(resultFileName, ".csv");
    Log(lInfo)<<"Saving result to file: "<<resultFileName;
    RoadRunnerData resultData(mEngine);

    ofstream fs(resultFileName.c_str());
    fs << resultData;
    fs.close();
    return true;
}
} //end of namespace



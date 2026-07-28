#pragma hdrstop
#include <iomanip>
#include <map>
#include <utility>
#include "rrLogger.h"
#include "rrSBMLModelSimulation.h"
#include "rrUtils.h"
#include "rrRoadRunner.h"
#include "CVODEIntegrator.h"
//---------------------------------------------------------------------------

using std::filesystem::path;

namespace rr
{
    /** @cond PRIVATE */

    SBMLModelSimulation::SBMLModelSimulation(std::filesystem::path  dataOutputFolder, std::filesystem::path  tempDataFilePath)
        :
        mModelFilePath(""),
        mModelFileName(""),
        mDataOutputFolder(std::move(dataOutputFolder)),
        mCompileIfDllExists(true),
        mTempDataFolder(std::move(tempDataFilePath)),
        mEngine(NULL)
    {
        // TODO mSettings.absolute    = 1.e-7;
        // TODO mSettings.relative    = 1.e-4;
    }

    SBMLModelSimulation::~SBMLModelSimulation()
    {}

    std::filesystem::path SBMLModelSimulation::GetTempDataFolder()
    {
        return mTempDataFolder;
    }

    void SBMLModelSimulation::ReCompileIfDllExists(const bool& doIt)
    {
        mCompileIfDllExists = doIt;
    }

    bool SBMLModelSimulation::SetModelFilePath(const std::filesystem::path& p)
    {
        mModelFilePath = p;
        return true;
    }

    RoadRunnerData SBMLModelSimulation::GetResult()
    {
        if (mEngine)
        {
            return RoadRunnerData(mEngine);
        }
        else
        {
            return RoadRunnerData();
        }
    }

    bool SBMLModelSimulation::SetModelFileName(const std::string& name)
    {
        if (!getFilePath(name).empty())
        {
            mModelFilePath = getFilePath(name);
        }

        mModelFileName = getFileName(name);

        path p = path(mModelFilePath) / mModelFileName;


        if (!std::filesystem::exists(p))
        {
            rrLog(Logger::LOG_ERROR) << "The file: " << p << " doesn't exist.";
            return false;
        }

        return true;
    }

    bool SBMLModelSimulation::SetDataOutputFolder(const std::filesystem::path& name)
    {
        mDataOutputFolder = name;
        return true;
    }

    std::filesystem::path  SBMLModelSimulation::GetModelsFullFilePath()
    {
        return mModelFilePath / mModelFileName;
    }

    std::filesystem::path  SBMLModelSimulation::GetDataOutputFolder()
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

    void SBMLModelSimulation::loadSBMLTolerances(std::filesystem::path const& filename)
    {
        if (filename.string().empty())
        {
            rrLog(Logger::LOG_ERROR) << "Empty file name for settings file";
        }
        else
        {
            std::map<std::string, std::string> options;
            std::map<std::string, std::string>::iterator it;
            //Read each line in the settings file
            std::vector<std::string> lines = getLinesInFile(filename);
            for (int i = 0; i < lines.size(); i++)
            {
                std::vector<std::string> line = splitString(lines[i], ":");
                if (line.size() == 2)
                {
                    options.insert(std::pair<std::string, std::string>(line[0], line[1]));
                }
                else
                {
                    rrLog(Logger::LOG_DEBUG) << "Empty line in settings file: " << lines[i];
                }
            }

            rrLog(Logger::LOG_DEBUG) << "Settings File =============";
            for (it = options.begin(); it != options.end(); it++)
            {
                rrLog(Logger::LOG_DEBUG) << (*it).first << " => " << (*it).second;
            }
            rrLog(Logger::LOG_DEBUG) << "===========================";

            //Assign values
            it = options.find("absolute");
            if (it != options.end())
            {
                mAbsolute = std::abs(toDouble((*it).second));
            }

            it = options.find("relative");
            if (it != options.end())
            {
                mRelative = std::abs(toDouble((*it).second));
            }
        }
    }

    bool SBMLModelSimulation::LoadSettings(const std::filesystem::path& settingsFName)
    {
        std::filesystem::path fName(settingsFName);

        if (fName.string().empty())
        {
            rrLog(Logger::LOG_ERROR) << "Empty file name for setings file";
            return false;
        }

        mAbsolute = 1e-10;
        mRelative = 1e-5;
        loadSBMLTolerances(fName);

        mSettings = SimulateOptions();
        mSettings.loadSBMLSettings(fName.string());

        if (mEngine)
        {
            mEngine->getIntegrator()->loadSBMLSettings(fName.string());
            // make a copy and tweak tolerances for integrator
            SimulateOptions opt = mSettings;
            if (mEngine->getIntegrator()->getName() == "cvode")
            {
                // Do the tolerance tweaking here.
                //opt.tweakTolerances();
                CVODEIntegrator* cvode_integrator = dynamic_cast<CVODEIntegrator*>(mEngine->getIntegrator());
                if (cvode_integrator)
                {
                    cvode_integrator->tweakTolerances();
                }
                else
                {
                    throw std::runtime_error("Cannot tweak tolerances of integrator because it is not CVODE.");
                }
            }
            mEngine->setSimulateOptions(opt);
        }

        return true;
    }

    bool SBMLModelSimulation::SetTimeStart(const double& startTime)
    {
        mSettings.setStartTime(startTime);
        return true;
    }

    bool SBMLModelSimulation::SetTimeEnd(const double& endTime)
    {
        mSettings.setDuration(endTime - mSettings.getStartTime());
        return true;
    }

    bool SBMLModelSimulation::SetNumberOfPoints(const int& steps)
    {
        mSettings.setSteps(steps);
        return true;
    }

    bool SBMLModelSimulation::SetSelectionList(const std::string& selectionList)
    {
        std::vector<std::string> vars = splitString(selectionList, ", ");
        for (u_int i = 0; i < vars.size(); i++)
        {
            mSettings.variables.push_back(trim(vars[i]));
        }

        mEngine->setSimulateOptions(mSettings);

        return true;
    }


    bool SBMLModelSimulation::LoadSBMLFromFile()                    //Use current file information to load sbml from file
    {
        if (!mEngine)
        {
            return false;
        }

        LoadSBMLOptions opt;
        opt.modelGeneratorOpt = mCompileIfDllExists ?
            opt.modelGeneratorOpt | LoadSBMLOptions::RECOMPILE :
            opt.modelGeneratorOpt & ~LoadSBMLOptions::RECOMPILE;

        mEngine->load(GetModelsFullFilePath().string(), &opt);
        return true;
    }

    bool SBMLModelSimulation::SaveModelAsXML(std::filesystem::path& folder)
    {
        if (!mEngine)
        {
            return false;
        }
        std::filesystem::path fName = folder / mModelFileName;
        fName = changeFileExtensionTo(fName.string(), "xml");

        std::fstream fs(fName.c_str(), std::fstream::out);

        if (!fs)
        {
            rrLog(Logger::LOG_ERROR) << "Failed writing sbml to file " << fName;
            return false;
        }
        fs << mEngine->getSBML();
        fs.close();
        return true;
    }

    bool SBMLModelSimulation::CreateModel()
    {
        if (!mEngine)
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
        if (!mEngine)
        {
            return false;
        }

        return mEngine->simulate(0) != 0;
    }

    bool SBMLModelSimulation::SaveResult()
    {
        string onlyNumber = mModelFileName;
        onlyNumber.replace(onlyNumber.find("-sbml"), 10, "");
        std::filesystem::path resultFileName(mDataOutputFolder / ("rr_result-" + onlyNumber));
        resultFileName = resultFileName.replace_extension(".csv");
        rrLog(lInfo) << "Saving result to file: " << resultFileName;
        RoadRunnerData resultData(mEngine);

        std::ofstream fs(resultFileName.c_str());
        RoadRunnerData::writeOnlyData(fs, resultData);
        fs.close();


        return true;
    }
    /** @endcond PRIVATE */
} //end of namespace



#include <iostream>
#include <fstream>
#include "rrLogger.h"
#include "rrException.h"
#include "rrUtils.h"
#include "Args.h"
#include "rrc_api.h"
#include "rrGetOptions.h"
#include "TestSuiteSimulation.h"
#include "rrException.h"
#include "rrRoadRunner.h"
#include "rrConfig.h"

using namespace std;
using namespace rr;

string     gTempFolder              = "";
string     gRRInstallFolder         = "";
string     gTSModelsPath            = "";
void ProcessCommandLineArguments(int argc, char* argv[], Args& args);

static void dispConfigFile()
{
    std::string conf = rr::Config::getConfigFilePath();
    if(conf.length())
    {
        cout << "config file: " << conf << endl;
    }
    else
    {
        cout << "could not find default config file, tried locations: " << endl;
        int llevel = rr::Logger::getLevel();
        rr::Logger::setLevel(rr::Logger::LOG_DEBUG);
        rr::Config::getConfigFilePath();
        rr::Logger::setLevel(llevel);
    }
}

//call with arguments, -m"modelFilePath" -r"resultFileFolder" -t"TempFolder"
int main(int argc, char* argv[])
{
    try
    {
        cout << "RoadRunner SBML Tester" << endl;

        cout << "Built on " << __DATE__ << " " << __TIME__ << endl;

        cout << RoadRunner::getExtendedVersionInfo() << endl;

        Logger::enableConsoleLogging();

        dispConfigFile();

        Logger::setLevel(Logger::LOG_NOTICE);

        Args args;
        ProcessCommandLineArguments(argc, argv, args);

        string thisExeFolder = getCurrentExeFolder();
        cout << "RoadRunner version: ";
        cout << "RoadRunner bin location is: "<<thisExeFolder<<endl;

        //Assume(!) this is the bin folder of roadrunner install
        gRRInstallFolder = getParentFolder(thisExeFolder);    //Go up one folder
        gTSModelsPath    = args.SBMLModelsFilePath;
        gTempFolder      = args.TempDataFolder;
        setInstallFolder(gRRInstallFolder.c_str());

        if(args.EnableLogging)
        {
            Logger::setLevel(Logger::LOG_DEBUG);
        }
        else
        {
            Logger::setLevel(Logger::LOG_NOTICE);
        }

        Log(Logger::LOG_NOTICE) << "Notice";

        // set full model path (read from cmd line)
        gTSModelsPath = joinPath(joinPath(gTSModelsPath, "cases"), "semantic");
        Log(lInfo)<<"Testing model: "<<args.ModelNumber;

        switch(args.ModelVersion)
        {
            case l2v4:
                RunTest("l2v4", args.ModelNumber, args.compiler);
            break;
            case l3v1:
                RunTest("l3v1", args.ModelNumber, args.compiler);
            break;

        }

        return 0;
    }
    catch (std::exception &e)
    {
        cout << "caught Exception: " << e.what() << "\n";
    }
    return -1;
}

void ProcessCommandLineArguments(int argc, char* argv[], Args& args)
{
    char c;
    while ((c = GetOptions(argc, argv, ("vi:a:m:t:c:"))) != (char)(-1))
    {
        switch (c)
        {
            case ('i'): args.ModelNumber                            = toInt(rrOptArg);                   break;
            case ('a'): args.ModelVersion                           = (ModelVersions) toInt(rrOptArg);   break;
            case ('m'): args.SBMLModelsFilePath                     = rrOptArg;                          break;
            case ('t'): args.TempDataFolder                         = rrOptArg;                          break;
            case ('v'): args.EnableLogging                          = true;                              break;
            case ('c'): args.compiler                               = rrOptArg;                          break;
            case ('?'): cout<<Usage(argv[0])<<endl;                                                      break;
            default:
            {
                string str = argv[rrOptInd-1];
                if(str != "-?")
                {
                    cout<<"*** Illegal option:\t"<<argv[rrOptInd-1]<<" ***\n\n";
                }
                exit(0);
            }
            break;
        }
    }

    //Check arguments, and choose to bail here if something is not right...
    if(argc < 2)
    {
        cout<<Usage(argv[0])<<endl;
        cout<<"\n";
        exit(0);
    }
}


#if defined(CG_IDE)

//#if defined(STATIC_RTL)
//#pragma comment(lib, "roadrunner_c_api-static.lib")
//#else
#pragma comment(lib, "roadrunner_c_api.lib")
//#endif

#if defined(STATIC_RR)
    #pragma comment(lib, "roadrunner-static.lib")
#else
    #pragma comment(lib, "roadrunner.lib")
#endif

#pragma comment(lib, "sundials_cvode")
#pragma comment(lib, "sundials_nvecserial")
#pragma comment(lib, "libf2c")
#pragma comment(lib, "blas")
#pragma comment(lib, "lapack")
#pragma comment(lib, "nleq-static")
#pragma comment(lib, "poco_foundation-static.lib")
#pragma comment(lib, "rr-libstruct-static.lib")
#pragma comment(lib, "unit_test-static.lib")
#endif



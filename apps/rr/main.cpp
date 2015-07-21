#include "rrLogger.h"
#include "rrRoadRunner.h"
#include "rrException.h"
#include "rrUtils.h"
#include "rrGetOptions.h"
#include "Args.h"
#include "Integrator.h"
#include "rrVersionInfo.h"

#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>
#include <stdexcept>



#if defined(_MSVER)
#include <direct.h>
#endif



using namespace std;
using namespace rr;

void ProcessCommandLineArguments(int argc, char* argv[], Args& args);
int main(int argc, char * argv[])
{
    string settingsFile;
    bool doContinue = true;
    Args args;
//     try
//     {
    Logger::enableConsoleLogging();

    if(argc < 2)
    {
        cout<<Usage(argv[0])<<endl;
        exit(0);
    }

    ProcessCommandLineArguments(argc, argv, args);

    Logger::setLevel(args.CurrentLogLevel);

    string thisExeFolder = getCurrentExeFolder();
    Log(Logger::LOG_DEBUG) << "RoadRunner bin location is: " << thisExeFolder;

    if(args.ModelFileName.size())
    {
        string logName = getFileName(args.ModelFileName);
        logName = changeFileExtensionTo(logName, ".log");
    }
    else
    {
        Logger::enableConsoleLogging(Logger::getLevel());
    }

    Log(Logger::LOG_INFORMATION) << "Current Log level is:"
      << Logger::getCurrentLevelAsString();

    if(!args.ModelFileName.size())
    {
        Log(lInfo)<<"Please supply a sbml model file name, using option -m<modelfilename>";
        exit(0);
    }

    //Creating roadrunner
    Log(Logger::LOG_DEBUG) << "Creating RoadRunner..." << endl;
    RoadRunner rr(args.ModelFileName);

    rr.setCompiler(args.compilerStr);

    SimulateOptions& opt = rr.getSimulateOptions();
    opt.start = args.StartTime;
    opt.duration = args.EndTime - args.StartTime;
    opt.steps = args.Steps;
    if(args.variableStep)
    {
        rr.getIntegrator()->setValue("variable_step_size", true);
    }

    ls::DoubleMatrix res = *rr.simulate();

    if(args.OutputFileName.size() >  0)
    {
        ofstream os(args.OutputFileName.c_str());
        os << res;
    }
    else
    {
      cout << res;
    }
//     catch(std::exception& ex)
//     {
//         Log(Logger::LOG_ERROR) << ex.what() << endl;
//     }


    return 0;
}

void ProcessCommandLineArguments(int argc, char* argv[], Args& args)
{
    char c;

    while ((c = GetOptions(argc, argv, (const char*) ("puo:c:v:n:d:t:l:m:s:e:z:"))) != (char)(-1))
    {
        switch (c)
        {
            case ('v'):
                if(string(rrOptArg) == "ersion" ) //hack to get rr -version working :)
                {
                    cout << "Version: " << rr::getVersionStr() << std::endl;
                    exit(-1);
                }
                else
                {
                    args.CurrentLogLevel                = Logger::stringToLevel(rrOptArg);
                }
                break;
            case ('c'): args.compilerStr                    = std::string(rrOptArg);               break;
            case ('x'): args.variableStep                   = true;                                break;
            case ('p'): args.Pause                          = true;                                break;
            case ('t'): args.TempDataFolder                 = rrOptArg;                            break;
            case ('d'): args.DataOutputFolder               = rrOptArg;                            break;
            case ('m'): args.ModelFileName                  = rrOptArg;                            break;
            case ('u'): args.UseOSTempFolder                = true;                                break;
            case ('l'): args.SelectionList                  = rrOptArg;                            break;
            case ('s'): args.StartTime                      = toDouble(rrOptArg);                  break;
            case ('e'): args.EndTime                        = toDouble(rrOptArg);                  break;
            case ('z'): args.Steps                          = toInt(rrOptArg);                     break;
            case ('o'): args.OutputFileName                 = rrOptArg;                            break;
            case ('?'):
            {
                    cout<<Usage(argv[0])<<endl;
                    exit(-1);
            }
            default:
            {
                string str = argv[rrOptInd-1];
                if(str != "-?")
                {
                    cout<<"*** Illegal option:\t"<<argv[rrOptInd-1]<<" ***\n"<<endl;
                }
                exit(-1);
            }
        }
    }
}

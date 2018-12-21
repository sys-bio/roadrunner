#include <iostream>
#include "rrRoadRunner.h"
#include "rrLogger.h"
#include "rrUtils.h"
#include "rrException.h"

using namespace rr;

int main(int argc, char** argv)
{

    try
    {
        LogOutput::mLogToConsole = true;

        //Use a list of roadrunners
        const char* rootPath = "..";

//        gLog.SetCutOffLogLevel(lDebug1);
        gLog.setLevel(lInfo);
        string tmpFolder = joinPath(rootPath, "temp");

        const string modelFile = joinPath(rootPath, "models", "feedback.xml");

        //Load modelFiles..
        Log(lInfo)<<" ---------- LOADING/GENERATING MODELS ------";

        RoadRunner rr1("", tmpFolder);
        RoadRunner rr2("", tmpFolder);
        rr1.load(modelFile);
        rr2.load(modelFile);

        Log(lInfo)<<" ---------- SIMULATE ---------------------";

        Log(lInfo)<<"Data:"<<rr1.simulate();
        Log(lInfo)<<"Data:"<<rr2.simulate();
    }
    catch(const Exception& ex)
    {
        Log(Logger::ERROR)<<"There was a  problem: "<<ex.getMessage();
    }

    //Pause(true);
    return 0;
}

#pragma comment(lib, "roadrunner.lib")
#pragma comment(lib, "poco_foundation-static.lib")
#pragma comment(lib, "rr-libstruct-static.lib")

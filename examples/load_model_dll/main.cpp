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
        const char* rootPath = "..";

        gLog.setLevel(lInfo);
        string tmpFolder = joinPath(rootPath, "temp");

        const string modelFile = joinPath(rootPath, "models", "test_1.xml");


        //Load modelFiles..
        Log(lInfo)<<" ---------- LOADING/GENERATING MODEL: "<<modelFile;

        RoadRunner rr1("", tmpFolder);
        if(!rr1.load(modelFile))    //This will generate a model DLL
        {

            Log(Logger::LOG_ERROR)<<"Failed to create model DLL....";
            return -1;
        }

    }
    catch(const Exception& ex)
    {

        Log(Logger::LOG_ERROR)<<"There was a  problem: "<<ex.getMessage();
    }

    //Pause(true);
    return 0;
}
#if defined(CG_IDE) || defined(MSVS)
#pragma comment(lib, "roadrunner-static.lib")
#pragma comment(lib, "poco_foundation-static.lib")
#pragma comment(lib, "rr-libstruct-static.lib")
#endif

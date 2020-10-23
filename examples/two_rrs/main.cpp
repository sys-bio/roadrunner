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
        //Use a list of roadrunners
        const char* rootPath = "..";

//        gLog.SetCutOffLogLevel(lDebug1);
        gLog.setLevel(lInfo);
        string tmpFolder = joinPath(rootPath, "temp");

        const string modelFile = joinPath(rootPath, "models", "feedback.xml");

        //Load modelFiles..
        Log(lInfo)<<" ---------- LOADING/GENERATING MODELS ------";

        RoadRunner rr1("");
        RoadRunner rr2("");
        rr1.load(modelFile);
        rr2.load(modelFile);

        Log(lInfo)<<" ---------- SIMULATE ---------------------";

        Log(lInfo)<<"Data:"<<rr1.simulate();
        Log(lInfo)<<"Data:"<<rr2.simulate();
    }
    catch(const Exception& ex)
    {
        Log(lError)<<"There was a  problem: "<<ex.getMessage();
    }

    //Pause(true);
    return 0;
}


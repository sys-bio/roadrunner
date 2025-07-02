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
        std::filesystem::path rootPath("..");

        //        gLog.SetCutOffLogLevel(lDebug1);
        gLog.setLevel(lInfo);
        std::filesystem::path modelFile(rootPath / "models" / "feedback.xml");

        //Load modelFiles..
        rrLog(lInfo) << " ---------- LOADING/GENERATING MODELS ------";

        RoadRunner rr1("");
        RoadRunner rr2("");
        rr1.load(modelFile.string());
        rr2.load(modelFile.string());

        rrLog(lInfo) << " ---------- SIMULATE ---------------------";

        rrLog(lInfo) << "Data:" << rr1.simulate();
        rrLog(lInfo) << "Data:" << rr2.simulate();
    }
    catch (const Exception& ex)
    {
        rrLog(lError) << "There was a problem: " << ex.getMessage();
    }

    //Pause(true);
    return 0;
}


#include <iostream>
#include "rrRoadRunner.h"
#include "rrException.h"
#include "rrUtils.h"
#include "rrLogger.h"
using namespace rr;

int main(int argc, char** argv)
{
    const char* rootPath = "..";

    try
    {
        gLog.setLevel(lInfo);
        string tmpFolder = joinPath(rootPath, "temp");

        const string modelFile = joinPath(rootPath, "models", "test_1.xml");

        //Load modelFiles..
        Log(lInfo)<<" ---------- LOADING/GENERATING MODELS ------";

        RoadRunner rr1("", tmpFolder);
        LoadSBMLOptions opt;
        opt.modelGeneratorOpt |= LoadSBMLOptions::RECOMPILE;
        if(!rr1.load(modelFile, &opt))
        {
            Log(Logger::LOG_ERROR)<<"There was a problem loading model in file: "<<modelFile;
            throw(Exception("Bad things in loadSBMLFromFile function"));
        }

        Log(lInfo)<<" ---------- SIMULATE ---------------------";

        rr1.simulate();
        Log(lInfo)<<"Data:"<<*(rr1.getSimulationData());
    }
    catch(const Exception& ex)
    {

        Log(Logger::LOG_ERROR)<<"There was a  problem: "<<ex.getMessage();
    }

    return 0;
}





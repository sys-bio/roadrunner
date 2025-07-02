#include <iostream>
#include <filesystem>
#include "rrRoadRunner.h"
#include "rrException.h"
#include "rrUtils.h"
#include "rrLogger.h"

using namespace rr;

int main(int argc, char** argv)
{
    std::filesystem::path rootPath("..");

    try
    {
        gLog.setLevel(lInfo);
        std::filesystem::path modelFile(rootPath / "models" / "test_1.xml");

        //Load modelFiles..
        rrLog(lInfo) << " ---------- LOADING/GENERATING MODELS ------";

        RoadRunner rr1("");
        LoadSBMLOptions opt;
        opt.modelGeneratorOpt |= LoadSBMLOptions::RECOMPILE;
        rr1.load(modelFile.string(), &opt);

        rrLog(lInfo) << " ---------- SIMULATE ---------------------";

        rr1.simulate();
        rrLog(lInfo) << "Data:" << *(rr1.getSimulationData());
    }
    catch (const Exception& ex)
    {

        rrLog(Logger::LOG_ERROR) << "There was a problem: " << ex.getMessage();
    }

    return 0;
}





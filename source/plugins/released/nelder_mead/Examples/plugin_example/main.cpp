#include <stdio.h>
#include <math.h>
#include "rrException.h"
#include "rrRoadRunnerData.h"
#include "rrRoadRunnerOptions.h"
#include "rrRoadRunner.h"
#include "../../lib/nmsimplex.h"
#include "telTelluriumData.h"
#include "telPluginManager.h"
#include "rrLogger.h"
#include "telPlugin.h"
using namespace rr;

void onStarted(void* data1, void* data2);
void onProgress(void* data1, void* data2);
void onFinished(void* data1, void* data2);

int main()
{
    double scale = 1;
    Logger::setLevel(lDebug);
    Logger::enableFileLogging("Test.log");

    try
    {
        PluginManager PM("..\\plugins");
        PM.load();

        Log(lInfo) << "The following plugins are loaded";
        Log(lInfo) << PM.getPluginNames();

        Plugin* test_model = PM.getPlugin("TestModel");
        if(!test_model)
        {
            throw(Exception("Test Model plugin is NULL!"));
        }

        test_model->execute();

        TelluriumData model(        (TelluriumData*) test_model->getPropertyValueHandle("SimulatedData"));
        TelluriumData experiment(   (TelluriumData*) test_model->getPropertyValueHandle("SimulatedDataWithNoise"));
        model.write("r:\\modelData.dat");
        experiment.write("r:\\expData.dat");

        Plugin* NMP = PM.getPlugin("Nelder-Mead");

        if(!NMP)
        {
            throw(Exception("NelderMead plugin is NULL.. Exiting"));
        }

        //Setup data structure
        NMP->setPropertyValue("SBML", test_model->getPropertyValueHandle("Model"));
        NMP->setPropertyValue("ExperimentalData", test_model->getPropertyValueHandle("SimulatedDataWithNoise"));

        //Setup parameters to fit
        Properties Parameters;
        Parameters.add(new Property<double>(12.3, "k1"));
        NMP->setPropertyValue("InputParameterList", &Parameters);
        NMP->setPropertyByString("ExperimentalDataSelectionList",   "S1,S2");
        NMP->setPropertyByString("FittedDataSelectionList",         "S1,S2");

        NMP->assignOnStartedEvent(onStarted, NMP);
        NMP->assignOnProgressEvent(onProgress, NMP);
        NMP->assignOnFinishedEvent(onFinished, NMP);
        NMP->execute();


        Log(lInfo) <<"======== RESULT ==========";
        Log(lInfo) << NMP->getResult();

//        Log(lInfo) << "Norms: "<<NMP->getPropertyValueAsString("Norms");

    }
    catch(const rr::Exception& e)
    {
        Log(lError) << "There was a problem: " << e.what();
    }
    return 0;
}

void onStarted(void* data1, void* data2)
{
    Log(lInfo) <<"Started ..";
}

void onProgress(void* data1, void* data2)
{
    Plugin* NMP = (Plugin*) data1;

    Log(lInfo) <<"Iteration, FuncIter, NORM: "<<NMP->getPropertyValueAsString("NrOfIter")<<", "<<NMP->getPropertyValueAsString("NrOfFuncIter")<<", "<<NMP->getPropertyValueAsString("Norm");
}

void onFinished(void* data1, void* data2)
{
    Log(lInfo) <<"Finished..";
}


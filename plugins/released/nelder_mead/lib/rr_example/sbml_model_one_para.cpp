#include <stdio.h>
#include <math.h>
#include "rrException.h"
#include "rrRoadRunnerData.h"
#include "rrRoadRunnerOptions.h"
#include "rrRoadRunner.h"
#include "nmsimplex.h"
#include "telTelluriumData.h"
#include "telPluginManager.h"
#include "rrLogger.h"
#include "telPlugin.h"

using namespace rr;
using rr::RoadRunner;
class MinimizationData
{
    public:

        double              mEpsilon;
        double              mScale;

        TelluriumData       mExperimentalData;
        rr::RoadRunner      mRRI;
        TelluriumData       mChiSquares;
        Properties          mParameters;        //Parameters to be minimized
        Plugin*             mChiSquarePlugin;

    public:
                            MinimizationData();
};

MinimizationData::MinimizationData()
:
mEpsilon(1.e-18),
mScale(1)
{}

double objfun(double par[], const void* userData)
{
    MinimizationData&   myData  = *((MinimizationData*) userData);
    TelluriumData&      expData = myData.mExperimentalData;
    RoadRunner&         rr      = myData.mRRI;
    Plugin&             chi     = *(myData.mChiSquarePlugin);

    //Reset roadrunner
    rr.reset();

    //Get current parameter values
    int nrOfParameters = myData.mParameters.count();
    for(int i =0; i < nrOfParameters; i++)
    {
        PropertyBase* para = myData.mParameters[i];
        double parValue = par[i];
        para->setValue( &parValue  );
        if(!rr.setValue(para->getName(), * (double*) para->getValueHandle()))
        {
            throw(Exception("Failed setting value of parameter"));
        }

    }

    rr::SimulateOptions opt;
    opt.start       = expData.getTimeStart();
    opt.duration    = expData.getTimeEnd() - expData.getTimeStart();
    opt.steps       = expData.rSize() -1;
    TelluriumData simData(rr.simulate(&opt));

    //Calculate Chi Square using the ChiSquare plugin
    chi.setPropertyValue("ModelData", &(simData));
    int nrOfParas = myData.mParameters.count();
    chi.setPropertyValue("NrOfModelParameters", &(nrOfParas));
    chi.execute(false);

    double chiSquare = * (double*) chi.getPropertyValueHandle("ChiSquare");
    return chiSquare;
}

int main()
{
    double scale = 1;
    Logger::setLevel(lDebug);
    Logger::enableFileLogging("Test.log");

    try
    {
        MinimizationData myData;

        PluginManager PM("..\\plugins");
        PM.load();

        Log(lInfo) << "The following plugins are loaded";
        Log(lInfo) << PM.getPluginNames();

        Plugin* model = PM.getPlugin("SBMLModel");
        if(!model)
        {
            throw(Exception("Model plugin is NULL!"));
        }

        Plugin* chiSquare = PM.getPlugin("ChiSquare");
        if(!chiSquare)
        {
            throw(Exception("ChiSquare plugin is NULL!"));
        }

        //Setup data structure
        myData.mRRI.load(model->getPropertyValueAsString("Model"));
        myData.mExperimentalData.read("ExperimentalData.dat");
        myData.mRRI.setSelections(StringList("Time, S1, S2"));
        myData.mParameters.add(new Property<double>(12.3, "k1"));
        myData.mChiSquarePlugin = chiSquare;

        chiSquare->setPropertyValue("ExperimentalData", &(myData.mExperimentalData));

        double* intialParameters = new double(myData.mParameters.count());
        intialParameters[0] = * (double*) myData.mParameters[0]->getValueHandle();
        //Execute minimizer
        double min = simplex2(  objfun,
                                &myData,
                                intialParameters,
                                myData.mParameters.count(),
                                myData.mEpsilon,
                                scale,
                                NULL
                                );//my_constraints);

    }
    catch(const rr::Exception& e)
    {
        Log(lError) << "There was a problem: " << e.what();
    }


    return 0;
}


//void my_constraints(double x[], int n)
//{
//    // rate contstants must be positive
//    int i;
//    for (i=0; i<n; i++)
//    {
//        if (x[i] < 0)
//        {
//            x[i] = fabs(x[i]);
//        }
//    }
//}




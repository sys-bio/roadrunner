#include <stdio.h>
#include <math.h>
#include "rrException.h"
#include "nmsimplex.h"
#include "rrRoadRunner.h"
#include "telTelluriumData.h"
#include "telPluginManager.h"
#include "rrLogger.h"
#include "telPlugin.h"
using namespace rr;
using rr::RoadRunner;
typedef struct
{
    TelluriumData mExperimentalData;
    rr::RoadRunner mRRI;

} nmDataStructure;

double objfun(double par[], const void* userData)
{
    double chiSquare = -1;
    nmDataStructure* myData =  (nmDataStructure*) userData;
    RoadRunner& rr = myData->mRRI;

    double k1 = par[0];

    rr.reset();

    if(!rr.setValue("k1", k1))
    {
        throw(Exception("Failed setting value of parameter"));
    }
    return chiSquare;
}

void my_constraints(double x[], int n)
{
    // rate contstants must be positive
    int i;
    for (i=0; i<n; i++)
    {
        if (x[i] < 0)
        {
            x[i] = fabs(x[i]);
        }
    }
}

int main()
{
    double start[] = {0.1};
    double min;
    int i;
    int dim = 1;
    double eps = 1.0e-8;
    double scale = 1.0;
    Logger::setLevel(lDebug);
    Logger::enableFileLogging("Test.log");

    try
    {
        PluginManager PM("..\\plugins");
        PM.load();

        Log(lInfo) << "The following plugins are loaded";
        Log(lInfo) << PM.getPluginNames();

        Plugin* model = PM.getPlugin("SBMLModel");
        if(!model)
        {
            throw(Exception("Model plugin is NULL!"));
        }

        nmDataStructure myData;

        TelluriumData expData;
        myData.mRRI.load(model->getPropertyValueAsString("Model"));

        myData.mRRI.setSelections(StringList("k1"));

        min = simplex2(objfun, &myData, start, dim, eps, scale, my_constraints);
    //
    //    for (i=0; i < dim; i++)
    //    {
    //        printf("%f\n", start[i]);
    //    }
    //    delete myData.mRRI;

    }
    catch(const rr::Exception& e)
    {
        Log(lError) << "There was a problem: " << e.what();
    }


    return 0;
}





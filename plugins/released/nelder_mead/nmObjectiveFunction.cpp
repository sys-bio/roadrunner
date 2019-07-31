#pragma hdrstop
#include "nmObjectiveFunction.h"
#include "nmNelderMead.h"
#include "telMathUtils.h"
#include "telProperties.h"
//---------------------------------------------------------------------------

using namespace rr;

double NelderMeadObjectiveFunction(double par[], const void* userData)
{
    NelderMead&         plugin      = *((NelderMead*) userData);
    TelluriumData&      expData     = plugin.mExperimentalData;
    RoadRunner*         rr          = plugin.getRoadRunner();

    if(!rr)
    {
        throw(Exception("RoadRunner is NULL in Nelder-Mead objective function"));
    }
    //Reset roadrunner
    rr->reset();

    //Get current parameter values
    Properties* inParas = (Properties*) plugin.mInputParameterList.getValueHandle();
    int nrOfParameters = inParas->count();

    for(int i = 0; i < nrOfParameters; i++)
    {
        PropertyBase* para = inParas->getPropertyAt(i);
        rr->setValue(para->getName(), par[i]);
    }

    rr::SimulateOptions opt;
    opt.start       = expData.getTimeStart();
    opt.duration    = expData.getTimeEnd() - expData.getTimeStart();
    opt.steps       = expData.rSize() -1;

    //Simulate
    rr->simulate(&opt);
    TelluriumData   simData;
    simData.setData(*rr->getSimulationData());

    StringList* species = (StringList*) plugin.mExperimentalDataSelectionList.getValueHandle();
    int nrOfSpecies = species->Count();

    //Calculate residuals
    vector<double> residuals(simData.rSize() * nrOfSpecies);

    int resIndex = 0;
    for(int specie = 0; specie < nrOfSpecies; specie++)
    {
        for(int timePoint = 0; timePoint < simData.rSize(); timePoint++)
        {
            residuals[resIndex] = expData(timePoint, specie + 1) - simData(timePoint, specie); //+1 because of time
            resIndex ++;
        }
    }

    //Calculate Norm
    double norm = getEuclideanNorm(residuals);
    plugin.mNorm.setValue(norm);
    plugin.mTheNorms.push_back(norm);
    //Assign data relevant to the progress
     plugin.mNrOfFuncIter.setValue(plugin.mNrOfFuncIter.getValue() + 1);

    //Call OnProgress
    if(plugin.hasProgressEvent())
    {
        //Pass trough event data
        pair<void*, void*> passTroughData = plugin.getWorkProgressData();
        plugin.WorkProgressEvent(passTroughData.first, passTroughData.second);
    }

    return norm;
}

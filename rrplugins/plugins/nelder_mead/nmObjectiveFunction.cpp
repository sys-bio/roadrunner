#pragma hdrstop
#include "nmObjectiveFunction.h"
#include "nmNelderMead.h"
#include "telMathUtils.h"
#include "telProperties.h"
//---------------------------------------------------------------------------

namespace nmfit
{
    extern rrc::THostInterface* gHostInterface;

    using namespace tlp;

    double NelderMeadObjectiveFunction(double par[], const void* userData)
    {
        nmfit::NelderMead& plugin = *((nmfit::NelderMead*)userData);
        TelluriumData& expData = plugin.mExperimentalData;
        //Reset roadrunner
        gHostInterface->reset(plugin.rrHandle);

        //Get current parameter values
        Properties* inParas = (Properties*)plugin.mInputParameterList.getValueHandle();
        int nrOfParameters = inParas->count();

        for (int i = 0; i < nrOfParameters; i++)
        {
            Property<double>* para = (Property<double>*) (inParas->getPropertyAt(i));
            gHostInterface->setValue(plugin.rrHandle, para->getName().c_str(), par[i]);
        }

        //Simulate
        double norm = DBL_MAX; //For if the simulation fails.
        bool ret = gHostInterface->simulateExNoReturn(plugin.rrHandle, expData.getTimeStart(), expData.getTimeEnd(), expData.rSize());
        if (ret)
        {
            DoubleMatrix* modelData = (DoubleMatrix*)gHostInterface->getSimulationResultAsDoubleMatrix(plugin.rrHandle);
            TelluriumData   simData;
            simData.setData(*modelData);

            StringList* species = (StringList*)plugin.mExperimentalDataSelectionList.getValueHandle();
            int nrOfSpecies = species->count();

            //Calculate residuals
            vector<double> residuals(simData.rSize() * nrOfSpecies);

            int resIndex = 0;
            for (int specie = 0; specie < nrOfSpecies; specie++)
            {
                for (int timePoint = 0; timePoint < simData.rSize(); timePoint++)
                {
                    residuals[resIndex] = expData(timePoint, specie + 1) - simData(timePoint, specie); //+1 because of time
                    resIndex++;
                }
            }

            //Calculate Norm
            norm = getEuclideanNorm(residuals);
        }
        plugin.mNorm.setValue(norm);
        plugin.mTheNorms.push_back(norm);
        //Assign data relevant to the progress
        plugin.mNrOfFuncIter.setValue(plugin.mNrOfFuncIter.getValue() + 1);

        //Call OnProgress
        if (plugin.hasProgressEvent())
        {
            //Pass trough event data
            pair<void*, void*> passTroughData = plugin.getWorkProgressData();
            plugin.WorkProgressEvent(passTroughData.first, passTroughData.second);
        }

        return norm;
    }
}
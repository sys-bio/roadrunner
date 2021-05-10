#pragma hdrstop
#include "lmObjectiveFunction.h"
#include "telMathUtils.h"
#include "lm.h"
#include "rrRoadRunner.h"

namespace lmfit
{
    using namespace std;
    using namespace tlp;
    extern rrc::THostInterface* gHostInterface;
    //---------------------------------------------------------------------------
    /* function evaluation, determination of residues */
    void lmObjectiveFunction(const double* par,//Property vector
        int           m_dat,         //Dimension of residue vector
        const void* userData,            //Data structure
        double* fvec,          //residue vector..
        int* userBreak      //Non zero value means termination
    )
    {
        LM* thePlugin = (LM*)userData;
        lmDataStructure* myData = &(thePlugin->mLMData);

        //Check if user have asked for termination..
        if (thePlugin->isBeingTerminated())
        {
            (*userBreak) = -1; //This signals lmfit algorithm to break
            return;
        }

        gHostInterface->reset(myData->mTheHost->rrHandle);

        for (int i = 0; i < myData->nrOfParameters; i++)
        {
            RRPLOG(lInfo) << myData->parameterLabels[i] << " = " << par[i] << "\n";
            gHostInterface->setValue(myData->mTheHost->rrHandle, myData->parameterLabels[i], par[i]);
        }

        bool ret = gHostInterface->simulateExNoReturn(myData->mTheHost->rrHandle, myData->timeStart, myData->timeEnd, myData->nrOfTimePoints);
        if (!ret)
        {
            setTerribleResiduals(fvec, myData->nrOfSpecies * myData->nrOfTimePoints, thePlugin);
            return;
        }
        DoubleMatrix* modelData = (DoubleMatrix*)gHostInterface->getSimulationResult(myData->mTheHost->rrHandle);

        if (!modelData)
        {
            setTerribleResiduals(fvec, myData->nrOfSpecies * myData->nrOfTimePoints, thePlugin);
            return;
        }
        //calculate fvec for each species
        int count = 0;
        vector<double> residuals(myData->nrOfSpecies * myData->nrOfTimePoints);
        for (int i = 0; i < myData->nrOfSpecies; i++)
        {
            fvec[count] = 0;
            for (int j = 0; j < myData->nrOfTimePoints; j++)
            {
                const double& modelValue = modelData->Element(j, i);

                if (!isNaN(myData->experimentalData[i][j]))
                {
                    fvec[count] = myData->experimentalData[i][j] - modelValue;
                    residuals[count] = fvec[count];

                    //If data points are weighted, weigh fvec data points
                    if (myData->experimentalDataWeights != NULL)
                    {
                        if (myData->experimentalDataWeights[i][j] != 0) //Cause the first column is time... :( loks ugly
                        {
                            double weight = myData->experimentalDataWeights[i][j];
                            fvec[count] = fvec[count] / weight;
                        }
                    }
                }
                else
                {
                    fvec[count] = 0;
                }
                count++;
            }
        }

        //Assign data relevant to the progress
        thePlugin->mNrOfIter.setValue(thePlugin->mNrOfIter.getValue() + 1);
        double norm = getEuclideanNorm(residuals);
        thePlugin->mNorm.setValue(norm);

        //Add norm to Norms property
        thePlugin->rNormsData(thePlugin->mNrOfIter.getValue() - 1, 0) = thePlugin->mNorm.getValue();

        if (thePlugin->hasProgressEvent())
        {
            //Pass trough event data
            pair<void*, void*> passTroughData = thePlugin->getWorkProgressData();
            thePlugin->WorkProgressEvent(passTroughData.first, passTroughData.second);
        }
    }

    void setTerribleResiduals(double* fvec, int len, LM* thePlugin)
    {
        for (int count = 0; count < len; count++)
        {
            fvec[count] = DBL_MAX;
        }
        thePlugin->mNrOfIter.setValue(thePlugin->mNrOfIter.getValue() + 1);
        thePlugin->mNorm.setValue(DBL_MAX);
        thePlugin->rNormsData(thePlugin->mNrOfIter.getValue() - 1, 0) = thePlugin->mNorm.getValue();
        if (thePlugin->hasProgressEvent())
        {
            //Pass trough event data
            pair<void*, void*> passTroughData = thePlugin->getWorkProgressData();
            thePlugin->WorkProgressEvent(passTroughData.first, passTroughData.second);
        }

    }
}

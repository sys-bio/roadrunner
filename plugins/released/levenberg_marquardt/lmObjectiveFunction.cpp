#pragma hdrstop
#include "lmObjectiveFunction.h"
#include "telMathUtils.h"
#include "lm.h"
#include "rrRoadRunner.h"

using namespace std;
using namespace lmfit;
using namespace rr;
//---------------------------------------------------------------------------
/* function evaluation, determination of residues */
void lmObjectiveFunction(const double *par,//Property vector
              int           m_dat,         //Dimension of residue vector
              const void   *userData,            //Data structure
              double       *fvec,          //residue vector..
              int          *userBreak      //Non zero value means termination
)
{
    LM*         thePlugin = (LM*) userData;
    lmDataStructure* myData = &(thePlugin->mLMData);

    //Check if user have asked for termination..
    if(thePlugin->isBeingTerminated())
    {
        (*userBreak) = -1; //This signals lmfit algorithm to break
        return;
    }

    if (!myData->roadrunner) {
        Log(lError) << "No roadrunner instance\n";
        throw std::runtime_error("No roadrunner instance");
    }

    //Reset RoadRunner
    myData->roadrunner->reset();

    for(int i = 0; i < myData->nrOfParameters; i++)
    {
        Log(lDebug) << myData->parameterLabels[i] << " = " << par[i] << "\n";
        myData->roadrunner->setValue(myData->parameterLabels[i], par[i]);
    }

    rr::SimulateOptions opt;
    opt.start       = myData->timeStart;
    opt.duration    = myData->timeEnd - opt.start;
    opt.steps       = myData->nrOfTimePoints;

    myData->roadrunner->simulate(&opt);

    if(!myData->roadrunner->getSimulationData())
    {
        stringstream msg;
        msg << "NULL data returned from RoadRunner simulate() function.";
        Log(lError)<<msg.str();
        throw std::runtime_error("NULL data returned from RoadRunner simulate() function");
    }

    //calculate fvec for each specie
    int count = 0;
    vector<double> residuals(myData->nrOfSpecies *  myData->nrOfTimePoints);
    for(int i = 0; i < myData->nrOfSpecies; i++)
    {
        fvec[count] = 0;
        for(int j = 0; j < myData->nrOfTimePoints; j++ )
        {
            double modelValue = (*myData->roadrunner->getSimulationData())(j,i);

            if(!isNaN(myData->experimentalData[i][j]))
            {
                fvec[count] = myData->experimentalData[i][j] - modelValue;
                residuals[count] = fvec[count];

                //If data points are weighted, weigh fvec data points
                if(myData->experimentalDataWeights != NULL)
                {
                    if(myData->experimentalDataWeights[i][j] != 0) //Cause the first column is time... :( loks ugly
                    {
                        double weight =  myData->experimentalDataWeights[i][j];
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
    thePlugin->rNormsData(thePlugin->mNrOfIter.getValue() -1, 0) = thePlugin->mNorm.getValue();

    if(thePlugin->hasProgressEvent())
    {
        //Pass trough event data
        pair<void*, void*> passTroughData = thePlugin->getWorkProgressData();
        thePlugin->WorkProgressEvent(passTroughData.first, passTroughData.second);
    }
}

#pragma hdrstop
#include "csChiWorker.h"
#include "rrLogger.h"
#include "rrException.h"
#include "telTelluriumData.h"
#include "csChiSquare.h"
#include "rrUtils.h"
#include "telMathUtils.h"
#include "telProperty.h"
//---------------------------------------------------------------------------
using namespace std;
using namespace rr;

ChiWorker::ChiWorker(ChiSquare& host)
:
mTheHost(host)
{}

bool ChiWorker::isRunning() const
{
    return mThread.isRunning();
}

void ChiWorker::start(bool runInThread)
{
    if(runInThread)
    {
        if(mThread.isRunning())
        {
            Log(lError)<<"Tried to start an already working thread!";
            return;
        }

        mThread.start(*this);    //Will spawn the run function below in a thread
    }
    else
    {
        run();
    }
}

void ChiWorker::run()
{
    workerStarted();

    //The user may have aborted the minization... check here..
    if(mTheHost.isBeingTerminated())
    {
        //user did set the terminate flag to true.. discard any data and get out of the
        //plugin execute code..
        Log(lInfo)<<"The ChiWorker was terminated.. aborting";
        workerFinished();
        return;
    }

    //Calculate ChiSquare
    TelluriumData& expData      = mTheHost.mExperimentalData.getValueReference();
    TelluriumData& modelData    = mTheHost.mModelData.getValueReference();

    double& chiSquare           = mTheHost.mChiSquare.getValueReference();
    double& redChiSquare        = mTheHost.mReducedChiSquare.getValueReference();

    TelluriumData mdlData = getDataSet(1, modelData);
    double chi = 0;

    //Get ChiSquare column by column and average them together
    int startCol = expData.isFirstColumnTime() ? 1 : 0;
    for(int n = startCol; n < expData.cSize(); n++)
    {
        vector<double> expDataN     = getValuesInColumn(n, expData);
        vector<double> variancesN   = getWeightValuesInColumn(n, expData);
        vector<double> modelDataN   = getValuesInColumn(n, mdlData);
        chi += getChiSquare(expDataN, modelDataN, variancesN);
    }

    int test = expData.isFirstColumnTime() ? 1 : 0;
    int nrOfCols = expData.cSize() -  test;
    int degreeOfFreedom = expData.rSize() * nrOfCols - mTheHost.mNrOfModelParameters.getValue();

    chiSquare =  chi;
    redChiSquare = chi/degreeOfFreedom;


    Log(lInfo)<<"Chi Square = "<<chiSquare;
    Log(lInfo)<<"Reduced Chi Square = "<<redChiSquare;

    //Post fitting data calculations
    workerFinished();
}

void ChiWorker::workerStarted()
{
    mTheHost.mIsWorking = true;
    if(mTheHost.mWorkStartedEvent)
    {
        mTheHost.mWorkStartedEvent(mTheHost.mWorkStartedData1, mTheHost.mWorkStartedData2);
    }
}

void ChiWorker::workerFinished()
{
    mTheHost.mIsWorking = false;//Set this flag before event so client can query plugin about termination
    if(mTheHost.mWorkFinishedEvent)
    {
        mTheHost.mWorkFinishedEvent(mTheHost.mWorkFinishedData1, mTheHost.mWorkFinishedData2);
    }
}


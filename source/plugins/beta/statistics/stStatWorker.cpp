#pragma hdrstop
#include "stStatWorker.h"
#include "telLogger.h"
#include "telException.h"
#include "telTelluriumData.h"
#include "stStat.h"
#include "telUtils.h"
#include "telMathUtils.h"
#include "telProperty.h"
//---------------------------------------------------------------------------
using namespace std;
using namespace tlp;

stStatWorker::stStatWorker(Stat& host)
:
mTheHost(host)
{}

bool stStatWorker::isRunning() const
{
    return mThread.isRunning();
}

void stStatWorker::start(bool runInThread)
{
    if(runInThread)
    {
        if(mThread.isRunning())
        {
            RRPLOG(lError)<<"Tried to start an already working thread!";
            return;
        }

        mThread.start(*this);    //Will spawn the run function below in a thread
    }
    else
    {
        run();
    }
}

void stStatWorker::run()
{
    workerStarted();

    //The user may have aborted the minization... check here..
    if(mTheHost.mTerminate)
    {
        //user did set the terminate flag to true.. discard any minimization data and get out of the
        //plugin execute code..
        RRPLOG(lInfo)<<"The stat worker was terminated.. aborting";
        workerFinished();
        return;
    }
    //Calculate standardized residuals
    TelluriumData& residuals = *(TelluriumData*) mTheHost.mResidualsData.getValueHandle();

    //Populate the standardized residuals
    TelluriumData& stdRes = *(TelluriumData*) mTheHost.mStandardizedResiduals.getValueHandle();
    stdRes = getStandardizedPopulations(residuals);

    //Create a probability plot for the residuals
    TelluriumData& probPlot = *(TelluriumData*) mTheHost.mNormalProbabilityOfResiduals.getValueHandle();
    probPlot = getNormalProbabilityPlot(stdRes);

    //Calculate ChiSquare(s)
    TelluriumData& modelData = *(TelluriumData*) mTheHost.mModelData.getValuePointer();
    TelluriumData& obsData = *(TelluriumData*) mTheHost.mExperimentalData.getValuePointer();

    double chiSquare = 0;
    //Get ChiSquare specie by specie and average them together

    for(int n = obsData.isFirstColumnTime() ? 1 : 0; n < obsData.cSize(); n++)
    {
        vector<double> obsDataN     = getValuesInColumn(n, obsData);
        vector<double> modelDataN   = getValuesInColumn(n, modelData);

        double stdDevOfResiduals = getStandardDeviation(getValuesInColumn(n, residuals));

        vector<double> variances(modelDataN.size(), pow(stdDevOfResiduals,2));

        chiSquare += getChiSquare(obsDataN, modelDataN, variances);
    }
    //Divide chiSquare with number of species
    int test = obsData.isFirstColumnTime() ? 1 : 0;
    int nrOfSpecies = obsData.cSize() -  test;

//    int degreeOfFreedom = obsData.rSize() * nrOfSpecies - mStatData.nrOfParameters;
//    mTheHost.mChiSquare.setValue(chiSquare);
//    mTheHost.mReducedChiSquare.setValue(chiSquare/degreeOfFreedom);

    RRPLOG(lInfo)<<"Chi Square = "<<chiSquare;

    //Post fitting data calculations
    workerFinished();
}

void stStatWorker::workerStarted()
{
    mTheHost.mIsWorking = true;
    if(mTheHost.mWorkStartedEvent)
    {
        mTheHost.mWorkStartedEvent(mTheHost.mWorkStartedData1, mTheHost.mWorkStartedData2);
    }
}

void stStatWorker::workerFinished()
{
    mTheHost.mIsWorking = false;//Set this flag before event so client can query plugin about termination
    if(mTheHost.mWorkFinishedEvent)
    {
        mTheHost.mWorkFinishedEvent(mTheHost.mWorkFinishedData1, mTheHost.mWorkFinishedData2);
    }
}

bool stStatWorker::setup()
{
    return true;
}



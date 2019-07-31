#pragma hdrstop
#include "psParameterScanWorker.h"
#include "rrLogger.h"
#include "rrRoadRunner.h"
#include "rrRoadRunnerData.h"
#include "rrException.h"
#include "telTelluriumData.h"
#include "psParameterScan.h"
#include "rrUtils.h"
#include "telMathUtils.h"
#include "telProperty.h"
//---------------------------------------------------------------------------
using namespace std;
using namespace rr;
using rr::RoadRunnerData;
ParameterScanWorker::ParameterScanWorker(ParameterScan& host)
:
mHost(host)
{}

bool ParameterScanWorker::isRunning() const
{
    return mThread.isRunning();
}

void ParameterScanWorker::start(bool runInThread)
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

void ParameterScanWorker::run()
{
    workerStarted();

    //The user may have aborted the minization... check here..
    if(mHost.isBeingTerminated())
    {
        //user did set the terminate flag to true.. discard any data and get out of the
        //plugin execute code..
        Log(lInfo)<<"The ParameterScanWorker was terminated.. aborting";
        workerFinished();
        return;
    }

    //Create arrayed data
    ArrayedParameter para(mHost.mParameter.getValueReference());

    StringList selList(mHost.mSelectionList.getValue());

    RoadRunner rr;
    rr.load(mHost.mSBML.getValue());

    //Make sure time is the first column in the selectin list
    if(selList.size() && compareNoCase(selList[0], "time") != true)
    {
        selList.InsertAt(0,"time");
    }
    rr.setSelections(selList);

    TelluriumData& data =     mHost.mOutputData.getValueReference();
    data.clear();
    for(int i = 0; i < para.getNumberOfIncrements() + 1; i++)
    {
        rr::SimulateOptions options;
        options.flags |= rr::SimulateOptions::RESET_MODEL;
        options.start = mHost.mStartTime.getValue();
        options.duration =  mHost.mEndTime.getValue() - mHost.mStartTime.getValue();
        options.steps = mHost.mNumberOfPointsPerSimulation.getValue() - 1;

        //Set parameter
        rr.setValue(para.getName(), para.getCurrentValue());
        const RoadRunnerData *rrData = rr.simulate(&options);
        TelluriumData *temp = new TelluriumData(*rrData);
        data.append(*temp);
        delete temp;
        para.increment();
    }

    //The Arrayed parameter need to be added to the data.
    data.setArrayedParameter(para);
    workerFinished();
}

void ParameterScanWorker::workerStarted()
{
    mHost.mIsWorking = true;
    if(mHost.mWorkStartedEvent)
    {
        mHost.mWorkStartedEvent(mHost.mWorkStartedData1, mHost.mWorkStartedData2);
    }
}

void ParameterScanWorker::workerFinished()
{
    mHost.mIsWorking = false;//Set this flag before event so client can query plugin about termination
    if(mHost.mWorkFinishedEvent)
    {
        mHost.mWorkFinishedEvent(mHost.mWorkFinishedData1, mHost.mWorkFinishedData2);
    }
}


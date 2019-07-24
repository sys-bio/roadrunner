#pragma hdrstop
#include "telTelluriumData.h"
#include "rrLogger.h"
#include "add_noise_worker.h"
#include "rrUtils.h"
#include "add_noise.h"
//---------------------------------------------------------------------------

namespace addNoise
{

AddNoiseWorker::AddNoiseWorker(AddNoise& host)
:
mTheHost(host),
mNoise(0,0)
{}

bool AddNoiseWorker::start(bool runInThread)
{
    if(runInThread)
    {
        if(mThread.isRunning())
        {
            Log(lError)<<"Tried to start a working thread!";
            return false;
        }

        //Run the worker in a thread
        mThread.start(*this);
    }
    else
    {
        run(); //Not running worker in a thread
    }
    return true;
}

bool AddNoiseWorker::isRunning() const
{
    return mThread.isRunning();
}

void AddNoiseWorker::run()
{
    if(mTheHost.mWorkStartedEvent)
    {
        mTheHost.mProgress.setValue((double) 0);
        mTheHost.mWorkStartedEvent(mTheHost.mWorkStartedData1, mTheHost.mWorkStartedData2);
    }

    TelluriumData& data = (mTheHost.mData.getValueReference());

    double sigma = mTheHost.mSigma.getValue();

    //Don't add noise to a column that has name 'time'
    bool ignoreFirstColumn = data.isFirstColumnTime();
    for(int row = 0; row < data.rSize(); row++)
    {
        for(int col = 0; col < data.cSize(); col++)
        {
                if(col == 0)
                {                
                    if(!ignoreFirstColumn)
                    {
                        data(row, col) = mNoise.getNoise(data(row, col), sigma);
                    }
                }
                else
                {
                    data(row, col) = mNoise.getNoise(data(row, col), sigma);                
                }
            }

            if(mTheHost.mWorkProgressEvent)
            {
                //The progress is communicated to the client trough the mProgress property
                double progress = (row * 100.0) /(data.rSize() -1.0);
                mTheHost.mProgress.setValue( progress );
                mTheHost.mWorkProgressEvent(mTheHost.mWorkProgressData1,  mTheHost.mWorkProgressData2);
            }
    }

    if(mTheHost.mWorkFinishedEvent)
    {
        mTheHost.mWorkFinishedEvent( mTheHost.mWorkFinishedData1, mTheHost.mWorkFinishedData2);
    }
}

}

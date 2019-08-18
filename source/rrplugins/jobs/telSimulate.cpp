#pragma hdrstop
#include "telSimulateThread.h"
#include "telSimulate.h"
//---------------------------------------------------------------------------

namespace rr
{

Simulate::Simulate(RoadRunnerList& rrs, const int& nrThreads)
:
ThreadPool()
{
    //create nrThreads that can load SBML models
    for(int i = 0; i < nrThreads; i++)
    {
        SimulateThread* sThread = new SimulateThread(NULL, false);
        mThreads.push_back(sThread);
    }

    //The following will add jobs and get them done too..
    for(int i = 0; i < rrs.count(); i++)
    {
        addJob(rrs[i]);
    }

    //This will start processing the jobs..
    if(!isJobQueueEmpty() && nrThreads > 0)
    {
        start();
    }
}

Simulate::~Simulate()
{
    list<RoadRunnerThread*>::iterator    iter;
    for(iter = mThreads.begin(); iter != mThreads.end(); iter++)
    {
        RoadRunnerThread* t = (*iter);
        delete t;
    }
}

}

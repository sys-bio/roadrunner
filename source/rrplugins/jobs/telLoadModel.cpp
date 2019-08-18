#pragma hdrstop
#include "telLoadModelThread.h"
#include "telLoadModel.h"

//---------------------------------------------------------------------------
namespace rr
{

LoadModel::LoadModel(RoadRunnerList& rrs, const string& sbml, const int& nrThreads, bool reCompile)
:
ThreadPool()
{
    //create nrThreads that can load SBML models
    for(int i = 0; i < nrThreads; i++)
    {
        LoadModelThread* lmThread = new LoadModelThread("", reCompile);
        lmThread->setSBML(sbml);
        mThreads.push_back(lmThread);
    }

    //The following will add jobs
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

LoadModel::~LoadModel()
{
    list<RoadRunnerThread*>::iterator    iter;
    for(iter = mThreads.begin(); iter != mThreads.end(); iter++)
    {
        RoadRunnerThread* t = (*iter);
        delete t;
    }
}

}
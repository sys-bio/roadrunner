#pragma hdrstop
#include "telLoadModelThread.h"
#include "../../rrLogger.h"
#include "../../rrRoadRunner.h"
#include "../../rrUtils.h"
#include "Poco/Mutex.h"
#include "Poco/ScopedLock.h"
//---------------------------------------------------------------------------

namespace rr
{

using namespace rr;
using namespace Poco;

int                 LoadModelThread::mNrOfWorkers = 0;
Poco::Mutex         LoadModelThread::mNrOfWorkersMutex;

list<RoadRunner*>   LoadModelThread::mJobs;
Poco::Mutex         LoadModelThread::mJobsMutex;
Poco::Condition     LoadModelThread::mJobsCondition;


LoadModelThread::LoadModelThread(const string& modelFile, bool recompileOnLoad, RoadRunner* rri, bool autoStart)
:
mModelFileName(modelFile)
{

//    mLoadSBMLOptions.modelGeneratorOpt  = mLoadSBMLOptions.modelGeneratorOpt | LoadSBMLOptions::ForceReCompile;
    if(rri)
    {
        addJob(rri);
    }

    if(autoStart && rri != NULL)
    {
        start();
    }
}

LoadModelThread::~LoadModelThread()
{
}
void LoadModelThread::setSBML(const string& sbml)
{
    mSBML = sbml;
}

void LoadModelThread::addJob(RoadRunner* rr)
{
    //getMutex
    Mutex::ScopedLock lock(mJobsMutex);
    mJobs.push_back(rr);
    mJobsCondition.signal();    //Tell the thread its time to go to work
}

unsigned int LoadModelThread::getNrOfJobsInQueue()
{
    Mutex::ScopedLock lock(mJobsMutex);
    return mJobs.size();
}

void LoadModelThread::signalExit()
{
    mJobsCondition.signal();
}

void LoadModelThread::signalAll()
{
    mJobsCondition.broadcast();
}

bool LoadModelThread::isAnyWorking()
{
       Mutex::ScopedLock lock(mNrOfWorkersMutex);
    return mNrOfWorkers > 0 ? true : false;
}

void LoadModelThread::worker()
{
    RoadRunner *rri = NULL;
    mWasStarted = true;
    mIsWorking  = true;

    //Any thread working on this Job list need to increment this one
    Mutex::ScopedLock lock(mNrOfWorkersMutex);
    mNrOfWorkers++;
    mNrOfWorkersMutex.unlock();

    while(!mIsTimeToDie)
    {
        {    //Scope for scoped lock
            Mutex::ScopedLock lock(mJobsMutex);
            if(mJobs.size() == 0)
            {
                Log(rr::lDebug5)<<"Waiting for jobs in loadSBML worker";
                //mJobsCondition.wait(mJobsMutex);
                break;
            }

            if(mIsTimeToDie)
            {
                break;    //ends the life of the thread..
            }
            else
            {
                //Get a job
                rri = mJobs.front();
                mJobs.pop_front();
            }
        }        //Causes the scoped lock to unlock

        //Do the job
        if(rri)
        {
            Log(lDebug2)<<"Loading model into instance: "<<rri->getInstanceID();
            if(mModelFileName.size())
            {
                rri->load(mModelFileName, &mLoadSBMLOptions);
            }
            else if(mSBML.size())
            {
                rri->load(mSBML, &mLoadSBMLOptions);
            }
        }
        else
        {
            Log(lError)<<"Null job pointer...!";
        }
    }

    Log(lDebug)<<"Exiting Load Model thread: "<<mThread.id();
    mIsWorking  = false;
    Mutex::ScopedLock lock2(mNrOfWorkersMutex);
    mNrOfWorkers--;

}

}


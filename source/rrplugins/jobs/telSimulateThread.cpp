#pragma hdrstop
#include "telSimulateThread.h"
#include "telLogger.h"
#include "rr/rrRoadRunner.h"
#include "rr/rrRoadRunnerOptions.h"
//---------------------------------------------------------------------------

namespace tlp
{
using namespace rr;

int                 SimulateThread::mNrOfWorkers = 0;
Poco::Mutex         SimulateThread::mNrOfWorkersMutex;

list<RoadRunner*>   SimulateThread::mJobs;
Mutex               SimulateThread::mJobsMutex;
Condition           SimulateThread::mJobsCondition;

SimulateThread::SimulateThread( RoadRunner* rri, const double& ts, const double& te,
                                const int& nrPoints, JobStartedCB f1, JobFinishedCB f2, void* userData, bool autoStart)
:
mSimulateEx(true),
mNrPoints(nrPoints),
mTimeStart(ts),
mTimeEnd(te),
mJobStartedCB(f1),
mJobFinishedCB(f2),
mUserData(userData),
RoadRunnerThread()
{
    if(rri)
    {
        addJob(rri);
    }

    if(autoStart && rri != NULL)
    {
        start();
    }
}

SimulateThread::~SimulateThread()
{}

void SimulateThread::addJob(RoadRunner* rr)
{
    //getMutex
    Mutex::ScopedLock lock(mJobsMutex);
    if(mSimulateEx)
    {
        SimulateOptions opt;
        opt.start = mTimeStart;
        opt.duration = mTimeEnd - mTimeStart;
        opt.steps = mNrPoints -1;
        rr->setSimulateOptions(opt);
    }
    mJobs.push_back(rr);
    mJobsCondition.signal();    //Tell the thread its time to go to work
}

bool SimulateThread::isAnyWorking()
{
    bool result = false;
    Mutex::ScopedLock lock(mNrOfWorkersMutex);
    return mNrOfWorkers > 0 ? true : false;
}

bool SimulateThread::isWorking()
{
    return mIsWorking;
}

void SimulateThread::worker()
{
    mWasStarted = true;
    mIsWorking  = true;

    if(mJobStartedCB)
    {
        mJobStartedCB(mUserData);
    }
    Mutex::ScopedLock lock(mNrOfWorkersMutex);
    mNrOfWorkers++;
    mNrOfWorkersMutex.unlock();

    RoadRunner *rri = NULL;
    //////////////////////////////////
    while(!mIsTimeToDie)
    {
        {    //Scope for the mutex lock...
            Mutex::ScopedLock lock(mJobsMutex);
            if(mJobs.size() == 0 )//|| mIsTimeToDie)
            {
                break;    //ends the life of the thread..
            }
                rri = mJobs.front();
                mJobs.pop_front();
         }        //Causes the scoped lock to unlock

        //Do the job
        if(rri)
        {
            RRPLOG(lInfo)<<"Simulating RR instance: "<<rri->getInstanceID();

            if(!rri->simulate())
            {
                RRPLOG(lError)<<"Failed simulating instance: "<<rri->getInstanceID();
            }
        }
        else
        {
            RRPLOG(lError)<<"Null job pointer...!";
        }
    }

    RRPLOG(lDebug)<<"Exiting Simulate thread: "<<mThread.id();

    mIsWorking  = false;
    Mutex::ScopedLock lock2(mNrOfWorkersMutex);
    mNrOfWorkers--;

    if(mJobFinishedCB)
    {
        mJobFinishedCB(mUserData);
    }
}

void SimulateThread::signalExit()
{
    mJobsCondition.signal();
}

unsigned int SimulateThread::getNrOfJobsInQueue()
{
    Mutex::ScopedLock lock(mJobsMutex);
    return mJobs.size();
}

void SimulateThread::signalAll()
{
    mJobsCondition.broadcast();
}

}

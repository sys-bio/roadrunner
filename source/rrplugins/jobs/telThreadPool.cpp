#pragma hdrstop
#include "../../rrLogger.h"
#include "telThreadPool.h"
//---------------------------------------------------------------------------

namespace rr
{
//using namespace rr;
ThreadPool::ThreadPool()
{}

ThreadPool::~ThreadPool()
{

}

void ThreadPool::addJob(RoadRunner* rri)
{
    //We add jobs to the threads (static) queue. Means that at least one thread has to exist
    if(mThreads.size() < 1)
    {
        return ;
    }
    else
    {
        if(mThreads.front() != NULL)
        {
            mThreads.front()->addJob(rri);
        }
    }
}

void ThreadPool::start()
{
    //We add jobs to the threads (static) queue. Means that at least one thread has to exist
    if(mThreads.size() < 1)
    {
        return ;
    }
    else
    {
        list<RoadRunnerThread*>::iterator    iter;
        for(iter = mThreads.begin(); iter != mThreads.end(); iter++)
        {
            RoadRunnerThread* t = (*iter);
            if(t != NULL)
            {
                if(!t->isActive())
                {
                    t->start();
                }
                else
                {
                    Log(lError)<<"Tried to start an active thread";
                }
            }
        }
    }
}

bool ThreadPool::isJobQueueEmpty()
{
    if(mThreads.front() != NULL)
    {
        bool val = mThreads.front()->getNrOfJobsInQueue() > 0 ? false : true;
        if(val == true)
        {
            Log(lInfo)<<"Job queue is empty!";
        }
        return val;
    }
    return true;
}

int ThreadPool::getNumberOfRemainingJobs()
{
    if(mThreads.front() != NULL)
    {
        int val = mThreads.front()->getNrOfJobsInQueue();
        return val;
    }
    return -1;
}

bool ThreadPool::isWorking()
{
    if(mThreads.size())
    {
        //Go trought the threads and see if there is any activity
        list<RoadRunnerThread*>::iterator    iter;

        for(iter = mThreads.begin(); iter != mThreads.end(); iter++)
        {
            RoadRunnerThread* aThread = (*iter);
            if(aThread->isActive())
            {
                return true;
            }
        }
    }
    return false;
}

void ThreadPool::exitAll()
{
    //Send each thread a time to die message
    list<RoadRunnerThread*>::iterator    iter;

    for(iter = mThreads.begin(); iter != mThreads.end(); iter++)
    {
        if((*iter) != NULL)
        {
            (*iter)->exit();
        }
    }

    iter = mThreads.begin();
    if((*iter) != NULL)
    {
        (*iter)->signalAll();
    }
}

void ThreadPool::waitForStart()
{
//    while(isWorking())
//    {
//        Poco::Thread::sleep(50);
//    };
}

void ThreadPool::waitForFinish()
{
    //This could be checked in a thread, and using a condition Variable
    while(isWorking())
    {
        Poco::Thread::sleep(5);
    };
}


}

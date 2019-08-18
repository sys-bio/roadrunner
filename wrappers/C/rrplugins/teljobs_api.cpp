#pragma hdrstop
#include "teljobs_api.h"
#include <string>
#include <iostream>
#include <sstream>
#include <fstream>
#include "rrRoadRunner.h"
#include "rrException.h"
#include "rrplugins/jobs/telLoadModel.h"
#include "rrplugins/jobs/telLoadModelFromFile.h"
#include "rrplugins/jobs/telSimulate.h"
#include "rrplugins/jobs/telLoadModelThread.h"
#include "rrplugins/jobs/telSimulateThread.h"
#include "rrUtils.h"
#include "telplugins_c_api.h"
#include "telplugins_cpp_support.h"     // Need to include this before the support header..
#include "tel_macros.h"
#include "telRoadRunnerList.h"
//---------------------------------------------------------------------------

namespace tlpc
{
using namespace std;
using namespace rr;


TELHandle tlp_cc loadSBMLFromFileJob(tlpc::TELHandle handle, const char* fileName)
{
    start_try
        //Check if file exists first
        if(!fileExists(fileName))
        {
            stringstream msg;
            msg<<"The file "<<fileName<<" do not exist";
            setError(msg.str());
            return NULL;
        }

        //RoadRunner* rr = tlp::castToRoadRunner(rrHandle);
        RoadRunner* rr = castHandle<RoadRunner>(handle, __FUNC__);
        LoadModelThread* loadThread = new LoadModelThread(fileName);

        if(!loadThread)
        {
            tlpc::setError("Failed to create a LoadModel Thread");
        }
        loadThread->addJob(rr);
        loadThread->start();
        return loadThread;
    catch_ptr_macro
}

TELHandle tlp_cc loadSBMLJob(tlpc::TELHandle handle, const char* sbml)
{
    start_try
        RoadRunner* rr = castHandle<RoadRunner>(handle, __FUNC__);
        LoadModelThread* loadThread = new LoadModelThread();
        loadThread->setSBML(sbml);

        if(!loadThread)
        {
            tlpc::setError("Failed to create a LoadModel Thread");
        }
        loadThread->addJob(rr);
        loadThread->start();
        return loadThread;
    catch_ptr_macro
}

TELHandle tlp_cc loadSBMLFromFileJobEx(tlpc::TELHandle handle, const char* fileName, bool recompile)
{
    start_try
        //Check if file exists first
        if(!fileExists(fileName))
        {
            stringstream msg;
            msg<<"The file "<<fileName<<" do not exist";
            tlpc::setError(msg.str());
            return NULL;
        }

        RoadRunner* rr = castHandle<RoadRunner>(handle, __FUNC__);
        LoadModelThread* loadThread = new LoadModelThread(fileName, recompile);

        if(!loadThread)
        {
            tlpc::setError("Failed to create a LoadModel Thread");
        }

        loadThread->addJob(rr);
        loadThread->start();
        return loadThread;
    catch_ptr_macro
}

TELHandle tlp_cc loadSBMLJobEx(tlpc::TELHandle handle, const char* sbml, bool recompile)
{
    start_try
        RoadRunner* rr = castHandle<RoadRunner>(handle, __FUNC__);
        LoadModelThread* loadThread = new LoadModelThread("", recompile);
        loadThread->setSBML(sbml);

        if(!loadThread)
        {
            tlpc::setError("Failed to create a LoadModel Thread");
        }

        loadThread->addJob(rr);
        loadThread->start();
        return loadThread;
    catch_ptr_macro
}

TELHandle tlp_cc loadSBMLFromFileJobs(RRInstanceListPtr _handles, const char* fileName, int nrOfThreads)
{
    start_try
        //Check if file exists first
        if(!fileExists(fileName))
        {
            stringstream msg;
            msg<<"The file "<<fileName<<" do not exist";
            tlpc::setError(msg.str());
            return NULL;
        }

        RoadRunnerList *rrs = getRRList(_handles);
        LoadModelFromFile* tp = new LoadModelFromFile(*rrs, fileName, nrOfThreads);

        if(!tp)
        {
            tlpc::setError("Failed to create a LoadModel Thread Pool");
        }
        return tp;
    catch_ptr_macro
}

TELHandle tlp_cc loadSBMLJobs(RRInstanceListPtr _handles, const char* sbml, int nrOfThreads)
{
    start_try
        RoadRunnerList *rrs = getRRList(_handles);
        LoadModel* tp = new LoadModel(*rrs, sbml, nrOfThreads);


        if(!tp)
        {
            tlpc::setError("Failed to create a LoadModel Thread Pool");
        }
        return tp;
    catch_ptr_macro
}

TELHandle tlp_cc loadSBMLJobsEx(RRInstanceListPtr _handles, const char* sbml, int nrOfThreads, bool reCompile)
{
    start_try
        RoadRunnerList *rrs = getRRList(_handles);
        LoadModel* tp = new LoadModel(*rrs, sbml, nrOfThreads, reCompile);


        if(!tp)
        {
            tlpc::setError("Failed to create a LoadModel Thread Pool");
        }
        return tp;
    catch_ptr_macro
}

bool tlp_cc waitForJob(TELHandle handle)
{
    start_try
        RoadRunnerThread* aThread = (RoadRunnerThread*) handle;
        if(aThread)
        {
            aThread->waitForFinish();
            return true;
        }
        return false;
    catch_bool_macro
}

bool tlp_cc waitForJobs(TELHandle handle)
{
    start_try
        ThreadPool* aTP = (ThreadPool*) handle;
        if(aTP)
        {
            aTP->waitForFinish();
            return true;
        }
        return false;
    catch_bool_macro
}

bool tlp_cc isJobFinished(TELHandle handle)
{
    start_try
        RoadRunnerThread* aT = (RoadRunnerThread*) handle;
        if(aT)
        {
            return ! aT->isActive();
        }
        return false;
    catch_bool_macro
}

bool tlp_cc areJobsFinished(TELHandle handle)
{
    start_try
        ThreadPool* aTP = (ThreadPool*) handle;
        if(aTP)
        {
            return ! aTP->isWorking();
        }
        return false;
    catch_bool_macro
}

int tlp_cc getNumberOfRemainingJobs(TELHandle handle)
{
    start_try
        ThreadPool* aTP = (ThreadPool*) handle;
        if(aTP)
        {
            return aTP->getNumberOfRemainingJobs();
        }
        return -1;
    catch_int_macro
}

TELHandle tlp_cc simulateJob(tlpc::TELHandle rrHandle)
{
    start_try
        //RoadRunner *rr = tlp::castToRoadRunner(rrHandle);
        RoadRunner *aRR = castHandle<RoadRunner>(rrHandle, __FUNC__);
        SimulateThread *t = new SimulateThread(NULL, false);

        if(!t)
        {
            tlpc::setError("Failed to create a Simulate Thread Pool");
        }
        t->addJob(aRR);
        t->start();
        return t;
    catch_ptr_macro
}

TELHandle tlp_cc simulateJobEx(    tlpc::TELHandle rrHandle,
                                        double timeStart,
                                        double timeEnd,
                                        int numberOfPoints,
                                        callBackFunc f1,
                                        callBackFunc f2,
                                        void* userData)
{
    start_try
//        RoadRunner *rr = tlp::castToRoadRunner(rrHandle);
        RoadRunner *aRR = castHandle<RoadRunner>(rrHandle, __FUNC__);
        SimulateThread *t = new SimulateThread(NULL, timeStart, timeEnd, numberOfPoints, f1, f2, userData, false);

        if(!t)
        {
            tlpc::setError("Failed to create a Simulate Thread Pool");
        }
        t->addJob(aRR);
        t->start();
        return t;
    catch_ptr_macro
}

TELHandle tlp_cc simulateJobs(RRInstanceListPtr _handles, int nrOfThreads)
{
    start_try
        RoadRunnerList *rrs = getRRList(_handles);
        Simulate* tp = new Simulate(*rrs, nrOfThreads);

        if(!tp)
        {
            tlpc::setError("Failed to create a Simulate Thread Pool");
        }
        return tp;
    catch_ptr_macro
}

TELHandle tlp_cc simulateJobsEx(    RRInstanceListPtr _handles,
                                        int nrOfThreads,
                                        double timeStart,
                                        double timeEnd,
                                        int numPoints,
                                        callBackFunc fn1,
                                        callBackFunc fn2,
                                        void* userData)
{
    start_try
        RoadRunnerList *rrs = getRRList(_handles);

//        for(int i = 0; i < rrs->count(); i++)
//        {
//            (*rrs)[i]->setTimeStart(timeStart);
//            (*rrs)[i]->setTimeEnd(timeEnd);
//            (*rrs)[i]->setNumPoints(numPoints);
//        }

        Simulate* tp = new Simulate(*rrs, nrOfThreads);

        if(!tp)
        {
            tlpc::setError("Failed to create a Simulate Thread Pool");
        }
        return tp;
    catch_ptr_macro
}


bool tlp_cc freeJob(TELHandle aJob, RRJobType jt)
{
    start_try
        switch(jt)
        {
            case jtLoadModel:
                delete (LoadModelThread*) aJob;
            break;

            case jtSimulate:
               delete (SimulateThread*) aJob;
            break;
            default:
            {
               Log(rr::lError)<<"Failed to free job handle";
               return false;
            }
        }
        return true;
    catch_bool_macro
}

bool tlp_cc freeJobs(TELHandle handle)
{
    start_try
        ThreadPool* jobs = (ThreadPool*) handle;
        if(jobs)
        {
           delete jobs;
        }
        return true;
    catch_bool_macro
}

}

#ifndef telLoadModelThreadH
#define telLoadModelThreadH
#include <list>
#include "telJobsExporter.h"
#include "telRoadRunnerThread.h"
#include "Poco/Condition.h"
#include "../../rrConstants.h"
#include "../../rrRoadRunnerOptions.h"
//---------------------------------------------------------------------------

namespace rr
{
using std::list;

class JOBS_DECLSPEC LoadModelThread : public RoadRunnerThread
{
    protected:
        static list<RoadRunner*>        mJobs;
        static Poco::Mutex              mJobsMutex;
        static Poco::Condition          mJobsCondition;
        static Poco::Mutex              mNrOfWorkersMutex;
        static int                      mNrOfWorkers;        //incremented when entering worker function and decremented when exiting

    protected:
        string                          mModelFileName;
        string                          mSBML;                //If not loading from file
        rr::LoadSBMLOptions             mLoadSBMLOptions;

        void                            signalAll();
        void                            signalExit();

    public:
                                        LoadModelThread(const string& modelFile = rr::gEmptyString, bool recompileOnLoad = true, RoadRunner* rri = NULL, bool autoStart = false);
                                       ~LoadModelThread();
        void                            setSBML(const string& sbml);
        void                            worker();
        void                            addJob(RoadRunner* rr);
        unsigned int                    getNrOfJobsInQueue();
        bool                            isAnyWorking();
};

}
#endif

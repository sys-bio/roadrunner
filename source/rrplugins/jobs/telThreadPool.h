#ifndef telThreadPoolH
#define telThreadPoolH
#include <list>
#include <vector>
#include "telRoadRunnerThread.h"
//---------------------------------------------------------------------------

namespace rr
{
using namespace std;

class JOBS_DECLSPEC ThreadPool
{
    protected:
        list<RoadRunnerThread*>         mThreads;

    public:
                                        ThreadPool();
        virtual                        ~ThreadPool();
        void                            addJob(RoadRunner* rri);
        bool                            isJobQueueEmpty();
        int                             getNumberOfRemainingJobs();
        bool                            isWorking();
        void                            start();
        void                            exitAll();
        void                            waitForStart();
        void                            waitForFinish();
};

}
#endif

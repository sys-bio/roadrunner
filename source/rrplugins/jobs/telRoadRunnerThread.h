#ifndef telRoadRunnerThreadH
#define telRoadRunnerThreadH
#include <string>
#include "Poco/Thread.h"
#include "Poco/Runnable.h"
#include "telJobsExporter.h"
//---------------------------------------------------------------------------

namespace rr
{
class RoadRunner;
}

namespace rr
{
using std::string;
using rr::RoadRunner;


class JOBS_DECLSPEC RoadRunnerThread : public Poco::Runnable
{
    protected:
        Poco::Thread                        mThread;
        bool                                mIsTimeToDie;
        bool                                mWasStarted;
        bool                                mIsWorking;

    public:
                                            RoadRunnerThread();
        virtual                            ~RoadRunnerThread();

        void                                setName(const string& name);
        string                              getName();
        void                                join();
        bool                                isActive();

        void                                start(RoadRunner* instance = NULL);
        virtual void                        run();
        void                                exit();    //Tell the thread to die
        void                                waitForFinish();
        void                                waitForStart();

        //Pure virtuals
        virtual void                        worker() = 0;
        virtual void                        addJob(RoadRunner* instance) = 0;
        virtual unsigned int                getNrOfJobsInQueue() = 0;

        //Useful individual thread states
        bool                                wasStarted();
        bool                                isWorking();
        bool                                didFinish();

        //When we are in a pool of threads...
        virtual bool                        isAnyWorking() = 0;

        //Todo: Gotta refine the exiting of threads later...
        virtual void                        signalExit() = 0;
        virtual void                        signalAll() = 0;
};

}
#endif

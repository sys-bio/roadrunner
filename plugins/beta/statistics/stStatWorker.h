#ifndef stStatWorkerH
#define stStatWorkerH
#include "Poco/Thread.h"
#include "Poco/Runnable.h"
#include "stStatUtils.h"
#include "telTelluriumData.h"
//---------------------------------------------------------------------------


class Stat;

using rr::TelluriumData;

class stStatWorker : public Poco::Runnable
{
    friend class Stat;

    protected:
        Poco::Thread                mThread;

        Stat&                       mTheHost;
        bool                        setupRoadRunner();
        bool                        setup();
        void                        createModelData(TelluriumData* data);
        void                        createResidualsData(TelluriumData* data);

        void                        workerStarted();
        void                        workerFinished();

    public:
                                    stStatWorker(Stat& host);
        void                        start(bool runInThread = true);
        void                        run();
        bool                        isRunning() const;
};

#endif

#ifndef csChiWorkerH
#define csChiWorkerH
#include "Poco/Thread.h"
#include "Poco/Runnable.h"
#include "telTelluriumData.h"
//---------------------------------------------------------------------------



namespace cs_ChiSquare {
    using tlp::TelluriumData;
    class ChiSquare;
    class ChiWorker : public Poco::Runnable
    {
        friend class ChiSquare;

    protected:
        Poco::Thread                mThread;

        ChiSquare&                  mTheHost;

        void                        workerStarted();
        void                        workerFinished();

    public:
                                    ChiWorker(ChiSquare& host);
        void                        start(bool runInThread = true);
        void                        run();
        bool                        isRunning() const;
    };
}
#endif

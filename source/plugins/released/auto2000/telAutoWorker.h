#ifndef telAutoWorkerH
#define telAutoWorkerH
#include "Poco/Thread.h"
#include "Poco/Runnable.h"
#include "rrRoadRunner.h"
#include "rrOSSpecifics.h"
#include "telTelluriumData.h"
#include "auto_utils.h"
#include "telAutoDataParser.h"
//---------------------------------------------------------------------------

namespace telauto
{
class AutoTellurimInterface;
}

class AutoPlugin;
using rr::TelluriumData;

class AutoWorker : public Poco::Runnable
{
    public:
                                                    AutoWorker(AutoPlugin& host);
                                                   ~AutoWorker();
        bool                                        start(bool runInThread = false);
        void                                        run();
        bool                                        isRunning() const;

    protected:
        Poco::Thread                                mThread;
        AutoPlugin&                                 mTheHost;
        telauto::AutoTellurimInterface&             mRRAuto;        //This class communicate with auto

        //Auto Data parser
        AutoDataParser                              mAutoDataParser;
        bool                                        removeTempFiles(const string& fldr);
        bool                                        setupAuto();
};

#endif

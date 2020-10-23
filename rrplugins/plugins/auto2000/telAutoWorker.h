#ifndef telAutoWorkerH
#define telAutoWorkerH
#include "Poco/Thread.h"
#include "Poco/Runnable.h"
#include "../../../source/rrOSSpecifics.h"
#include "telTelluriumData.h"
#include "auto_utils.h"
#include "rrplugins/core/tel_api.h"
#include "telAutoDataParser.h"
#include "../../../wrappers/C/rrc_types.h"
//---------------------------------------------------------------------------

namespace telauto
{
class AutoTellurimInterface;
}

class AutoPlugin;
using tlp::TelluriumData;

class AutoWorker : public Poco::Runnable
{
    public:
                                                    AutoWorker(AutoPlugin& host);
                                                   ~AutoWorker();
        bool                                        start(bool runInThread = false);
        void                                        run();
        bool                                        isRunning() const;
        void                                        assignRoadRunner(rrc::RRHandle _rrHandle,rrc::THostInterface* tHostInterface);
        rrc::THostInterface*                        mhostInterface;
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

#ifndef bsWorkerH
#define bsWorkerH
#include <vector>
#include "Poco/Thread.h"
#include "Poco/Runnable.h"

#include "telPluginManager.h"
#include "telplugins_c_api.h"
#include "telTelluriumData.h"
#include "telRandom.h"
#include "telProperties.h"
//---------------------------------------------------------------------------

class MonteCarlo;
using rr::TelluriumData;
using rr::Properties;
using tlpc::TELHandle;

class bsWorker : public Poco::Runnable
{
    friend class MonteCarlo;

    public:
                                        bsWorker(MonteCarlo& plugin);
        void                            start(bool runInThread = true);
        void                            run();
        bool                            isRunning() const;

    protected:
        Poco::Thread                    mThread;
        MonteCarlo&                     mParent;

        bool                            setup();

        void                            workerStarted();
        void                            workerProgress();
        void                            workerFinished();


    private:
        const rr::PluginManager*       mPM;
        rr::Random                     mRandom;

        TELHandle                       mMinimizerPlugin;
        std::vector<double>             mResiduals;
        std::vector<TelluriumData*>     mMCDataSets;
        std::vector<Properties>         mMCParameters;

        void                            reset();

        bool                            createInitialResiduals();
        bool                            createMonteCarloDataSets();
        Properties                      getParameters(TelluriumData* data);

};

#endif

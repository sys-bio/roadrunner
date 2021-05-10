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

namespace bsmc
{
    class MonteCarlo;
    using tlp::TelluriumData;
    using tlp::Properties;
    using tlpc::TELHandle;

    class bsWorker : public Poco::Runnable
    {
        friend class MonteCarlo;

    public:
        bsWorker(MonteCarlo& plugin);
        ~bsWorker();
        void                            start(bool runInThread = true);
        void                            run();
        bool                            isRunning() const;

    protected:
        Poco::Thread                    mThread;
        MonteCarlo& mParent;

        bool                            setup();

        void                            workerStarted();
        void                            workerProgress();
        void                            workerFinished();


    private:
        tlp::Random*                    mRandom;

        tlp::Plugin*                    mMinimizerPlugin;
        std::vector<double>             mResiduals;
        std::vector<TelluriumData*>     mMCDataSets;
        std::vector<Properties>         mMCParameters;

        void                            reset();

        bool                            createInitialResiduals();
        bool                            createMonteCarloDataSets();
        Properties                      getParameters(TelluriumData* data);

    };

}
#endif

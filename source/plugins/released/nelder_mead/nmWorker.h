#ifndef nmWorkerH
#define nmWorkerH
#include <vector>
#include "Poco/Thread.h"
#include "Poco/Runnable.h"
#include "rrRoadRunner.h"
#include "nmUtils.h"
#include "telTelluriumData.h"
#include "telProperties.h"
//---------------------------------------------------------------------------

using std::vector;
using rr::TelluriumData;
using rr::Properties;

class NelderMead;

class nmWorker : public Poco::Runnable
{
    friend class NelderMead;

    public:
                                    nmWorker(NelderMead& host);
        void                        start(bool runInThread = true);
        void                        run();
        bool                        isRunning() const;

    protected:
        Poco::Thread                mThread;
        NelderMead&                 mHost;
        bool                        setupRoadRunner();

        void                        createModelData(TelluriumData* data);
        void                        createResidualsData(TelluriumData* data);
        void                        workerStarted();
        void                        workerFinished();

        void                        postFittingWork();
        void                        calculateChiSquare();
        void                        calculateHessian();
        void                        calculateCovariance();
        void                        calculateConfidenceLimits();
        double                      getChi(const Properties& parameters);
        int                         getNumberOfParameters();

};

#endif

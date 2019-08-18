#ifndef deWorkerH
#define deWorkerH
#include <vector>
#include "Poco/Thread.h"
#include "Poco/Runnable.h"
#include "rrRoadRunner.h"
#include "deUtils.h"
#include "telTelluriumData.h"
#include "telProperties.h"
//---------------------------------------------------------------------------

using std::vector;
using rr::TelluriumData;
using rr::Properties;

class DifferentialEvolution;

class deWorker : public Poco::Runnable
{
    friend class DifferentialEvolution;

    public:
                                    deWorker(DifferentialEvolution& host);
        void                        start(bool runInThread = true);
        void                        run();
        bool                        isRunning() const;

    protected:
        Poco::Thread                mThread;
        DifferentialEvolution&                 mHost;
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

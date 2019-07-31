#ifndef lmWorkerH
#define lmWorkerH
#include <vector>
#include "Poco/Thread.h"
#include "Poco/Runnable.h"
#include "rrRoadRunner.h"
#include "lmUtils.h"
#include "../common/telTelluriumData.h"
#include "telProperties.h"
//---------------------------------------------------------------------------

namespace lmfit
{

using std::vector;
class LM;
using rr::TelluriumData;
using rr::Properties;

class lmWorker : public Poco::Runnable
{
    friend class LM;

    protected:
        rr::RoadRunner             *mRRI;

        lmDataStructure             mLMData;        //LevenbergMarq.. data structure
        Poco::Thread                mThread;

        LM&                         mTheHost;
        bool                        setupRoadRunner();
        bool                        setup();
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

    public:
                                    lmWorker(LM& host);
        void                        start(bool runInThread = true);
        void                        run();
        bool                        isRunning() const;
};

}
#endif

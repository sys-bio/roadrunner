#ifndef bsMonteCarloH
#define bsMonteCarloH
#include <vector>
#include "telProperty.h"
#include "telCPPPlugin.h"
#include "telTelluriumData.h"
#include "bsWorker.h"
#include "Poco/ClassLibrary.h"

namespace bsmc
{
    //---------------------------------------------------------------------------
    using namespace tlp;
    using std::string;

    class MonteCarlo : public CPPPlugin
    {
        friend class bsWorker;

    protected:
        Property<string>                        mSBML;                    //This is the model
        Property<TelluriumData>				    mExperimentalData;
        Property<Properties>                    mInputParameterList;      //Parameters to fit
        Property<Properties>                    mCurrentParameters;       //This property can be checked in OnProgress event
        Property<TelluriumData>                 mMonteCarloParameters;    //Obtained parameter values from each MC run
        Property<Properties>                    mMeans;                   //Final mean values for each parameter
        Property<Properties>                    mConfidenceIntervals;     //Final Confidence Intervals for each parameter
        Property<Properties>                    mPercentiles;             //Final percentiles for each parameter
        Property<StringList>                    mExperimentalDataSelectionList;
        Property<StringList>                    mModelDataSelectionList;
        Property<int>                           mNrOfMCRuns;
        Property<string>                        mMinimizerPlugin;
        Property<unsigned long>                 mSeed;

        //Utility functions for the thread
        string                                  getSBML();

    protected:

        //The worker is doing the work
        bsWorker                                mWorker;

    public:
        MonteCarlo();
        ~MonteCarlo();

        bool                                    execute(bool inThread = false);
        string                                  getResult();
        bool                                    resetPlugin();
        string                                  getImplementationLanguage();
        string                                  getStatus();
        bool                                    isWorking() const;

        unsigned char* getManualAsPDF() const;
        size_t                                  getPDFManualByteSize();
        tlp::StringList                         getExperimentalDataSelectionList();
        void                                    assignPropertyDescriptions();
    };

#ifdef EXPORT_MONTE_CARLO_BS
    extern "C"
    {
        TLP_DS MonteCarlo* plugins_cc    createPlugin();
        TLP_DS const char* plugins_cc    getImplementationLanguage();
        TLP_DS void plugins_cc setHostInterface(rrc::THostInterface* _hostInterface);
        TLP_DS void plugins_cc setPluginManager(TELHandle manager);
    }
#endif

}


#endif
